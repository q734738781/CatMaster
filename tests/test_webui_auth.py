from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from catmaster.runtime.self_evolution import (
    LearningCandidate,
    SelfEvolutionStore,
)
from catmaster.runtime.self_evolution.storage import hash_tree, utc_now
from catmaster.tools.base import system_root
from catmaster.webui.auth import SESSION_COOKIE_NAME, session_cookie_name
from catmaster.webui.server import create_app
from catmaster.webui.thread_models import MessagePart, ThreadMessage, ToolCallPart
from catmaster.webui.thread_store import ThreadStore


def _captcha_answer(question: str) -> str:
    numbers = [int(value) for value in re.findall(r"\d+", question)]
    return str(sum(numbers))


def _register(client: TestClient, username: str, password: str = "correct-password-123") -> dict:
    captcha = client.get("/api/auth/captcha")
    assert captcha.status_code == 200
    captcha_payload = captcha.json()
    response = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "captcha_id": captcha_payload["captcha_id"],
            "captcha_answer": _captcha_answer(captcha_payload["question"]),
        },
    )
    assert response.status_code == 200
    return response.json()


def test_login_mode_requires_authentication_for_webui_api(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is True
    assert status.json()["authenticated"] is False
    assert status.json()["registration_enabled"] is True
    assert status.json()["self_evolution_available"] is True

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 401


def test_register_hashes_password_and_bootstraps_locked_user_root(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    payload = _register(client, "Alice_1")

    assert payload["authenticated"] is True
    assert payload["username"] == "alice_1"
    assert SESSION_COOKIE_NAME in client.cookies

    with sqlite3.connect(str(tmp_path / ".webui_auth" / "auth.sqlite")) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username = ?", ("alice_1",)).fetchone()
    assert row is not None
    password_hash = str(row[0])
    assert password_hash.startswith("pbkdf2_sha256$")
    assert password_hash != "correct-password-123"

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    boot_payload = bootstrap.json()
    user_root = tmp_path.resolve() / "users" / "alice_1"
    assert "workspace_root" not in boot_payload
    assert boot_payload["workspace_root_locked"] is True
    assert boot_payload["workspace_name"] == ""
    assert boot_payload["workspaces"] == []
    assert boot_payload["auth"]["username"] == "alice_1"
    assert not (user_root / "default").exists()

    outside = tmp_path / "outside-root"
    refresh = client.post(
        f"/api/session/{boot_payload['ctx']}/workspace/refresh",
        json={"root_path": str(outside), "workspace": "default"},
    )
    assert refresh.status_code == 404
    assert not outside.exists()
    assert not (user_root / "default").exists()
    created = client.post(f"/api/session/{boot_payload['ctx']}/workspace/create", json={"workspace": "project"})
    assert created.status_code == 200
    refresh = client.post(
        f"/api/session/{boot_payload['ctx']}/workspace/refresh",
        json={"root_path": str(outside), "workspace": "project"},
    )
    assert refresh.status_code == 200
    assert "workspace_root" not in refresh.json()
    assert refresh.json()["workspace_name"] == "project"
    assert (user_root / "project" / "files").is_dir()
    assert not outside.exists()


def test_instance_scoped_cookies_keep_two_login_instances_independent(tmp_path: Path) -> None:
    first_app = create_app(
        project_space_root=str(tmp_path / "first"),
        instance_id="7970",
    )
    second_app = create_app(
        project_space_root=str(tmp_path / "second"),
        instance_id="7990",
    )
    first = TestClient(first_app, base_url="http://catmaster.local:7970")
    second = TestClient(second_app, base_url="http://catmaster.local:7990")

    _register(first, "alice")
    _register(second, "bob")
    first_cookie = session_cookie_name("7970")
    second_cookie = session_cookie_name("7990")
    first_token = first.cookies.get(first_cookie)
    second_token = second.cookies.get(second_cookie)

    assert first_cookie != second_cookie
    assert first_token
    assert second_token
    combined_cookie_header = (
        f"{first_cookie}={first_token}; {second_cookie}={second_token}"
    )
    first_bootstrap = TestClient(
        first_app,
        base_url="http://catmaster.local:7970",
    ).get("/api/bootstrap", headers={"cookie": combined_cookie_header})
    second_bootstrap = TestClient(
        second_app,
        base_url="http://catmaster.local:7990",
    ).get("/api/bootstrap", headers={"cookie": combined_cookie_header})

    assert first_bootstrap.status_code == 200
    assert first_bootstrap.json()["auth"]["username"] == "alice"
    assert second_bootstrap.status_code == 200
    assert second_bootstrap.json()["auth"]["username"] == "bob"

    second_logout = TestClient(
        second_app,
        base_url="http://catmaster.local:7990",
    ).post("/api/auth/logout", headers={"cookie": combined_cookie_header})
    assert second_logout.status_code == 200
    assert second_logout.headers["set-cookie"].startswith(f"{second_cookie}=")
    first_after_second_logout = TestClient(
        first_app,
        base_url="http://catmaster.local:7970",
    ).get("/api/bootstrap", headers={"cookie": combined_cookie_header})
    assert first_after_second_logout.status_code == 200
    assert first_after_second_logout.json()["auth"]["username"] == "alice"


def test_authenticated_tool_part_fields_are_exactly_pageable_and_cursor_bound(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    assert client.get("/api/bootstrap").status_code == 200
    workspace = tmp_path / "users" / "alice" / "default"
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread_id = store.create_thread(title="Large tool fields").thread_id
    long_input = "input-segment-" * 8_000
    long_output = "output-segment-" * 8_000
    tool_input = {
        "query": long_input,
        "api_key": "secret-input-value",
        "nested": {"password": "secret-password", "visible": "kept"},
    }
    tool_output = {
        "result": long_output,
        "authorization": "Bearer secret-output-value",
        "status": "complete",
    }
    store.append_message(
        ThreadMessage(
            id="msg_large_tool_fields",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                ToolCallPart(
                    id="part_large_tool_fields",
                    tool_call_id="call_large_tool_fields",
                    tool="large_tool",
                    input=tool_input,
                    output=tool_output,
                    status="completed",
                )
            ],
        )
    )
    route = (
        f"/api/threads/{thread_id}/messages/msg_large_tool_fields/"
        "parts/part_large_tool_fields/content"
    )
    expected = {
        "input": json.dumps(
            {
                "query": long_input,
                "api_key": "[REDACTED]",
                "nested": {"password": "[REDACTED]", "visible": "kept"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        "output": json.dumps(
            {
                "result": long_output,
                "authorization": "[REDACTED]",
                "status": "complete",
            },
            ensure_ascii=False,
            indent=2,
        ),
    }

    projected = client.get(f"/api/threads/{thread_id}/messages")
    assert projected.status_code == 200
    projected_part = projected.json()["messages"][0]["parts"][0]
    assert projected_part["input_ref"].endswith("/content?field=input")
    assert projected_part["output_ref"].endswith("/content?field=output")

    first_input_cursor = ""
    for field_name in ("input", "output"):
        chunks: list[str] = []
        cursor = ""
        expected_start = 0
        while True:
            response = client.get(
                route,
                params={"field": field_name, "cursor": cursor, "limit": 4_096},
            )
            assert response.status_code == 200
            payload = response.json()
            page = payload["page"]
            assert page["shown_count"] == expected_start + len(payload["text"])
            assert page["full_content_ref"].endswith(f"/content?field={field_name}")
            chunks.append(payload["text"])
            expected_start = page["shown_count"]
            if not cursor and field_name == "input":
                first_input_cursor = page["next_cursor"]
            if not page["truncated"]:
                break
            cursor = page["next_cursor"]
            assert cursor
        reconstructed = "".join(chunks)
        assert reconstructed == expected[field_name]
        assert "secret-input-value" not in reconstructed
        assert "secret-password" not in reconstructed
        assert "secret-output-value" not in reconstructed

    assert first_input_cursor
    wrong_field = client.get(
        route,
        params={"field": "output", "cursor": first_input_cursor, "limit": 4_096},
    )
    assert wrong_field.status_code == 400
    store.update_part(
        thread_id,
        "msg_large_tool_fields",
        "part_large_tool_fields",
        input={**tool_input, "query": long_input + "changed"},
    )
    stale = client.get(
        route,
        params={"field": "input", "cursor": first_input_cursor, "limit": 4_096},
    )
    assert stale.status_code == 400

    anonymous = TestClient(app)
    assert anonymous.get(route, params={"field": "input"}).status_code == 401


def test_authenticated_skill_canary_records_webui_actor_and_note(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap").json()
    workspace = tmp_path / "users" / "alice" / "default"
    store = SelfEvolutionStore(workspace, project_id="default")
    candidate = LearningCandidate(
        candidate_id="sec_review_preview",
        project_id="default",
        run_id="run-one",
        thread_id="thread-one",
        action="skill",
        status="review",
        group="materials_worker",
        name="human-review-test",
        review={
            "recommendation": "approve",
            "summary": "Add one bounded workspace workflow.",
            "change_points": [],
            "scope_assessment": "Workspace only.",
            "proportionality_assessment": {"status": "pass", "explanation": "Bounded."},
            "concerns": [],
            "human_checks": [],
            "rationale": "Independent review completed.",
        },
        created_at=utc_now(),
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    proposed = root / "proposed" / candidate.group / candidate.name
    proposed.mkdir(parents=True)
    (proposed / "SKILL.md").write_text(
        """---
name: human-review-test
description: Use this skill to test an authenticated bounded promotion.
license: project-local
compatibility: local
---
# Human review test

## Overview

Bounded authenticated promotion test.

## Quick Start

Read this workflow before use.

## Workflow

1. Follow the bounded workflow.

## Method-critical defaults

Keep the workflow workspace-scoped.

## Output Contract

Return a concise result.

## References

No external references.
""",
        encoding="utf-8",
    )
    candidate.bundle_hash = hash_tree(proposed)
    store.write_candidate(candidate)

    response = client.post(
        (
            f"/api/session/{bootstrap['ctx']}/self-evolution/candidates/"
            f"{candidate.candidate_id}/revisions/1/start-canary"
        ),
        json={
            "project_space": "default",
            "scope_kind": "run",
            "scope_id": "run-one",
            "rationale": "I inspected the exact diff.",
        },
    )

    assert response.status_code == 200
    assert response.json()["candidate"]["status"] == "canary"
    audit_events = [json.loads(line) for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert audit_events[-1]["event"] == "canary_started"
    assert audit_events[-1]["actor"] == "alice"
    assert audit_events[-1]["candidate_hash"] == candidate.bundle_hash
    assert audit_events[-1]["rationale"] == "I inspected the exact diff."


def test_authenticated_effective_skills_manager_pages_and_mutates_exact_state(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap").json()
    assert client.post(f"/api/session/{bootstrap['ctx']}/workspace/create", json={"workspace": "default"}).status_code == 200
    base = f"/api/session/{bootstrap['ctx']}/self-evolution"

    listing = client.get(
        f"{base}/skills",
        params={"project_space": "default", "limit": 200},
    )
    assert listing.status_code == 200
    catalog = listing.json()
    assert catalog["total_count"] >= len(catalog["skills"]) > 1
    assert catalog["activation"] == "next_run"
    skill = next(item for item in catalog["skills"] if item["enabled_control"])
    assert skill["selected_version"] == "base"
    assert skill["update_policy"] == "follow_auto"

    changed = client.patch(
        f"{base}/skills/state",
        json={
            "project_space": "default",
            "target": skill["target"],
            "expected_selected_version": "base",
            "enabled": False,
        },
    )
    assert changed.status_code == 200
    assert changed.json()["skill"]["enabled"] is False

    detail = client.get(
        f"{base}/skills/detail",
        params={"project_space": "default", "target": skill["target"]},
    )
    assert detail.status_code == 200
    exact = detail.json()["skill"]
    assert exact["enabled"] is False
    assert exact["versions"][-1]["version"] == "base"
    assert exact["history"][0]["source"] == "gui"
    assert exact["history"][0]["actor"] == "alice"
    assert exact["history"][0]["before"]["enabled"] is True
    assert exact["history"][0]["after"]["enabled"] is False

    stale = client.patch(
        f"{base}/skills/state",
        json={
            "project_space": "default",
            "target": skill["target"],
            "expected_selected_version": "stale",
            "enabled": True,
        },
    )
    assert stale.status_code == 409

    next_mode = "auto" if catalog["mode"] != "auto" else "observe"
    preview = client.post(
        f"{base}/settings/preview",
        json={"project_space": "default", "mode": next_mode},
    )
    assert preview.status_code == 200
    assert preview.json()["mode_before"] == catalog["mode"]
    assert preview.json()["mode_after"] == next_mode
    applied = client.patch(
        f"{base}/settings",
        json={
            "project_space": "default",
            "mode": next_mode,
            "expected_mode": catalog["mode"],
        },
    )
    assert applied.status_code == 200
    assert applied.json()["mode"] == next_mode

    anonymous = TestClient(app)
    assert anonymous.get(f"{base}/skills", params={"project_space": "default"}).status_code == 401


def test_authenticated_candidate_revision_files_page_complete_review_evidence(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap").json()
    workspace = tmp_path / "users" / "alice" / "default"
    store = SelfEvolutionStore(workspace, project_id="default")
    concerns = [
        {
            "concern": f"Concern {index}: " + ("complete concern evidence " * 20),
            "evidence_refs": [f"run:run-review#event:{index + 1}"],
        }
        for index in range(120)
    ]
    candidate = LearningCandidate(
        candidate_id="sec_complete_review_files",
        project_id="default",
        run_id="run-review",
        thread_id="thread-review",
        episode_id="episode-review",
        action="memory",
        route="workspace_preference",
        status="review",
        rationale="A complete review must remain reachable.",
        review={
            "recommendation": "needs_revision",
            "summary": "Review all concerns.",
            "concerns": concerns,
            "human_checks": [f"Check {index}" for index in range(120)],
        },
        revision=1,
        created_at=utc_now(),
    )
    store.write_candidate(candidate)
    store.write_revision_json(
        candidate.candidate_id,
        1,
        "proposal.json",
        {
            "rationale": "proposal rationale " * 4_000,
            "raw_response_ref": "proposer_response.txt",
        },
    )
    store.write_revision_text(
        candidate.candidate_id,
        1,
        "proposer_response.txt",
        "raw proposer evidence " * 6_000,
    )
    store.write_revision_text(
        candidate.candidate_id,
        1,
        "reviewer_response.txt",
        "raw reviewer evidence " * 6_000,
    )

    listing = client.get(
        f"/api/session/{bootstrap['ctx']}/self-evolution/candidates",
        params={"project_space": "default"},
    )
    assert listing.status_code == 200
    projected = next(
        item
        for item in listing.json()["candidates"]
        if item["candidate_id"] == candidate.candidate_id
    )
    assert {
        "candidate.json",
        "proposal.json",
        "proposer_response.txt",
        "review.json",
        "reviewer_response.txt",
    }.issubset(set(projected["revision_file_refs"]))

    route_base = (
        f"/api/session/{bootstrap['ctx']}/self-evolution/candidates/"
        f"{candidate.candidate_id}/revisions/1/files/content"
    )
    for file_name in ("review.json", "proposer_response.txt", "reviewer_response.txt"):
        route = projected["revision_file_refs"][file_name]
        cursor = ""
        chunks: list[str] = []
        expected_start = 0
        while True:
            response = client.get(
                route,
                params={
                    "project_space": "default",
                    "file": file_name,
                    "cursor": cursor,
                    "limit": 4_096,
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["page"]["range_start"] == expected_start
            chunks.append(payload["text"])
            expected_start = payload["page"]["range_end"]
            if not payload["page"]["truncated"]:
                break
            cursor = payload["page"]["next_cursor"]
            assert cursor
        expected = (
            store.revision_dir(candidate.candidate_id, 1) / file_name
        ).read_text(encoding="utf-8")
        assert "".join(chunks) == expected

    assert client.get(
        route_base,
        params={"project_space": "default", "file": "../candidate.json"},
    ).status_code == 400


def test_authenticated_user_can_retry_only_an_explicitly_selected_failed_job(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap").json()
    workspace = tmp_path / "users" / "alice" / "default"
    run_dir = system_root(workspace) / "runs" / "run-failed-reflection"
    run_dir.mkdir(parents=True)
    store = SelfEvolutionStore(workspace, project_id="default")
    queued = store.enqueue_job(
        trigger_kind="post_run",
        run_id="run-failed-reflection",
        run_dir=run_dir,
        thread_id="thread-failed-reflection",
        payload={"user_prompt": "Inspect the completed run."},
    )
    claimed = store.claim_jobs(owner="test-worker")[0]
    failed = store.finish_job(
        claimed,
        status="error",
        error="RuntimeError: reviewer unavailable",
        owner="test-worker",
    )
    assert failed.job_id == queued.job_id

    response = client.post(
        (
            f"/api/session/{bootstrap['ctx']}/self-evolution/jobs/"
            f"{failed.job_id}/retry"
        ),
        json={"project_space": "default"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    retry = payload["job"]
    assert retry["predecessor_job_id"] == failed.job_id
    assert retry["run_id"] == failed.run_id
    assert retry["title"] == "Selected job retry"
    persisted_retry = store.read_job(retry["job_id"])
    assert persisted_retry is not None
    assert persisted_retry.trigger_kind == "selected_retry"
    assert persisted_retry.run_dir == failed.run_dir
    unchanged = store.read_job(failed.job_id)
    assert unchanged is not None and unchanged.status == "error"
    assert unchanged.error == "RuntimeError: reviewer unavailable"

    duplicate = client.post(
        (
            f"/api/session/{bootstrap['ctx']}/self-evolution/jobs/"
            f"{failed.job_id}/retry"
        ),
        json={"project_space": "default"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["job"]["job_id"] == retry["job_id"]
    audit_events = [
        json.loads(line)
        for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert audit_events[-1]["event"] == "self_evolution_job_retry_queued"
    assert audit_events[-1]["actor"] == "alice"

    retry_claimed = store.claim_jobs(owner="retry-worker")[0]
    retry_failed = store.finish_job(
        retry_claimed,
        status="error",
        error="RuntimeError: retry also failed",
        owner="retry-worker",
    )
    chained = client.post(
        (
            f"/api/session/{bootstrap['ctx']}/self-evolution/jobs/"
            f"{retry_failed.job_id}/retry"
        ),
        json={"project_space": "default"},
    )
    assert chained.status_code == 200
    chained_job = store.read_job(chained.json()["job"]["job_id"])
    assert chained_job is not None
    assert chained_job.predecessor_job_id == retry_failed.job_id
    assert chained_job.payload["original_trigger_kind"] == "post_run"

    anonymous = TestClient(app)
    unauthorized = anonymous.post(
        (
            f"/api/session/{bootstrap['ctx']}/self-evolution/jobs/"
            f"{failed.job_id}/retry"
        ),
        json={"project_space": "default"},
    )
    assert unauthorized.status_code == 401


def test_thread_learn_entry_resolves_host_run_and_injects_authenticated_actor(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    workspace_name = bootstrap.json()["workspace_name"]
    workspace = tmp_path / "users" / "alice" / workspace_name
    thread_id = ThreadStore(
        workspace=workspace,
        workspace_id=workspace_name,
    ).create_thread(title="Evidence thread", entrypoint="research").thread_id
    run_id = "run_host_selected"
    run_dir = system_root(workspace) / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "webui_thread_id": thread_id,
                "entrypoint": "research",
                "status": "done",
                "user_prompt": "Keep future reports concise.",
            }
        ),
        encoding="utf-8",
    )
    ThreadStore(workspace=workspace, workspace_id=workspace_name).append_message(
        ThreadMessage(
            id="msg_assistant_result",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id="part_result",
                    type="text",
                    text="Completed.",
                    status="completed",
                )
            ],
            meta={"run_id": run_id},
        )
    )

    response = client.post(
        f"/api/threads/{thread_id}/self-evolution/learn",
        json={
            "note": "Use concise Chinese summaries for this workspace.",
            "run_id": "run_client_spoof",
            "run_dir": "/tmp/client-spoof",
            "thread_id": "thread-client-spoof",
            "actor": "mallory",
            "route_hint": "new_skill",
        },
    )

    assert response.status_code == 200
    store = SelfEvolutionStore(workspace, project_id=workspace_name)
    jobs = store.list_jobs(project_id=workspace_name)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.run_id == run_id
    assert Path(job.run_dir) == run_dir.resolve()
    assert job.thread_id == thread_id
    assert job.payload["actor"] == "alice"
    assert "route_hint" not in job.payload
    assert "episode_projection" not in job.payload
    events = [
        json.loads(line)
        for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "explicit_learn_queued"
    assert events[-1]["actor"] == "alice"


def test_candidate_revision_history_is_read_only_and_stale_actions_conflict(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap").json()
    workspace = tmp_path / "users" / "alice" / "default"
    store = SelfEvolutionStore(workspace, project_id="default")
    candidate_id = "sec_revision_history"

    first = LearningCandidate(
        candidate_id=candidate_id,
        project_id="default",
        run_id="run-one",
        thread_id="thread-one",
        action="memory",
        status="review",
        route="workspace_preference",
        rationale="Prefer concise reports.",
        evidence_ids=["obs-one"],
        revision=1,
        created_at=utc_now(),
    )
    root_one = store.reset_candidate_dir(candidate_id)
    (root_one / "current").mkdir()
    (root_one / "memories").mkdir()
    (root_one / "current/AGENTS.md").write_text(
        "# Memory\n\n- Use detailed reports.\n",
        encoding="utf-8",
    )
    (root_one / "memories/AGENTS.md").write_text(
        "# Memory\n\n- Use concise reports.\n",
        encoding="utf-8",
    )
    store.write_candidate(first)
    store.write_revision_json(
        candidate_id,
        1,
        "proposal.json",
        {"evidence_ids": ["obs-one"]},
    )

    second = LearningCandidate.from_dict(
        {
            **first.to_dict(),
            "status": "review",
            "rationale": "Prefer concise generated reports, except quotations.",
            "evidence_ids": ["obs-one", "obs-two"],
            "revision": 2,
        }
    )
    root_two = store.create_revision_dir(candidate_id, 2)
    (root_two / "current").mkdir()
    (root_two / "memories").mkdir()
    (root_two / "current/AGENTS.md").write_text(
        "# Memory\n\n- Use concise reports.\n",
        encoding="utf-8",
    )
    (root_two / "memories/AGENTS.md").write_text(
        "# Memory\n\n- Use concise generated reports except quotations.\n",
        encoding="utf-8",
    )
    store.write_candidate(second)
    store.write_revision_json(
        candidate_id,
        2,
        "proposal.json",
        {"evidence_ids": ["obs-one", "obs-two"]},
    )

    base = (
        f"/api/session/{bootstrap['ctx']}/self-evolution/candidates/"
        f"{candidate_id}/revisions"
    )
    historical = client.get(f"{base}/1", params={"project_space": "default"})
    current = client.get(f"{base}/2", params={"project_space": "default"})
    historical_diff = client.get(
        f"{base}/1/diff",
        params={"project_space": "default"},
    )
    current_diff = client.get(
        f"{base}/2/diff",
        params={"project_space": "default"},
    )
    stale_action = client.post(
        f"{base}/1/reject",
        json={
            "project_space": "default",
            "rationale": "Rejecting an old revision must be refused.",
        },
    )

    assert historical.status_code == 200
    assert historical.json()["candidate"]["revision"] == 1
    assert historical.json()["read_only"] is True
    assert historical.json()["current_revision"] == 2
    assert historical.json()["candidate"]["allowed_actions"] == []
    assert current.status_code == 200
    assert current.json()["read_only"] is False
    assert historical_diff.status_code == 200
    assert historical_diff.json()["read_only"] is True
    assert "Use detailed reports." in historical_diff.json()["diff"]
    assert current_diff.status_code == 200
    assert current_diff.json()["read_only"] is False
    assert stale_action.status_code == 409
    assert "newer revision" in stale_action.json()["detail"]


def test_same_ctx_is_isolated_between_authenticated_users(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    alice = TestClient(app)
    bob = TestClient(app)

    _register(alice, "alice")
    _register(bob, "bob")

    alice_boot = alice.get("/api/bootstrap", params={"ctx": "ctx_shared_001"})
    assert alice_boot.status_code == 200
    ctx = alice_boot.json()["ctx"]
    alice_create = alice.post(f"/api/session/{ctx}/workspace/create", json={"workspace": "private"})
    assert alice_create.status_code == 200
    assert alice_create.json()["ok"] is True

    bob_boot = bob.get("/api/bootstrap", params={"ctx": "ctx_shared_001"})
    assert bob_boot.status_code == 200
    bob_payload = bob_boot.json()

    alice_root = tmp_path.resolve() / "users" / "alice"
    bob_root = tmp_path.resolve() / "users" / "bob"
    assert "workspace_root" not in alice_boot.json()
    assert "workspace_root" not in bob_payload
    assert alice_boot.json()["workspace_root_locked"] is True
    assert bob_payload["workspace_root_locked"] is True
    assert (alice_root / "private").is_dir()
    assert not (bob_root / "private").exists()
    assert "private" not in {item["value"] for item in bob_payload["workspaces"]}


def test_disable_registration_rejects_signup_but_preserves_existing_login(tmp_path: Path) -> None:
    bootstrap_client = TestClient(create_app(project_space_root=str(tmp_path)))
    _register(bootstrap_client, "existing_user")

    client = TestClient(create_app(project_space_root=str(tmp_path), disable_registration=True))
    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is True
    assert status.json()["authenticated"] is False
    assert status.json()["registration_enabled"] is False
    assert status.json()["has_users"] is True

    captcha = client.get("/api/auth/captcha")
    assert captcha.status_code == 403
    assert captcha.json()["detail"] == "Registration is disabled."

    register = client.post(
        "/api/auth/register",
        json={
            "username": "blocked_user",
            "password": "correct-password-123",
            "captcha_id": "",
            "captcha_answer": "",
        },
    )
    assert register.status_code == 403
    assert register.json()["detail"] == "Registration is disabled."

    login = client.post(
        "/api/auth/login",
        json={"username": "existing_user", "password": "correct-password-123"},
    )
    assert login.status_code == 200
    assert login.json()["authenticated"] is True
    assert login.json()["registration_enabled"] is False
    assert client.get("/api/bootstrap").status_code == 200


def test_no_login_has_no_workspace_until_explicit_creation(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is False
    assert status.json()["authenticated"] is True
    assert status.json()["username"] == "admin"
    assert status.json()["registration_enabled"] is False
    assert status.json()["self_evolution_available"] is True

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert "workspace_root" not in payload
    assert payload["workspace_root_locked"] is True
    assert payload["workspace_name"] == ""
    assert payload["workspaces"] == []
    assert not (tmp_path / "admin").exists()
    assert client.post("/api/workspaces/admin/threads", json={}).status_code == 404
    created = client.post(f"/api/session/{payload['ctx']}/workspace/create", json={"workspace": "catalysts"})
    assert created.status_code == 200
    opened = client.get("/api/bootstrap", params={"project_space": "catalysts"}).json()
    assert opened["workspace_name"] == "catalysts"
    assert (tmp_path / "catalysts" / "files").is_dir()
    assert not (tmp_path / "admin").exists()
