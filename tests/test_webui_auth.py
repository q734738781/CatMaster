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
from catmaster.webui.auth import SESSION_COOKIE_NAME
from catmaster.webui.server import create_app
from catmaster.webui.thread_models import MessagePart, ThreadMessage
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
    assert boot_payload["workspace_name"] == "default"
    assert boot_payload["auth"]["username"] == "alice_1"
    assert (user_root / "default" / "files").is_dir()

    outside = tmp_path / "outside-root"
    refresh = client.post(
        f"/api/session/{boot_payload['ctx']}/workspace/refresh",
        json={"root_path": str(outside), "workspace": "default"},
    )
    assert refresh.status_code == 200
    assert "workspace_root" not in refresh.json()
    assert refresh.json()["workspace_name"] == "default"


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


def test_thread_learn_entry_resolves_host_run_and_injects_authenticated_actor(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    _register(client, "alice")
    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    workspace_name = bootstrap.json()["workspace_name"]
    thread_response = client.post(
        f"/api/workspaces/{workspace_name}/threads",
        json={"title": "Evidence thread", "entrypoint": "research"},
    )
    assert thread_response.status_code == 200
    thread_id = thread_response.json()["thread"]["thread_id"]
    workspace = tmp_path / "users" / "alice" / workspace_name
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


def test_no_login_uses_admin_workspace_without_auth_cookie(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is False
    assert status.json()["authenticated"] is True
    assert status.json()["username"] == "admin"
    assert status.json()["registration_enabled"] is False

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert "workspace_root" not in payload
    assert payload["workspace_root_locked"] is True
    assert payload["workspace_name"] == "admin"
    assert (tmp_path / "admin" / "files").is_dir()
