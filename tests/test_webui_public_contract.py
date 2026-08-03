from __future__ import annotations

import asyncio
import json
import os
import runpy
import threading
import time
import tracemalloc
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.runtime.self_evolution import SelfEvolutionStore
from catmaster.storage import workspace_db
from catmaster.storage.workspace_db import connect_workspace_db, workspace_database_path
from catmaster.tools.base import ensure_project_space_layout
from catmaster.specialists.streaming_runner import CatMasterStreamTranslator
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.projections.errors import project_error_part
from catmaster.webui.projections.messages import project_message
from catmaster.webui.projections.monitor import project_monitor_snapshot
from catmaster.webui.projections.self_evolution import (
    project_self_evolution_candidate,
)
from catmaster.webui.server import create_app
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import MessagePart, ThreadMessage, ToolCallPart
from catmaster.webui.thread_store import ThreadStore


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    return workspace


def _new_thread(client: TestClient, title: str = "Public contract") -> str:
    response = client.post("/api/workspaces/default/threads", json={"title": title})
    assert response.status_code == 200
    return str(response.json()["thread"]["thread_id"])


def _read_diagnostics_json(
    client: TestClient,
    url: str,
    *,
    limit: int = 64_000,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cursor = ""
    chunks: list[str] = []
    pages: list[dict[str, object]] = []
    expected_start = 0
    while True:
        response = client.get(
            url,
            params={"cursor": cursor, "limit": limit},
        )
        assert response.status_code == 200
        assert len(response.content) < 1_200_000
        payload = response.json()
        page = payload["page"]
        assert page["range_start"] == expected_start
        assert page["range_end"] - page["range_start"] == page["shown_count"]
        chunks.append(payload["content"])
        pages.append(page)
        expected_start = page["range_end"]
        if not page["truncated"]:
            break
        assert page["next_cursor"]
        repeated = client.get(
            url,
            params={"cursor": page["next_cursor"], "limit": limit},
        )
        assert repeated.status_code == 200
        cursor = page["next_cursor"]
    serialized = "".join(chunks)
    assert len(serialized) == pages[-1]["total_count"]
    return json.loads(serialized), pages


def test_ordinary_message_projection_hides_raw_and_diagnostics_recovers_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    monkeypatch.setenv("CATMASTER_WEBUI_DEVELOPER_DIAGNOSTICS", "1")
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    thread_id = _new_thread(client)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    store.append_message(
        ThreadMessage(
            id="msg_projection",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id="part_tool",
                    type="tool-call",
                    status="completed",
                    meta={
                        "tool": "remote_submission",
                        "input": {
                            "api_key": "secret-value",
                            "run_path": str(workspace / "metadata" / "runs" / "run_private"),
                            "task": "Relax the selected structure",
                        },
                        "output": {"status": "completed", "submission_hash": "private-hash"},
                        "agent_name": "materials_worker",
                        "agent_run_id": "task:surface_relaxation:runtime-1",
                    },
                )
            ],
            meta={"run_id": "run_private", "provider_payload": {"secret": "raw"}},
        )
    )

    ordinary = client.get(f"/api/threads/{thread_id}/messages")
    assert ordinary.status_code == 200
    ordinary_text = json.dumps(ordinary.json(), ensure_ascii=False)
    assert "secret-value" not in ordinary_text
    assert "private-hash" not in ordinary_text
    assert str(workspace) not in ordinary_text
    assert "provider_payload" not in ordinary_text
    assert "remote_submission" not in ordinary_text
    public_part = ordinary.json()["messages"][0]["parts"][0]
    assert public_part["title"] == "Materials · Remote submission"
    assert public_part["activity_group_title"] == "Materials"
    assert public_part["activity_group_id"].startswith("activity_")
    assert "task:surface_relaxation" not in ordinary_text

    diagnostics = client.get(
        f"/api/diagnostics/threads/{thread_id}/messages/msg_projection"
    )
    assert diagnostics.status_code == 200
    diagnostics_text = json.dumps(diagnostics.json(), ensure_ascii=False)
    assert "secret-value" in diagnostics_text
    assert "provider_payload" in diagnostics_text


def test_tool_projection_hides_truncated_json_and_opaque_agent_namespace() -> None:
    projected = project_message(
        ThreadMessage(
            id="msg_structured",
            thread_id="thread_structured",
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id="part_structured",
                    type="tool-call",
                    status="completed",
                    text='{"meta":{"count":9470},"results":[{"title":"paper"}]...[truncated]',
                    meta={
                        "tool": "acquire_literature_source",
                        "output": '{"meta":{"count":9470},"results":[{"title":"paper"}]...[truncated]',
                        "agent_name": "tools:da002fa7-6e8f-a97a-2045-fd9eb51d8b06",
                    },
                ),
                MessagePart(
                    id="part_plan",
                    type="tool-call",
                    status="completed",
                    meta={
                        "tool": "write_todos",
                        "input": {"todos": [{"content": "Read the evidence", "status": "pending"}]},
                        "agent_name": "tools:da002fa7-6e8f-a97a-2045-fd9eb51d8b06",
                    },
                ),
            ],
        )
    )

    serialized = projected.model_dump_json()
    assert '"title":"paper"' not in serialized
    assert "publication_year" not in serialized
    assert "da002fa7" not in serialized
    assert projected.parts[0].summary == "Structured results are available in details."
    assert projected.parts[1].title == "Specialist plan"


def test_reasoning_and_root_tools_share_public_activity_identity() -> None:
    projected = project_message(
        ThreadMessage(
            id="msg_root_activity",
            thread_id="thread_root_activity",
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id="part_reasoning",
                    type="reasoning",
                    status="completed",
                    text="Inspect the workspace first.",
                    meta={"source": "research_specialist"},
                ),
                MessagePart(
                    id="part_tool",
                    type="tool-call",
                    status="completed",
                    meta={
                        "tool": "read_file",
                        "agent_name": "research_specialist",
                        "agent_run_id": "agent:research_specialist",
                    },
                ),
            ],
        )
    )

    assert projected.parts[0].activity_group_title == "Research"
    assert projected.parts[0].activity_group_id == projected.parts[1].activity_group_id


def test_unknown_persisted_part_has_safe_fallback_and_raw_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    monkeypatch.setenv("CATMASTER_WEBUI_DEVELOPER_DIAGNOSTICS", "1")
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    thread_id = _new_thread(client)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    store.append_message(
        ThreadMessage(
            id="msg_unknown_part",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id="part_provider_future",
                    type="provider-future-payload",
                    status="completed",
                    text="LLM_RAW_RESPONSE",
                    meta={"engine_path": str(workspace / "metadata" / "private")},
                )
            ],
        )
    )

    ordinary = client.get(f"/api/threads/{thread_id}/messages")
    assert ordinary.status_code == 200
    part = ordinary.json()["messages"][0]["parts"][0]
    assert part["type"] == "unknown"
    assert part["title"] == "This activity cannot be displayed yet"
    assert "LLM_RAW_RESPONSE" not in json.dumps(part)
    assert str(workspace) not in json.dumps(part)

    diagnostics = client.get(
        f"/api/diagnostics/threads/{thread_id}/messages/msg_unknown_part"
    )
    assert diagnostics.status_code == 200
    assert "provider-future-payload" in json.dumps(diagnostics.json())


def test_raw_session_aliases_are_not_ordinary_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path)
    monkeypatch.delenv("CATMASTER_WEBUI_LEGACY_ROUTES", raising=False)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    ctx = client.get("/api/bootstrap", params={"project_space": "default"}).json()["ctx"]

    assert client.get(f"/api/session/{ctx}/snapshot").status_code == 404
    assert client.get(f"/api/session/{ctx}/events").status_code == 404
    assert client.get(f"/api/session/{ctx}/observability").status_code == 404
    assert client.get(f"/api/session/{ctx}/memory").status_code == 404


def test_legacy_synthetic_evaluation_payload_is_not_publicly_projected() -> None:
    candidate = project_self_evolution_candidate(
        {
            "candidate_id": "sec_diagnostic",
            "revision": 1,
            "evaluation": {
                "passed": True,
                "execution_mode": "deterministic_policy_replay",
                "full_task_executed": True,
                "verified_task_outcomes": True,
            },
        }
    )

    assert "evaluation" not in candidate
    assert "deterministic_policy_replay" not in json.dumps(candidate)


def test_self_evolution_candidate_projects_one_exact_target_evidence_list() -> None:
    candidate = project_self_evolution_candidate(
        {
            "candidate_id": "sec_surface",
            "revision": 2,
            "route": "amend_existing_skill",
            "evidence": [
                {
                    "observation_id": "obs-one",
                    "signal_kind": "skill_revision",
                    "status": "consolidated",
                    "claim": "Revise the surface-selection boundary.",
                    "evidence_refs": [
                        {
                            "source_ref": "run:one#event:8",
                            "reason": "agent_decision",
                            "excerpt": "The complete episode shows the wrong branch.",
                        }
                    ],
                }
            ],
            # Old polarity arrays must not leak back into the public contract.
            "supporting_evidence": [{"observation_id": "legacy-support"}],
            "counterexamples": [{"observation_id": "legacy-counter"}],
        }
    )

    assert len(candidate["evidence"]) == 1
    assert candidate["evidence"][0]["signal_label"] == "Existing skill revision"
    assert candidate["evidence"][0]["status_label"] == "Included in a candidate revision"
    assert candidate["evidence_summary"] == (
        "1 complete episode observation for this exact target."
    )
    assert "supporting_evidence" not in candidate
    assert "counterexamples" not in candidate


def test_public_openapi_contains_nonnullable_request_and_projection_models(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    schema = create_app(project_space_root=str(tmp_path), no_login=True).openapi()
    models = schema["components"]["schemas"]

    for name in (
        "ThreadCreateRequest",
        "ThreadPatchRequest",
        "ThreadResumeAction",
        "PublicEventData",
        "PublicMessagePageEnvelope",
    ):
        assert name in models
        assert '"null"' not in json.dumps(models[name])
    assert models["ThreadPatchRequest"]["properties"]["metadata"]["type"] == "object"
    assert models["ThreadResumeAction"]["properties"]["fields"]["type"] == "object"


def test_message_cursor_is_stable_across_restarts_and_has_no_duplicates(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread(title="paging")
    for index in range(125):
        store.append_message(
            ThreadMessage(
                id=f"msg_{index:04d}",
                thread_id=thread.thread_id,
                role="assistant",
                status="completed",
                parts=[
                    MessagePart(
                        id=f"part_{index:04d}",
                        type="text",
                        text=f"message {index}",
                        status="completed",
                    )
                ],
            )
        )

    newest = store.list_messages_page(thread.thread_id, limit=50)
    restarted = ThreadStore(workspace=workspace, workspace_id="default")
    older = restarted.list_messages_page(
        thread.thread_id,
        before=newest.next_cursor,
        limit=50,
    )
    repeated = restarted.list_messages_page(
        thread.thread_id,
        before=newest.next_cursor,
        limit=50,
    )

    assert [row.id for row in newest.messages] == [
        f"msg_{index:04d}" for index in range(75, 125)
    ]
    assert [row.id for row in older.messages] == [
        f"msg_{index:04d}" for index in range(25, 75)
    ]
    assert [row.id for row in repeated.messages] == [row.id for row in older.messages]
    assert not ({row.id for row in newest.messages} & {row.id for row in older.messages})


def test_message_parts_and_text_use_recoverable_opaque_pagination(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    thread_id = _new_thread(client)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    store.append_message(
        ThreadMessage(
            id="msg_many_parts",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id=f"part_{index:03d}",
                    type="text",
                    text=("segment-" + str(index) + " ") * 6_000,
                    status="completed",
                )
                for index in range(45)
            ],
        )
    )

    initial = client.get(f"/api/threads/{thread_id}/messages").json()
    message = initial["messages"][0]
    assert len(message["parts"]) == 20
    assert message["parts_page"]["truncated"] is True
    assert message["parts_page"]["shown_count"] == 20
    assert not message["parts_page"]["next_cursor"].isdigit()

    next_parts = client.get(
        f"/api/threads/{thread_id}/messages/msg_many_parts/parts",
        params={"cursor": message["parts_page"]["next_cursor"], "limit": 20},
    )
    assert next_parts.status_code == 200
    assert next_parts.json()["page"]["shown_count"] == 40
    assert next_parts.json()["parts"][0]["id"] == "part_020"

    first_part = message["parts"][0]
    assert first_part["truncation"]["truncated"] is True
    text_page = client.get(
        f"/api/threads/{thread_id}/messages/msg_many_parts/parts/part_000/content",
        params={"cursor": first_part["truncation"]["next_cursor"]},
    )
    assert text_page.status_code == 200
    assert text_page.json()["page"]["shown_count"] > first_part["truncation"]["shown_count"]


def test_todo_items_page_without_silently_truncating_each_label(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    thread_id = _new_thread(client)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    long_label = "Preserve this complete plan item: " + ("unbroken-token-" * 80)
    todos = [
        {
            "content": long_label if index == 0 else f"plan item {index}",
            "status": "in_progress" if index == 0 else "pending",
        }
        for index in range(205)
    ]
    store.append_message(
        ThreadMessage(
            id="msg_many_todos",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                MessagePart(
                    id="part_many_todos",
                    type="tool-call",
                    status="completed",
                    meta={
                        "tool": "write_todos",
                        "input": {"todos": todos},
                    },
                )
            ],
        )
    )

    initial = client.get(f"/api/threads/{thread_id}/messages").json()
    part = initial["messages"][0]["parts"][0]
    assert part["items"][0]["label"] == long_label
    assert len(part["items"]) == 200
    assert part["truncation"]["truncated"] is True
    assert part["truncation"]["total_count"] == 205

    remaining = client.get(
        "/api/threads/"
        f"{thread_id}/messages/msg_many_todos/parts/part_many_todos/items",
        params={"cursor": part["truncation"]["next_cursor"], "limit": 200},
    )
    assert remaining.status_code == 200
    assert [item["label"] for item in remaining.json()["items"]] == [
        f"plan item {index}" for index in range(200, 205)
    ]


def test_text_deltas_are_batched_and_concurrent_threads_do_not_lose_text(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread_a = store.create_thread(title="a")
    thread_b = store.create_thread(title="b")
    for thread, message_id, part_id in (
        (thread_a, "msg_a", "part_a"),
        (thread_b, "msg_b", "part_b"),
    ):
        store.append_message(
            ThreadMessage(
                id=message_id,
                thread_id=thread.thread_id,
                role="assistant",
                status="streaming",
                parts=[MessagePart(id=part_id, type="text", status="streaming")],
            )
        )

    flush_count = 0
    flush_count_lock = threading.Lock()
    original_flush = store._flush_message_deltas_locked

    def counted_flush(thread_id: str, message_id: str) -> None:
        nonlocal flush_count
        with flush_count_lock:
            flush_count += 1
        original_flush(thread_id, message_id)

    store._flush_message_deltas_locked = counted_flush  # type: ignore[method-assign]

    def stream(thread_id: str, message_id: str, part_id: str, token: str) -> None:
        for _ in range(100):
            store.add_text_delta(thread_id, message_id, part_id, token)

    workers = [
        threading.Thread(
            target=stream,
            args=(thread_a.thread_id, "msg_a", "part_a", "A"),
        ),
        threading.Thread(
            target=stream,
            args=(thread_b.thread_id, "msg_b", "part_b", "B"),
        ),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    time.sleep(0.12)

    assert store.get_message(thread_a.thread_id, "msg_a").parts[0].text == "A" * 100
    assert store.get_message(thread_b.thread_id, "msg_b").parts[0].text == "B" * 100
    assert 2 <= flush_count <= 4


def test_public_sse_deduplicates_message_and_thread_failure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    thread_id = _new_thread(client)
    broker = ThreadEventBroker(workspace=workspace)
    broker.emit(
        thread_id,
        "thread.status",
        status="running",
        data={"run_id": "run_failure_test", "status": "running"},
    )
    start_seq = 1
    broker.emit(
        thread_id,
        "message.failed",
        message_id="msg_failure",
        status="failed",
        data={"run_id": "run_failure_test", "error": "Calculation stopped."},
    )
    broker.emit(
        thread_id,
        "thread.updated",
        status="error",
        data={
            "status": "error",
            "run_id": "run_failure_test",
            "thread": {
                "active_message_id": "msg_failure",
                "error": "Calculation stopped.",
            },
        },
    )

    response = client.get(
        f"/api/threads/{thread_id}/stream",
        params={"last_seq": str(start_seq), "once": "true"},
    )
    assert response.status_code == 200
    assert response.text.count("event: run.failed") == 1
    assert "Calculation stopped." in response.text


def test_files_are_isolated_paginated_and_archived_without_internal_state(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    for index in range(205):
        (workspace / "files" / f"result_{index:03d}.txt").write_text(
            str(index),
            encoding="utf-8",
        )
    (workspace / "files" / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (workspace / "metadata" / "private.json").write_text("{}", encoding="utf-8")
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    ctx = client.get("/api/bootstrap", params={"project_space": "default"}).json()["ctx"]

    first = client.get(
        f"/api/session/{ctx}/files/tree",
        params={"project_space": "default", "limit": 200},
    )
    assert first.status_code == 200
    assert first.json()["page"]["truncated"] is True
    assert first.json()["page"]["total_count"] == 205
    assert ".hidden.txt" not in {row["name"] for row in first.json()["children"]}

    second = client.get(
        f"/api/session/{ctx}/files/tree",
        params={
            "project_space": "default",
            "cursor": first.json()["page"]["next_cursor"],
            "limit": 200,
        },
    )
    assert second.status_code == 200
    assert len(second.json()["children"]) == 5
    assert client.get(
        f"/api/session/{ctx}/files/content",
        params={"project_space": "default", "path": "metadata/private.json"},
    ).status_code == 404
    assert client.get(
        f"/api/session/{ctx}/files/content",
        params={"project_space": "default", "path": "files/.hidden.txt"},
    ).status_code == 404

    archive = client.get(
        f"/api/session/{ctx}/files/archive",
        params={"project_space": "default"},
    )
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content)) as bundle:
        names = bundle.namelist()
    assert names
    assert all(name.startswith("files/") for name in names)
    assert not any("metadata" in name or "/." in name for name in names)


def test_files_projection_hides_tool_offload_and_known_transient_extract(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "large_tool_results").mkdir()
    (workspace / "files" / "large_tool_results" / "call_x").write_text(
        "internal",
        encoding="utf-8",
    )
    literature = workspace / "files" / "literature"
    literature.mkdir()
    (literature / "core_extract.json").write_text("{}", encoding="utf-8")
    (literature / "selected_evidence.md").write_text(
        "# Evidence",
        encoding="utf-8",
    )
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    ctx = client.get(
        "/api/bootstrap",
        params={"project_space": "default"},
    ).json()["ctx"]

    root = client.get(
        f"/api/session/{ctx}/files/tree",
        params={"project_space": "default"},
    ).json()
    assert {row["name"] for row in root["children"]} == {"literature"}
    nested = client.get(
        f"/api/session/{ctx}/files/tree",
        params={"project_space": "default", "path": "files/literature"},
    ).json()
    assert {row["name"] for row in nested["children"]} == {
        "selected_evidence.md"
    }
    assert client.get(
        f"/api/session/{ctx}/files/content",
        params={
            "project_space": "default",
            "path": "files/literature/core_extract.json",
        },
    ).status_code == 404

    archive = client.get(
        f"/api/session/{ctx}/files/archive",
        params={"project_space": "default"},
    )
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content)) as bundle:
        names = bundle.namelist()
    assert any(name.endswith("selected_evidence.md") for name in names)
    assert not any(
        "large_tool_results" in name or name.endswith("core_extract.json")
        for name in names
    )


def test_workspace_sqlite_uses_wal_only_for_verified_local_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE", raising=False)
    monkeypatch.setattr(workspace_db, "filesystem_type", lambda _path: "nfs4")
    assert workspace_db.workspace_journal_mode(tmp_path) == "DELETE"
    monkeypatch.setattr(workspace_db, "filesystem_type", lambda _path: "ext4")
    assert workspace_db.workspace_journal_mode(tmp_path) == "WAL"
    monkeypatch.setenv("CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE", "DELETE")
    assert workspace_db.workspace_journal_mode(tmp_path) == "DELETE"


@pytest.mark.skipif(
    os.getenv("CATMASTER_RUN_LARGE_WEBUI_BENCHMARK") != "1",
    reason="Set CATMASTER_RUN_LARGE_WEBUI_BENCHMARK=1 for the 100 MiB release gate.",
)
def test_100_mib_thread_release_gate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread(title="100 MiB release gate")
    body = "long-context-fixture-" * 5_000

    def rows():
        for index in range(1_100):
            message = ThreadMessage(
                id=f"msg_large_{index:04d}",
                thread_id=thread.thread_id,
                role="assistant",
                status="completed",
                parts=[
                    MessagePart(
                        id=f"part_large_{index:04d}",
                        type="text",
                        text=body,
                        status="completed",
                    )
                ],
            )
            payload = message.model_dump(mode="json")
            yield (
                message.thread_id,
                message.id,
                message.created_at,
                message.updated_at,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )

    with connect_workspace_db(workspace) as connection:
        connection.executemany(
            """
            INSERT INTO thread_messages(
                thread_id, message_id, created_at, updated_at, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows(),
        )
    assert workspace_database_path(workspace).stat().st_size >= 95 * 1024 * 1024

    timings = []
    tracemalloc.start()
    for _ in range(7):
        start = time.perf_counter()
        page = store.list_messages_page(thread.thread_id, limit=50)
        timings.append(time.perf_counter() - start)
        assert len(page.messages) == 50
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p95 = sorted(timings)[-1]
    assert p95 < 0.300
    assert peak < 40 * 1024 * 1024

    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    response = client.get(f"/api/threads/{thread.thread_id}/messages", params={"limit": 50})
    assert response.status_code == 200
    assert len(response.content) < 2 * 1024 * 1024
    print(
        json.dumps(
            {
                "database_bytes": workspace_database_path(workspace).stat().st_size,
                "newest_50_p95_ms": round(p95 * 1_000, 3),
                "tracemalloc_peak_bytes": peak,
                "response_bytes": len(response.content),
            },
            sort_keys=True,
        )
    )
