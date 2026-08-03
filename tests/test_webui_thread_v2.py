from __future__ import annotations

import base64
import json
import multiprocessing
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command

from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, ObservabilityStore
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui import server
from catmaster.webui.artifact_registry import ArtifactRegistry, infer_renderer
from catmaster.webui.projections import project_event
from catmaster.webui.projections.tools import project_tool_part
from catmaster.webui.server import create_app
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import (
    ArtifactPart,
    MessagePart,
    ThreadMessage,
    ThreadStopRequest,
    ThreadSubmitRequest,
)
from catmaster.webui.thread_store import ThreadStore, new_id
from catmaster.specialists.runtime import SpecialistUsageCallbackHandler
from catmaster.specialists.streaming_runner import (
    CatMasterStreamTranslator,
    StreamingSpecialistRunner,
    _agent_run_id,
    _extract_sidecar_artifact_paths,
    _extract_workspace_paths_from_text,
    _json_safe,
)


def _register_artifact_process(
    workspace: str,
    path: str,
    thread_id: str,
    barrier: Any,
    results: Any,
) -> None:
    try:
        barrier.wait(timeout=10)
        record = ArtifactRegistry(
            workspace=Path(workspace),
            workspace_id="default",
        ).register_path(path, thread_id=thread_id)
        results.put({"artifact_id": record.artifact_id})
    except Exception as exc:  # pragma: no cover - surfaced in the parent
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def _wait_for_thread_event_process(
    workspace: str,
    thread_id: str,
    ready: Any,
    results: Any,
) -> None:
    try:
        broker = ThreadEventBroker(workspace=Path(workspace))
        ready.set()
        events, cursor = broker.wait_for_events(
            thread_id,
            last_seq=0,
            timeout_s=5,
        )
        results.put(
            {
                "events": [event.event for event in events],
                "cursor": cursor,
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced in the parent
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    return workspace


def test_thread_request_models_keep_model_config_as_an_api_alias() -> None:
    default_request = ThreadSubmitRequest(text="hello")
    selected_request = ThreadSubmitRequest(text="hello", model_config="configs/custom.yaml")

    assert default_request.llm_config == ""
    assert selected_request.llm_config == "configs/custom.yaml"
    assert ThreadSubmitRequest(text="hello", llm_config="configs/by-name.yaml").llm_config == "configs/by-name.yaml"
    assert "model_config" in ThreadSubmitRequest.model_json_schema()["properties"]
    assert "llm_config" not in ThreadSubmitRequest.model_json_schema()["properties"]
    assert ThreadStopRequest().emergency is False
    assert ThreadStopRequest().reason == ""


def test_thread_store_persists_messages_and_events_replay(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread(title="hello")
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="user",
        status="completed",
        parts=[MessagePart(id=new_id("part_text"), type="text", text="hi", status="completed")],
    )
    store.append_message(message)

    assert store.get_thread(thread.thread_id).title == "hello"
    assert store.list_messages(thread.thread_id)[0].parts[0].text == "hi"

    broker = ThreadEventBroker(workspace=workspace)
    first = broker.emit(thread.thread_id, "message.created", message_id=message.id, data={"message_id": message.id})
    second = broker.emit(thread.thread_id, "message.completed", message_id=message.id)

    assert first.seq == 1
    assert second.seq == 2
    replay = broker.replay(thread.thread_id, last_seq=1)
    assert [event.event for event in replay] == ["message.completed"]
    assert not broker.events_path(thread.thread_id).exists()


def test_thread_event_broker_persists_stream_events_to_observability_store(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread(title="observe")
    run_dir = system_root(workspace) / "runs" / "run_observe"

    broker = ThreadEventBroker(workspace=workspace)
    broker.emit(thread.thread_id, "reasoning.delta", message_id="msg_1", data={"run_id": "run_observe", "part_id": "part_reasoning"})

    assert (run_dir / OBSERVABILITY_DB_NAME).exists()
    rows = ObservabilityStore(run_dir).read_thread_events_page(thread.thread_id)
    assert [row["name"] for row in rows] == ["reasoning.delta"]
    assert rows[0]["channel"] == "thread"
    assert rows[0]["message_id"] == "msg_1"

    restarted = ThreadEventBroker(workspace=workspace)
    replay = restarted.replay(thread.thread_id)
    assert [event.event for event in replay] == ["reasoning.delta"]


def test_thread_event_broker_wakes_a_different_process(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    thread_id = ThreadStore(
        workspace=workspace,
        workspace_id="default",
    ).create_thread(title="cross-process").thread_id
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    process = context.Process(
        target=_wait_for_thread_event_process,
        args=(str(workspace), thread_id, ready, results),
    )
    process.start()
    assert ready.wait(timeout=10)

    emitted = ThreadEventBroker(workspace=workspace).emit(
        thread_id,
        "message.created",
        message_id="msg_cross_process",
    )
    outcome = results.get(timeout=10)
    process.join(timeout=10)

    assert process.exitcode == 0
    assert outcome == {
        "events": ["message.created"],
        "cursor": emitted.seq,
    }


def test_thread_stream_route_replays_observability_events_after_reconnect(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    created = client.post("/api/workspaces/default/threads", json={"title": "SSE replay"})
    assert created.status_code == 200
    thread_id = created.json()["thread"]["thread_id"]

    broker = ThreadEventBroker(workspace=workspace)
    broker.emit(thread_id, "reasoning.delta", message_id="msg_1", data={"run_id": "run_sse", "part_id": "part_reasoning"})
    broker.emit(thread_id, "message.delta", message_id="msg_1", data={"run_id": "run_sse", "part_id": "part_text"})

    response = client.get(f"/api/threads/{thread_id}/stream", params={"last_seq": "1", "once": "true"})

    assert response.status_code == 200
    assert "event: message.delta" in response.text


def test_thread_stream_deduplicates_the_same_failure_across_reconnects(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post(
        "/api/workspaces/default/threads",
        json={"title": "SSE failure reconnect"},
    ).json()["thread"]["thread_id"]
    broker = ThreadEventBroker(workspace=workspace)
    first = broker.emit(
        thread_id,
        "message.failed",
        message_id="msg_failure_domain",
        data={"error": "Remote calculation failed."},
    )

    initial = client.get(
        f"/api/threads/{thread_id}/stream",
        params={"last_seq": str(first.seq - 1), "once": "true"},
    )
    assert initial.text.count("event: run.failed") == 1

    duplicate = broker.emit(
        thread_id,
        "error",
        message_id="msg_failure_domain",
        data={"error": "Remote calculation failed."},
    )
    reconnected = client.get(
        f"/api/threads/{thread_id}/stream",
        params={"last_seq": str(first.seq), "once": "true"},
        headers={"Last-Event-ID": str(first.seq)},
    )
    assert duplicate.seq > first.seq
    assert reconnected.text.count("event: run.failed") == 0

    broker.emit(
        thread_id,
        "thread.status",
        message_id="msg_failure_domain",
        status="running",
        data={"status": "running"},
    )
    next_failure = broker.emit(
        thread_id,
        "message.failed",
        message_id="msg_failure_domain",
        data={"error": "A later retry failed."},
    )
    retried = client.get(
        f"/api/threads/{thread_id}/stream",
        params={"last_seq": str(duplicate.seq), "once": "true"},
    )
    assert next_failure.seq > duplicate.seq
    assert retried.text.count("event: run.failed") == 1


def test_thread_store_migrates_legacy_chat_sessions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    legacy_session = system_root(workspace) / "chat_sessions" / "session_a"
    legacy_session.mkdir(parents=True)
    (legacy_session / "session.json").write_text(json.dumps({"title": "Legacy chat"}), encoding="utf-8")
    rows = [
        {"message_id": "msg_user", "role": "user", "content": "Run O2.", "created_at": 10.0},
        {
            "message_id": "msg_result",
            "role": "assistant",
            "kind": "run_result",
            "source_run_id": "run_o2",
            "content": "O2 done.",
            "created_at": 11.0,
        },
    ]
    (legacy_session / "messages.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.get_thread("thread_session_a")
    messages = store.list_messages(thread.thread_id)

    assert thread.title == "Legacy chat"
    assert thread.meta["legacy_chat_session_id"] == "session_a"
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].parts[0].text == "Run O2."
    assert messages[1].parts[0].text == "O2 done."
    assert messages[1].meta["legacy_kind"] == "run_result"
    assert messages[1].meta["run_id"] == "run_o2"


def test_artifact_registry_renderer_and_path_safety(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")

    record = registry.register_path("table.csv", thread_id="thread_x", message_id="msg_x")

    assert record.path == "files/table.csv"
    assert record.renderer == "csv"
    assert infer_renderer("POSCAR") == "structure"
    assert infer_renderer("figure.png", "image/png") == "image"

    try:
        registry.register_path("../secret.txt")
    except ValueError as exc:
        assert "invalid" in str(exc) or "escapes" in str(exc)
    else:
        raise AssertionError("unsafe artifact path was accepted")


def test_artifact_registry_reuses_one_thread_artifact_for_the_same_path(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")

    first = registry.register_path(
        "table.csv",
        thread_id="thread_x",
        message_id="msg_one",
        tool_call_id="call_one",
    )
    second = registry.register_path(
        "table.csv",
        thread_id="thread_x",
        message_id="msg_two",
        tool_call_id="call_two",
    )

    assert second.artifact_id == first.artifact_id
    assert [record.artifact_id for record in registry.list_artifacts(thread_id="thread_x")] == [
        first.artifact_id
    ]


def test_artifact_registry_imports_legacy_index_once_without_rewriting_it(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = workspace / "files" / "legacy-index.txt"
    output.write_text("legacy", encoding="utf-8")
    legacy_root = system_root(workspace) / "artifacts"
    legacy_root.mkdir(parents=True)
    index_path = legacy_root / "index.jsonl"
    original = (
        json.dumps(
            {
                "artifact_id": "art_legacy_index",
                "thread_id": "thread_legacy",
                "workspace_id": "default",
                "path": "files/legacy-index.txt",
                "title": "Legacy index",
            }
        )
        + "\n"
    )
    index_path.write_text(original, encoding="utf-8")

    first = ArtifactRegistry(workspace=workspace, workspace_id="default")
    second = ArtifactRegistry(workspace=workspace, workspace_id="default")

    assert first.get("art_legacy_index") is not None
    assert [item.artifact_id for item in second.list_artifacts()] == [
        "art_legacy_index"
    ]
    assert index_path.read_text(encoding="utf-8") == original


def test_artifact_registry_concurrent_processes_do_not_lose_updates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = ["first.txt", "second.txt"]
    for path in paths:
        (workspace / "files" / path).write_text(path, encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(paths) + 1)
    results = context.Queue()
    processes = [
        context.Process(
            target=_register_artifact_process,
            args=(str(workspace), path, f"thread_{index}", barrier, results),
        )
        for index, path in enumerate(paths)
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    outcomes = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert all("error" not in outcome for outcome in outcomes)
    records = ArtifactRegistry(
        workspace=workspace,
        workspace_id="default",
    ).list_artifacts()
    assert {record.path for record in records} == {
        "files/first.txt",
        "files/second.txt",
    }
    assert not (system_root(workspace) / "artifacts" / "index.jsonl").exists()


def test_artifact_registry_renderer_mapping_covers_domain_artifacts(tmp_path: Path) -> None:
    assert infer_renderer("POSCAR") == "structure"
    assert infer_renderer("trajectory.traj") == "structure"
    assert infer_renderer("figure.svg", "image/svg+xml") == "image"
    assert infer_renderer("table.tsv") == "csv"
    assert infer_renderer("report.rst") == "markdown"
    assert infer_renderer("paper.pdf", "application/pdf") == "pdf"
    assert infer_renderer("stdout.log") == "text"
    assert infer_renderer("bundle.zip") == "archive"


def test_artifact_registry_skips_missing_run_state_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "notes").mkdir(parents=True)
    (workspace / "files" / "notes" / "summary.json").write_text("{}", encoding="utf-8")
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")

    records = registry.register_from_run_state(
        {
            "thread_id": "thread_x",
            "run_id": "run_x",
            "artifacts": [
                {"path": "fmax=0.02 eV/Å", "description": "not a file"},
                {"path": "notes/summary.json", "description": "real output"},
            ],
        },
        thread_id="thread_x",
        run_id="run_x",
    )

    assert [record.path for record in records] == ["files/notes/summary.json"]


def test_artifact_registry_rejects_missing_tool_artifact_paths(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")

    with pytest.raises(ValueError, match="existing workspace file"):
        registry.register_path(
            "task_config.fmax",
            thread_id="thread_x",
            tool_call_id="call_task_spec",
            meta={"source": "tool_artifact"},
        )

    assert registry.list_artifacts(thread_id="thread_x") == []


def test_artifact_registry_hides_index_records_after_file_disappears(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = workspace / "files" / "temporary.csv"
    output.write_text("x\n1\n", encoding="utf-8")
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    record = registry.register_path("temporary.csv", thread_id="thread_x")

    output.unlink()

    assert registry.get(record.artifact_id) is None
    assert registry.list_artifacts(thread_id="thread_x") == []


def test_artifact_registry_migrates_legacy_run_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "legacy.csv").write_text("x\n1\n", encoding="utf-8")
    run_dir = system_root(workspace) / "runs" / "run_legacy"
    run_dir.mkdir(parents=True)
    run_state_text = json.dumps(
        {
            "status": "done",
            "thread_id": "thread_legacy",
            "artifacts": [{"path": "legacy.csv", "summary": "legacy table"}],
        },
        indent=2,
    )
    (run_dir / "run_state.json").write_text(run_state_text, encoding="utf-8")
    checkpoint_path = system_root(workspace) / "deepagent_threads.sqlite"
    checkpoint_path.write_bytes(b"legacy checkpoint")
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")

    records = registry.list_artifacts(thread_id="thread_legacy")

    assert len(records) == 1
    assert records[0].path == "files/legacy.csv"
    assert records[0].renderer == "csv"
    assert records[0].run_id == "run_legacy"
    assert records[0].meta["source"] == "run_state"
    assert (run_dir / "run_state.json").read_text(encoding="utf-8") == run_state_text
    assert checkpoint_path.read_bytes() == b"legacy checkpoint"


def test_server_thread_routes_and_artifact_preview(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "note.md").write_text("# Result\n\nok\n", encoding="utf-8")
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    created = client.post("/api/workspaces/default/threads", json={"title": "T"})
    assert created.status_code == 200
    thread_id = created.json()["thread"]["thread_id"]

    listed = client.get("/api/workspaces/default/threads")
    assert listed.status_code == 200
    assert listed.json()["threads"][0]["thread_id"] == thread_id

    assert client.get(f"/api/threads/{thread_id}").status_code == 200
    empty_page = client.get(f"/api/threads/{thread_id}/messages").json()
    assert empty_page["messages"] == []
    assert empty_page["page"]["shown_count"] == 0
    assert empty_page["page"]["total_count"] == 0
    assert empty_page["page"]["total_unknown"] is False
    assert empty_page["page"]["truncated"] is False
    assert empty_page["page"]["next_cursor"] == ""

    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    artifact = registry.register_path("note.md", thread_id=thread_id, message_id="msg_x")
    preview = client.get(f"/api/artifacts/{artifact.artifact_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["kind"] == "markdown"
    assert "Result" in preview.json()["preview_text"]

    malformed = client.post(f"/api/threads/{thread_id}/resume", json={"decisions": [{"type": "deny"}]})
    assert malformed.status_code == 400


def test_server_hides_historical_artifact_parts_without_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "real.csv").write_text("x\n1\n", encoding="utf-8")
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={"title": "T"}).json()["thread"]["thread_id"]
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    real = registry.register_path("real.csv", thread_id=thread_id, message_id="msg_artifacts")
    store = ThreadStore(workspace=workspace, workspace_id="default")
    store.append_message(
        ThreadMessage(
            id="msg_artifacts",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=[
                ArtifactPart(
                    id="part_real",
                    artifact_id=real.artifact_id,
                    path=real.path,
                ),
                ArtifactPart(
                    id="part_missing",
                    artifact_id="art_task_config_fmax",
                    path="files/task_config.fmax",
                ),
            ],
        )
    )

    response = client.get(f"/api/threads/{thread_id}/messages")

    assert response.status_code == 200
    artifact_parts = [
        part
        for part in response.json()["messages"][0]["parts"]
        if part["type"] == "artifact"
    ]
    assert [part["artifact_id"] for part in artifact_parts] == [real.artifact_id]


def test_thread_permission_mode_create_patch_and_interrupt_mapping(tmp_path: Path) -> None:
    _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    default_created = client.post("/api/workspaces/default/threads", json={"title": "default"})
    assert default_created.status_code == 200
    default_thread = default_created.json()["thread"]
    assert default_thread["permission_mode"] == "auto"
    assert "meta" not in default_thread
    assert server._thread_permission_mode(SimpleNamespace(meta={})) == "auto"

    created = client.post("/api/workspaces/default/threads", json={"title": "auto", "permission_mode": "auto-approve"})
    assert created.status_code == 200
    thread = created.json()["thread"]
    assert thread["permission_mode"] == "auto"
    assert server._interrupt_on_for_permission_mode(thread["permission_mode"]) == {}

    patched = client.patch(f"/api/threads/{thread['thread_id']}", json={"permission_mode": "review"})
    assert patched.status_code == 200
    thread = patched.json()["thread"]
    assert thread["permission_mode"] == "hitl"
    assert server._interrupt_on_for_permission_mode(thread["permission_mode"]) == server.default_thread_interrupt_on()

    invalid = client.patch(f"/api/threads/{thread['thread_id']}", json={"permission_mode": "bad"})
    assert invalid.status_code == 400


def test_thread_entrypoint_api_preserves_all_lanes_and_validates_values(tmp_path: Path) -> None:
    _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    entrypoints = client.get("/api/entrypoints")
    assert entrypoints.status_code == 200
    ids = [item["id"] for item in entrypoints.json()["entrypoints"]]
    assert ids == ["research", "experiment", "writing", "peer_review", "literature_review"]

    created = client.post("/api/workspaces/default/threads", json={"title": "Write", "entrypoint": "writing"})
    assert created.status_code == 200
    thread = created.json()["thread"]
    assert thread["entrypoint"] == "writing"

    patched = client.patch(f"/api/threads/{thread['thread_id']}", json={"entrypoint": "peer-review"})
    assert patched.status_code == 200
    assert patched.json()["thread"]["entrypoint"] == "peer_review"

    alias = client.patch(f"/api/threads/{thread['thread_id']}", json={"entrypoint": "literature"})
    assert alias.status_code == 200
    assert alias.json()["thread"]["entrypoint"] == "literature_review"

    invalid = client.patch(f"/api/threads/{thread['thread_id']}", json={"entrypoint": "unknown"})
    assert invalid.status_code == 400


def test_resume_marks_pending_interrupt_part_resolved(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]
    store = ThreadStore(workspace=workspace, workspace_id="default")
    captured: dict[str, object] = {}
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread_id,
        role="assistant",
        status="interrupted",
        parts=[
            MessagePart(
                id="part_interrupt",
                type="interrupt",
                status="pending",
                text="Review required.",
                meta={
                    "interrupt_id": "interrupt_x",
                    "status": "pending",
                    "payload": {
                        "interrupts": [
                            {
                                "value": {
                                    "action_requests": [
                                        {"name": "write_file", "args": {"file_path": "/a.txt", "content": "a"}},
                                        {"name": "write_file", "args": {"file_path": "/b.txt", "content": "b"}},
                                    ],
                                    "review_configs": [
                                        {"action_name": "write_file", "allowed_decisions": ["approve", "edit", "reject", "respond"]},
                                        {"action_name": "write_file", "allowed_decisions": ["approve", "edit", "reject", "respond"]},
                                    ],
                                }
                            }
                        ]
                    },
                },
            )
        ],
    )
    store.append_message(message)
    store.update_thread(thread_id, status="interrupted", active_message_id=message.id)

    async def _fake_aresume(self, *, thread_id, message_id, text_part_id, decisions, **kwargs):
        captured["decisions"] = decisions
        captured["resume_tool_inputs"] = kwargs.get("resume_tool_inputs")
        self.thread_store.update_message(thread_id, message_id, status="completed")
        self.thread_store.update_thread(thread_id, status="idle", active_message_id="", active_run_id="")
        return {"status": "done"}

    monkeypatch.setattr(server.StreamingSpecialistRunner, "aresume", _fake_aresume)
    monkeypatch.setattr(
        server,
        "build_specialist_runner",
        lambda **_kwargs: SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        ),
    )

    response = client.post(f"/api/threads/{thread_id}/resume", json={"decisions": [{"type": "approve"}]})

    assert response.status_code == 200
    saved = store.get_message(thread_id, message.id)
    assert saved.parts[0].status == "resolved"
    assert saved.parts[0].meta["resolution"]["decisions"] == [{"type": "approve"}, {"type": "approve"}]
    assert captured["decisions"] == [{"type": "approve"}, {"type": "approve"}]
    assert captured["resume_tool_inputs"] == [
        {"name": "write_file", "args": {"file_path": "/a.txt", "content": "a"}, "source": "interrupt_review"},
        {"name": "write_file", "args": {"file_path": "/b.txt", "content": "b"}, "source": "interrupt_review"},
    ]


def test_resume_reject_decision_is_preserved_for_model_feedback(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]
    store = ThreadStore(workspace=workspace, workspace_id="default")
    captured: dict[str, object] = {}
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread_id,
        role="assistant",
        status="interrupted",
        parts=[
            MessagePart(
                id="part_interrupt",
                type="interrupt",
                status="pending",
                text="Review required.",
                meta={
                    "interrupt_id": "interrupt_x",
                    "status": "pending",
                    "payload": {
                        "interrupts": [
                            {
                                "value": {
                                    "action_requests": [
                                        {"name": "write_file", "args": {"file_path": "/a.txt", "content": "a"}},
                                    ],
                                    "review_configs": [
                                        {"action_name": "write_file", "allowed_decisions": ["approve", "edit", "reject", "respond"]},
                                    ],
                                }
                            }
                        ]
                    },
                },
            )
        ],
    )
    store.append_message(message)
    store.update_thread(thread_id, status="interrupted", active_message_id=message.id)

    async def _fake_aresume(self, *, thread_id, message_id, text_part_id, decisions, **_kwargs):
        captured["decisions"] = decisions
        self.thread_store.update_message(thread_id, message_id, status="completed")
        self.thread_store.update_thread(thread_id, status="idle", active_message_id="", active_run_id="")
        return {"status": "done"}

    monkeypatch.setattr(server.StreamingSpecialistRunner, "aresume", _fake_aresume)
    monkeypatch.setattr(
        server,
        "build_specialist_runner",
        lambda **_kwargs: SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        ),
    )

    decision = {"type": "reject", "message": "Do not write outside files/."}
    response = client.post(f"/api/threads/{thread_id}/resume", json={"decisions": [decision]})

    assert response.status_code == 200
    saved = store.get_message(thread_id, message.id)
    assert saved.parts[0].status == "resolved"
    assert saved.parts[0].meta["resolution"]["decisions"] == [decision]
    assert captured["decisions"] == [decision]


def test_submit_creates_user_and_running_assistant_message(tmp_path: Path, monkeypatch) -> None:
    _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]

    async def _fake_arun_turn(self, *, thread_id, message_id, text_part_id, **_kwargs):
        self.thread_store.add_text_delta(thread_id, message_id, text_part_id, "done")
        self.thread_store.update_message(thread_id, message_id, status="completed")
        self.thread_store.update_thread(thread_id, status="idle", active_message_id="", active_run_id="")
        return {"status": "done"}

    monkeypatch.setattr(server.StreamingSpecialistRunner, "arun_turn", _fake_arun_turn)
    monkeypatch.setattr(
        server,
        "build_specialist_runner",
        lambda **_kwargs: SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        ),
    )

    submitted = client.post(f"/api/threads/{thread_id}/submit", json={"text": "hello", "permission_mode": "auto"})

    assert submitted.status_code == 200
    payload = submitted.json()
    assert payload["message"]["role"] == "user"
    assert payload["assistant_message"]["role"] == "assistant"
    assert "meta" not in payload["assistant_message"]
    assert payload["thread"]["permission_mode"] == "auto"


def test_submit_image_attachment_registers_artifact_without_persisting_data_url(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]
    captured: dict[str, object] = {}

    async def _fake_arun_turn(self, *, prompt, thread_id, message_id, text_part_id, **_kwargs):
        captured["prompt"] = prompt
        captured["content"] = _kwargs.get("content")
        self.thread_store.add_text_delta(thread_id, message_id, text_part_id, "ok")
        self.thread_store.update_message(thread_id, message_id, status="completed")
        self.thread_store.update_thread(thread_id, status="idle", active_message_id="", active_run_id="")
        return {"status": "done"}

    monkeypatch.setattr(server.StreamingSpecialistRunner, "arun_turn", _fake_arun_turn)
    monkeypatch.setattr(
        server,
        "build_specialist_runner",
        lambda **_kwargs: SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        ),
    )

    image_data = base64.b64encode(b"fake-png").decode("ascii")
    submitted = client.post(
        f"/api/threads/{thread_id}/submit",
        json={
            "text": "inspect this image",
            "attachments": [
                {
                    "type": "image",
                    "filename": "figure.png",
                    "mime_type": "image/png",
                    "data": f"data:image/png;base64,{image_data}",
                }
            ],
        },
    )

    assert submitted.status_code == 200
    user_message = submitted.json()["message"]
    artifact_parts = [part for part in user_message["parts"] if part["type"] == "artifact"]
    assert len(artifact_parts) == 1
    assert artifact_parts[0]["path"].startswith("files/attachments/")
    assert "data:image/png" not in json.dumps(user_message)
    assert "data:image/png" not in str(captured["prompt"])
    content = captured["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "figure.png" in content[0]["text"]
    # The checked-in profile uses the local codex_oauth adapter. Its installed
    # LangChain bridge currently accepts text blocks only, so the file must be
    # retained as an artifact without claiming it was sent to the model.
    assert len(content) == 1
    assert (workspace / artifact_parts[0]["path"]).exists()
    multimodal_events = [event for event in ThreadEventBroker(workspace=workspace).replay(thread_id) if event.event == "multimodal.prepared"]
    assert multimodal_events
    event_data = multimodal_events[-1].data
    assert event_data["attachment_count"] == 1
    assert event_data["attachments"][0]["sent_to_model"] is False
    assert event_data["attachments"][0]["sent_as"] == "stored_only"
    assert "does not enable image blocks" in event_data["attachments"][0]["warnings"][0]
    assert "base64" not in json.dumps(event_data)


def test_submit_pdf_attachment_passes_bounded_text_without_persisting_data_url(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]
    captured: dict[str, object] = {}

    async def _fake_arun_turn(self, *, prompt, thread_id, message_id, text_part_id, **_kwargs):
        captured["prompt"] = prompt
        captured["content"] = _kwargs.get("content")
        self.thread_store.add_text_delta(thread_id, message_id, text_part_id, "ok")
        self.thread_store.update_message(thread_id, message_id, status="completed")
        self.thread_store.update_thread(thread_id, status="idle", active_message_id="", active_run_id="")
        return {"status": "done"}

    monkeypatch.setattr(server.StreamingSpecialistRunner, "arun_turn", _fake_arun_turn)
    monkeypatch.setattr(
        "catmaster.webui.agent_loop.read_document",
        lambda *_args, **_kwargs: "PDF source: `/attachments/paper.pdf`\n\n--- Page 1 ---\nbounded PDF text",
    )
    monkeypatch.setattr(
        server,
        "build_specialist_runner",
        lambda **_kwargs: SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        ),
    )

    pdf_data = base64.b64encode(b"%PDF-1.4 fake").decode("ascii")
    submitted = client.post(
        f"/api/threads/{thread_id}/submit",
        json={
            "text": "inspect this PDF",
            "attachments": [
                {
                    "type": "file",
                    "filename": "paper.pdf",
                    "mime_type": "application/pdf",
                    "data": f"data:application/pdf;base64,{pdf_data}",
                }
            ],
        },
    )

    assert submitted.status_code == 200
    user_message = submitted.json()["message"]
    assert "data:application/pdf" not in str(user_message)
    content = captured["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "text"
    assert "bounded PDF text" in content[1]["text"]
    assert pdf_data not in str(content)
    multimodal_events = [
        event
        for event in ThreadEventBroker(workspace=workspace).replay(thread_id)
        if event.event == "multimodal.prepared"
    ]
    assert multimodal_events
    event_data = multimodal_events[-1].data
    assert event_data["attachments"][0]["sent_to_model"] is True
    assert event_data["attachments"][0]["sent_as"] == "text_excerpt"
    assert "base64" not in json.dumps(event_data)


def test_submit_docx_attachment_passes_parsed_text_instead_of_office_bytes(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]
    captured: dict[str, object] = {}

    async def _fake_arun_turn(self, *, prompt, thread_id, message_id, text_part_id, **_kwargs):
        captured["content"] = _kwargs.get("content")
        self.thread_store.add_text_delta(thread_id, message_id, text_part_id, "ok")
        self.thread_store.update_message(thread_id, message_id, status="completed")
        self.thread_store.update_thread(thread_id, status="idle", active_message_id="", active_run_id="")
        return {"status": "done"}

    monkeypatch.setattr(server.StreamingSpecialistRunner, "arun_turn", _fake_arun_turn)
    monkeypatch.setattr(
        "catmaster.webui.agent_loop.read_document",
        lambda *_args, **_kwargs: "DOCX source: `/attachments/report.docx`\n\nParsed Word paragraph",
    )
    monkeypatch.setattr(
        server,
        "build_specialist_runner",
        lambda **_kwargs: SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        ),
    )

    office_bytes = base64.b64encode(b"PK fake docx bytes").decode("ascii")
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    submitted = client.post(
        f"/api/threads/{thread_id}/submit",
        json={
            "text": "inspect this report",
            "attachments": [
                {
                    "type": "file",
                    "filename": "report.docx",
                    "mime_type": mime,
                    "data": f"data:{mime};base64,{office_bytes}",
                }
            ],
        },
    )

    assert submitted.status_code == 200
    content = captured["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "text"
    assert "Parsed Word paragraph" in content[1]["text"]
    assert office_bytes not in str(content)


def test_agent_loop_service_launches_turn_and_queues_steering(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread(meta={"permission_mode": "hitl"})
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    tasks: dict[str, object] = {}
    stop_flags: set[str] = set()
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["runner_init"] = kwargs

        async def arun_turn(self, **kwargs):
            captured["arun_turn"] = kwargs
            return {"status": "done"}

    def fake_build_runner(**kwargs):
        captured["build_runner"] = kwargs
        return SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        )

    service = ThreadAgentLoopService(
        workspace=workspace,
        workspace_id="default",
        store=store,
        broker=broker,
        artifact_registry=registry,
        thread_tasks=tasks,  # type: ignore[arg-type]
        thread_stop_flags=stop_flags,
        build_runner=fake_build_runner,
        streaming_runner_cls=FakeRunner,
        permission_mode_for_thread=lambda thread, override="": str(override or thread.meta.get("permission_mode") or "hitl"),
        interrupt_on_for_permission_mode=lambda mode: {"write_file": True} if mode == "hitl" else {},
        normalize_entrypoint=lambda value: value or "research",
        should_stop=lambda _thread_id: False,
    )

    result = awaitable_result(service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="hello", attachments=[], entrypoint="research", llm_config="", permission_mode="hitl")))

    assert result["queued"] is False
    assert "build_runner" in captured
    assert captured["build_runner"]["interrupt_on"] == {"write_file": True}
    assert result["assistant_message"].role == "assistant"

    running_thread = store.update_thread(thread.thread_id, status="running")
    tasks[thread.thread_id] = SimpleNamespace(done=lambda: False)
    queued = awaitable_result(service.submit(thread_id=running_thread.thread_id, payload=SimpleNamespace(text="steer", attachments=[], entrypoint="research", llm_config="", permission_mode="hitl")))

    assert queued["queued"] is True
    assert store.get_thread(thread.thread_id).pending_steering[0]["text"] == "steer"
    assert store.get_thread(thread.thread_id).pending_steering[0]["model_config"] == ""


def test_first_research_turn_binds_a_graph_before_freezing_runtime_context(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    first_thread = store.create_thread(entrypoint="research")
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    captured_contexts: list[dict[str, str]] = []
    captured_contents: list[object] = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.thread_store = kwargs["thread_store"]

        async def arun_turn(self, **kwargs):
            captured_contents.append(kwargs.get("content"))
            self.thread_store.update_message(
                kwargs["thread_id"],
                kwargs["message_id"],
                status="completed",
            )
            self.thread_store.update_thread(
                kwargs["thread_id"],
                status="idle",
                active_message_id="",
                active_run_id="",
            )
            return {"status": "done"}

    def fake_build_runner(**kwargs):
        captured_contexts.append(dict(kwargs["runtime_context"]))
        return SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(
                run_id=f"run_{len(captured_contexts)}",
                run_dir=tmp_path / f"run_{len(captured_contexts)}",
            ),
        )

    service = ThreadAgentLoopService(
        workspace=workspace,
        workspace_id="default",
        store=store,
        broker=broker,
        artifact_registry=registry,
        thread_tasks={},  # type: ignore[arg-type]
        thread_stop_flags=set(),
        build_runner=fake_build_runner,
        streaming_runner_cls=FakeRunner,
        permission_mode_for_thread=lambda _thread, override="": override or "hitl",
        interrupt_on_for_permission_mode=lambda _mode: {},
        normalize_entrypoint=lambda value: value or "research",
        should_stop=lambda _thread_id: False,
    )

    prompt = "Determine which ceramic ablation mechanism is consistent with the evidence."
    first = awaitable_result(
        service.submit(
            thread_id=first_thread.thread_id,
            payload=SimpleNamespace(
                text=prompt,
                attachments=[],
                entrypoint="research",
                llm_config="",
                permission_mode="hitl",
            ),
        )
    )

    graphs = ResearchGraphStore(workspace).list_graphs(include_archived=False)
    assert len(graphs) == 1
    graph_id = graphs[0]["graph_id"]
    assert graphs[0]["question"] == prompt
    assert store.get_thread(first_thread.thread_id).active_research_graph_id == graph_id
    assert captured_contexts[0] == {
        "research_graph_id": graph_id,
        "research_focus_node_id": "",
        "research_launch_id": "",
    }
    assert first["assistant_message"].meta["research_graph_id"] == graph_id
    assert "# Active Research Graph: partial focus snippet" in str(captured_contents[0])
    assert prompt in str(captured_contents[0])

    second_thread = store.create_thread(entrypoint="research")
    awaitable_result(
        service.submit(
            thread_id=second_thread.thread_id,
            payload=SimpleNamespace(
                text="Continue this workspace research from another thread.",
                attachments=[],
                entrypoint="research",
                llm_config="",
                permission_mode="hitl",
            ),
        )
    )
    assert store.get_thread(second_thread.thread_id).active_research_graph_id == graph_id
    assert len(ResearchGraphStore(workspace).list_graphs(include_archived=False)) == 1
    assert captured_contexts[1]["research_graph_id"] == graph_id

    graph_store = ResearchGraphStore(workspace)
    graph_store.create_graph(title="Another route", question="Test another route")
    ambiguous_thread = store.create_thread(entrypoint="research")
    awaitable_result(
        service.submit(
            thread_id=ambiguous_thread.thread_id,
            payload=SimpleNamespace(
                text="Continue the relevant research.",
                attachments=[],
                entrypoint="research",
                llm_config="",
                permission_mode="hitl",
            ),
        )
    )
    assert store.get_thread(ambiguous_thread.thread_id).active_research_graph_id == ""
    assert captured_contexts[2] == {
        "research_graph_id": "",
        "research_focus_node_id": "",
        "research_launch_id": "",
    }
    assert len(graph_store.list_graphs(include_archived=False)) == 2


def test_agent_loop_persists_submit_entrypoint_and_passes_it_to_runner(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread(entrypoint="research", meta={"permission_mode": "hitl"})
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.thread_store = kwargs["thread_store"]

        async def arun_turn(self, **kwargs):
            captured["arun_turn"] = kwargs
            self.thread_store.update_message(kwargs["thread_id"], kwargs["message_id"], status="completed")
            self.thread_store.update_thread(kwargs["thread_id"], status="idle", active_message_id="", active_run_id="")
            return {"status": "done"}

    def fake_build_runner(**kwargs):
        captured["build_runner"] = kwargs
        return SimpleNamespace(
            runner=object(),
            run_context=SimpleNamespace(run_id="run_fake", run_dir=tmp_path / "run_fake"),
        )

    service = ThreadAgentLoopService(
        workspace=workspace,
        workspace_id="default",
        store=store,
        broker=broker,
        artifact_registry=registry,
        thread_tasks={},  # type: ignore[arg-type]
        thread_stop_flags=set(),
        build_runner=fake_build_runner,
        streaming_runner_cls=FakeRunner,
        permission_mode_for_thread=lambda thread, override="": str(override or thread.meta.get("permission_mode") or "hitl"),
        interrupt_on_for_permission_mode=lambda _mode: {},
        normalize_entrypoint=lambda value: value if value in {"research", "writing"} else "research",
        should_stop=lambda _thread_id: False,
    )

    result = awaitable_result(service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="draft", attachments=[], entrypoint="writing", llm_config="", permission_mode="hitl")))

    assert result["queued"] is False
    assert store.get_thread(thread.thread_id).entrypoint == "writing"
    assert captured["build_runner"]["preferred_entrypoint"] == "writing"
    assert captured["build_runner"]["runtime_context"] == {
        "research_graph_id": "",
        "research_focus_node_id": "",
        "research_launch_id": "",
    }
    assert captured["arun_turn"]["entrypoint"] == "writing"
    assert result["assistant_message"].meta["entrypoint"] == "writing"
    stored_message = store.get_message(
        thread.thread_id,
        result["assistant_message"].id,
    )
    assert stored_message.meta == {
        "permission_mode": "hitl",
        "entrypoint": "writing",
        "research_graph_id": "",
        "research_focus_node_id": "",
        "research_launch_id": "",
        "run_id": "run_fake",
    }


def test_agent_loop_applies_queued_steering_at_safe_boundary(tmp_path: Path) -> None:
    import asyncio

    async def _run() -> None:
        workspace = _workspace(tmp_path)
        store = ThreadStore(workspace=workspace, workspace_id="default")
        thread = store.create_thread(meta={"permission_mode": "hitl"})
        broker = ThreadEventBroker(workspace=workspace)
        registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
        tasks: dict[str, asyncio.Task] = {}
        stop_flags: set[str] = set()
        started = asyncio.Event()
        release = asyncio.Event()
        prompts: list[str] = []

        class FakeRunner:
            def __init__(self, **kwargs):
                self.thread_store = kwargs["thread_store"]

            async def arun_turn(self, **kwargs):
                prompt = str(kwargs.get("prompt") or "")
                prompts.append(prompt)
                if len(prompts) == 1:
                    started.set()
                    await release.wait()
                self.thread_store.update_message(kwargs["thread_id"], kwargs["message_id"], status="completed")
                self.thread_store.update_thread(kwargs["thread_id"], status="idle", active_message_id="", active_run_id="")
                return {"status": "steered" if len(prompts) == 1 else "done"}

        def fake_build_runner(**_kwargs):
            return SimpleNamespace(
                runner=object(),
                run_context=SimpleNamespace(run_id=f"run_{len(prompts) + 1}", run_dir=tmp_path / f"run_{len(prompts) + 1}"),
            )

        service = ThreadAgentLoopService(
            workspace=workspace,
            workspace_id="default",
            store=store,
            broker=broker,
            artifact_registry=registry,
            thread_tasks=tasks,  # type: ignore[arg-type]
            thread_stop_flags=stop_flags,
            build_runner=fake_build_runner,
            streaming_runner_cls=FakeRunner,
            permission_mode_for_thread=lambda thread, override="": str(override or thread.meta.get("permission_mode") or "hitl"),
            interrupt_on_for_permission_mode=lambda _mode: {},
            normalize_entrypoint=lambda value: value or "research",
            should_stop=lambda _thread_id: False,
        )

        first = await service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="first", attachments=[], entrypoint="research", llm_config="", permission_mode="hitl"))
        assert first["queued"] is False
        await started.wait()
        queued = await service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="steer", attachments=[], entrypoint="research", llm_config="", permission_mode="hitl"))
        assert queued["queued"] is True
        assert store.get_thread(thread.thread_id).pending_steering
        release.set()
        for _ in range(20):
            if len(prompts) >= 2 and not store.get_thread(thread.thread_id).pending_steering:
                break
            await asyncio.sleep(0.01)
        assert prompts == ["first", "steer"]
        assert store.get_thread(thread.thread_id).pending_steering == []

    asyncio.run(_run())


def test_agent_loop_stop_cancels_active_turn_and_discards_queued_steering(
    tmp_path: Path,
) -> None:
    import asyncio

    async def _run() -> None:
        workspace = _workspace(tmp_path)
        store = ThreadStore(workspace=workspace, workspace_id="default")
        thread = store.create_thread(meta={"permission_mode": "hitl"})
        broker = ThreadEventBroker(workspace=workspace)
        registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
        tasks: dict[str, asyncio.Task] = {}
        started = asyncio.Event()
        prompts: list[str] = []

        class FakeRunner:
            def __init__(self, **_kwargs):
                pass

            async def arun_turn(self, **kwargs):
                prompts.append(str(kwargs.get("prompt") or ""))
                started.set()
                await asyncio.Event().wait()

        service = ThreadAgentLoopService(
            workspace=workspace,
            workspace_id="default",
            store=store,
            broker=broker,
            artifact_registry=registry,
            thread_tasks=tasks,
            thread_stop_flags=set(),
            build_runner=lambda **_kwargs: SimpleNamespace(
                runner=object(),
                run_context=SimpleNamespace(run_id="run_stop", run_dir=tmp_path / "run_stop"),
            ),
            streaming_runner_cls=FakeRunner,
            permission_mode_for_thread=lambda _thread, override="": override or "hitl",
            interrupt_on_for_permission_mode=lambda _mode: {},
            normalize_entrypoint=lambda value: value or "research",
            should_stop=lambda _thread_id: False,
        )

        await service.submit(
            thread_id=thread.thread_id,
            payload=SimpleNamespace(
                text="first",
                attachments=[],
                entrypoint="research",
                llm_config="",
                permission_mode="hitl",
            ),
        )
        await started.wait()
        await service.submit(
            thread_id=thread.thread_id,
            payload=SimpleNamespace(
                text="stale steering",
                attachments=[],
                entrypoint="research",
                llm_config="",
                permission_mode="hitl",
            ),
        )

        result = await service.stop(
            thread_id=thread.thread_id,
            payload=SimpleNamespace(emergency=False, reason="user stop"),
        )
        assert result["status"] == "stopped"
        assert prompts == ["first"]
        stopped = store.get_thread(thread.thread_id)
        assert stopped.status == "stopped"
        assert stopped.pending_steering == []
        assert thread.thread_id not in tasks

    asyncio.run(_run())


def test_agent_loop_stop_preserves_remote_receipt_parts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[
            MessagePart(id="part_text", type="text", text="", status="streaming"),
            MessagePart(
                id="part_receipt_dp_test",
                type="receipt",
                text=".deepagents/dpdispatcher/receipts/dp_test.json",
                status="updated",
                meta={"remote_context_id": "dp_test", "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_test.json"},
            ),
        ],
    )
    store.append_message(message)
    store.update_thread(thread.thread_id, status="running", active_message_id=message.id, active_run_id="run_receipt")

    service = ThreadAgentLoopService(
        workspace=workspace,
        workspace_id="default",
        store=store,
        broker=broker,
        artifact_registry=registry,
        thread_tasks={},  # type: ignore[arg-type]
        thread_stop_flags=set(),
        build_runner=lambda **_kwargs: None,
        streaming_runner_cls=object,
        permission_mode_for_thread=lambda thread, override="": str(override or thread.meta.get("permission_mode") or "hitl"),
        interrupt_on_for_permission_mode=lambda _mode: {},
        normalize_entrypoint=lambda value: value or "research",
        should_stop=lambda _thread_id: False,
    )

    result = awaitable_result(service.stop(thread_id=thread.thread_id, payload=SimpleNamespace(emergency=False, reason="user stop")))

    assert result["status"] == "stopped"
    saved = store.get_message(thread.thread_id, message.id)
    receipt_parts = [part for part in saved.parts if part.type == "receipt"]
    assert len(receipt_parts) == 1
    assert receipt_parts[0].meta["remote_context_id"] == "dp_test"


def test_streaming_runner_emits_usage_updated_after_summary_write(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "reports").mkdir(parents=True)
    (workspace / "files" / "reports" / "sidecar.md").write_text("sidecar artifact", encoding="utf-8")
    run_dir = tmp_path / "run_usage"
    run_dir.mkdir()
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    staged_assets: list[dict[str, str]] = []
    model_message = AIMessage(
        id="resp_usage_stream",
        content="streamed answer",
        response_metadata={"model_name": "test-model"},
        usage_metadata={
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
            "input_token_details": {"cache_read": 1},
            "output_token_details": {"reasoning": 2},
        },
    )

    class FakeAgent:
        async def astream_events(self, _payload, config=None, version="v3"):
            yield {
                "event": "on_chat_model_end",
                "run_id": "llm_usage_stream",
                "metadata": {"lc_agent_name": "experiment_specialist"},
                "data": {"output": model_message},
            }
            yield {
                "method": "values",
                "params": {
                    "data": {
                        "messages": [model_message],
                        "structured_sidecar": {"artifact_refs": [{"path": "reports/sidecar.md"}]},
                    }
                },
            }

    class FakeRunner:
        run_context = SimpleNamespace(
            workspace=workspace,
            run_id="run_usage",
            run_dir=run_dir,
            project_id="default",
        )

        def _stage_deepagent_assets(self, files_root, *, thread_id):
            staged_assets.append(
                {
                    "files_root": str(files_root),
                    "thread_id": str(thread_id),
                }
            )

        def _new_usage_callback(self):
            return SpecialistUsageCallbackHandler()

        def _emit(self, *_args, **_kwargs):
            return None

        def _write_run_state(self, payload):
            (run_dir / "run_state.json").write_text(json.dumps(payload), encoding="utf-8")

        @asynccontextmanager
        async def _open_agent_runtime(self, *, files_root):
            yield {}

        async def _build_entry_agent(
            self,
            *,
            entrypoint,
            runtime,
            thread_id,
            tool_thread_id,
        ):
            assert tool_thread_id == thread.thread_id
            return FakeAgent()

        def _langchain_callbacks(self, *, usage_handler, default_agent_name):
            return [usage_handler]

        def _finalize_report(self, report):
            return report

        def _coerce_report(self, *, raw):
            return {
                "text": "Final answer.",
                "summary": "ok",
                "facts": [],
                "files": [],
                "review_target": "",
            }

        def _artifact_rows(self, _reported_files):
            return [{"path": path} for path in _reported_files]

        def _research_kernel_state_fields(self, **_kwargs):
            return {}

        def _research_goal_state_fields(self, **_kwargs):
            return {}

        def _write_usage_summary(self, usage_handler):
            ObservabilityStore(run_dir).record_event(
                source="test",
                channel="callback",
                name="LLM_CALL_END",
                category="llm",
                ts=1.0,
                seq=None,
                run_id="run_usage",
                task_id="",
                step_id=None,
                payload={
                    "model": "test-model",
                    "callback_run_id": "usage-callback",
                    "usage": usage_handler.usage_metadata["test-model"],
                },
            )
            (run_dir / "usage_summary.json").write_text(
                json.dumps(
                    {
                        "source": "langchain_usage_metadata",
                        "input_tokens": usage_handler.usage_metadata["test-model"]["input_tokens"],
                        "input_uncached_tokens": 2,
                        "input_cached_tokens": 1,
                        "input_cache_write_tokens": 0,
                        "output_tokens": usage_handler.usage_metadata["test-model"]["output_tokens"],
                        "reasoning_tokens": 2,
                        "total_tokens": usage_handler.usage_metadata["test-model"]["total_tokens"],
                        "calls": 1,
                    }
                ),
                encoding="utf-8",
            )
            return {
                "source": "langchain_usage_metadata",
                "input_tokens": usage_handler.usage_metadata["test-model"]["input_tokens"],
                "input_uncached_tokens": 2,
                "input_cached_tokens": 1,
                "input_cache_write_tokens": 0,
                "output_tokens": usage_handler.usage_metadata["test-model"]["output_tokens"],
                "reasoning_tokens": 2,
                "total_tokens": usage_handler.usage_metadata["test-model"]["total_tokens"],
                "calls": 1,
                "raw_usage_metadata": dict(usage_handler.usage_metadata),
                "call_counts_by_model": dict(usage_handler.call_counts_by_model),
                "raw_usage_metadata_by_role": dict(usage_handler.usage_metadata_by_role),
                "call_counts_by_role": dict(usage_handler.call_counts_by_role),
            }

    runner = StreamingSpecialistRunner(
        runner=FakeRunner(),  # type: ignore[arg-type]
        thread_store=store,
        event_broker=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
    )

    result = awaitable_result(
        runner.arun_turn(
            prompt="hello",
            entrypoint="experiment",
            thread_id=thread.thread_id,
            message_id=message.id,
            text_part_id="part_text",
            deepagent_thread_id=thread.deepagent_thread_id,
        )
    )

    assert result["status"] == "done"
    assert staged_assets == [
        {
            "files_root": str(workspace / "files"),
            "thread_id": thread.thread_id,
        }
    ]
    assert result["artifact_ids"]
    saved = store.get_message(thread.thread_id, message.id)
    artifact_parts = [part for part in saved.parts if part.type == "artifact"]
    assert len(artifact_parts) == 1
    assert artifact_parts[0].path == "files/reports/sidecar.md"
    usage_events = [event for event in broker.replay(thread.thread_id) if event.event == "usage.updated"]
    assert len(usage_events) == 1
    completed_event = next(event for event in broker.replay(thread.thread_id) if event.event == "message.completed")
    assert usage_events[0].seq < completed_event.seq
    assert usage_events[0].data["run_id"] == "run_usage"
    assert usage_events[0].data["usage_summary"]["input_tokens"] == 3
    assert usage_events[0].data["usage_summary"]["output_tokens"] == 5
    assert usage_events[0].data["usage_summary"]["total_tokens"] == 8


def awaitable_result(awaitable):
    import asyncio

    async def _run():
        result = await awaitable
        await asyncio.sleep(0)
        return result

    return asyncio.run(_run())


def test_stream_translator_persists_interrupt_state(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_x",
    )

    translator.apply_v3_event(
        {
            "method": "updates",
            "params": {
                "interrupts": [
                    SimpleNamespace(
                        value={
                            "action_requests": [{"name": "write_file", "args": {"path": "x.txt", "content": "x"}}],
                            "review_configs": [{"action_name": "write_file", "allowed_decisions": ["approve"]}],
                        },
                        id="interrupt-sdk",
                    )
                ]
            },
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    assert saved.status == "interrupted"
    assert store.get_thread(thread.thread_id).status == "interrupted"
    interrupt_parts = [part for part in saved.parts if part.type == "interrupt"]
    assert interrupt_parts and interrupt_parts[0].status == "pending"
    assert interrupt_parts[0].meta["payload"]["interrupts"][0]["value"]["action_requests"][0]["name"] == "write_file"
    refreshed_store = ThreadStore(workspace=workspace, workspace_id="default")
    refreshed = refreshed_store.get_message(thread.thread_id, message.id)
    refreshed_interrupt_parts = [part for part in refreshed.parts if part.type == "interrupt"]
    assert refreshed_interrupt_parts and refreshed_interrupt_parts[0].status == "pending"
    assert [event.event for event in broker.replay(thread.thread_id, last_seq=0)] == ["interrupt.created", "thread.status"]


def test_stream_translator_projects_each_artifact_once_per_message(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "evidence.pdf").write_bytes(b"%PDF-1.4\n")
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_artifact",
    )
    first = registry.register_path(
        "evidence.pdf",
        thread_id=thread.thread_id,
        message_id=message.id,
        tool_call_id="call_one",
    )
    repeated = registry.register_path(
        "evidence.pdf",
        thread_id=thread.thread_id,
        message_id=message.id,
        tool_call_id="call_two",
    )

    assert translator.publish_artifact(first) is True
    assert translator.publish_artifact(repeated) is False
    saved = store.get_message(thread.thread_id, message.id)
    assert len([part for part in saved.parts if part.type == "artifact"]) == 1
    assert [event.event for event in broker.replay(thread.thread_id, last_seq=0)] == [
        "artifact.created"
    ]


def test_extract_workspace_paths_from_natural_final_text() -> None:
    paths = _extract_workspace_paths_from_text(
        "Files written: `o2_smoke/O2.xyz`, `o2_smoke/o2_report.md`; not a path: `Summary`."
    )

    assert paths == ["o2_smoke/O2.xyz", "o2_smoke/o2_report.md"]


def test_extract_workspace_paths_from_final_text_ignores_units_and_params() -> None:
    paths = _extract_workspace_paths_from_text(
        "Converged at `fmax=0.02 eV/Å` with max force `0.01556 eV/Å`; "
        "files: `structures/o2_relaxed.extxyz`, `notes/o2_mace_relax_summary.json`."
    )

    assert paths == ["structures/o2_relaxed.extxyz", "notes/o2_mace_relax_summary.json"]


def test_extract_sidecar_artifact_paths_ignores_artifact_ids() -> None:
    paths = _extract_sidecar_artifact_paths(
        {
            "structured_sidecar": {
                "artifact_ids": ["art_existing"],
                "artifact_refs": [
                    {"artifact_id": "art_skip", "path": "reports/sidecar.md"},
                    "tables/results.csv",
                ],
                "artifact_paths": ["figures/plot.png"],
            }
        }
    )

    assert paths == ["reports/sidecar.md", "tables/results.csv", "figures/plot.png"]


def test_stream_translator_ignores_historical_tool_messages_from_state_snapshots(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    previous = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="completed",
        parts=[
            MessagePart(
                id="part_tool_old",
                type="tool-call",
                text="old output",
                status="completed",
                meta={"tool_call_id": "tc_old", "tool": "ls", "input": {"path": "/"}, "output": "old output"},
            )
        ],
    )
    store.append_message(previous)
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_snapshot",
    )

    translator.apply_v3_event(
        {
            "method": "values",
            "params": {
                "data": {
                    "messages": [
                        ToolMessage(content="old output", tool_call_id="tc_old", name="ls"),
                    ]
                }
            },
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    assert [part for part in saved.parts if part.type == "tool-call"] == []
    assert broker.replay(thread.thread_id) == []
    assert translator.last_values["messages"]


def test_stream_translator_deduplicates_repeated_tool_messages_without_message_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_snapshot_dedupe",
    )

    translator.apply_v3_event(
        {"method": "tool_calls", "params": {"data": {"id": "tc_new", "name": "ls", "args": {"path": "/new"}}}}
    )
    for _ in range(2):
        translator.apply_v3_event(
            {
                "method": "values",
                "params": {"data": {"messages": [ToolMessage(content="new output", tool_call_id="tc_new", name="ls")]}},
            }
        )

    saved = store.get_message(thread.thread_id, message.id)
    tool_parts = [part for part in saved.parts if part.type == "tool-call"]
    completed_events = [event for event in broker.replay(thread.thread_id) if event.event == "tool_call.completed"]
    assert len(tool_parts) == 1
    assert len(completed_events) == 1
    assert tool_parts[0].meta["input"] == {"path": "/new"}


def test_stream_translator_merges_token_tool_and_artifact_parts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "out.csv").write_text("x\n1\n", encoding="utf-8")
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_x",
    )

    translator.apply_v3_event({"method": "messages", "params": {"data": {"role": "assistant", "content": "hel"}}})
    translator.apply_v3_event({"method": "messages", "params": {"data": {"role": "assistant", "content": "lo"}}})
    translator.apply_v3_event({"method": "tool_calls", "params": {"data": {"id": "tc1", "name": "make_csv", "args": {"x": 1}}}})
    translator._handle_tool_message(
        ToolMessage(
            content="wrote file",
            tool_call_id="tc1",
            artifact={"data": {"output_path": "out.csv"}},
        )
    )
    store.append_part(
        thread.thread_id,
        message.id,
        MessagePart(
            id="part_reasoning",
            type="reasoning",
            text="working",
            status="streaming",
        ),
    )
    translator.complete("hello", sidecar={"summary": "ok"})

    saved = store.get_message(thread.thread_id, message.id)
    assert saved.parts[0].text == "hello"
    assert any(part.type == "tool-call" for part in saved.parts)
    artifact_parts = [part for part in saved.parts if part.type == "artifact"]
    assert artifact_parts
    assert registry.get(artifact_parts[0].artifact_id).renderer == "csv"
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.meta["input"] == {"x": 1}
    completed = [event for event in broker.replay(thread.thread_id) if event.event == "tool_call.completed"]
    assert completed[-1].data["input"] == {"x": 1}
    message_completed = [event for event in broker.replay(thread.thread_id) if event.event == "message.completed"][-1]
    assert message_completed.data["text"] == "hello"
    assert message_completed.data["message"]["status"] == "completed"
    assert message_completed.data["message"]["parts"][0]["text"] == "hello"
    assert next(
        part
        for part in message_completed.data["message"]["parts"]
        if part["id"] == "part_reasoning"
    )["status"] == "completed"
    public_event = project_event(message_completed, workspace=workspace)
    assert public_event.data.message is not None
    assert public_event.data.message.status == "completed"
    assert public_event.data.message.parts[0].text == "hello"


def test_message_part_pages_keep_ref_and_todos_use_full_current_turn(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post(
        "/api/workspaces/default/threads",
        json={"title": "Long turn"},
    ).json()["thread"]["thread_id"]
    store = ThreadStore(workspace=workspace, workspace_id="default")
    store.append_message(
        ThreadMessage(
            id="msg_user_long_parts",
            thread_id=thread_id,
            role="user",
            status="completed",
            parts=[
                MessagePart(
                    id="part_user_long_parts",
                    type="text",
                    text="Run the bounded review.",
                    status="completed",
                )
            ],
        )
    )
    parts = [
        MessagePart(
            id=f"part_long_{index:03d}",
            type="reasoning",
            text=f"step {index}",
            status="completed",
        )
        for index in range(85)
    ]
    parts[10] = MessagePart(
        id="part_todo_initial",
        type="tool-call",
        status="completed",
        meta={
            "tool": "write_todos",
            "agent_name": "litreview_agent",
            "input": {
                "todos": [
                    {"content": "Search representative work", "status": "completed"},
                    {"content": "Write synthesis", "status": "in_progress"},
                ]
            },
        },
    )
    parts[84] = MessagePart(
        id="part_todo_final",
        type="tool-call",
        status="completed",
        meta={
            "tool": "write_todos",
            "agent_name": "litreview_agent",
            "input": {
                "todos": [
                    {"content": "Search representative work", "status": "completed"},
                    {"content": "Write synthesis", "status": "completed"},
                ]
            },
        },
    )
    store.append_message(
        ThreadMessage(
            id="msg_assistant_long_parts",
            thread_id=thread_id,
            role="assistant",
            status="completed",
            parts=parts,
        )
    )

    message_page = client.get(f"/api/threads/{thread_id}/messages").json()
    assistant = message_page["messages"][-1]
    assert assistant["parts_page"]["shown_count"] == 20
    assert assistant["parts_page"]["total_count"] == 85
    assert message_page["todo_parts"][0]["summary"] == "2 of 2 items complete."
    assert message_page["todo_parts"][0]["id"] == "part_todo_final"

    ref = assistant["parts_page"]["full_content_ref"]
    cursor = assistant["parts_page"]["next_cursor"]
    shown_counts = [20]
    while cursor:
        response = client.get(ref, params={"cursor": cursor, "limit": 20})
        assert response.status_code == 200
        page = response.json()["page"]
        shown_counts.append(page["shown_count"])
        cursor = page["next_cursor"]
        if cursor:
            assert page["full_content_ref"] == ref
    assert shown_counts == [20, 40, 60, 80, 85]


def test_completed_message_pushes_only_closed_canonical_todo_plans(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post(
        "/api/workspaces/default/threads",
        json={"title": "Terminal todo"},
    ).json()["thread"]["thread_id"]
    store = ThreadStore(workspace=workspace, workspace_id="default")
    store.append_message(
        ThreadMessage(
            id="msg_user_terminal_todo",
            thread_id=thread_id,
            role="user",
            status="completed",
            parts=[MessagePart(id="part_user_terminal_todo", type="text", text="Run it.", status="completed")],
        )
    )
    assistant = ThreadMessage(
        id="msg_assistant_terminal_todo",
        thread_id=thread_id,
        role="assistant",
        status="completed",
        parts=[
            MessagePart(
                id="part_root_todo",
                type="tool-call",
                status="completed",
                meta={
                    "tool": "write_todos",
                    "agent_name": "litreview_agent",
                    "input": {"todos": [{"content": "Write synthesis", "status": "completed"}]},
                },
            ),
            MessagePart(
                id="part_child_todo",
                type="tool-call",
                status="completed",
                meta={
                    "tool": "write_todos",
                    "agent_name": "general-purpose",
                    "input": {
                        "todos": [
                            {"content": "Collect papers", "status": "completed"},
                            {"content": "Return handoff", "status": "in_progress"},
                        ]
                    },
                },
            ),
        ],
    )
    store.append_message(assistant)

    message_page = client.get(f"/api/threads/{thread_id}/messages").json()
    assert [part["id"] for part in message_page["todo_parts"]] == ["part_root_todo"]

    event = ThreadEventBroker(workspace=workspace).emit(
        thread_id,
        "message.completed",
        message_id=assistant.id,
        status="completed",
        data={"message": assistant.model_dump(mode="json")},
    )
    projected = project_event(event, workspace=workspace)
    assert [part.id for part in projected.data.todo_parts] == ["part_root_todo"]


def test_task_result_projects_subagent_markdown_instead_of_command_repr(
    tmp_path: Path,
) -> None:
    content = """## Bounded evidence handoff

### Judgment

The matched evidence favors a coupled Pt-O-Ce mechanism over either isolated variable.

### Evidence constraints

The isotope and operando measurements were not performed in one synchronized experiment.
"""
    command = Command(
        update={
            "files": {},
            "messages": [ToolMessage(content=content, tool_call_id="call_task_result")],
        }
    )
    structured = _json_safe(command)
    assert structured["type"] == "Command"
    assert structured["update"]["messages"][0]["content"] == content

    for output in (structured, str(command)):
        projected = project_tool_part(
            {
                "id": "part_task_result",
                "type": "tool-call",
                "status": "completed",
                "meta": {
                    "tool": "task",
                    "input": {"subagent_type": "general-purpose"},
                    "output": output,
                },
            },
            workspace=tmp_path,
            thread_id="thread_task_result",
            message_id="msg_task_result",
        )
        assert projected.title == "Research assistant · Bounded evidence handoff"
        assert projected.summary.startswith("The matched evidence favors")
        assert [item.label for item in projected.items] == ["Judgment", "Evidence constraints"]
        assert "Command(update=" not in projected.summary
        assert projected.fields == []


def test_stream_translator_registers_only_existing_files_from_tool_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "files" / "result.csv").write_text("x\n1\n", encoding="utf-8")
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=ThreadEventBroker(workspace=workspace),
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_artifact_filter",
    )

    translator._handle_tool_message(
        ToolMessage(
            content="task spec",
            tool_call_id="call_task_spec",
            artifact={
                "data": {
                    "task_fields": [
                        {"path": "task_config.fmax"},
                        {"path": "task_config.steps"},
                    ],
                    "output_path": "result.csv",
                }
            },
        )
    )

    records = registry.list_artifacts(thread_id=thread.thread_id)
    assert [record.path for record in records] == ["files/result.csv"]
    artifact_parts = [
        part
        for part in store.get_message(thread.thread_id, message.id).parts
        if part.type == "artifact"
    ]
    assert [part.path for part in artifact_parts] == ["files/result.csv"]


def test_stream_translator_surfaces_tool_call_model_text_as_progress(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_tool_progress",
    )

    translator.apply_v3_event(
        {
            "event": "on_chat_model_end",
            "metadata": {"lc_agent_name": "materials_worker", "langgraph_checkpoint_ns": "task:materials_worker"},
            "data": {
                "output": AIMessage(
                    content="Preflight: managed `mace_relax_dir` is available.",
                    tool_calls=[{"id": "tc_progress", "name": "write_todos", "args": {"todos": ["prepare O2"]}}],
                )
            },
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    text_part = next(part for part in saved.parts if part.id == "part_text")
    reasoning_part = next(part for part in saved.parts if part.type == "reasoning")
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert text_part.text == ""
    assert reasoning_part.text == "Preflight: managed `mace_relax_dir` is available."
    assert tool_part.meta["tool"] == "write_todos"
    assert tool_part.meta["input"] == {"todos": ["prepare O2"]}
    assert tool_part.meta["subagent_source"] == "materials_worker"
    events = broker.replay(thread.thread_id)
    assert [event for event in events if event.event == "reasoning.delta"][-1].data["delta"] == "Preflight: managed `mace_relax_dir` is available."
    assert [event for event in events if event.event == "tool_call.started"][-1].data["input"] == {"todos": ["prepare O2"]}


def test_stream_translator_merges_incremental_tool_call_args(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_chunked_tool",
    )

    translator._handle_tool_call_payload({"id": "tc_chunk", "index": 0, "name": "write_file", "args": "{\"path\":\"notes/"})
    translator._handle_tool_call_payload({"index": 0, "args": "summary.md\",\"content\":\"ok\"}"})
    translator._handle_tool_message(ToolMessage(content="done", tool_call_id="tc_chunk", name="write_file"))

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.status == "completed"
    assert tool_part.meta["tool"] == "write_file"
    assert tool_part.meta["input"] == {"path": "notes/summary.md", "content": "ok"}
    events = broker.replay(thread.thread_id)
    delta_events = [event for event in events if event.event == "tool_call.delta"]
    completed_events = [event for event in events if event.event == "tool_call.completed"]
    assert delta_events[-1].data["input"] == {"path": "notes/summary.md", "content": "ok"}
    assert completed_events[-1].data["input"] == {"path": "notes/summary.md", "content": "ok"}


def test_stream_translator_omits_injected_runtime_from_v3_tool_input(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_injected_runtime",
    )

    class _RuntimeLike:
        def __str__(self) -> str:
            raise AssertionError("Injected ToolRuntime must not be serialized")

    translator.apply_v3_event(
        {
            "method": "tools",
            "params": {
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "tc_runtime",
                    "tool_name": "read_file",
                    "input": {
                        "file_path": "/notes/report.md",
                        "runtime": _RuntimeLike(),
                        "config": {"tags": ["internal"]},
                    },
                }
            },
        }
    )

    expected = {"file_path": "/notes/report.md"}
    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.meta["input"] == expected
    started = [event for event in broker.replay(thread.thread_id) if event.event == "tool_call.started"][-1]
    assert started.data["input"] == expected


def test_stream_translator_labels_tool_calls_with_agent_source(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_tool_source",
    )

    translator._handle_tool_call_payload(
        {"id": "tc_todos", "name": "write_todos", "args": {"todos": [{"content": "plan", "status": "in_progress"}]}},
        metadata={"lc_agent_name": "experiment_specialist", "langgraph_node": "model"},
    )

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    started = [event for event in broker.replay(thread.thread_id) if event.event == "tool_call.started"][-1]
    assert tool_part.meta["agent_name"] == "experiment_specialist"
    assert tool_part.meta["subagent_source"] == "experiment_specialist"
    assert started.data["agent_name"] == "experiment_specialist"
    assert started.data["subagent_source"] == "experiment_specialist"


def test_stream_translator_backfills_tool_agent_source_from_callback_observation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_dir = tmp_path / "run_tool_observed_source"
    run_id = "run_tool_observed_source"
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    observability = ObservabilityStore(run_dir)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id=run_id,
        observability_store=observability,
    )
    todos = [{"content": "Prepare input", "status": "in_progress"}]

    translator._handle_tool_call_payload({"id": "tc_todos", "name": "write_todos", "args": {"todos": todos}})
    observability.record_event(
        source="langchain_callback",
        channel="callback",
        name="TOOL_CALL_START",
        category="tool",
        ts=1.0,
        seq=None,
        run_id=run_id,
        task_id="",
        step_id=None,
        payload={
            "tool": "write_todos",
            "tool_name": "write_todos",
            "agent_name": "materials_worker",
            "params_compact": json.dumps({"todos": todos}),
        },
    )
    translator._handle_tool_message(ToolMessage(content="Updated todo list", tool_call_id="tc_todos", name="write_todos"))

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    completed = [event for event in broker.replay(thread.thread_id) if event.event == "tool_call.completed"][-1]
    assert tool_part.meta["agent_name"] == "materials_worker"
    assert tool_part.meta["subagent_source"] == "materials_worker"
    assert completed.data["agent_name"] == "materials_worker"
    assert completed.data["subagent_source"] == "materials_worker"


def test_stream_translator_backfills_tool_args_from_chat_model_end(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_tool_end_args",
    )

    translator._handle_tool_call_payload({"id": "tc_end", "name": "ls", "args": {}})
    translator.apply_v3_event(
        {
            "event": "on_chat_model_end",
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[{"id": "tc_end", "name": "ls", "args": {"path": "/"}}],
                )
            },
        }
    )
    translator._handle_tool_message(ToolMessage(content="['/.deepagents/']", tool_call_id="tc_end", name="ls"))

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.meta["input"] == {"path": "/"}
    events = broker.replay(thread.thread_id)
    delta_events = [event for event in events if event.event == "tool_call.delta"]
    completed_events = [event for event in events if event.event == "tool_call.completed"]
    assert delta_events[-1].data["input"] == {"path": "/"}
    assert completed_events[-1].data["input"] == {"path": "/"}


def test_stream_translator_backfills_tool_args_from_observability_store(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_dir = system_root(workspace) / "runs" / "run_observed_args"
    ObservabilityStore(run_dir).record_event(
        source="langchain_callback",
        channel="callback",
        category="llm",
        name="LLM_RAW_RESPONSE",
        ts=1.0,
        seq=None,
        run_id="run_observed_args",
        task_id="",
        step_id=None,
        payload={
            "generations": [
                {
                    "parsed_tool_calls": [
                        {
                            "id": "tc_observed",
                            "name": "ls",
                            "args_json": "{\"path\":\"/files\"}",
                            "raw": {"id": "tc_observed", "name": "ls", "args": {"path": "/files"}, "type": "tool_call"},
                        }
                    ]
                }
            ]
        },
    )
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_observed_args",
        observability_store=ObservabilityStore(run_dir),
    )

    translator._handle_tool_call_payload({"id": "tc_observed", "name": "ls", "args": {}})
    translator._handle_tool_message(ToolMessage(content="[]", tool_call_id="tc_observed", name="ls"))

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.meta["input"] == {"path": "/files"}
    events = broker.replay(thread.thread_id)
    assert [event for event in events if event.event == "tool_call.started"][-1].data["input"] == {"path": "/files"}
    assert [event for event in events if event.event == "tool_call.completed"][-1].data["input"] == {"path": "/files"}


def test_stream_translator_backfills_resumed_approved_tool_input(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_resumed_args",
        resume_tool_inputs=[
            {
                "name": "write_file",
                "args": {"file_path": "/notes/hitl.txt", "content": "approved"},
            }
        ],
    )

    translator._handle_tool_message(ToolMessage(content="Updated file /notes/hitl.txt", tool_call_id="tc_resume", name="write_file"))

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    expected = {"file_path": "/notes/hitl.txt", "content": "approved"}
    assert tool_part.meta["input"] == expected
    events = broker.replay(thread.thread_id)
    assert [event for event in events if event.event == "tool_call.started"][-1].data["input"] == expected
    assert [event for event in events if event.event == "tool_call.completed"][-1].data["input"] == expected


def test_stream_activity_id_uses_task_lifecycle_and_groups_root_tools_by_agent() -> None:
    assert _agent_run_id(
        {"namespace": ["task:candidate_a", "tools:1"]},
        source="materials_worker",
    ) == "task:candidate_a"
    assert _agent_run_id(
        {"namespace": ["tools:root_call"]},
        source="research_specialist",
    ) == "agent:research_specialist"


def test_stream_translator_attaches_subagent_source_to_tool_calls(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_subagent_tool_source",
    )

    translator.apply_v3_event(
        {
            "event": "on_tool_start",
            "name": "write_file",
            "metadata": {"lc_agent_name": "materials_worker", "namespace": ["task:o2_relax", "tools:1"]},
            "data": {"input": {"file_path": "/notes/o2.txt", "content": "ok"}},
        }
    )
    translator.apply_v3_event(
        {
            "event": "on_tool_end",
            "name": "write_file",
            "metadata": {"lc_agent_name": "materials_worker", "namespace": ["task:o2_relax", "tools:1"]},
            "data": {"output": "Updated file /notes/o2.txt"},
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.meta["subagent_source"] == "materials_worker"
    assert tool_part.meta["agent_run_id"] == "task:o2_relax"
    assert tool_part.meta["stream_namespace"] == ["task:o2_relax", "tools:1"]
    events = broker.replay(thread.thread_id)
    started = [event for event in events if event.event == "tool_call.started"][-1]
    completed = [event for event in events if event.event == "tool_call.completed"][-1]
    assert started.data["subagent_source"] == "materials_worker"
    assert started.data["agent_run_id"] == "task:o2_relax"
    assert completed.data["subagent_source"] == "materials_worker"
    assert completed.data["agent_run_id"] == "task:o2_relax"


def test_stream_translator_creates_tool_part_for_orphan_tool_message(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_orphan_tool",
    )

    translator._handle_tool_message(ToolMessage(content="orphan output", tool_call_id="tc_orphan", name="read_file"))

    saved = store.get_message(thread.thread_id, message.id)
    tool_parts = [part for part in saved.parts if part.type == "tool-call"]
    assert len(tool_parts) == 1
    assert tool_parts[0].status == "completed"
    assert tool_parts[0].text == "orphan output"
    event_names = [event.event for event in broker.replay(thread.thread_id)]
    assert event_names[-2:] == ["tool_call.started", "tool_call.completed"]


def test_stream_translator_separates_reasoning_and_internal_subagent_streams(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_stream_classification",
    )

    translator.apply_v3_event(
        {
            "event": "on_chat_model_stream",
            "metadata": {"lc_agent_name": "research_specialist", "langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "reasoning", "text": "thinking\n"}, {"type": "text", "text": "Visible answer."}])},
        }
    )
    translator.apply_v3_event(
        {
            "event": "on_chat_model_stream",
            "metadata": {"lc_agent_name": "materials_worker", "langgraph_checkpoint_ns": "task:materials_worker"},
            "data": {"chunk": AIMessageChunk(content="Internal worker progress.")},
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    text_part = next(part for part in saved.parts if part.id == "part_text")
    reasoning_parts = [part for part in saved.parts if part.type == "reasoning"]
    subagent_parts = [part for part in saved.parts if part.type == "subagent"]
    assert text_part.text == "Visible answer."
    assert reasoning_parts and reasoning_parts[0].text == "thinking\n"
    assert subagent_parts and subagent_parts[0].text == "Internal worker progress."
    assert "Internal worker progress" not in text_part.text
    event_names = [event.event for event in broker.replay(thread.thread_id)]
    assert "message.delta" in event_names
    assert "reasoning.delta" in event_names
    assert "subagent.started" in event_names
    assert "subagent.delta" in event_names


def test_stream_translator_keeps_same_named_subagent_invocations_separate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_parallel_materials",
    )

    for task_id, text in (
        ("candidate_a", "Inspecting candidate A."),
        ("candidate_b", "Inspecting candidate B."),
    ):
        translator.apply_v3_event(
            {
                "method": "messages",
                "params": {
                    "namespace": [f"task:{task_id}", "model_request:1"],
                    "metadata": {"lc_agent_name": "materials_worker"},
                    "data": [
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": text},
                        }
                    ],
                },
            }
        )

    saved = store.get_message(thread.thread_id, message.id)
    subagent_parts = [part for part in saved.parts if part.type == "subagent"]
    assert len(subagent_parts) == 2
    assert [part.meta["source"] for part in subagent_parts] == ["materials_worker", "materials_worker"]
    assert [part.meta["agent_run_id"] for part in subagent_parts] == [
        "task:candidate_a",
        "task:candidate_b",
    ]
    assert [part.text for part in subagent_parts] == ["Inspecting candidate A.", "Inspecting candidate B."]


def test_stream_translator_surfaces_reasoning_text_and_content_fields(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_reasoning_fields",
    )

    translator.apply_v3_event(
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="", additional_kwargs={"reasoning_text": "Plan: "})},
        }
    )
    translator.apply_v3_event(
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Plan: inspect files"})},
        }
    )
    translator.apply_v3_event(
        {
            "method": "messages",
            "params": {
                "data": [
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "reasoning-delta", "reasoning": " and stream to audit"},
                    }
                ]
            },
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    text_part = next(part for part in saved.parts if part.id == "part_text")
    reasoning_part = next(part for part in saved.parts if part.type == "reasoning")
    assert text_part.text == ""
    assert reasoning_part.text == "Plan: inspect files and stream to audit"
    reasoning_events = [event for event in broker.replay(thread.thread_id) if event.event == "reasoning.delta"]
    assert [event.data["delta"] for event in reasoning_events] == ["Plan: ", "inspect files", " and stream to audit"]


def test_stream_translator_surfaces_model_end_reasoning_content(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_model_end_reasoning",
    )

    translator.apply_v3_event(
        {
            "event": "on_chat_model_end",
            "metadata": {"lc_agent_name": "research_specialist", "langgraph_node": "model"},
            "data": {
                "output": AIMessage(
                    content="",
                    additional_kwargs={"reasoning_content": "Need to plan the MACE relaxation before tools."},
                    tool_calls=[{"id": "tc_plan", "name": "write_todos", "args": {"todos": ["plan"]}}],
                )
            },
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    reasoning_part = next(part for part in saved.parts if part.type == "reasoning")
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert reasoning_part.text == "Need to plan the MACE relaxation before tools."
    assert tool_part.meta["tool"] == "write_todos"
    reasoning_events = [event for event in broker.replay(thread.thread_id) if event.event == "reasoning.delta"]
    assert reasoning_events[-1].data["delta"] == "Need to plan the MACE relaxation before tools."


def test_stream_translator_flushes_callback_reasoning_text(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_dir = tmp_path / "run_observed_reasoning"
    run_id = "run_observed_reasoning"
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    observability = ObservabilityStore(run_dir)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id=run_id,
        observability_store=observability,
    )
    observability.record_event(
        source="langchain_callback",
        channel="callback",
        name="LLM_CALL_END",
        category="llm",
        ts=1.0,
        seq=None,
        run_id=run_id,
        task_id="",
        step_id=None,
        payload={
            "agent_name": "materials_worker",
            "callback_run_id": "llm_1",
            "node": "model",
            "reasoning_text": "**Planning** Use MACE then audit outputs.",
        },
    )

    translator.flush_observed_reasoning()
    translator.flush_observed_reasoning()

    saved = store.get_message(thread.thread_id, message.id)
    reasoning_part = next(part for part in saved.parts if part.type == "reasoning")
    assert reasoning_part.text == "**Planning** Use MACE then audit outputs."
    assert reasoning_part.meta["source"] == "materials_worker"
    reasoning_events = [event for event in broker.replay(thread.thread_id) if event.event == "reasoning.delta"]
    assert [event.data["delta"] for event in reasoning_events] == ["**Planning** Use MACE then audit outputs."]


def test_stream_translator_handles_v3_content_block_delta_messages(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_v3_delta",
    )

    translator.apply_v3_event(
        {
            "method": "messages",
            "params": {
                "namespace": [],
                "data": [
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "I will inspect the staged inputs first. "},
                    }
                ],
            },
        }
    )
    translator.apply_v3_event(
        {
            "method": "messages",
            "params": {
                "namespace": ["tools:abc123", "model_request:def456"],
                "data": [
                    {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "Subagent is preparing the MACE stage."},
                    }
                ],
            },
        }
    )

    saved = store.get_message(thread.thread_id, message.id)
    text_part = next(part for part in saved.parts if part.id == "part_text")
    subagent_parts = [part for part in saved.parts if part.type == "subagent"]
    assert text_part.text == "I will inspect the staged inputs first. "
    assert subagent_parts and subagent_parts[0].text == "Subagent is preparing the MACE stage."
    assert "Subagent is preparing" not in text_part.text
    event_names = [event.event for event in broker.replay(thread.thread_id)]
    assert event_names == ["message.delta", "subagent.started", "subagent.delta"]


def test_stream_translator_projects_native_web_search_and_citations(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="default"),
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_native_search",
    )
    response = AIMessage(
        content=[
            {
                "type": "server_tool_call",
                "name": "web_search",
                "args": {"type": "search", "query": "OpenAI web search docs"},
                "id": "ws_1",
            },
            {
                "type": "server_tool_result",
                "tool_call_id": "ws_1",
                "status": "success",
                "output": {"sources": [{"url": "https://developers.openai.com/api/docs/guides/tools-web-search"}]},
            },
            {
                "type": "text",
                "text": "OpenAI documents hosted web search.",
                "annotations": [
                    {
                        "type": "citation",
                        "url": "https://developers.openai.com/api/docs/guides/tools-web-search",
                        "title": "Web search | OpenAI API",
                        "start_index": 0,
                        "end_index": 39,
                    }
                ],
            },
        ]
    )

    translator._handle_message_event((response, {"lc_agent_name": "experiment_specialist"}))
    translator._handle_message_event((response, {"lc_agent_name": "experiment_specialist"}))

    saved = store.get_message(thread.thread_id, message.id)
    tool_parts = [part for part in saved.parts if part.type == "tool-call"]
    assert len(tool_parts) == 1
    assert tool_parts[0].status == "completed"
    assert tool_parts[0].meta["tool"] == "web_search"
    assert tool_parts[0].meta["server_side"] is True
    assert tool_parts[0].meta["input"]["query"] == "OpenAI web search docs"
    assert translator.citations == [
        {
            "url": "https://developers.openai.com/api/docs/guides/tools-web-search",
            "title": "Web search | OpenAI API",
            "start_index": 0,
            "end_index": 39,
        }
    ]
    event_names = [event.event for event in broker.replay(thread.thread_id)]
    assert event_names.count("tool_call.started") == 1
    assert event_names.count("tool_call.completed") == 1
    assert "tool_call.delta" not in event_names


def test_stream_translator_persists_remote_receipt_parts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_receipt",
    )

    translator.apply_v3_event({"method": "tool_calls", "params": {"data": {"id": "tc_remote", "name": "remote_submission", "args": {"work_dir": "stage/mace"}}}})
    translator._handle_tool_message(
        ToolMessage(
            content="remote submitted",
            tool_call_id="tc_remote",
            artifact={
                "tool_name": "remote_submission",
                "data": {
                    "remote_context_id": "dp_test",
                    "submission_hash": "abc123",
                    "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_test.json",
                    "jobs": [{"job_id": "42", "status": "running"}],
                },
            },
        )
    )

    saved = store.get_message(thread.thread_id, message.id)
    receipt_parts = [part for part in saved.parts if part.type == "receipt"]
    assert len(receipt_parts) == 1
    assert receipt_parts[0].meta["remote_context_id"] == "dp_test"
    assert receipt_parts[0].meta["submission_hash"] == "abc123"
    events = broker.replay(thread.thread_id)
    receipt_events = [event for event in events if event.event == "task_receipt.updated"]
    assert len(receipt_events) == 1
    assert receipt_events[0].data["receipt"]["receipt_rel"].endswith("dp_test.json")


def test_streaming_adapter_falls_back_to_astream_event_schema(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_fallback",
    )

    class FakeFallbackAgent:
        async def astream_events(self, _payload, config=None, version="v3"):
            raise RuntimeError("provider does not support v3 stream events")

        async def astream(self, _payload, config=None, stream_mode=None):
            yield ("messages", {"role": "assistant", "content": "hel"})
            yield ("messages", {"role": "assistant", "content": "lo"})

    runner = StreamingSpecialistRunner(
        runner=SimpleNamespace(run_context=SimpleNamespace(run_dir=tmp_path, run_id="run_fallback")),  # type: ignore[arg-type]
        thread_store=store,
        event_broker=broker,
        artifact_registry=registry,
    )

    awaitable_result(
        runner._consume_agent_stream(
            FakeFallbackAgent(),
            input_payload={"messages": [{"role": "user", "content": "hello"}]},
            config={},
            translator=translator,
        )
    )

    saved = store.get_message(thread.thread_id, message.id)
    assert saved.parts[0].text == "hello"
    assert [event.event for event in broker.replay(thread.thread_id)] == ["message.delta", "message.delta"]


def test_streaming_adapter_emits_v3_tool_start_before_tool_finishes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_v3_tool_lifecycle",
    )

    import asyncio

    waiting_for_tool = asyncio.Event()
    release_tool = asyncio.Event()
    tool_message = ToolMessage(content="STREAM_DONE", tool_call_id="call_slow", name="execute")

    class FakeV3Agent:
        async def astream_events(self, _payload, config=None, version="v3"):
            yield {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-started",
                        "tool_call_id": "call_slow",
                        "tool_name": "execute",
                        "input": {"command": "sleep 12", "timeout": 30},
                    },
                },
            }
            waiting_for_tool.set()
            await release_tool.wait()
            yield {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-finished",
                        "tool_call_id": "call_slow",
                        "tool_name": "execute",
                        "output": tool_message,
                    },
                },
            }
            yield {"method": "values", "params": {"namespace": [], "data": {"messages": [tool_message]}}}

    runner = StreamingSpecialistRunner(
        runner=SimpleNamespace(run_context=SimpleNamespace(run_dir=tmp_path, run_id="run_v3_tool_lifecycle")),  # type: ignore[arg-type]
        thread_store=store,
        event_broker=broker,
        artifact_registry=registry,
    )

    async def _run() -> None:
        task = asyncio.create_task(
            runner._consume_agent_stream(
                FakeV3Agent(),
                input_payload={"messages": [{"role": "user", "content": "Run the slow tool."}]},
                config={},
                translator=translator,
            )
        )
        await asyncio.wait_for(waiting_for_tool.wait(), timeout=1.0)
        live_events = [event.event for event in broker.replay(thread.thread_id)]
        assert live_events == ["tool_call.started"]
        running_message = store.get_message(thread.thread_id, message.id)
        running_part = next(part for part in running_message.parts if part.type == "tool-call")
        assert running_part.status == "running"
        assert running_part.meta["input"] == {"command": "sleep 12", "timeout": 30}
        release_tool.set()
        await task

    asyncio.run(_run())

    final_events = [event.event for event in broker.replay(thread.thread_id)]
    assert final_events == ["tool_call.started", "tool_call.completed"]
    saved = store.get_message(thread.thread_id, message.id)
    tool_parts = [part for part in saved.parts if part.type == "tool-call"]
    assert len(tool_parts) == 1
    assert tool_parts[0].status == "completed"
    assert tool_parts[0].text == "STREAM_DONE"


def test_streaming_adapter_yields_for_steering_after_completed_tool_boundary(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_steering_boundary",
    )
    steering = {"queued": False}
    calls: list[object] = []

    class FakeV3Agent:
        async def astream_events(
            self,
            payload,
            config=None,
            version="v3",
            interrupt_after=None,
        ):
            calls.append(payload)
            assert interrupt_after == ["tools"]
            yield {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-started",
                        "tool_call_id": "call_safe",
                        "tool_name": "execute",
                        "input": {"task": "bounded work"},
                    },
                },
            }
            steering["queued"] = True
            yield {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "tool-finished",
                        "tool_call_id": "call_safe",
                        "tool_name": "execute",
                        "output": ToolMessage(
                            content="completed before steering",
                            tool_call_id="call_safe",
                            name="execute",
                        ),
                    },
                },
            }

        async def aget_state(self, _config):
            return SimpleNamespace(next=("model",))

    runner = StreamingSpecialistRunner(
        runner=SimpleNamespace(
            run_context=SimpleNamespace(
                run_dir=tmp_path,
                run_id="run_steering_boundary",
            )
        ),  # type: ignore[arg-type]
        thread_store=store,
        event_broker=broker,
        artifact_registry=registry,
        should_steer=lambda _thread_id: steering["queued"],
    )

    yielded = awaitable_result(
        runner._consume_agent_stream(
            FakeV3Agent(),
            input_payload={"messages": [{"role": "user", "content": "first"}]},
            config={"configurable": {"thread_id": "deepagent-safe"}},
            translator=translator,
        )
    )

    assert yielded is True
    assert len(calls) == 1
    assert [event.event for event in broker.replay(thread.thread_id)] == [
        "tool_call.started",
        "tool_call.completed",
    ]
    tool_part = next(
        part
        for part in store.get_message(thread.thread_id, message.id).parts
        if part.type == "tool-call"
    )
    assert tool_part.status == "completed"
    assert tool_part.text == "completed before steering"


def test_streaming_adapter_flushes_observed_reasoning_before_stop(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_id = "run_stop_reasoning"
    run_dir = tmp_path / run_id
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id=run_id,
        observability_store=ObservabilityStore(run_dir),
    )

    class FakeV3Agent:
        async def astream_events(self, _payload, config=None, version="v3"):
            ObservabilityStore(run_dir).record_event(
                source="langchain_callback",
                channel="callback",
                name="LLM_CALL_END",
                category="llm",
                ts=1.0,
                seq=None,
                run_id=run_id,
                task_id="",
                step_id=None,
                payload={
                    "agent_name": "experiment_specialist",
                    "callback_run_id": "llm_stop",
                    "node": "model",
                    "reasoning_text": "Need to inspect the remote stage before submitting.",
                },
            )
            yield {"method": "messages", "params": {"data": []}}

    runner = StreamingSpecialistRunner(
        runner=SimpleNamespace(run_context=SimpleNamespace(run_dir=run_dir, run_id=run_id)),  # type: ignore[arg-type]
        thread_store=store,
        event_broker=broker,
        artifact_registry=registry,
        should_stop=lambda _thread_id: True,
    )

    import asyncio
    async def _run() -> str:
        try:
            await runner._consume_agent_stream(
                FakeV3Agent(),
                input_payload={"messages": [{"role": "user", "content": "Run O2."}]},
                config={},
                translator=translator,
            )
        except asyncio.CancelledError:
            return "cancelled"
        return "completed"

    assert asyncio.run(_run()) == "cancelled"
    saved = store.get_message(thread.thread_id, message.id)
    reasoning_part = next(part for part in saved.parts if part.type == "reasoning")
    assert reasoning_part.text == "Need to inspect the remote stage before submitting."
    assert [event.event for event in broker.replay(thread.thread_id)] == [
        "message.part.created",
        "reasoning.delta",
    ]


def test_streaming_adapter_consumes_deepagents_v3_protocol_message_deltas(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    message = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", text="", status="streaming")],
    )
    store.append_message(message)
    broker = ThreadEventBroker(workspace=workspace)
    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    translator = CatMasterStreamTranslator(
        store=store,
        events=broker,
        artifact_registry=registry,
        thread_id=thread.thread_id,
        message_id=message.id,
        text_part_id="part_text",
        run_id="run_v3_protocol",
    )

    class FakeV3Agent:
        async def astream_events(self, _payload, config=None, version="v3"):
            yield {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": [
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": "I will check the structure first. "},
                        }
                    ],
                },
            }
            yield {
                "method": "messages",
                "params": {
                    "namespace": ["task:o2_relax", "model_request:1"],
                    "data": [
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": "Preparing the MACE relaxation stage."},
                        }
                    ],
                },
            }

    runner = StreamingSpecialistRunner(
        runner=SimpleNamespace(run_context=SimpleNamespace(run_dir=tmp_path, run_id="run_v3_protocol")),  # type: ignore[arg-type]
        thread_store=store,
        event_broker=broker,
        artifact_registry=registry,
    )

    awaitable_result(
        runner._consume_agent_stream(
            FakeV3Agent(),
            input_payload={"messages": [{"role": "user", "content": "Run O2."}]},
            config={},
            translator=translator,
        )
    )

    saved = store.get_message(thread.thread_id, message.id)
    text_part = next(part for part in saved.parts if part.id == "part_text")
    subagent_parts = [part for part in saved.parts if part.type == "subagent"]
    assert text_part.text == "I will check the structure first. "
    assert subagent_parts and subagent_parts[0].text == "Preparing the MACE relaxation stage."
    assert [event.event for event in broker.replay(thread.thread_id)] == [
        "message.delta",
        "subagent.started",
        "subagent.delta",
    ]
