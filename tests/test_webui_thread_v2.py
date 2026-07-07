from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, ObservabilityStore
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui import server
from catmaster.webui.artifact_registry import ArtifactRegistry, infer_renderer
from catmaster.webui.server import create_app
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import MessagePart, ThreadMessage
from catmaster.webui.thread_store import ThreadStore, new_id
from catmaster.specialists.streaming_runner import CatMasterStreamTranslator, StreamingSpecialistRunner, _extract_sidecar_artifact_paths, _extract_workspace_paths_from_text


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    return workspace


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
    assert client.get(f"/api/threads/{thread_id}/messages").json() == {"messages": []}

    registry = ArtifactRegistry(workspace=workspace, workspace_id="default")
    artifact = registry.register_path("note.md", thread_id=thread_id, message_id="msg_x")
    preview = client.get(f"/api/artifacts/{artifact.artifact_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["kind"] == "markdown"
    assert "Result" in preview.json()["preview_text"]

    malformed = client.post(f"/api/threads/{thread_id}/resume", json={"decisions": [{"type": "deny"}]})
    assert malformed.status_code == 400


def test_thread_permission_mode_create_patch_and_interrupt_mapping(tmp_path: Path) -> None:
    _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    default_created = client.post("/api/workspaces/default/threads", json={"title": "default"})
    assert default_created.status_code == 200
    default_thread = default_created.json()["thread"]
    assert default_thread["meta"]["permission_mode"] == "auto"
    assert server._thread_permission_mode(SimpleNamespace(meta={})) == "auto"

    created = client.post("/api/workspaces/default/threads", json={"title": "auto", "permission_mode": "auto-approve"})
    assert created.status_code == 200
    thread = created.json()["thread"]
    assert thread["meta"]["permission_mode"] == "auto"
    assert server._interrupt_on_for_permission_mode(thread["meta"]["permission_mode"]) == {}

    patched = client.patch(f"/api/threads/{thread['thread_id']}", json={"permission_mode": "review"})
    assert patched.status_code == 200
    thread = patched.json()["thread"]
    assert thread["meta"]["permission_mode"] == "hitl"
    assert server._interrupt_on_for_permission_mode(thread["meta"]["permission_mode"]) == server.default_thread_interrupt_on()

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
    assert payload["assistant_message"]["meta"]["permission_mode"] == "auto"
    assert payload["thread"]["meta"]["permission_mode"] == "auto"


def test_submit_image_attachment_registers_artifact_without_persisting_data_url(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    thread_id = client.post("/api/workspaces/default/threads", json={}).json()["thread"]["thread_id"]
    captured: dict[str, str] = {}

    async def _fake_arun_turn(self, *, prompt, thread_id, message_id, text_part_id, **_kwargs):
        captured["prompt"] = prompt
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
    assert "data:image/png" not in user_message.model_dump_json() if hasattr(user_message, "model_dump_json") else "data:image/png" not in str(user_message)
    assert "data:image/png" not in captured["prompt"]
    assert "figure.png" in captured["prompt"]
    assert (workspace / artifact_parts[0]["path"]).exists()


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

    result = awaitable_result(service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="hello", attachments=[], entrypoint="research", model_config="", permission_mode="hitl")))

    assert result["queued"] is False
    assert "build_runner" in captured
    assert captured["build_runner"]["interrupt_on"] == {"write_file": True}
    assert result["assistant_message"].role == "assistant"

    running_thread = store.update_thread(thread.thread_id, status="running")
    tasks[thread.thread_id] = SimpleNamespace(done=lambda: False)
    queued = awaitable_result(service.submit(thread_id=running_thread.thread_id, payload=SimpleNamespace(text="steer", attachments=[], entrypoint="research", model_config="", permission_mode="hitl")))

    assert queued["queued"] is True
    assert store.get_thread(thread.thread_id).pending_steering[0]["text"] == "steer"


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

    result = awaitable_result(service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="draft", attachments=[], entrypoint="writing", model_config="", permission_mode="hitl")))

    assert result["queued"] is False
    assert store.get_thread(thread.thread_id).entrypoint == "writing"
    assert captured["build_runner"]["preferred_entrypoint"] == "writing"
    assert captured["arun_turn"]["entrypoint"] == "writing"
    assert result["assistant_message"].meta["entrypoint"] == "writing"


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
            def __init__(self, **_kwargs):
                pass

            async def arun_turn(self, **kwargs):
                prompts.append(str(kwargs.get("prompt") or ""))
                if len(prompts) == 1:
                    started.set()
                    await release.wait()
                self.thread_store.update_message(kwargs["thread_id"], kwargs["message_id"], status="completed")
                self.thread_store.update_thread(kwargs["thread_id"], status="idle", active_message_id="", active_run_id="")
                return {"status": "done"}

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

        first = await service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="first", attachments=[], entrypoint="research", model_config="", permission_mode="hitl"))
        assert first["queued"] is False
        await started.wait()
        queued = await service.submit(thread_id=thread.thread_id, payload=SimpleNamespace(text="steer", attachments=[], entrypoint="research", model_config="", permission_mode="hitl"))
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

    class FakeAgent:
        async def astream_events(self, _payload, config=None, version="v3"):
            yield {"method": "messages", "params": {"data": AIMessage(content="streamed answer")}}
            yield {
                "method": "values",
                "params": {
                    "data": {
                        "messages": [AIMessage(content="streamed answer")],
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

        def _stage_deepagent_assets(self, _files_root):
            return None

        def _new_usage_callback(self):
            return SimpleNamespace(usage_metadata={"test-model": {"input_tokens": 3, "output_tokens": 5}})

        def _emit(self, *_args, **_kwargs):
            return None

        def _write_run_state(self, payload):
            (run_dir / "run_state.json").write_text(json.dumps(payload), encoding="utf-8")

        @asynccontextmanager
        async def _open_agent_runtime(self, *, files_root):
            yield {}

        async def _build_entry_agent(self, *, entrypoint, runtime, thread_id):
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
                        "output_tokens": usage_handler.usage_metadata["test-model"]["output_tokens"],
                        "calls": 1,
                    }
                ),
                encoding="utf-8",
            )

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
    assert result["artifact_ids"]
    saved = store.get_message(thread.thread_id, message.id)
    artifact_parts = [part for part in saved.parts if part.type == "artifact"]
    assert len(artifact_parts) == 1
    assert artifact_parts[0].path == "files/reports/sidecar.md"
    usage_events = [event for event in broker.replay(thread.thread_id) if event.event == "usage.updated"]
    assert len(usage_events) == 1
    assert usage_events[0].data["run_id"] == "run_usage"
    assert usage_events[0].data["usage_summary"]["input_tokens"] == 3
    assert usage_events[0].data["usage_summary"]["output_tokens"] == 5


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
    assert tool_part.meta["stream_namespace"] == ["task:o2_relax", "tools:1"]
    events = broker.replay(thread.thread_id)
    started = [event for event in events if event.event == "tool_call.started"][-1]
    completed = [event for event in events if event.event == "tool_call.completed"][-1]
    assert started.data["subagent_source"] == "materials_worker"
    assert completed.data["subagent_source"] == "materials_worker"


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
