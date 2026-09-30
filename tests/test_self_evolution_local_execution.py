"""Learning admission and evidence for the local DBOS thread runtime."""
import asyncio
import json
import threading
from contextlib import asynccontextmanager

import pytest
from deepagents import create_deep_agent
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from catmaster.runtime.execution import ExecutionHost
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.self_evolution import ReflectionResult, SelfEvolutionCoordinator, SelfEvolutionStore
from catmaster.runtime.self_evolution.query import EvolutionHistoryScope, EvolutionTraceScope
from catmaster.runtime.self_evolution.trace import collect_turn_trace
from catmaster.webui import server
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import MessagePart, ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore


class Queue:
    async def runs(self, *args, **kwargs):
        return []


def service_for(workspace):
    return LocalThreadService(
        workspace=workspace, workspace_id=workspace.name,
        store=ThreadStore(workspace=workspace), broker=ThreadEventBroker(workspace=workspace),
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id=workspace.name),
        normalize_entrypoint=lambda x: x or "writing", permission_mode_for_thread=lambda *_: "auto",
        execution=Queue(),
    )


def finish(service, packet, *, status="completed", resumable=False):
    run_dir = service.workspace / "metadata/runs" / packet["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({"run_id": packet["run_id"]}))
    ObservabilityStore(run_dir).record_raw_callback(
        "TOOL_RAW_OUTPUT", category="tool", run_id=packet["run_id"],
        payload={"tool": "read_file", "raw_output": "Exact evidence for this run.",
                 "projection": {"content_text": "Exact evidence for this run."}},
    )
    answer = service.store.get_message(packet["thread_id"], packet["assistant_message_id"])
    service.store.update_message(answer.thread_id, answer.id, status=status,
        meta={**answer.meta, "checkpoint_resume_available": resumable},
        parts=[MessagePart(id=answer.id + "_text", type="text", status=status,
                           text="Original final answer.")])
    return run_dir


def test_native_trace_keeps_selected_turn_and_resumed_episode(tmp_path):
    async def scenario():
        service = service_for(tmp_path / "workspace")
        thread = await service.create_thread(entrypoint="writing")
        first = await service._prepare_turn(thread.thread_id, ThreadSubmitRequest(
            text="Original user requirement.", entrypoint="writing"))
        first_dir = finish(service, first, status="interrupted", resumable=True)
        coordinator = SelfEvolutionCoordinator(workspace=service.workspace, mode="auto")
        assert coordinator.enqueue_post_run(run_id=first["run_id"], thread_id=thread.thread_id,
            terminal_status="interrupted", run_dir=first_dir) is None

        resumed = await service._prepare_turn(thread.thread_id, ThreadSubmitRequest(
            text="", entrypoint="writing"), command={})
        run_dir = finish(service, resumed)
        args = dict(run_id=resumed["run_id"], thread_id=thread.thread_id,
                    terminal_status="success", run_dir=run_dir)
        queued = coordinator.enqueue_post_run(**args)
        assert coordinator.enqueue_post_run(**args).job_id == queued.job_id
        assert queued.episode_id == first["input_message_id"]
        assert queued.payload["message_id"] == first["input_message_id"]
        assert queued.payload["execution_status"] == "done"
        assert "run_state_diagnostic" not in queued.payload

        newer = await service._prepare_turn(thread.thread_id, ThreadSubmitRequest(
            text="Newer unrelated requirement.", entrypoint="writing"))
        finish(service, newer)
        other = await service.create_thread(entrypoint="writing")
        other_packet = await service._prepare_turn(other.thread_id, ThreadSubmitRequest(
            text="Another thread.", entrypoint="writing"))
        finish(service, other_packet)

        learned = coordinator.enqueue_explicit_learn(run_id=resumed["run_id"],
            run_dir=run_dir, thread_id=thread.thread_id, note="Keep this correction.")
        assert learned.episode_id == queued.episode_id
        history = EvolutionHistoryScope(db_path=coordinator.store.db_path)
        selected_dir, trace = history.open_run(resumed["run_id"])
        assert trace.user_prompt == "Original user requirement."
        assert trace.final_answer == "Original final answer."
        assert trace.status == "done"
        assert trace.source_message_id == first["input_message_id"]
        assert trace.assistant_message_id == resumed["assistant_message_id"]
        assert trace.prior_assistant_message_id == first["assistant_message_id"]
        assert not trace.diagnostics
        scope = EvolutionTraceScope({trace.run_id: (selected_dir, trace)})
        events = scope.execute("SELECT id FROM trajectory_events")
        assert events["rows"]
        handle = f"run:{trace.run_id}#event:{events['rows'][0]['id']}"
        assert "Exact evidence" in json.dumps(scope.read_event(handle=handle))
        assert not (run_dir / "run_state.json").exists()
        assert SelfEvolutionCoordinator(workspace=service.workspace, mode="off").enqueue_post_run(**args) is None
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["running", "pending", "capacity_wait"])
def test_nonterminal_native_status_does_not_queue(tmp_path, status):
    coordinator = SelfEvolutionCoordinator(workspace=tmp_path / "workspace", mode="auto")
    assert coordinator.enqueue_post_run(run_id="native:turn", terminal_status=status,
                                        run_dir=tmp_path / "absent") is None


def test_learn_route_selects_older_native_answer(tmp_path):
    async def prepare():
        service = service_for(tmp_path / "workspace")
        thread = await service.create_thread(entrypoint="writing")
        first = await service._prepare_turn(thread.thread_id, ThreadSubmitRequest(
            text="Selected original request.", entrypoint="writing"))
        finish(service, first)
        second = await service._prepare_turn(thread.thread_id, ThreadSubmitRequest(
            text="Newer unrelated request.", entrypoint="writing"))
        finish(service, second)
        return service, thread, first

    service, thread, selected = asyncio.run(prepare())
    client = TestClient(server.create_app(project_space_root=str(tmp_path), no_login=True))
    response = client.post(f"/api/threads/{thread.thread_id}/self-evolution/learn", json={
        "message_id": selected["assistant_message_id"], "note": "Remember this reporting preference.",
        "run_id": "untrusted:run", "run_dir": "/tmp/untrusted-run",
    })
    assert response.status_code == 200, response.text
    job = SelfEvolutionStore(service.workspace).list_jobs()[0]
    assert job.run_id == selected["run_id"]
    assert job.episode_id == selected["input_message_id"]
    trace = collect_turn_trace(run_dir=job.run_dir,
        fallback={"run_id": job.run_id, "thread_id": job.thread_id, **job.payload})
    assert trace.user_prompt == "Selected original request."
    assert trace.explicit_correction == "Remember this reporting preference."
    assert trace.final_answer == "Original final answer."
    assert not trace.diagnostics


def test_finalized_failure_overrides_native_success_and_resumable_error_waits(tmp_path):
    async def scenario():
        service = service_for(tmp_path / "workspace")
        thread = await service.create_thread(entrypoint="writing")
        packet = await service._prepare_turn(thread.thread_id, ThreadSubmitRequest(
            text="Deliver an answer.", entrypoint="writing"))
        run_dir = finish(service, packet, status="failed", resumable=True)
        answer = service.store.get_message(thread.thread_id, packet["assistant_message_id"])
        service.store.update_message(thread.thread_id, answer.id,
            meta={**answer.meta, "native_status": "success"})
        trace = collect_turn_trace(run_dir=run_dir,
            fallback={"run_id": packet["run_id"], "thread_id": thread.thread_id})
        assert trace.status == "error"
        coordinator = SelfEvolutionCoordinator(workspace=service.workspace, mode="auto")
        assert coordinator.enqueue_post_run(run_id=packet["run_id"], thread_id=thread.thread_id,
            terminal_status="error", run_dir=run_dir) is None
    asyncio.run(scenario())


def test_local_completion_runs_learning_worker_and_publishes_chat_notice(tmp_path, monkeypatch):
    """Real DBOS, native graph, server callback/worker/outbox; model calls are fake."""
    notice_ready = threading.Event()
    seen = []

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class Proposer:
        def reflect(self, *, trace_scope, **kwargs):
            rows = trace_scope.execute("SELECT user_prompt, final_answer, status FROM trajectory_runs")
            seen.append(rows["rows"])
            return ReflectionResult(kind="no_change", rationale="No durable change is needed."), {}

    def coordinator(**kwargs):
        return SelfEvolutionCoordinator(**kwargs, mode="auto", proposer=Proposer(), reviewer=object())

    @asynccontextmanager
    async def graph_factory(service, packet):
        run_dir = service.workspace / "metadata/runs" / packet["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(json.dumps({"run_id": packet["run_id"]}))
        ObservabilityStore(run_dir)
        async with AsyncSqliteSaver.from_conn_string(str(service.workspace / "metadata/test.sqlite")) as saver:
            yield create_deep_agent(Model(responses=[AIMessage(content="Delivered report.")]), checkpointer=saver)

    original_service = server.LocalThreadService
    def local_service(**kwargs):
        return original_service(**kwargs, graph_factory=graph_factory)

    original_emit = ThreadEventBroker.emit
    def emit(self, thread_id, event, **kwargs):
        result = original_emit(self, thread_id, event, **kwargs)
        meta = kwargs.get("data", {}).get("message", {}).get("meta", {})
        if meta.get("kind") == "self_evolution":
            notice_ready.set()
        return result

    monkeypatch.setattr(server, "SelfEvolutionCoordinator", coordinator)
    monkeypatch.setattr(server, "LocalThreadService", local_service)
    monkeypatch.setattr(ThreadEventBroker, "emit", emit)

    async def scenario():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        host = ExecutionHost(tmp_path / "execution.sqlite", lambda *_: None)
        app = server.create_app(project_space_root=str(tmp_path), no_login=True, execution_host=host)
        async with app.router.lifespan_context(app):
            service = host.factory(workspace, workspace.name)
            thread = await service.create_thread(entrypoint="writing")
            submitted = await service.submit(thread_id=thread.thread_id,
                payload=ThreadSubmitRequest(text="Finish the existing report.", entrypoint="writing"))
            handle = await host.client.retrieve_workflow_async(submitted["run_id"])
            result = await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 25)
            assert result["status"] == "success"
            assert await asyncio.to_thread(notice_ready.wait, 20)
            jobs = SelfEvolutionStore(workspace).list_jobs()
            assert len(jobs) == 1 and jobs[0].status == "done"
            assert jobs[0].run_id == submitted["run_id"]
            assert jobs[0].payload["message_id"] == submitted["message"].id
            assert seen == [[{"user_prompt": "Finish the existing report.",
                             "final_answer": "Delivered report.", "status": "done"}]]
            messages = service.store.list_messages(thread.thread_id)
            notices = [m for m in messages if m.meta.get("kind") == "self_evolution"]
            assert len(notices) == 1
            events = service.broker.replay(thread.thread_id)
            assert any(e.event == "message.created" and e.message_id == notices[0].id for e in events)
            trace = collect_turn_trace(run_dir=jobs[0].run_dir,
                fallback={"run_id": jobs[0].run_id, "thread_id": thread.thread_id}, include_events=False)
            assert trace.final_answer == "Delivered report."  # Ignore the newer learning notice for this run.
            # Callback replay must not create another job or researcher turn.
            await service.prepare_completion(submitted["assistant_message"].structured_sidecar["execution"], result)
            assert len(SelfEvolutionStore(workspace).list_jobs()) == 1
            assert len([m for m in service.store.list_messages(thread.thread_id) if m.role == "user"]) == 1
    asyncio.run(scenario())
