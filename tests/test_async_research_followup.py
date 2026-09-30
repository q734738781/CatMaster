import asyncio
import uuid
from contextlib import asynccontextmanager

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from catmaster.runtime.execution import ExecutionHost
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore
from test_local_execution import Model


@pytest.mark.parametrize("strategy", ["enqueue", "interrupt"])
def test_agent_followup_delivery_preserves_context_and_reports_accepted_turn(tmp_path, strategy):
    async def exercise():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        store = ThreadStore(workspace=workspace)
        started, release = asyncio.Event(), asyncio.Event()
        resumed, finish_followup, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        count = 0
        resumed_messages = []

        class CapturingModel(Model):
            async def _agenerate(self, messages, **kwargs):
                if any(isinstance(m, HumanMessage) and m.content == "new source" for m in messages):
                    resumed_messages.append(messages)
                return await super()._agenerate(messages, **kwargs)

        @asynccontextmanager
        async def factory(service, packet):
            thread = store.get_thread(packet["thread_id"])
            message = store.get_message(thread.thread_id, packet["input_message_id"]).parts[0].text
            if thread.parent_thread_id:
                @tool
                def saved_evidence() -> str:
                    """Read evidence that remains usable after a correction."""
                    return "accepted original evidence"

                @tool
                async def work() -> str:
                    """Wait for the original evidence-gathering operation."""
                    nonlocal count
                    count += 1
                    started.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    return "original evidence"

                @tool
                async def use_followup() -> str:
                    """Incorporate the accepted instruction using saved evidence."""
                    resumed.set()
                    await finish_followup.wait()
                    return "new source incorporated"

                tools = [saved_evidence, work, use_followup]
                responses = ([AIMessage(content="", tool_calls=[{"name": "saved_evidence", "args": {}, "id": "saved"}]),
                              AIMessage(content="", tool_calls=[{"name": "work", "args": {}, "id": "work"}]),
                              AIMessage(content="Original investigation finished")]
                             if message == "original" else [AIMessage(content="", tool_calls=[
                                 {"name": "use_followup", "args": {}, "id": "use-followup"}]),
                                 AIMessage(content="New source incorporated")])
            else:
                @tool
                async def await_start() -> str:
                    """Synchronize the test with already executing evidence work."""
                    await started.wait()
                    return "started"
                task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, packet["run_id"] + ":start"))
                tools = [*service.background_tools(packet), await_start]
                responses = [AIMessage(content="", tool_calls=[{"id": "start", "name": "start_async_task",
                    "args": {"agent": "research_specialist", "description": "original", "on_completion": "notify"}}]),
                    AIMessage(content="", tool_calls=[{"id": "wait", "name": "await_start", "args": {}}]),
                    AIMessage(content="", tool_calls=[{"id": "followup", "name": "update_async_task",
                        "args": {"task_id": task_id, "message": "new source", "strategy": strategy}}]),
                    AIMessage(content="Follow-up accepted")]
            async with AsyncSqliteSaver.from_conn_string(str(workspace / "metadata/deepagent_threads.sqlite")) as saver:
                yield create_deep_agent(CapturingModel(responses=responses), tools=tools, checkpointer=saver)

        host = ExecutionHost(tmp_path / "execution.sqlite", lambda *_: service)
        service = LocalThreadService(workspace=workspace, workspace_id="w", store=store,
            broker=ThreadEventBroker(workspace=workspace), artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="w"),
            normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto", execution=host, graph_factory=factory)
        async def finish(rid):
            handle = await host.client.retrieve_workflow_async(rid)
            return await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 25)
        await host.start()
        try:
            root = await service.create_thread()
            initial = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="launch"))
            assert (await finish(initial["run_id"]))["status"] == "success"
            child = next(t for t in store.list_threads() if t.parent_thread_id)
            runs = await host.runs(workspace, child.thread_id)
            assert len(runs) == 2
            assert count == 1 and not release.is_set()
            if strategy == "enqueue":
                assert {r.status for r in runs} == {"PENDING", "ENQUEUED"}
                assert not cancelled.is_set() and not resumed.is_set()
                task = await service.task(root.thread_id, child.thread_id)
                assert task["instructions"]["followup_status"] == "pending"
                assert (await service.active_subagent_parts(root.thread_id))[0].task_followup_status == "pending"
                release.set()
            else:
                assert cancelled.is_set()
                assert "CANCELLED" in {r.status for r in runs}
            await asyncio.wait_for(resumed.wait(), 15)
            task = await service.task(root.thread_id, child.thread_id)
            assert task["instructions"] == {"original": "original", "followup": "new source", "followup_status": "running"}
            assert (await service.active_subagent_parts(root.thread_id))[0].task_followup_status == "running"
            assert any(isinstance(m, ToolMessage) and m.content == "accepted original evidence" for m in resumed_messages[0])
            assert any(isinstance(m, HumanMessage) and m.content == "original" for m in resumed_messages[0])
            finish_followup.set()
            assert (await finish(child.meta["task_followup_run_id"]))["status"] == "success"
            task = await service.task(root.thread_id, child.thread_id)
            assert task["instructions"]["followup_status"] == "success"
            card = store.get_message_part(root.thread_id, child.meta["parent_message_id"], "part_subagent_" + child.thread_id)
            assert card.meta["task_followup_status"] == "success"
            assert count == 1
            # Old instruction text remains visible without claiming it was processed.
            fresh = store.get_thread(child.thread_id)
            store.update_thread(child.thread_id, meta={k: v for k, v in fresh.meta.items() if k != "task_followup_run_id"})
            assert (await service.task(root.thread_id, child.thread_id))["instructions"]["followup_status"] == ""
        finally:
            release.set()
            finish_followup.set()
            await host.close()
    asyncio.run(exercise())


def test_research_descendant_shares_single_slot_pool_and_returns_to_original_parent(tmp_path):
    from catmaster.runtime.research_capacity import ResearchPoolConfig
    async def exercise():
        workspace = tmp_path / 'workspace'
        (workspace / 'files').mkdir(parents=True)
        store = ThreadStore(workspace=workspace)
        started, release = asyncio.Event(), asyncio.Event()
        count = 0

        @asynccontextmanager
        async def factory(service, packet):
            thread = store.get_thread(packet['thread_id'])
            message = store.get_message(thread.thread_id, packet['input_message_id']).parts[0].text
            tools = service.background_tools(packet)
            if message in {'launch', 'branch A'}:
                name = 'branch A' if message == 'launch' else 'branch B'
                responses = [AIMessage(content='', tool_calls=[{'name': 'start_async_task', 'id': 'child',
                    'args': {'agent': 'research_specialist', 'description': name, 'task_cost': 'low'}}]),
                    AIMessage(content='Independent work is pending; this turn yields.')]
            elif message == 'branch B':
                @tool
                async def evidence() -> str:
                    """Produce independent evidence while occupying the only research slot."""
                    nonlocal count
                    count += 1
                    assert service.execution.capacity.snapshot()['active'] == 1
                    started.set()
                    await release.wait()
                    return 'distinct mechanism evidence'
                tools = [evidence]
                responses = [AIMessage(content='', tool_calls=[{'name': 'evidence', 'args': {}, 'id': 'evidence'}]),
                    AIMessage(content='Branch B has finished its investigation.')]
            else:
                responses = [AIMessage(content='Integrated the completed independent investigation.')]
            async with AsyncSqliteSaver.from_conn_string(str(workspace / 'metadata/deepagent_threads.sqlite')) as saver:
                yield create_deep_agent(Model(responses=responses), tools=tools, checkpointer=saver)

        host = ExecutionHost(tmp_path / 'execution.sqlite', lambda *_: service,
            capacity_config=ResearchPoolConfig(1, {'low': 1, 'medium': 1, 'high': 1}))
        service = LocalThreadService(workspace=workspace, workspace_id='w', store=store,
            broker=ThreadEventBroker(workspace=workspace), artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id='w'),
            normalize_entrypoint=lambda x: x or 'research', permission_mode_for_thread=lambda *_: 'auto',
            execution=host, graph_factory=factory)
        async def finish(rid):
            handle = await host.client.retrieve_workflow_async(rid)
            return await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 25)
        await host.start()
        try:
            root = await service.create_thread()
            initial = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text='launch'))
            await finish(initial['run_id'])
            await asyncio.wait_for(started.wait(), 15)
            a = next(t for t in store.list_threads() if t.parent_thread_id == root.thread_id)
            b = next(t for t in store.list_threads() if t.parent_thread_id == a.thread_id)
            await finish(a.meta['last_run_id'])
            task = await service.task(root.thread_id, a.thread_id)
            assert task['status'] == 'pending' and task['capacity_state'] == 'waiting_children'
            assert len(await host.runs(workspace, root.thread_id)) == 1
            assert len(await service.active_subagent_parts(root.thread_id)) == 1
            release.set()
            await finish(b.meta['last_run_id'])
            for run in await host.runs(workspace, a.thread_id, active=True):
                await finish(run.workflow_id)
            for run in await host.runs(workspace, root.thread_id, active=True):
                await finish(run.workflow_id)
            assert len(await host.runs(workspace, root.thread_id)) == 2
            assert len(await host.runs(workspace, a.thread_id)) == 2
            assert (await service.task(root.thread_id, a.thread_id))['status'] == 'success'
            assert count == 1 and host.capacity.snapshot()['active'] == 0
            assert a.active_research_graph_id == b.active_research_graph_id == store.get_thread(root.thread_id).active_research_graph_id
        finally:
            release.set()
            await host.close()
    asyncio.run(exercise())
