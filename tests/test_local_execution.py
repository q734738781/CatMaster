"""Exercise the actual DBOS service with native DeepAgents and disk checkpoints."""
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages

from catmaster.runtime.execution import ExecutionHost, thread_key
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest, ThreadStopRequest, ThreadCheckpointContinueRequest
from catmaster.webui.thread_store import ThreadStore


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize('outcome', ['success', 'error', 'cancel'])
def test_native_nested_stream_identity_and_terminal_status(tmp_path, outcome):
    waiting = asyncio.Event()

    class StreamModel(FakeListChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            # Match Responses: an empty provider-ID opener, then chunks whose
            # IDs BaseChatModel supplies, and a full provider-ID state message.
            yield ChatGenerationChunk(message=AIMessageChunk(id='resp_nested', content=''))
            yield ChatGenerationChunk(message=AIMessageChunk(content='Saved '))
            yield ChatGenerationChunk(message=AIMessageChunk(content='evidence'))

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
            if outcome == 'error':
                raise RuntimeError('after partial evidence')
            if outcome == 'cancel':
                waiting.set()
                await asyncio.Event().wait()

    class StreamState(TypedDict):
        messages: Annotated[list, add_messages]

    async def exercise():
        workspace = tmp_path / 'workspace'
        (workspace / 'files').mkdir(parents=True)
        store = ThreadStore(workspace=workspace)

        @asynccontextmanager
        async def factory(service, packet):
            model = StreamModel(responses=['unused'])

            async def answer(state):
                result = await model.ainvoke(state['messages'])
                return {'messages': [result]}

            child = StateGraph(StreamState).add_node('model', answer).add_edge(START, 'model').add_edge('model', END).compile()
            async with AsyncSqliteSaver.from_conn_string(str(workspace / 'metadata/deepagent_threads.sqlite')) as saver:
                yield StateGraph(StreamState).add_node('worker', child).add_edge(START, 'worker').add_edge('worker', END).compile(checkpointer=saver)

        service = None
        host = ExecutionHost(tmp_path / 'execution.sqlite', lambda *_: service)
        service = LocalThreadService(workspace=workspace, workspace_id='workspace', store=store,
            broker=ThreadEventBroker(workspace=workspace),
            artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id='workspace'),
            normalize_entrypoint=lambda x: x or 'research', permission_mode_for_thread=lambda *_: 'auto',
            execution=host, graph_factory=factory)
        await host.start()
        try:
            thread = await service.create_thread()
            submitted = await service.submit(thread_id=thread.thread_id, payload=ThreadSubmitRequest(text='Prepare candidates'))
            if outcome == 'cancel':
                await asyncio.wait_for(waiting.wait(), 15)
                await service.stop(thread_id=thread.thread_id, payload=ThreadStopRequest())
            else:
                handle = await host.client.retrieve_workflow_async(submitted['run_id'])
                result = await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 25)
                assert result['status'] == outcome
            messages = [m for m in store.list_messages(thread.thread_id) if m.meta.get('namespace')]
            assert len(messages) == 1
            assert messages[0].parts[0].text == 'Saved evidence'
            expected = {'success': 'completed', 'error': 'failed', 'cancel': 'interrupted'}[outcome]
            assert messages[0].status == expected
            assert messages[0].parts[0].status == expected
            assert len(messages[0].meta['native_message_ids']) == 2
        finally:
            await host.close()

    asyncio.run(exercise())


def test_parallel_detached_interaction_notify_and_checkpoint(tmp_path):
    async def exercise():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        store = ThreadStore(workspace=workspace)
        broker = ThreadEventBroker(workspace=workspace)
        started, released = {x: asyncio.Event() for x in "ABC"}, asyncio.Event()
        foreground_started, foreground_release = asyncio.Event(), asyncio.Event()
        tool_count = {x: 0 for x in "ABC"}

        @asynccontextmanager
        async def graph_factory(service, packet):
            thread = store.get_thread(packet["thread_id"])
            if thread.parent_thread_id:
                name = thread.meta["task_description"]
                if name == "B" and "entrypoint" in packet:
                    assert packet['entrypoint'] == 'research_challenger'
                    assert packet['research']['research_graph_id'] == thread.active_research_graph_id
                @tool
                async def bounded_work() -> str:
                    """Perform this isolated branch's work."""
                    tool_count[name] += 1
                    started[name].set()
                    await released.wait()
                    return "evidence " + name
                responses = [AIMessage(content="", tool_calls=[{"name": "bounded_work", "args": {}, "id": "work-"+name}]),
                             AIMessage(content="result " + name)]
                tools = [bounded_work]
            else:
                message = store.get_message(thread.thread_id, packet["input_message_id"])
                text = message.parts[0].text if message else ""
                tools = service.background_tools(packet)
                responses = [AIMessage(content="handled: " + text)]
                if text.startswith("launch"):
                    responses = [AIMessage(content="", tool_calls=[{
                        "name": "start_async_task", "id": "start-"+x,
                        "args": {"agent": "research_challenger" if x == "B" else "writing_specialist", "description": x,
                                 "on_completion": "notify" if x == "C" else "resume_parent"}}
                        for x in "ABC"]), AIMessage(content="Background tasks accepted.")]
                elif text == "foreground question":
                    @tool
                    async def foreground_work() -> str:
                        """Complete the current foreground request."""
                        foreground_started.set()
                        await foreground_release.wait()
                        return "foreground evidence"
                    tools = [*tools, foreground_work]
                    responses = [AIMessage(content="", tool_calls=[{
                        "name":"foreground_work", "args":{}, "id":"foreground-work"}]),
                        AIMessage(content="Foreground request answered.")]
            async with AsyncSqliteSaver.from_conn_string(str(workspace / "metadata/deepagent_threads.sqlite")) as saver:
                yield create_deep_agent(Model(responses=responses), tools=tools, checkpointer=saver)

        service = None
        host = ExecutionHost(tmp_path / "execution.sqlite", lambda *_: service)
        service = LocalThreadService(workspace=workspace, workspace_id="workspace", store=store, broker=broker,
            artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="workspace"),
            normalize_entrypoint=lambda x: x if x in {'research', 'writing'} else 'research',
            permission_mode_for_thread=lambda *_: "auto", execution=host, graph_factory=graph_factory)
        await host.start()
        async def finish(run_id):
            handle = await host.client.retrieve_workflow_async(run_id)
            return await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 25)
        try:
            root = await service.create_thread()
            initial = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="launch PARENT_ONLY_CONTEXT"))
            first = await finish(initial["run_id"])
            assert first["status"] == "success"
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 15)
            assert not released.is_set()
            reply = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="foreground question"))
            await asyncio.wait_for(foreground_started.wait(), 15)
            assert not released.is_set()
            children = [t for t in store.list_threads() if t.parent_thread_id == root.thread_id]
            assert len(children) == 3
            assert len({t.deepagent_thread_id for t in children}) == 3
            cards = await service.active_subagent_parts(root.thread_id)
            assert len(cards) == 3
            assert all(card.status == "running" for card in cards)
            released.set()
            for child in children:
                result = await finish(child.meta["last_run_id"])
                assert result["status"] == "success"
            assert await service.active_subagent_parts(root.thread_id) == []
            active = await host.runs(workspace, root.thread_id, active=True)
            assert len(active) == 3  # Current answer plus two queued completion turns.
            assert sum(run.status == "PENDING" for run in active) == 1
            foreground_release.set()
            assert (await finish(reply["run_id"]))["status"] == "success"
            for run in await host.runs(workspace, root.thread_id, active=True):
                await finish(run.workflow_id)
            turns = await host.runs(workspace, root.thread_id)
            assert len(turns) == 4  # launch, user question, A + B completion
            completion = next(m for m in store.list_messages(root.thread_id)
                if m.structured_sidecar.get("execution", {}).get("metadata", {}).get("catmaster_async_completion_run_id"))
            await host.enqueue(completion.structured_sidecar["execution"])
            assert len(await host.runs(workspace, root.thread_id)) == 4
            quiet = next(c for c in children if c.meta["task_description"] == "C")
            assert (await service.task(root.thread_id, quiet.thread_id))["result"] == "result C"
            from catmaster.webui.projections.events import project_event
            updates = [project_event(e) for e in broker.replay(root.thread_id)
                       if e.event == "message.part.updated"]
            assert updates and all(e.data.part.type == "subagent" for e in updates)
            assert any(e.data.part.text == "result C" for e in updates)
            assert any(any(f.value == "bounded_work" for f in e.data.part.fields) for e in updates)
            assert tool_count == {x: 1 for x in "ABC"}
            async with graph_factory(service, initial["assistant_message"].structured_sidecar["execution"]) as graph:
                state = await graph.aget_state({"configurable": {"thread_id": root.deepagent_thread_id}})
                users = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
                assert len(users) == 4
                assert users[1].content.endswith("foreground question")
            for child in children:
                assert child.active_research_graph_id == store.get_thread(root.thread_id).active_research_graph_id
                async with graph_factory(service, {"thread_id": child.thread_id}) as graph:
                    state = await graph.aget_state({"configurable": {"thread_id": child.deepagent_thread_id}})
                    assert not any("foreground question" in str(m.content) for m in state.values["messages"])
                    assert not any("PARENT_ONLY_CONTEXT" in str(m.content) for m in state.values["messages"])
                    child_inputs = [m.content for m in state.values['messages'] if isinstance(m, HumanMessage)]
                    assert len(child_inputs) == 1
                    assert child.active_research_graph_id in child_inputs[0]
                    assert child_inputs[0].endswith(child.meta['task_description'])
        finally:
            released.set()
            foreground_release.set()
            await host.close()
    asyncio.run(exercise())


def test_actual_specialist_builder_keeps_native_state_and_tool_topology(tmp_path, monkeypatch):
    from catmaster.specialists.runtime import SpecialistRunner
    from catmaster.runtime.checkpoint_compat import RetainedCheckpointState
    from langchain_core.messages import ToolMessage
    models=[]
    specs=[]
    build = SpecialistRunner._create_deep_agent
    def capture_build(self, **kwargs):
        specs.append(kwargs)
        return build(self, **kwargs)
    monkeypatch.setattr(SpecialistRunner, '_create_deep_agent', capture_build)
    class CapturingModel(Model):
        def bind_tools(self, tools, **kwargs):
            models.append({(tool.get('name') or tool.get('function', {}).get('name'))
                if isinstance(tool, dict) else tool.name:tool for tool in tools})
            return self
    monkeypatch.setattr(SpecialistRunner, '_build_deepagent_chat_model',
        lambda *args,**kwargs:CapturingModel(responses=[AIMessage(content='Finished this requested stage.',
            response_metadata={'model_name':'native-test-model'},
            usage_metadata={'input_tokens':12,'output_tokens':3,'total_tokens':15})]))
    async def scenario():
        workspace=tmp_path/'workspace';(workspace/'files').mkdir(parents=True)
        store=ThreadStore(workspace=workspace)
        service=LocalThreadService(workspace=workspace,workspace_id='workspace',store=store,
            broker=ThreadEventBroker(workspace=workspace),artifact_registry=ArtifactRegistry(workspace=workspace,workspace_id='workspace'),
            normalize_entrypoint=lambda x:x or 'research',permission_mode_for_thread=lambda *_:'auto',execution=None)
        root=await service.create_thread(entrypoint='writing')
        packet={'thread_id':root.thread_id,'run_id':'builder-test','assistant_message_id':'builder-answer','entrypoint':'writing',
            'model_config':'','permission_mode':'auto','research':{},'input_message_id':''}
        async with service._graph(packet) as graph:
            result=await graph.ainvoke({'messages':[HumanMessage(content='Describe the available writing tools briefly.')]},
                {'configurable':{'thread_id':root.thread_id}})
            assert result['messages'][-1].text=='Finished this requested stage.'
            saved=await graph.aget_state({'configurable':{'thread_id':root.thread_id}})
            assert saved.values['messages'][-1].text==result['messages'][-1].text
            # Usage must be durable and published before the runtime closes.
            from catmaster.runtime.usage_stats import load_usage_summary
            summary = load_usage_summary(workspace/'metadata/runs/builder-test')
            assert summary['calls'] == 1 and summary['total_tokens'] == 15
            assert any(e.event == 'usage.updated' for e in service.broker.replay(root.thread_id))
        # Reopening the native runtime preserves the previous run's usage.
        async with service._graph(packet) as graph:
            await graph.ainvoke({'messages':[HumanMessage(content='One more short reply.')]},
                {'configurable':{'thread_id':root.thread_id}})
        summary = load_usage_summary(workspace/'metadata/runs/builder-test')
        assert summary['calls'] == 2 and summary['total_tokens'] == 30
        assert any('task' in group for group in models)
        # Public model schemas, not only Pydantic internals.
        packet.update(entrypoint='research',run_id='research-builder')
        async with service._graph(packet) as graph:
            assert 'files' in graph.channels and 'async_tasks' in graph.channels
            tools=service.background_tools(packet)
            start=next(t for t in tools if t.name=='start_async_task')
            schema=start.tool_call_schema.model_json_schema()
            assert schema['properties']['on_completion']['enum']==['resume_parent','notify']
            assert {'research_specialist', 'research_challenger'} <= set(schema['properties']['agent']['enum'])
            assert schema['properties']['task_cost']['enum'] == ['low', 'medium', 'high']
            assert schema['properties']['task_cost']['default'] == 'medium'
            update = next(t for t in tools if t.name == 'update_async_task')
            strategy = update.tool_call_schema.model_json_schema()['properties']['strategy']
            assert strategy['default'] == 'enqueue'
            assert strategy['enum'] == ['enqueue', 'interrupt']
            assert strategy['type'] == 'string' and strategy['description']
            assert 'runtime' not in schema['properties']
            # Research retains general-purpose; experiment still builds its workers.
            assert 'tools' in graph.nodes
        # A real research branch retains native domain workers and removes only
        # root-scoped graph controls, including with native provider tool dicts.
        store.update_thread(root.thread_id, meta={**store.get_thread(root.thread_id).meta,
            'research_branch': True, 'background_task': True})
        packet.update(task_cost='low',run_id='branch-builder')
        async with service._graph(packet) as graph:
            await graph.ainvoke({'messages': [HumanMessage(content='Finish this branch.')]},
                {'configurable': {'thread_id': 'branch-builder'}})
            bound = models[-1]
            assert 'set_research_task_cost' in bound and 'task' in bound
            assert 'start_async_task' in bound
            assert 'set_research_graph_completion' not in bound
            assert 'update_research_graph_scope' not in bound
            assert 'experiment_specialist' in bound['task'].description
            assert 'litreview_agent' in bound['task'].description
            branch_spec = next(s for s in reversed(specs) if s.get('name') == 'research_specialist')
            general = next(s for s in branch_spec['subagents'] if s['name'] == 'general-purpose')
            assert general['tools']
            assert all(getattr(t, 'name', '') != 'set_research_task_cost' for t in general['tools'])
        from catmaster.research.knowledge_graph.service import ResearchGraphService
        from catmaster.research.knowledge_graph.models import GraphCreateRequest
        graph_service = ResearchGraphService(workspace=workspace, workspace_id='workspace')
        graph_id = graph_service.create_graph(GraphCreateRequest(question='Test direction recovery'))['graph']['graph_id']
        store.update_thread(root.thread_id, active_research_graph_id=graph_id)
        packet.update(entrypoint='research_challenger', run_id='challenger-builder',
            research={'research_graph_id': graph_id})
        async with service._graph(packet) as graph:
            await graph.ainvoke({'messages': [HumanMessage(content='Assess the supplied evidence.')]},
                {'configurable': {'thread_id': 'challenger-builder'}})
            bound = models[-1]
            assert {'read_file', 'glob', 'grep', 'query_research_graph_sql',
                    'record_research_review', 'revise_research_claim'} <= set(bound)
            assert not ({'execute', 'write_file', 'edit_file', 'task', 'start_async_task',
                         'set_research_graph_completion'} & set(bound))
            challenger = next(s for s in reversed(specs) if s.get('name') == 'research_challenger')
            assert challenger['subagents'] == []
            assert any('research_reasoning' in root for root in challenger['skills'])
        packet.pop('task_cost')
        packet.update(entrypoint='experiment',run_id='experiment-builder')
        async with service._graph(packet) as graph:
            assert 'tools' in graph.nodes
    asyncio.run(scenario())
