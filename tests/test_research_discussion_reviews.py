"""Persistent root chooses follow-ups; discussion itself never schedules work."""
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from deepagents import create_deep_agent
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from catmaster.research.discussions import DiscussionPostRequest, ResearchDiscussions
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
from catmaster.runtime.execution import ExecutionHost
from catmaster.specialists.research_collaboration import ResearchCollaborationMiddleware
from catmaster.storage import connect_workspace_db
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore
from test_local_execution import Model
from test_research_discussions import setup_graph


def build_service(tmp_path, workspace, factory):
    host = ExecutionHost(tmp_path / 'execution.sqlite', lambda *_: service)
    service = LocalThreadService(workspace=workspace, workspace_id='workspace', store=ThreadStore(workspace=workspace),
        broker=ThreadEventBroker(workspace=workspace), artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id='workspace'),
        normalize_entrypoint=lambda x: x or 'research', permission_mode_for_thread=lambda *_: 'auto', execution=host, graph_factory=factory)
    return host, service


async def finish(host, run_id):
    handle = await host.client.retrieve_workflow_async(run_id)
    return await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 30)


def test_scope_follows_persistent_ancestry_not_research_entrypoint_or_shared_graph(tmp_path):
    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    discussions = ResearchDiscussions(workspace)
    host, service = build_service(tmp_path, workspace, None)
    ordinary = service.store.create_thread(title='Independent normal Research', entrypoint='research')
    service.store.update_thread(ordinary.thread_id, active_research_graph_id=gid)
    for thread in (root, a, b):
        packet = {'thread_id': thread.thread_id, 'run_id': 'surface'}
        assert service.research_collaboration_middleware(packet) is not None
        tools = {t.name: t for t in service.background_tools(packet)}
        schema = tools['post_research_message'].tool_call_schema.model_json_schema()
        assert not {'request_review', 'review_reason', 'runtime'} & schema['properties'].keys()
        assert 'anyOf' not in schema['properties']['resolves_message_id']
    packet = {'thread_id': ordinary.thread_id, 'run_id': 'surface'}
    assert service.research_collaboration_middleware(packet) is None
    assert 'post_research_message' not in {t.name for t in service.background_tools(packet)}
    assert {'start_async_task', 'check_async_task', 'update_async_task'} <= {t.name for t in service.background_tools(packet)}
    ordinary_tools = {t.name: t for t in service.background_tools(packet)}
    assert 'scope' not in ordinary_tools['list_async_tasks'].tool_call_schema.model_json_schema()['properties']
    with pytest.raises(HTTPException):
        asyncio.run(service.read_task(ordinary.thread_id, b.thread_id))
    with pytest.raises(HTTPException):
        asyncio.run(service.list_research_tasks(ordinary.thread_id, scope='research_graph'))
    with pytest.raises(ValueError, match='Persistent Research'):
        discussions.post(gid, DiscussionPostRequest(title='Wrong scope', body='Ordinary Research'), author_thread_id=ordinary.thread_id)
    first = discussions.post(gid, DiscussionPostRequest(title='Control question', body='Original methods and source details\n' * 1000,
        target_task_id=b.thread_id), author_thread_id=a.thread_id)
    reply = discussions.post(gid, DiscussionPostRequest(reply_to=first['message_id'], body='Compare the gas residence time.'), author_thread_id=b.thread_id)
    assert reply['target_task_id'] == a.thread_id
    assert discussions.notice(gid, root.thread_id, 0, all_graph=True)['count'] == 2
    assert discussions.notice(gid, b.thread_id, 0)['count'] == 1
    with pytest.raises(ValueError, match='main research'):
        discussions.post(gid, DiscussionPostRequest(reply_to=first['message_id'], resolves_message_id=first['message_id'], body='Peer decision'), author_thread_id=b.thread_id)
    decision = discussions.post(gid, DiscussionPostRequest(reply_to=first['message_id'], resolves_message_id=first['message_id'],
        body='Existing source resolves the apparatus delay; no additional work needed.'), author_thread_id=root.thread_id, message_id='decision')
    assert discussions.messages(gid)['messages'][0]['review_response_id'] == decision['message_id']
    assert discussions.graphs.get_graph(gid)['revision'] == graph['graph']['revision']
    # Even an accidentally supplied middleware cannot change an ordinary entry.
    middleware = ResearchCollaborationMiddleware(discussions=discussions, graph_id=gid, thread_id=ordinary.thread_id, publish_progress=AsyncMock())
    state = {'messages': [AIMessage(content='Ordinary answer')]}
    assert asyncio.run(middleware.abefore_model(state, None)) is None
    assert asyncio.run(middleware.aafter_model(state, None)) is None
    # History remains readable through the normal graph SQL surface.
    assert ResearchGraphSQLQuery(workspace).execute(graph_id=gid, sql=f"SELECT body FROM research_discussions WHERE message_id='{first['message_id']}'")['rows'][0]['body'] == first['body']


def test_root_reads_peer_question_then_resumes_original_child_and_receives_completion(tmp_path):
    async def exercise():
        workspace, graph, root, a, b = setup_graph(tmp_path)
        gid = graph['graph']['graph_id']
        discussions = ResearchDiscussions(workspace)
        child_release, child_entered = asyncio.Event(), asyncio.Event()
        child_calls, sql_reads, decisions = [], [], []
        question = None

        class ChildModel(Model):
            async def _agenerate(self, messages, **kwargs):
                child_calls.append(messages)
                if len(child_calls) > 1:
                    child_entered.set()
                    await child_release.wait()
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=
                    'Transport-control baseline retained.' if len(child_calls) == 1 else 'Existing methods support only an unresolved memory signal.'))])

        @asynccontextmanager
        async def factory(service, packet):
            tools = service.background_tools(packet)
            if packet['thread_id'] == root.thread_id:
                @tool
                def query_research_graph_sql(sql: str) -> dict:
                    """Read the full bound scientific discussion and records."""
                    result = ResearchGraphSQLQuery(workspace).execute(graph_id=gid, sql=sql)
                    sql_reads.append(result)
                    return result
                tools = [*tools, query_research_graph_sql]
                query = AIMessage(content='', tool_calls=[{'id': 'read-discussion', 'name': 'query_research_graph_sql',
                    'args': {'sql': 'SELECT * FROM research_discussions ORDER BY seq'}}])
                if packet['metadata'].get('catmaster_async_completion_run_id'):
                    responses = [query, AIMessage(content='Synthesis corrected using the returned control assessment.')]
                else:
                    responses = [query,
                        AIMessage(content='', tool_calls=[{'id': 'continue-original', 'name': 'update_async_task',
                            'args': {'task_id': b.thread_id, 'message': f"Investigate the apparatus-control question in discussion {question['message_id']}. Use existing sources and your baseline; no new calculations."}}]),
                        AIMessage(content='', tool_calls=[{'id': 'record-decision', 'name': 'post_research_message',
                            'args': {'reply_to': question['message_id'], 'resolves_message_id': question['message_id'],
                                'target_task_id': b.thread_id, 'review_outcome': 'follow_up',
                                'body': f'Continuing {b.thread_id} to check the existing controls; the interpretation remains open until its assessment returns.'}}]),
                        AIMessage(content='A bounded follow-up is active.'), AIMessage(content='Waiting for the accepted follow-up; no duplicate investigation needed.')]
                model = Model(responses=responses)
                decisions.append(packet)
            else:
                model = ChildModel(responses=[])
            middleware = service.research_collaboration_middleware(packet)
            async with AsyncSqliteSaver.from_conn_string(str(workspace / 'metadata/deepagent_threads.sqlite')) as saver:
                agent = create_deep_agent(model, tools=tools, middleware=[middleware] if middleware else [], checkpointer=saver)
                from langgraph.channels import DeltaChannel
                assert isinstance(agent.channels['messages'], DeltaChannel)
                yield agent

        host, service = build_service(tmp_path, workspace, factory)
        await host.start()
        try:
            # Seed the completed child's real native context without a root turn.
            service.store.update_thread(b.thread_id, meta={**b.meta, 'on_completion': 'notify'})
            baseline = await service.submit(thread_id=b.thread_id, payload=ThreadSubmitRequest(text='Keep the prior transport-control baseline.'))
            assert (await finish(host, baseline['run_id']))['status'] == 'success'
            old = service._thread(b.thread_id)
            service.store.update_thread(b.thread_id, meta={**old.meta, 'on_completion': 'resume_parent'})
            question = discussions.post(gid, DiscussionPostRequest(title='Surface memory or transport?',
                body='The existing report did not exclude apparatus residence time. Does that weaken the surface-memory interpretation?',
                target_task_id=b.thread_id), author_thread_id=a.thread_id)
            # Peer discussion and recovery do not wake either idle agent.
            await service.reconcile_async_subagents()
            await service.reconcile_research_graph_updates(gid, root.thread_id)
            assert await host.runs(workspace, root.thread_id) == []
            assert len(await host.runs(workspace, b.thread_id)) == 1
            run = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(
                text='Review the current interpretation and discuss whether the source controls suffice.', entrypoint='persistent_research'))
            assert (await finish(host, run['run_id']))['status'] == 'success'
            await asyncio.wait_for(child_entered.wait(), 15)
            assert len(sql_reads) == 1 and sql_reads[0]['rows'][0]['body'] == question['body']
            followup = (await host.runs(workspace, b.thread_id))[0]
            assert followup.workflow_id != baseline['run_id']
            assert any(isinstance(m, HumanMessage) and 'prior transport-control baseline' in str(m.content) for m in child_calls[1])
            assert any(isinstance(m, AIMessage) and 'baseline retained' in str(m.content) for m in child_calls[1])
            assert await host.runs(workspace, a.thread_id) == []
            with pytest.raises(HTTPException):
                service._child(a.thread_id, b.thread_id)
            saved = discussions.messages(gid)['messages'][0]
            assert saved['review_status'] == 'follow_up'
            assert len(await host.runs(workspace, root.thread_id)) == 1
            child_release.set()
            assert (await finish(host, followup.workflow_id))['status'] == 'success'
            completion = next(r for r in await host.runs(workspace, root.thread_id) if r.workflow_id != run['run_id'])
            assert (await finish(host, completion.workflow_id))['status'] == 'success'
            assert len(await host.runs(workspace, b.thread_id)) == 2
            assert len(await host.runs(workspace, root.thread_id)) == 2
            assert len(decisions) == 2 and len(sql_reads) == 2
        finally:
            child_release.set()
            await host.close()
    asyncio.run(exercise())


def test_closeout_offers_one_scientific_choice_without_mandatory_replies(tmp_path):
    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    discussions = ResearchDiscussions(workspace)
    discussions.post(gid, DiscussionPostRequest(title='Open method question', body='Check whether the existing evidence already answers this.', target_task_id=b.thread_id), author_thread_id=a.thread_id)
    middleware = ResearchCollaborationMiddleware(discussions=discussions, graph_id=gid, thread_id=root.thread_id, publish_progress=AsyncMock(), run_id='root-turn')
    state = {'messages': [AIMessage(content='Current synthesis')]}
    reminder = asyncio.run(middleware.aafter_model(state, None))
    assert reminder['jump_to'] == 'model'
    assert reminder['messages'][0].additional_kwargs['catmaster_in_turn']
    # The main researcher may judge no new work is needed. No forced tool call,
    # per-comment status gate, repeated reminder, or fabricated peer wakeup.
    next_state = {**state, **reminder, 'messages': [AIMessage(content='The existing source suffices; no further task is needed.')]}
    assert asyncio.run(middleware.aafter_model(next_state, None)) is None
    graph_row = discussions.graphs.get_graph(gid)
    discussions.graphs.update_graph(gid, expected_revision=graph_row['revision'], changes={'completed': True})
    next_turn = ResearchCollaborationMiddleware(discussions=discussions, graph_id=gid, thread_id=root.thread_id, publish_progress=AsyncMock(), run_id='later-turn')
    assert asyncio.run(next_turn.aafter_model(next_state, None)) is None


def test_discussion_policy_is_role_local_system_guidance_and_not_repeated_in_notices(tmp_path, monkeypatch):
    from catmaster.llm.config import LLMConfig, LLMProfile
    import catmaster.specialists.runtime as runtime_mod

    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    discussions = ResearchDiscussions(workspace)
    monkeypatch.setattr(runtime_mod, 'build_chat_model', lambda cfg: Model(responses=[AIMessage(content='done')]))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, '_load_create_deep_agent', staticmethod(lambda: lambda **kw: kw))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, '_research_subagents', lambda *args, **kwargs: [])
    profile = LLMProfile(models={'test': LLMConfig(model='gpt-6-astra')}, agents={
        role: 'test' for role in ('research_lead', 'task_runner', 'summary', 'memory_patch', 'proposal', 'director')})
    built = runtime_mod.build_specialist_runner(
        workspace=workspace, llm_profile=profile, reporter=None, run_control=None,
        project_id='workspace', preferred_entrypoint='research',
    )
    discussions.post(gid, DiscussionPostRequest(title='Evidence question', body='Review the source conditions.',
        target_task_id=b.thread_id), author_thread_id=a.thread_id)
    policies = []
    for thread in (root, b):
        middleware = ResearchCollaborationMiddleware(discussions=discussions, graph_id=gid,
            thread_id=thread.thread_id, publish_progress=AsyncMock(), run_id='test-turn')
        policy = middleware.system_guidance()
        policies.append(policy)
        runtime = {'checkpointer': object(), 'store': object(), 'backend': object(),
                   'collaboration_middleware': middleware}
        kwargs = asyncio.run(built.runner._build_entry_agent(
            entrypoint=thread.entrypoint, runtime=runtime, thread_id=thread.thread_id))
        assert kwargs['system_prompt'].count(policy) == 1
        assert middleware in kwargs['middleware']
        # Raw context helpers must not inherit the researcher's discussion authority.
        for spec in kwargs['subagents']:
            assert policy not in spec.get('system_prompt', '')
        notice = asyncio.run(middleware.abefore_model({'messages': []}, None))
        message = notice['messages'][0]
        assert isinstance(message, HumanMessage)
        assert policy not in message.content
        assert 'SELECT * FROM research_discussions' in message.content
        assert ('target_task_id =' in message.content) is (thread == b)
        assert asyncio.run(middleware.abefore_model({**notice, 'messages': []}, None)) is None
    assert policies[0] != policies[1]
    unbound = asyncio.run(built.runner._build_entry_agent(
        entrypoint='research', runtime={'checkpointer': object(), 'store': object(), 'backend': object()},
        thread_id='unbound'))
    assert not any(policy in unbound['system_prompt'] for policy in policies)


def test_api_posts_are_passive_and_old_review_data_stays_readable(tmp_path, monkeypatch):
    from catmaster.webui.server import create_app
    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    enqueue = AsyncMock(side_effect=AssertionError('Discussion must not enqueue'))
    monkeypatch.setattr(ExecutionHost, 'enqueue', enqueue)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    base = f'/api/workspaces/workspace/research-graphs/{gid}/discussions'
    result = client.post(base, json={'title': 'Question for the next synthesis', 'body': 'Check the existing methods.'})
    assert result.status_code == 200, result.text
    message_id = result.json()['message_id']
    assert client.get(base).json()['collaboration_enabled']
    assert client.post(base, json={'title': 'No escalation flag', 'body': 'text', 'request_review': True}).status_code == 422
    with connect_workspace_db(workspace) as conn:
        conn.execute("UPDATE research_discussions SET review_status='pending', review_reason='Previous impact note', review_notified_run_id='old-run' WHERE message_id=?", (message_id,))
    history = client.get(base).json()['messages'][0]
    assert history['review_reason'] == 'Previous impact note' and history['body'] == 'Check the existing methods.'
    reply = client.post(base, json={'body': 'Existing methods suffice; no further investigation.', 'reply_to': message_id,
        'resolves_message_id': message_id, 'review_outcome': 'addressed'})
    assert reply.status_code == 200, reply.text
    enqueue.assert_not_awaited()
    # Switching out of Persistent Research removes communication admission but
    # does not delete discussion history or turn it into a completion gate.
    store = ThreadStore(workspace=workspace)
    store.update_thread(root.thread_id, entrypoint='research')
    assert not client.get(base).json()['collaboration_enabled']
    assert client.post(base, json={'title': 'Ordinary research', 'body': 'No persistent channel'}).status_code == 409
    assert client.get(base).json()['messages'][0]['review_response_id'] == reply.json()['message_id']


def test_old_automatic_root_review_packet_does_not_restart(tmp_path):
    async def exercise():
        workspace, graph, root, a, b = setup_graph(tmp_path)
        host, service = build_service(tmp_path, workspace, AsyncMock(side_effect=AssertionError('No automatic root wake')))
        packet = await service._prepare_turn(root.thread_id, ThreadSubmitRequest(text='Old automatic review', entrypoint=root.entrypoint),
            run_metadata={'catmaster_discussion_review_graph': graph['graph']['graph_id']}, identity='old-review')
        assert (await service.execute_turn(packet))['status'] == 'interrupted'
    asyncio.run(exercise())
