import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from deepagents import create_deep_agent
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from catmaster.research.discussions import DiscussionPostRequest, ResearchDiscussions
from catmaster.research.knowledge_graph.models import GraphCreateRequest
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.runtime.execution import ExecutionHost
from catmaster.specialists.research_collaboration import ResearchCollaborationMiddleware
from catmaster.storage import connect_workspace_db
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore
from test_local_execution import Model


def setup_graph(tmp_path):
    workspace = tmp_path / 'workspace'
    (workspace / 'files').mkdir(parents=True)
    service = ResearchGraphService(workspace=workspace)
    graph = service.create_graph(GraphCreateRequest(question='Cu CO2 selectivity', orchestration_mode='auto', initial_hypotheses=[{'claim': 'Ion history changes selectivity'}]))
    store = ThreadStore(workspace=workspace)
    graph_id = graph['graph']['graph_id']
    root = store.create_thread(title='Root', entrypoint='persistent_research')
    a = store.create_thread(title='A', entrypoint='research', parent_thread_id=root.thread_id,
                            meta={'background_task': True, 'task_description': 'Ion identity', 'research_branch': True})
    b = store.create_thread(title='B', entrypoint='research', parent_thread_id=root.thread_id,
                            meta={'background_task': True, 'task_description': 'Ion history', 'research_branch': True})
    for t in (root, a, b):
        store.update_thread(t.thread_id, active_research_graph_id=graph_id)
    service.store.set_orchestration_thread(graph_id, root.thread_id)
    return workspace, graph, root, a, b


def test_discussion_full_bodies_replies_sql_scope_and_concurrent_posts(tmp_path):
    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    node = graph['nodes'][0]['node_id']
    discussions = ResearchDiscussions(workspace)
    body = 'partial evidence with conditions\n' * 2000
    request = DiscussionPostRequest(title='Separate identity and history', body=body, node_id=node,
                                    target_task_id=b.thread_id, references=['doi:10.123/example'])
    first = discussions.post(gid, request, author_thread_id=a.thread_id, message_id='first')
    assert first['body'] == body.strip()
    assert discussions.post(gid, request, author_thread_id=a.thread_id, message_id='first') == first
    reply = discussions.post(gid, DiscussionPostRequest(body='Which baseline?', reply_to='first'), author_thread_id=b.thread_id)
    assert reply['discussion_id'] == 'first' and reply['target_task_id'] == a.thread_id and reply['node_id'] == node
    assert discussions.notice(gid, a.thread_id, 0)['count'] == 1
    assert discussions.notice(gid, a.thread_id, reply['seq'])['count'] == 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda i: discussions.post(gid, DiscussionPostRequest(title=f'Question {i}', body='Independent question'),
                                                      author_thread_id=b.thread_id), range(24)))
    assert len({r['seq'] for r in rows}) == 24
    assert discussions.graphs.get_graph(gid)['revision'] == graph['graph']['revision']
    latest = discussions.messages(gid, limit=7)
    older = discussions.messages(gid, limit=100, before_seq=latest['next_before_seq'])
    assert len(latest['messages']) + len(older['messages']) == 26
    assert latest['messages'][0]['seq'] > older['messages'][-1]['seq']
    query = ResearchGraphSQLQuery(workspace)
    assert query.execute(graph_id=gid, sql="SELECT body FROM research_discussions WHERE message_id='first'")['rows'] == [{'body': body.strip()}]
    other = ResearchGraphService(workspace=workspace).create_graph(GraphCreateRequest(question='Unrelated graph'))['graph']['graph_id']
    assert query.execute(graph_id=other, sql='SELECT * FROM research_discussions')['rows'] == []
    with pytest.raises(ValueError, match='different graph'):
        discussions.post(other, request, author_thread_id=a.thread_id)
    with pytest.raises(ValueError, match='not in'):
        discussions.post(gid, DiscussionPostRequest(title='bad', body='bad', node_id='absent'), author_thread_id=a.thread_id)
    with connect_workspace_db(workspace) as conn:
        assert {json.loads(r[0])['change'] for r in conn.execute("SELECT payload_json FROM ui_events WHERE payload_json LIKE '%discussion.posted%'")} == {'discussion.posted'}


def test_notice_only_next_model_boundary_native_checkpoint_and_explicit_progress(tmp_path):
    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    discussions = ResearchDiscussions(workspace)
    progress = AsyncMock()
    middleware = ResearchCollaborationMiddleware(discussions=discussions, graph_id=gid, thread_id=a.thread_id, publish_progress=progress)

    async def exercise():
        @tool
        async def notify_progress(summary: str, next_step: str = '') -> str:
            """Publish the researcher's explicit current work."""
            discussions.post(gid, DiscussionPostRequest(title='Compare methods', body='Peer-only evidence', target_task_id=a.thread_id), author_thread_id=b.thread_id)
            return 'recorded'
        responses = [AIMessage(content='', tool_calls=[{'id': 'p', 'name': 'notify_progress', 'args': {'summary': 'Check ion history', 'next_step': 'Compare baseline'}}]), AIMessage(content='Result with conditions')]
        config = {'configurable': {'thread_id': a.thread_id}}
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'checkpoint.sqlite')) as saver:
            agent = create_deep_agent(Model(responses=responses), tools=[notify_progress], middleware=[middleware], checkpointer=saver)
            # Custom middleware must retain DeepAgents native incremental message channel.
            from langgraph.channels import DeltaChannel
            assert isinstance(agent.channels['messages'], DeltaChannel)
            state = await agent.ainvoke({'messages': [HumanMessage(content='Investigate ion effects')]}, config)
            notices = [m for m in state['messages'] if isinstance(m, HumanMessage) and m.additional_kwargs.get('catmaster_notification') == 'research_discussion']
            assert len(notices) == 1
            assert 'Peer-only evidence' not in notices[0].content
            assert state['research_discussion_notice_seq'] > 0
            progress.assert_awaited_once_with('p', {'summary': 'Check ion history', 'next_step': 'Compare baseline'})
        # Rebuild with the same native SQLite checkpoint: no second full notice or tool replay.
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'checkpoint.sqlite')) as saver:
            agent = create_deep_agent(Model(responses=[AIMessage(content='Continued')]), tools=[notify_progress], middleware=[middleware], checkpointer=saver)
            state = await agent.ainvoke({'messages': [HumanMessage(content='Summarize')]}, config)
            assert sum(m.additional_kwargs.get('catmaster_notification') == 'research_discussion' for m in state['messages']) == 1
            assert sum(isinstance(m, ToolMessage) for m in state['messages']) == 1
        progress.reset_mock()
        await middleware.abefore_model({'messages': [AIMessage(content='', tool_calls=[{'id': 'failed', 'name': 'notify_progress', 'args': {'summary': 'invalid'}}]), ToolMessage(content='failed', tool_call_id='failed', status='error')], 'research_discussion_notice_seq': state['research_discussion_notice_seq']}, None)
        progress.assert_not_awaited()
    asyncio.run(exercise())


def test_api_shared_discussion_has_no_submit_and_preserves_long_content(tmp_path):
    workspace, graph, root, a, b = setup_graph(tmp_path)
    from catmaster.webui.server import create_app
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    base = f"/api/workspaces/workspace/research-graphs/{graph['graph']['graph_id']}/discussions"
    response = client.post(base, json={'title': 'User evidence', 'body': 'method detail ' * 2000, 'target_task_id': a.thread_id})
    assert response.status_code == 200, response.text
    assert response.json()['author_kind'] == 'user'
    assert client.get(base).json()['messages'][0]['body'] == response.json()['body']
    assert not ThreadStore(workspace=workspace).list_messages(a.thread_id)
    assert client.post(base, json={'title': 'wrong node', 'body': 'text', 'node_id': 'missing'}).status_code == 409
    tasks = client.get(base.removesuffix('/discussions') + '/tasks')
    assert tasks.status_code == 200, tasks.text
    assert {t['task_id'] for t in tasks.json()['tasks']} == {a.thread_id, b.thread_id}
    assert all(t['entrypoint'] == 'research' for t in tasks.json()['tasks'])


def test_running_peer_discovery_exchange_and_idle_message_does_not_wake(tmp_path):
    async def exercise():
        workspace, graph, root, a, b = setup_graph(tmp_path)
        gid = graph['graph']['graph_id']
        store = ThreadStore(workspace=workspace)
        discussions = ResearchDiscussions(workspace)
        working, release = asyncio.Event(), asyncio.Event()
        notices = []

        @asynccontextmanager
        async def factory(service, packet):
            @tool
            async def evidence_work() -> str:
                """Wait for the bounded source read to finish."""
                working.set()
                await release.wait()
                return 'Source evidence'
            middleware = ResearchCollaborationMiddleware(discussions=discussions, graph_id=gid, thread_id=packet['thread_id'],
                publish_progress=lambda mid, data: service.publish_research_progress(packet, mid, data))
            async with AsyncSqliteSaver.from_conn_string(str(workspace / 'metadata/deepagent_threads.sqlite')) as saver:
                agent = create_deep_agent(Model(responses=[AIMessage(content='', tool_calls=[{'id': 'read', 'name': 'evidence_work', 'args': {}}]), AIMessage(content='Finished')]), tools=[evidence_work], middleware=[middleware], checkpointer=saver)
                yield agent
                state = await agent.aget_state({'configurable': {'thread_id': store.get_thread(packet['thread_id']).deepagent_thread_id}})
                notices.extend(m for m in state.values['messages'] if m.additional_kwargs.get('catmaster_notification') == 'research_discussion')

        host = ExecutionHost(tmp_path / 'execution.sqlite', lambda *_: service)
        service = LocalThreadService(workspace=workspace, workspace_id='workspace', store=store,
            broker=ThreadEventBroker(workspace=workspace), artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id='workspace'),
            normalize_entrypoint=lambda x: x or 'research', permission_mode_for_thread=lambda *_: 'auto', execution=host, graph_factory=factory)
        await host.start()
        try:
            # Notify keeps the test's root idle. Peer discussion must not bypass it.
            store.update_thread(a.thread_id, meta={**a.meta, 'on_completion': 'notify'})
            run = await service.submit(thread_id=a.thread_id, payload=ThreadSubmitRequest(text='Read evidence'))
            await asyncio.wait_for(working.wait(), 15)
            await service.publish_research_progress({'thread_id': a.thread_id, 'run_id': run['run_id']}, 'p1', {'summary': 'Reading ion-history methods', 'next_step': 'Compare controls'})
            peer = await service.read_task(b.thread_id, a.thread_id)
            assert peer['status'] == 'running' and not peer['result']
            assert peer['progress']['next_step'] == 'Compare controls'
            from catmaster.webui.projections.messages import project_part
            part = project_part({'id': 'progress', 'type': 'subagent', 'status': 'running',
                'meta': {'task_id': a.thread_id, 'research_progress': peer['progress']}},
                workspace=workspace, thread_id=root.thread_id, message_id='parent-answer')
            assert part.progress_summary == 'Reading ion-history methods'
            assert part.progress_next_step == 'Compare controls'
            assert len((await service.list_research_tasks(b.thread_id, scope='research_graph'))['tasks']) == 2
            with pytest.raises(HTTPException):
                service._child(b.thread_id, a.thread_id)
            tools = {t.name: t for t in service.background_tools({'thread_id': b.thread_id, 'run_id': 'discussion-run'})}
            schema = tools['post_research_message'].tool_call_schema.model_json_schema()
            assert 'runtime' not in schema['properties']
            for name in ('title', 'reply_to', 'node_id', 'target_task_id', 'references'):
                assert 'anyOf' not in schema['properties'][name]
            posted = discussions.post(gid, DiscussionPostRequest(title='Overlap?', body='Check the same control?', target_task_id=a.thread_id), author_thread_id=b.thread_id)
            assert not release.is_set() and not notices
            assert len(await host.runs(workspace, a.thread_id)) == 1
            release.set()
            handle = await host.client.retrieve_workflow_async(run['run_id'])
            assert (await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 25))['status'] == 'success'
            assert len(notices) == 1
            # Later discussion addressed to a completed task stays stored; tick also ignores it.
            discussions.post(gid, DiscussionPostRequest(body='Later evidence', reply_to=posted['message_id']), author_thread_id=b.thread_id)
            store.update_thread(root.thread_id, meta={'last_run_id': 'old-run', 'permission_mode': 'auto'})
            await service.reconcile_research_graph_updates(gid, root.thread_id)
            assert len(await host.runs(workspace, a.thread_id)) == 1
            assert await host.runs(workspace, root.thread_id) == []
        finally:
            release.set()
            await host.close()
    asyncio.run(exercise())


def test_discussion_is_a_durable_message_source_without_conversation_copy(tmp_path):
    workspace, graph, root, a, b = setup_graph(tmp_path)
    gid = graph['graph']['graph_id']
    discussions = ResearchDiscussions(workspace)
    message = discussions.post(gid, DiscussionPostRequest(title='Control warning', body='Gas-line delay is not surface memory'), author_thread_id=a.thread_id)
    service = ResearchGraphService(workspace=workspace)
    ref = service.validate_ref({'ref_kind': 'message', 'ref_id': message['message_id']})
    resolved = service.resolve_ref(ref)
    assert resolved['available'] and resolved['discussion_id'] == message['discussion_id']
    assert 'thread_id' not in resolved  # Clicking opens the discussion, not an unrelated chat.
    assert not ThreadStore(workspace=workspace).list_messages(a.thread_id)
    # Explicitly citing evidence from another graph makes just that source reachable.
    other = service.create_graph(GraphCreateRequest(question='Follow-up', initial_hypotheses=[{'claim': 'A control changes interpretation', 'refs': [ref]}]))
    rows = ResearchGraphSQLQuery(workspace).execute(graph_id=other['graph']['graph_id'], sql='SELECT message_id, body FROM research_discussions')['rows']
    assert rows == [{'message_id': message['message_id'], 'body': message['body']}]
