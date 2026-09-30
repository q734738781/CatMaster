import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from catmaster.webui.async_activity import AsyncActivityProjection
from catmaster.webui.projections.events import project_event
from catmaster.webui.projections.messages import project_message, project_part
from test_run_projection import _service




def test_nested_activity_reasoning_updates_tools_and_replay(tmp_path):
    service, _, store = _service(tmp_path)
    child = store.create_thread(title='Writing', thread_id='child-thread')
    changes = []
    def make():
        return AsyncActivityProjection(store=store, broker=service.broker, thread_id=child.thread_id,
            run_id='child-run', source='writing_specialist', on_update=lambda *args: changes.append(args))
    projector = make()
    metadata = {'lc_agent_name': 'writing_worker_agent'}
    frames = [
        {'type': 'AIMessageChunk', 'id': 'native-worker', 'content': [{'type': 'reasoning', 'reasoning': 'Compare the '} ]},
        {'type': 'AIMessageChunk', 'id': 'native-worker', 'content': [{'type': 'reasoning', 'reasoning': 'Cu evidence.'}]},
    ]
    for message in frames:
        projector.process({'event': 'messages|tools:worker', 'data': [message, metadata]})
    rendered = project_message(store.list_messages(child.thread_id)[0])
    assert rendered.source == 'Writing Worker'
    assert rendered.parts[0].text == 'Compare the Cu evidence.'
    # A new subscriber replays the same native run without doubling text.
    restarted = make()
    for message in frames:
        restarted.process({'event': 'messages|tools:worker', 'data': [message, metadata]})
    assert store.list_messages(child.thread_id)[0].parts[0].text == 'Compare the Cu evidence.'
    call = {'type': 'ai', 'id': 'progress', 'content': 'Evidence is ready.', 'tool_calls': [
        {'id': 'notify', 'name': 'notify_progress', 'args': {'summary': '已找到需要保留的铜表面条件差异。', 'next_step': '整理图表'}}]}
    restarted.process({'type': 'messages', 'ns': ['tools:worker'], 'data': [call, metadata]})
    restarted.process({'event': 'updates|tools:worker', 'data': {'tools': {'messages': [
        {'type': 'tool', 'id': 'result', 'tool_call_id': 'notify', 'content': 'Progress update recorded.'}]}}})
    assert changes[-1][0] == '已找到需要保留的铜表面条件差异。'
    message = store.get_message(child.thread_id, 'activity_progress')
    assert message.parts[-1].meta['output'] == 'Progress update recorded.'
    assert project_message(message).parts[-1].type == 'progress'
    restarted.process({'event': 'messages', 'data': [
        {'type': 'AIMessageChunk', 'id': 'internal', 'content': 'SESSION INTENT'}, {'lc_source': 'summarization'}]})
    assert not any('SESSION INTENT' in part.text for message in store.list_messages(child.thread_id) for part in message.parts)
    public_events = [project_event(event).model_dump(mode='json') for event in service.broker.replay(child.thread_id)]
    assert any(event['event'] == 'message.updated' for event in public_events)
    restarted.finish('success')
    assert all(message.status == 'completed' for message in store.list_messages(child.thread_id))


def test_finished_subagent_retains_inspection_without_steer():
    part = project_part({'id': 'child', 'type': 'subagent', 'status': 'completed',
        'meta': {'task_id': 'native-child', 'source': 'writing_specialist', 'native_status': 'success'}},
        workspace=None, thread_id='root', message_id='reply')
    assert part.type == 'subagent'
    assert part.detail_ref == '/api/threads/root/async-subagents/native-child'
    assert [action.id for action in part.actions] == ['open_async_subagent']


def test_snapshot_publishes_latest_activity_once_and_replay_keeps_history(tmp_path):
    service, _, store = _service(tmp_path)
    child = store.create_thread(title='Writing', thread_id='child-thread')
    changes = []
    projection = AsyncActivityProjection(store=store, broker=service.broker,
        thread_id=child.thread_id, run_id='run', source='Writing',
        on_update=lambda *args: changes.append(args))

    def call(name):
        return {'type': 'ai', 'id': name, 'content': '', 'tool_calls': [
            {'id': name + '-call', 'name': name, 'args': {}}]}

    snapshot = {'messages': [call('read_sources'), call('write_report')]}
    projection.restore(snapshot)
    assert [change[2] for change in changes] == ['write_report']
    # Replaying old root and nested frames reconstructs the entire transcript
    # without replacing the current card with past tool names.
    projection.process({'event': 'values', 'data': {'messages': [call('read_sources')]}})
    projection.process({'event': 'values|tools:worker', 'data': {'messages': [call('old_nested_tool')]}})
    projection.process({'event': 'values', 'data': snapshot})
    assert [change[2] for change in changes] == ['write_report']
    assert store.get_message(child.thread_id, 'activity_old_nested_tool') is not None
    # Duplicate full states and streamed prefixes cannot rewind saved messages.
    projection.process({'event': 'messages', 'data': [
        {'type': 'AIMessageChunk', 'id': 'read_sources', 'content': 'old prefix'}, {}]})
    projection.process({'event': 'values', 'data': snapshot})
    assert len(changes) == 1
    projection.process({'event': 'updates|tools:worker', 'data': {'model': {'messages': [call('compile_pdf')]}}})
    assert [change[2] for change in changes] == ['write_report', 'compile_pdf']
    projection.process({'event': 'values', 'data': {'messages': [*snapshot['messages'],
        {'type': 'ai', 'id': 'final', 'content': 'Reports completed.'}]}})
    assert changes[-1][0] == 'Reports completed.'
    projection.finish('success')
    assert len(store.list_messages(child.thread_id)) == 5
    assert all(message.status == 'completed' for message in store.list_messages(child.thread_id))


def test_async_instruction_excerpts_survive_public_projection():
    part = project_part({'id': 'child', 'type': 'subagent', 'status': 'running',
        'meta': {'task_id': 'child', 'source': 'writing_specialist',
                 'task_description': '整理陶瓷元数据报告', 'task_followup': '逐篇介绍研究内容，不要封面',
                 'task_followup_status': 'pending'}},
        workspace=None, thread_id='root', message_id='reply')
    assert part.task_description == '整理陶瓷元数据报告'
    assert part.task_followup == '逐篇介绍研究内容，不要封面'
    assert part.task_followup_status == 'pending'
    assert part.detail_ref.endswith('/async-subagents/child')




def test_historical_child_cards_are_not_a_current_activity_source(tmp_path):
    from catmaster.webui.thread_models import ThreadMessage, MessagePart
    from catmaster.webui.projections.messages import project_current_active_parts
    _, _, store = _service(tmp_path)
    thread = store.create_thread(title='Background work')
    store.append_message(ThreadMessage(id='launch', thread_id=thread.thread_id, role='assistant', parts=[
        MessagePart(id='old-tool', type='tool-call', status='running', meta={'tool': 'read_file'}),
        MessagePart(id='child', type='subagent', status='running', meta={'task_id': 'child'}),
    ]))
    store.append_message(ThreadMessage(id='followup', thread_id=thread.thread_id, role='user', parts=[]))
    messages = store.list_current_turn_messages(thread.thread_id)
    assert [message.id for message in messages] == ['followup']
    # Even the complete history cannot revive a child from a historical card.
    assert project_current_active_parts(store.list_messages(thread.thread_id)) == []
    saved = store.get_message_part(thread.thread_id, 'launch', 'child')
    assert saved.status == 'running'  # Transcript history was not rewritten.


def test_native_interruption_keeps_partial_output_without_failure_alert(tmp_path):
    from catmaster.webui.run_projection import RunProjection
    from catmaster.webui.thread_models import ThreadMessage, MessagePart
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Steer')
    store.append_message(ThreadMessage(id='answer', thread_id=thread.thread_id, role='assistant',
        status='streaming', parts=[MessagePart(id='text', type='text', text='已整理的部分结果', status='streaming')]))
    projection = RunProjection(store=store, broker=service.broker,
        artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
        run_id='run', assistant_message_id='answer', text_part_id='text')
    result = projection.finalize(native_status='interrupted', state={'values': {}})
    assert result.status == 'interrupted'
    assert result.parts[0].status == 'interrupted'
    assert result.parts[0].text == '已整理的部分结果'
    events = [project_event(event) for event in service.broker.replay(thread.thread_id)]
    assert not any(event.event == 'run.failed' for event in events)
    assert events[-1].event == 'message.updated'
    assert events[-1].data.message.status == 'interrupted'
@pytest.mark.parametrize('provider_first', [True, False])
def test_stream_message_id_changes_keep_one_message_and_tool_result(tmp_path, provider_first):
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Candidates')
    updates = []

    def make():
        return AsyncActivityProjection(store=store, broker=service.broker,
            thread_id=thread.thread_id, run_id='run', source='research',
            on_update=lambda *args: updates.append(args))

    ns = ('tools:worker',)
    metadata = {'lc_agent_name': 'general-purpose', 'langgraph_checkpoint_ns': 'tools:worker|model:call'}
    provider = 'resp_provider'
    temporary = 'lc_run--temporary'
    frames = [
        AIMessageChunk(id=provider if provider_first else temporary, content=''),
        AIMessageChunk(id=temporary, content=[{'type': 'reasoning', 'reasoning': 'Check structures.'}],
            tool_call_chunks=[{'index': 0, 'id': 'call-one', 'name': 'execute', 'args': '{"command":'}]),
        AIMessageChunk(id=temporary, content='',
            tool_call_chunks=[{'index': 0, 'id': None, 'name': None, 'args': '"enumerate"}'}]),
        AIMessageChunk(id=provider, content='', chunk_position='last'),
    ]
    final = AIMessage(id=provider, content=[{'type': 'reasoning', 'reasoning': 'Check structures.'}],
        tool_calls=[{'id': 'call-one', 'name': 'execute', 'args': {'command': 'enumerate'}}])
    projection = make()
    for frame in frames:
        projection.process({'type': 'messages', 'ns': ns, 'data': [frame, metadata]})
    projection.process({'type': 'updates', 'ns': ns, 'data': {'model': {'messages': [final]}}})
    projection.process({'type': 'updates', 'ns': ns, 'data': {'tools': {'messages': [
        {'type': 'tool', 'tool_call_id': 'call-one', 'content': '1665 rows', 'id': 'result'}]}}})
    messages = store.list_messages(thread.thread_id)
    assert len(messages) == 1
    message = messages[0]
    assert message.status == 'completed'
    assert message.parts[0].text == 'Check structures.'
    assert message.parts[-1].status == 'completed'
    assert message.parts[-1].meta['input'] == {'command': 'enumerate'}
    assert message.parts[-1].meta['output'] == '1665 rows'
    assert set(message.meta['native_message_ids']) == {provider, temporary}
    assert len(updates) == 1
    # Reconnect after completion: aliases and tool ownership must survive.
    restarted = make()
    for frame in frames:
        restarted.process({'type': 'messages', 'ns': ns, 'data': [frame, metadata]})
    restarted.state({'messages': [final]}, ns)
    assert len(store.list_messages(thread.thread_id)) == 1
    assert store.get_message(thread.thread_id, message.id).parts == message.parts
    assert len(updates) == 1
    created = [event for event in service.broker.replay(thread.thread_id) if event.event == 'message.created']
    assert len(created) == 1


def test_parallel_workers_and_successive_calls_keep_distinct_outputs(tmp_path):
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Parallel')
    projection = AsyncActivityProjection(store=store, broker=service.broker,
        thread_id=thread.thread_id, run_id='run', source='research')

    def emit(worker, native_id, content, last=False):
        projection.process({'type': 'messages', 'ns': [f'tools:{worker}'], 'data': [
            AIMessageChunk(id=native_id, content=content, chunk_position='last' if last else None),
            {'lc_agent_name': 'general-purpose', 'langgraph_checkpoint_ns': f'tools:{worker}|model:call'}]})

    emit('a', 'resp_a', '')
    emit('b', 'resp_b', '')
    emit('a', 'lc_run--a', 'Alpha ')
    emit('b', 'lc_run--b', 'Beta')
    # Reattach mid-stream and replay the received prefix without doubling it.
    projection = AsyncActivityProjection(store=store, broker=service.broker,
        thread_id=thread.thread_id, run_id='run', source='research')
    emit('a', 'resp_a', '')
    emit('a', 'lc_run--a', 'Alpha ')
    emit('a', 'lc_run--a', 'done', last=True)
    emit('b', 'resp_b', '')
    emit('b', 'lc_run--b', 'Beta', last=True)
    # A second call may reuse the graph node's metadata after the last chunk.
    emit('a', 'resp_next', '')
    emit('a', 'lc_run--next', 'Next', last=True)
    for worker, native_id, text in [('a', 'resp_a', 'Alpha done'), ('b', 'resp_b', 'Beta'), ('a', 'resp_next', 'Next')]:
        projection.state({'messages': [AIMessage(id=native_id, content=text)]}, (f'tools:{worker}',))
    messages = store.list_messages(thread.thread_id)
    assert len(messages) == 3
    assert [m.parts[0].text for m in messages] == ['Alpha done', 'Beta', 'Next']
    assert all(m.status == 'completed' for m in messages)


@pytest.mark.parametrize(('outcome', 'expected'), [('success', 'completed'), ('interrupted', 'interrupted'), ('error', 'failed')])
def test_finish_keeps_partial_output_and_actual_outcome(tmp_path, outcome, expected):
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Partial')
    for run in ['previous', 'current']:
        projection = AsyncActivityProjection(store=store, broker=service.broker,
            thread_id=thread.thread_id, run_id=run, source='general-purpose')
        projection.message({'type': 'AIMessageChunk', 'id': run, 'content': 'Saved evidence'}, {})
    projection.finish(outcome)
    messages = store.list_messages(thread.thread_id)
    assert messages[0].status == 'streaming'
    assert messages[1].status == expected
    assert messages[1].parts[0].text == 'Saved evidence'
    assert messages[1].parts[0].status == expected
