"""JSON checkpoint summaries must work through actual DeepAgents compaction."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from catmaster.llm.config import AgentRuntimeConfig
from catmaster.runtime.deepagent_summarization import RestoringSummarizationMiddleware
from catmaster.specialists.runtime import SpecialistRunner


class Model(FakeMessagesListChatModel):
    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        assert all(isinstance(message, BaseMessage) for message in messages)
        self.calls.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_summary_restoration_preserves_content_identity_and_state():
    model = Model(responses=[AIMessage(content='Unused')])
    summary = HumanMessage(id='old-summary', name='summary', content=[
        {'type': 'text', 'text': 'Ni coordination is conditional on the six-membered ring.'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.org/diagram.png'}},
    ], additional_kwargs={'lc_source': 'summarization'}, response_metadata={'note': 'retained'})
    event = {'cutoff_index': 450, 'summary_message': summary.model_dump(mode='json'),
             'file_path': '/conversation_history/old.md'}
    state = {'messages': [], '_summarization_event': event, 'todos': [{'task': 'keep', 'status': 'pending'}]}
    original = json.loads(json.dumps(state))
    request = ModelRequest(model=model, messages=[], state=state)
    restored = RestoringSummarizationMiddleware._restore_request(request)
    assert restored.state['_summarization_event']['summary_message'] == summary
    assert restored.state['_summarization_event']['cutoff_index'] == 450
    assert restored.state['_summarization_event']['file_path'] == event['file_path']
    assert restored.state['messages'] is state['messages']
    assert restored.state['todos'] is state['todos']
    assert state == original
    assert RestoringSummarizationMiddleware._restore_request(restored) is restored
    empty = ModelRequest(model=model, messages=[], state={})
    assert RestoringSummarizationMiddleware._restore_request(empty) is empty


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('cap,compacts', [(258_000, False), (200_000, True), (None, True)])
def test_imported_summary_and_new_compaction_use_one_upstream_summarizer(tmp_path, async_mode, cap, compacts):
    model = Model(responses=[AIMessage(content='Renewed summary'), AIMessage(content='Requested answer')])
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = SimpleNamespace(agent_runtime=AgentRuntimeConfig(deepagent_context_trigger_token_cap=cap))
    graph = runner._create_deep_agent(model=model, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'restored'}}
    hidden = [HumanMessage(id='prior-user', content='Old request'),
              AIMessage(id='prior-call', content='', tool_calls=[{'id': 'read', 'name': 'read_file', 'args': {}}]),
              ToolMessage(id='prior-result', content='Old result', tool_call_id='read')]
    tail = []
    for index in range(8):
        tail.extend([HumanMessage(id=f'q{index}', content=f'Question {index}'),
                     AIMessage(id=f'a{index}', content=f'Result {index}')])
    tail[-1] = AIMessage(id='a7', content='Latest result',
        usage_metadata={'input_tokens': 203_468, 'output_tokens': 1, 'total_tokens': 203_469},
        response_metadata={'model_provider': model._get_ls_params()['ls_provider']})
    tail.append(HumanMessage(id='latest', content='Prepare the script only.'))
    original = hidden + tail
    # The root channel has already restored message objects; only the nested
    # summary is still a JSON dict when the model node reads imported state.
    seed = {'messages': original,
            '_summarization_event': {'cutoff_index': 3, 'file_path': '/conversation_history/old.md',
                'summary_message': HumanMessage(id='old-summary', content='Previously summarized Ni evidence').model_dump(mode='json')}}
    history = tmp_path / 'conversation_history' / 'old.md'
    history.parent.mkdir()
    history.write_text('Original detailed evidence')
    if async_mode:
        asyncio.run(graph.ainvoke(Command(update=seed), config))
    else:
        graph.invoke(Command(update=seed), config)
    result = graph.get_state(config).values
    assert len(model.calls) == (2 if compacts else 1)
    assert [message.id for message in result['messages'][:len(original)]] == [message.id for message in original]
    assert history.read_text() == 'Original detailed evidence'
    last_input = model.calls[-1]
    assert last_input[-1].id == 'latest'
    assert not {'prior-user', 'prior-call', 'prior-result'} & {message.id for message in last_input}
    if compacts:
        event = result['_summarization_event']
        assert isinstance(event['summary_message'], HumanMessage)
        assert event['cutoff_index'] > 3
        assert 'Renewed summary' in event['summary_message'].content
        offloaded = (tmp_path / event['file_path'].lstrip('/')).read_text()
        assert 'Previously summarized Ni evidence' in offloaded
        assert 'Question 0' in offloaded
        assert result['messages'][-1].content == 'Requested answer'
    else:
        assert last_input[1].id == 'old-summary'  # system policy precedes effective history
        assert result['_summarization_event'] == seed['_summarization_event']


def test_long_tool_trace_reaches_summary_model_instead_of_empty_placeholder(tmp_path):
    model = Model(responses=[AIMessage(content='Ni conditions retained'), AIMessage(content='Ready')])
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = SimpleNamespace(agent_runtime=AgentRuntimeConfig(deepagent_context_trigger_token_cap=100))
    graph = runner._create_deep_agent(model=model, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True))
    messages = [HumanMessage(content='Earlier hidden request'), HumanMessage(content='Read existing evidence'),
        AIMessage(content='', tool_calls=[{'id': 'read', 'name': 'read_file', 'args': {}}]),
        ToolMessage(content='Condition-specific evidence ' * 15_000, tool_call_id='read'),
        *[AIMessage(content=f'Observation {index}') for index in range(12)],
        HumanMessage(content='Prepare the script')]
    seed = {'messages': messages, '_summarization_event': {'cutoff_index': 1,
        'summary_message': HumanMessage(content='Prior Ni summary').model_dump(mode='json'), 'file_path': None}}
    result = asyncio.run(graph.ainvoke(Command(update=seed)))
    assert len(model.calls) == 2
    assert 'Condition-specific evidence' in model.calls[0][-1].content
    assert any('Ni conditions retained' in message.content for message in model.calls[-1])
    assert result['messages'][-1].content == 'Ready'


class FileModel(Model):
    model_name: str = 'file-test-model'


def _file_history(*, reported=56_000, model='file-test-model'):
    return [HumanMessage(content='Compare the two workbooks'),
        AIMessage(content='', tool_calls=[
            {'id': 'a', 'name': 'read_file', 'args': {'file_path': '/a.xlsx'}},
            {'id': 'b', 'name': 'read_file', 'args': {'file_path': '/b.xlsx'}},
        ]),
        *[ToolMessage(tool_call_id=key, content=[{
            'type': 'file', 'source_type': 'base64', 'data': 'A' * 600_000,
            'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        }]) for key in ('a', 'b')],
        AIMessage(content='Both tables consumed',
            usage_metadata={'input_tokens': reported - 100, 'output_tokens': 100, 'total_tokens': reported},
            response_metadata={'model_name': model}),
        HumanMessage(content='Continue the comparison')]


@pytest.mark.parametrize('async_mode', [False, True])
def test_consumed_workbooks_do_not_repeat_compaction_or_change_model_input(tmp_path, async_mode):
    import httpx
    from test_provider_request_capture import make_model, response_stream
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=response_stream())
    model = make_model(handler)
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = SimpleNamespace(agent_runtime=AgentRuntimeConfig())
    graph = runner._create_deep_agent(model=model, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True))
    original = _file_history(model='gpt-6-astra')
    result = (asyncio.run(graph.ainvoke({'messages': original})) if async_mode
              else graph.invoke({'messages': original}))
    assert len(requests) == 1
    assert result['messages'][-1].text == 'OK'
    def file_data(value):
        if isinstance(value, dict):
            if value.get('type') == 'input_file':
                yield value['file_data']
            else:
                for child in value.values():
                    yield from file_data(child)
        elif isinstance(value, list):
            for child in value:
                yield from file_data(child)
    sent_files = list(file_data(requests[0]))
    assert len(sent_files) == 2
    assert all(data.split(',', 1)[-1] == 'A' * 600_000 for data in sent_files)
    assert [m.content for m in result['messages'] if isinstance(m, ToolMessage)] == [
        m.content for m in original if isinstance(m, ToolMessage)]
    assert not result.get('_summarization_event')


def test_file_count_retains_threshold_and_unmeasured_growth(tmp_path):
    from langchain_core.messages import SystemMessage
    model = FileModel(responses=[AIMessage(content='Unused')])
    middleware = RestoringSummarizationMiddleware(model=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), trigger=('tokens', 258_000))
    messages = _file_history()
    system = SystemMessage(content='Preserve the requested comparison')
    count = middleware._count_tokens(messages, system, [])
    assert 56_000 < count < 60_000
    # New large text or a new unmeasured binary still triggers. An old usage
    # record never makes later payloads disappear from the estimate.
    for content in ['evidence ' * 130_000, [{'type': 'file', 'source_type': 'base64', 'data': 'A' * 1_100_000}]]:
        growing = [*messages, HumanMessage(content=content)]
        assert middleware._should_summarize(growing, middleware._count_tokens(growing, system, []))
    for history in [_file_history(reported=260_000), _file_history(model='other-model'), messages[:-2]]:
        assert middleware._should_summarize(history, middleware._count_tokens(history, system, []))
    # Current schemas/system remain counted even if the preceding call used a
    # different tool binding. Plain text continues to use the upstream counter.
    assert middleware._count_tokens(messages, SystemMessage(content='S' * 900_000), []) > 258_000
    plain = [HumanMessage(content='text ' * 100)]
    assert middleware._count_tokens(plain, system, []) == middleware.token_counter([system, *plain], tools=[])


@pytest.mark.parametrize('content', [
    'Existing text evidence',
    [{'type': 'image', 'source_type': 'base64', 'data': 'AA==', 'mime_type': 'image/png'}],
    [{'type': 'image_url', 'image_url': {'url': 'https://example.org/page.png'}}],
    [{'type': 'file', 'source_type': 'base64', 'data': 'AA==', 'mime_type': 'application/pdf'}],
])
def test_actual_usage_triggers_even_with_codex_provider_alias(tmp_path, content):
    from test_provider_request_capture import make_model
    model = make_model(lambda _: pytest.fail('No provider call is needed for a trigger check'))
    middleware = RestoringSummarizationMiddleware(model=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), trigger=('tokens', 258_000))
    messages = [HumanMessage(content=content), AIMessage(content='Read the figures',
        usage_metadata={'input_tokens': 406_401, 'output_tokens': 100, 'total_tokens': 406_501},
        response_metadata={'model_name': 'gpt-6-astra', 'model_provider': 'openai'}),
        HumanMessage(content='Revise the report')]
    # This is the actual adapter's trace label, not a fake provider comparison.
    assert model._get_ls_params()['ls_provider'] == 'openai-codex'
    assert middleware.token_counter(messages) < 1000
    assert middleware._should_summarize(messages, middleware._count_tokens(messages, None, []))

    # Previous usage cannot determine this model's cost after a model switch.
    changed = [messages[0], messages[1].model_copy(update={
        'response_metadata': {'model_name': 'different-model', 'model_provider': 'openai'},
    }), messages[2]]
    assert not middleware._should_summarize(changed, middleware._count_tokens(changed, None, []))


@pytest.mark.parametrize('async_mode', [False, True])
def test_token_retention_advances_past_summary_and_does_not_loop(tmp_path, async_mode):
    from langchain.agents.middleware.types import ModelResponse
    from test_provider_request_capture import make_model
    model = make_model(lambda _: pytest.fail('No provider call in compaction replay'))
    model.profile = {'max_input_tokens': 1_050_000}

    class Summarizer(RestoringSummarizationMiddleware):
        summary_inputs = []

        def _create_summary(self, messages):
            self.summary_inputs.append(messages)
            return 'Retained scientific findings; details in the history file.'

        async def _acreate_summary(self, messages):
            return self._create_summary(messages)

    middleware = Summarizer(model=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        trigger=('tokens', 258_000), keep=('fraction', 0.1))
    messages = [HumanMessage(content='Old summary ' * 600)]
    for i in range(52):
        messages.extend([
            AIMessage(content='', tool_calls=[{'name': 'read_file', 'id': str(i), 'args': {}}]),
            ToolMessage(content='Scientific detail ' * 430, tool_call_id=str(i)),
        ])
    messages += [AIMessage(content='Prior observation',
        response_metadata={'model_name': 'gpt-6-astra', 'model_provider': 'openai'},
        usage_metadata={'input_tokens': 277_146, 'output_tokens': 1, 'total_tokens': 277_147}),
        HumanMessage(content='Continue')]
    native_cutoff = middleware._lc_helper._determine_cutoff_index(messages)
    cutoff = middleware._determine_cutoff_index(messages)
    assert cutoff > max(1, native_cutoff)
    assert not isinstance(messages[cutoff], ToolMessage)
    original_helper_counter = middleware._lc_helper.token_counter
    state = {'messages': messages}
    sent = []

    def handler(request):
        sent.append(request.messages)
        return ModelResponse(result=[AIMessage(content='Continue the analysis',
            response_metadata={'model_name': 'gpt-6-astra', 'model_provider': 'openai'},
            usage_metadata={'input_tokens': 104_000, 'output_tokens': 10, 'total_tokens': 104_010})])

    async def ahandler(request):
        return handler(request)

    async def run():
        for _ in range(3):
            request = ModelRequest(model=model, messages=state['messages'], state=state)
            response = (await middleware.awrap_model_call(request, ahandler) if async_mode
                        else middleware.wrap_model_call(request, handler))
            if getattr(response, 'command', None):
                state.update(response.command.update)
                result = response.model_response.result
            else:
                result = response.result
            state['messages'] = [*state['messages'], *result, HumanMessage(content='Next')]

    asyncio.run(run())
    assert len(middleware.summary_inputs) == 1
    assert len(middleware.summary_inputs[0]) > 1
    assert state['_summarization_event']['cutoff_index'] == cutoff
    assert len(sent[0]) < len(messages)
    assert state['messages'][:len(messages)] == messages
    archive = tmp_path / state['_summarization_event']['file_path'].lstrip('/')
    assert 'Scientific detail' in archive.read_text()
    assert middleware._lc_helper.token_counter is original_helper_counter
    model.http_client.close()
    asyncio.run(model.http_async_client.aclose())
