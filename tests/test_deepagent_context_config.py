"""Context compaction configuration stays separate from execution budgets."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from catmaster.llm.config import AgentRuntimeConfig, LLMProfile
from catmaster.runtime.deepagent_summarization import RestoringSummarizationMiddleware
from catmaster.specialists.runtime import SpecialistRunner


def test_compaction_config_yaml_environment_and_native_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv('CATMASTER_DEEPAGENT_CONTEXT_TRIGGER_TOKEN_CAP', raising=False)
    assert AgentRuntimeConfig.from_dict({}).deepagent_context_trigger_token_cap == 258_000
    monkeypatch.setenv('CATMASTER_DEEPAGENT_CONTEXT_TRIGGER_TOKEN_CAP', '240000')
    assert LLMProfile.from_env().agent_runtime.deepagent_context_trigger_token_cap == 240_000
    template = Path('configs/llm_codex_oauth.template.yaml').read_text()
    config = tmp_path / 'llm.yaml'
    config.write_text(template)
    assert LLMProfile.from_env_or_file(str(config)).agent_runtime.deepagent_context_trigger_token_cap == 258_000
    config.write_text(template.replace('deepagent_context_trigger_token_cap: 258000', 'deepagent_context_trigger_token_cap: null'))
    assert LLMProfile.from_env_or_file(str(config)).agent_runtime.deepagent_context_trigger_token_cap is None
    with pytest.raises(ValueError, match='max_tool_calls'):
        AgentRuntimeConfig.from_dict({'max_tool_calls': 12})


@pytest.mark.parametrize('window,cap,reported,compacts', [
    (None, 258_000, 203_469, False),
    (None, 200_000, 203_469, True),
    (None, None, 203_469, True),
    (128_000, 258_000, 110_000, True),
    (1_000_000, 258_000, 258_001, True),
])
def test_native_compaction_honors_config_and_known_smaller_windows(tmp_path, monkeypatch, window, cap, reported, compacts):
    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    profile = None if window is None else {'max_input_tokens': window}
    model = Model(profile=profile, responses=[AIMessage(content='Context or answer')])
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = SimpleNamespace(agent_runtime=AgentRuntimeConfig(deepagent_context_trigger_token_cap=cap))
    configured = []
    factory = runner._load_create_deep_agent()
    def capture(**kwargs):
        configured.extend(item for item in kwargs.get('middleware', []) if isinstance(item, SummarizationMiddleware))
        return factory(**kwargs)
    monkeypatch.setattr(runner, '_load_create_deep_agent', lambda: capture)
    graph = runner._create_deep_agent(
        model=model, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    # Reported usage recreates the legacy threshold trigger without paid tokens.
    messages = []
    for index in range(8):
        messages.extend([HumanMessage(content=f'Question {index}'), AIMessage(content=f'Result {index}')])
    messages[-1] = AIMessage(content='Latest result',
        usage_metadata={'input_tokens': reported - 1, 'output_tokens': 1, 'total_tokens': reported},
        response_metadata={'model_provider': model._get_ls_params()['ls_provider']})
    messages.append(HumanMessage(content='Continue'))
    if window is not None:
        # Small test messages already fit the native fractional retention window;
        # check its trigger decision without allocating a 100k-token fake history.
        assert configured[0]._should_summarize(messages, 0) is compacts
        assert model.profile == profile
        return

    async def scenario():
        sources = []
        async for chunk in graph.astream({'messages': messages}, stream_mode='messages', version='v2'):
            sources.append(chunk['data'][1].get('lc_source'))
        assert ('summarization' in sources) is compacts
        assert model.profile == profile

    asyncio.run(scenario())


@pytest.mark.parametrize('cap', [258_000, 200_000, None])
def test_raw_subagents_use_configured_compaction_in_the_native_stack(tmp_path, monkeypatch, cap):
    import deepagents.middleware.subagents as subagents

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = SimpleNamespace(agent_runtime=AgentRuntimeConfig(deepagent_context_trigger_token_cap=cap))
    model = Model(responses=[AIMessage(content='Context or answer')])
    small_model = Model(profile={'max_input_tokens': 128_000}, responses=[AIMessage(content='Answer')])
    native_children = {}
    original = subagents.create_sub_agent

    def capture(spec, **kwargs):
        graph = original(spec, **kwargs)
        native_children[spec['name']] = (spec, graph)
        return graph

    monkeypatch.setattr(subagents, 'create_sub_agent', capture)
    supplied = [
        {'name': 'general-purpose', 'description': 'Read sources', 'system_prompt': 'Read sources'},
        {'name': 'worker', 'description': 'Read sources', 'system_prompt': 'Read sources',
         'model': model, 'middleware': [], 'tools': []},
        {'name': 'small-worker', 'description': 'Read sources', 'system_prompt': 'Read sources',
         'model': small_model},
    ]
    runner._create_deep_agent(
        model=model, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), subagents=supplied,
    )
    summarizers = []
    for name, (spec, _) in native_children.items():
        stack = [item for item in spec['middleware'] if isinstance(item, SummarizationMiddleware)]
        assert len(stack) == 1
        summary = stack[0]
        assert isinstance(summary, RestoringSummarizationMiddleware)
        summarizers.append(summary)
        threshold = 108_800 if name == 'small-worker' else (cap or 170_000)
        assert not summary._should_summarize([], threshold - 1)
        assert summary._should_summarize([], threshold)
        assert summary.model is (small_model if name == 'small-worker' else model)
    assert len({id(item) for item in summarizers}) == len(supplied)
    assert 'middleware' not in supplied[0]
    assert supplied[1]['middleware'] == []
    assert supplied[1]['tools'] == []
    assert small_model.profile == {'max_input_tokens': 128_000}

    # Run the actual compiled leaf at either side of the configured boundary,
    # using provider usage and fake responses rather than a paid large request.
    async def scenario():
        threshold = cap or 170_000
        for reported in (threshold - 1, threshold):
            messages = []
            for index in range(8):
                messages.extend([HumanMessage(content=f'Question {index}'), AIMessage(content=f'Answer {index}')])
            messages[-1] = AIMessage(content='Latest answer',
                usage_metadata={'input_tokens': reported - 1, 'output_tokens': 1, 'total_tokens': reported},
                response_metadata={'model_provider': model._get_ls_params()['ls_provider']})
            messages.append(HumanMessage(content='Continue'))
            sources = []
            async for chunk in native_children['worker'][1].astream(
                {'messages': messages}, stream_mode='messages', version='v2',
            ):
                sources.append(chunk['data'][1].get('lc_source'))
            assert ('summarization' in sources) is (reported >= threshold)

    asyncio.run(scenario())
