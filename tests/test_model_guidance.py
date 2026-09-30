from __future__ import annotations

import asyncio
from dataclasses import fields

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.specialists.runtime import SpecialistRunner
from catmaster.runtime.prompts.model_harness import (
    model_prompt_bundle, register_model_harness,
)
from catmaster.runtime.prompts.renderer import render_prompt_bundle
from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model


@pytest.fixture(autouse=True)
def isolate_harness_registry(monkeypatch):
    from deepagents.profiles.harness import harness_profiles

    SpecialistRunner._load_create_deep_agent()
    harness_profiles._ensure_harness_profiles_loaded()
    monkeypatch.setattr(harness_profiles, "_HARNESS_PROFILES", {
        key: value for key, value in harness_profiles._HARNESS_PROFILES.items()
        if "mimo" not in key
    })


class CapturingModel(FakeMessagesListChatModel):
    model_name: str
    observed: list[list] = []
    bound_tool_names: list[list[str]] = []
    choices: list = []

    def bind_tools(self, tools, **kwargs):
        self.choices.append(kwargs.get("tool_choice"))
        self.bound_tool_names.append([t.name if hasattr(t, "name") else t["name"] for t in tools])
        return self

    def _generate(self, messages, *args, **kwargs):
        self.observed.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def guidance():
    return render_prompt_bundle("catmaster.model.mimo")


def test_shared_default_reaches_role_prompts_once_without_model_specific_text():
    runner = SpecialistRunner
    prompts = [
        *(runner._base_system_prompt(role) for role in (
            "research", "persistent_research", "experiment", "writing", "peer_review")),
        runner._materials_worker_prompt(), runner._ml_worker_prompt(),
        runner._dynamics_worker_prompt(), runner._orca_xtb_worker_prompt(),
        runner._general_purpose_child_prompt(), runner._litreview_wrapper_prompt(),
        runner._litreview_worker_prompt(), runner._writing_worker_prompt(),
        runner._presentation_worker_prompt(), runner._plot_worker_prompt(),
        runner._peer_review_worker_prompt(), runner._research_challenger_prompt(),
        runner._hypothesis_proposer_prompt(), runner._experiment_pair_comparator_prompt(),
    ]
    shared = render_prompt_bundle("catmaster.runtime.guidance")
    for prompt in prompts:
        assert prompt.count(shared) == 1
        assert guidance() not in prompt


@pytest.mark.parametrize("model_name", ["xiaomi/mimo-v2.6-pro", "gpt-6-astra"])
@pytest.mark.parametrize("specialist_first", [False, True])
@pytest.mark.parametrize("formal", [False, True])
@pytest.mark.parametrize("role", ["reflector", "proposer", "reviewer"])
def test_self_evolution_and_investigator_guidance_is_independent_of_entry_order(
    tmp_path, monkeypatch, model_name, specialist_first, formal, role,
):
    import deepagents.middleware.subagents as native_subagents
    from catmaster.runtime.self_evolution.agents import (
        _agent_response, _build_self_evolution_deep_agent, _load_prompt,
    )
    from catmaster.runtime.self_evolution.models import (
        ReflectionBatch, ProposerResult, ReviewerResult, TextResult,
    )

    schema, args = {
        "reflector": (ReflectionBatch, {"items": [{"kind": "no_change", "rationale": "Already covered."}]}),
        "proposer": (ProposerResult, {"action": "ignore", "rationale": "No candidate needed."}),
        "reviewer": (ReviewerResult, {"recommendation": "reject", "rationale": "No supporting evidence."}),
    }[role]
    prose = "The evidence does not support a change."
    response = (AIMessage(content="", tool_calls=[{"name": schema.__name__, "args": args, "id": "final"}])
                if formal else AIMessage(content=prose))
    model = CapturingModel(model_name=model_name, responses=[response])
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = LLMProfile()
    child_graphs = {}
    original = native_subagents.create_sub_agent

    def capture(spec, **kwargs):
        graph = original(spec, **kwargs)
        child_graphs[spec["name"]] = graph
        return graph

    monkeypatch.setattr(native_subagents, "create_sub_agent", capture)

    def specialist():
        return runner._create_deep_agent(
            model=model, system_prompt=runner._base_system_prompt("research"),
            subagents=[], backend=backend,
        )

    if specialist_first:
        specialist()
    else:
        assert guidance() not in (_harness_profile_for_model(model, None).system_prompt_suffix or "")
    authored = _load_prompt(role)
    agent = _build_self_evolution_deep_agent(
        model=model, backend=backend, tools=[], investigator_tools=[], system_prompt=authored,
        response_schema=schema, name=role, filesystem_tools=("read_file",), allow_mutations=False,
    )
    result = agent.invoke({"messages": [HumanMessage(content="Assess this episode.")]})
    expected = schema.model_validate(args) if formal else TextResult(text=prose)
    assert _agent_response(result, schema) == expected
    assert len(model.observed) == 1
    assert model.choices == ["auto"]
    assert schema.__name__ in model.bound_tool_names[0]
    assert authored in str(model.observed[0][0].content)

    # The investigator inherits the model guidance, but has no decision tool.
    model.responses = [AIMessage(content="Evidence inspected.")]
    model.i = 0
    child = child_graphs["general-purpose"]
    child.invoke({"messages": [HumanMessage(content="Inspect evidence.")]})
    assert schema.__name__ not in model.bound_tool_names[-1]
    asyncio.run(specialist().ainvoke({"messages": [HumanMessage(content="Summarize the current finding.")]}))
    assert len(model.observed) == 3
    shared = render_prompt_bundle("catmaster.runtime.guidance")
    for messages in model.observed:
        system = "\n".join(str(m.content) for m in messages if m.type == "system")
        assert system.count(shared) == 1
        assert system.count(guidance()) == (1 if "mimo" in model_name else 0)


def test_guidance_uses_existing_yaml_model_without_extra_configuration(tmp_path):
    path = tmp_path / "llm.yaml"
    path.write_text("""models:
  astra-label:
    provider: openrouter
    model: xiaomi/mimo-v2.6-pro
  mimo-label:
    provider: codex_oauth
    model: gpt-6-astra
agents:
  proposal: astra-label
  director: astra-label
  task_runner: mimo-label
  memory_patch: mimo-label
  summary: mimo-label
""")
    profile = LLMProfile.from_env_or_file(str(path))
    for role, applies in (("research_lead", True), ("task_runner", False)):
        model = CapturingModel(model_name=profile.config_for_role(role).model,
                               responses=[AIMessage(content="done")])
        register_model_harness(model)
        assert (_harness_profile_for_model(model, None).system_prompt_suffix == guidance()) is applies


@pytest.mark.parametrize("model,applies", [
    ("xiaomi/mimo-v2.6-pro", True), ("mimo-v2.6-pro", True),
    ("xiaomi/mimo-v2-pro", True), ("xiaomi/mimo-v2.6-pro-20260921", True),
    ("gpt-6-astra", False), ("openai/gpt-6-astra", False),
    ("gpt-6-luna", False), ("not-mimo-v2.6-pro", False),
])
def test_guidance_model_selection(model, applies):
    assert bool(model_prompt_bundle(model)) is applies


@pytest.mark.parametrize("automatic_child", [False, True])
@pytest.mark.parametrize("root_label,child_label", [("tuned", "plain"), ("plain", "tuned")])
def test_guidance_reaches_native_root_and_children_using_their_own_model(
    tmp_path, monkeypatch, root_label, child_label, automatic_child,
):
    import catmaster.specialists.runtime as runtime
    import deepagents.middleware.subagents as native_subagents

    monkeypatch.setattr(runtime, "build_chat_model", lambda cfg: CapturingModel(
        model_name=cfg.model, responses=[AIMessage(content="done")]))
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = LLMProfile(
        models={"tuned": LLMConfig(model="xiaomi/mimo-v2.6-pro"), "plain": LLMConfig(model="gpt-6-astra")},
        agents={"research_lead": root_label, "task_runner": child_label},
    )
    root = runner._build_role_chat_model("research_lead")
    child = runner._build_role_chat_model("task_runner")
    child_graphs = {}
    original = native_subagents.create_sub_agent

    def capture(spec, **kwargs):
        graph = original(spec, **kwargs)
        child_graphs[spec["name"]] = graph
        return graph

    monkeypatch.setattr(native_subagents, "create_sub_agent", capture)
    specs = [
        {"name": "general-purpose", "description": "Inspect evidence", "system_prompt": "Inherited child"},
        {"name": "worker", "description": "Inspect evidence", "system_prompt": "Explicit child", "model": child},
    ]
    graph = runner._create_deep_agent(
        model=root, system_prompt="Root instructions", subagents=specs[1:] if automatic_child else specs,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )

    async def scenario():
        for target in (graph, child_graphs["general-purpose"], child_graphs["worker"]):
            result = await target.ainvoke({"messages": [HumanMessage(content="Inspect evidence")]})
            # System instructions do not become accumulated conversation turns.
            assert not any(isinstance(m, SystemMessage) for m in result["messages"])

    asyncio.run(scenario())
    for model, label, expected_calls in ((root, root_label, 2), (child, child_label, 1)):
        assert len(model.observed) == expected_calls
        for messages in model.observed:
            system = "\n".join(str(m.content) for m in messages if m.type == "system")
            assert system.count(guidance()) == (1 if label == "tuned" else 0)
    assert "task" in root.bound_tool_names[0]
    assert "read_file" in root.bound_tool_names[0]
    assert "read_file" in child.bound_tool_names[0]
    assert "Root instructions" in str(root.observed[0][0].content)
    if not automatic_child:
        assert "Inherited child" in str(root.observed[1][0].content)
    assert "Explicit child" in str(child.observed[0][0].content)
    assert specs[0]["system_prompt"] == "Inherited child"
    assert specs[1]["system_prompt"] == "Explicit child"


def test_native_guidance_preserves_blocks_and_does_not_accumulate(tmp_path):
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = LLMProfile()
    source = SystemMessage(content=[{"type": "text", "text": "original", "cache_control": {"type": "ephemeral"}}])
    model = CapturingModel(model_name="mimo-v2.6-pro", responses=[AIMessage(content="done")])
    for _ in range(2):
        graph = runner._create_deep_agent(
            model=model, system_prompt=source, subagents=[],
            backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        )
        asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="Inspect evidence")]}))
    assert len(source.content) == 1
    for messages in model.observed:
        system = next(m for m in messages if m.type == "system")
        assert source.content[0] in system.content
        text = "\n".join(block["text"] for block in system.content if block["type"] == "text")
        assert text.count(guidance()) == 1


@pytest.mark.parametrize("provider,model_name", [
    ("openrouter", "xiaomi/mimo-v2.6-pro"),
    ("oai_compatible", "mimo-v2.6-pro"),
])
def test_real_adapter_identity_selects_native_profile(provider, model_name):
    model = build_chat_model(LLMConfig.from_dict({
        "provider": provider, "model": model_name, "api_key": "offline-test",
        "base_url": "https://example.invalid/v1",
    }))
    before = _harness_profile_for_model(model, None)
    register_model_harness(model)
    after = _harness_profile_for_model(model, None)
    assert after.system_prompt_suffix == guidance()
    assert after.base_system_prompt == before.base_system_prompt
    assert after.excluded_tools == before.excluded_tools
    for field in fields(after):
        if field.name != "system_prompt_suffix":
            assert getattr(after, field.name) == getattr(before, field.name)


def test_model_guidance_is_prompt_text_not_provider_parameter():
    config = LLMConfig.from_dict({
        "provider": "openrouter", "model": "xiaomi/mimo-v2.6-pro", "api_key": "offline-test",
        "reasoning": {"effort": "high"},
        "temperature": None,
    })
    runner = SpecialistRunner.__new__(SpecialistRunner)
    runner.llm_profile = LLMProfile(models={"tuned": config})
    model = runner._attach_model_label_metadata(build_chat_model(config), model_label="tuned")
    register_model_harness(model)
    from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt
    prompt = _apply_profile_prompt(_harness_profile_for_model(model, None), "Base instructions")
    history = AIMessage(content="", additional_kwargs={"reasoning_content": "preserve historical reasoning"},
                        tool_calls=[{"name": "inspect", "args": {}, "id": "call-1"}])
    messages, params = model._create_message_dicts([SystemMessage(content=prompt), history], None)
    assert guidance() in str(messages[0]["content"])
    assert messages[1]["reasoning"] == "preserve historical reasoning"
    assert "system_prompt_suffix" not in params
    assert params["reasoning"] == {"effort": "high"}
    assert "temperature" not in params
