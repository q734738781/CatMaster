from __future__ import annotations

import asyncio
import json
import re
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import openai
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import catmaster.specialists.runtime as runtime_mod
from catmaster.specialists.runtime import (
    RUN_STATE_FILE,
    _BOUND_RESEARCH_EXECUTION_TOOL_ALLOWLIST,
    _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES,
    _DYNAMICS_WORKER_TOOL_ALLOWLIST,
    _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST,
    _LITREVIEW_LOCAL_TOOL_ALLOWLIST,
    _MATERIALS_WORKER_TOOL_ALLOWLIST,
    _ML_WORKER_TOOL_ALLOWLIST,
    _ORCA_XTB_WORKER_TOOL_ALLOWLIST,
    _RESEARCH_TOOL_ALLOWLIST,
    _WRITING_WORKER_TOOL_ALLOWLIST,
    _WRITING_TOOL_ALLOWLIST,
    build_specialist_runner,
)
from catmaster.runtime.usage_stats import load_usage_summary
from catmaster.runtime.artifact_callback import UIEventHandler
from catmaster.runtime.run_control import RunControl
from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.tools.registry import get_tool_registry


class _FakeProfile:
    def config_for_role(self, role: str) -> SimpleNamespace:
        return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)


class _FakeProfileWithRuntime(_FakeProfile):
    agent_runtime = SimpleNamespace(
        deepagent_context_trigger_token_cap=256_000,
    )


class _FakeToolStrategy:
    def __init__(self, schema, handle_errors: bool = False) -> None:
        self.schema = schema
        self.handle_errors = handle_errors


class _FakeSubAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCompiledSubAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeSummarizationMiddleware:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCompactConversationMiddleware:
    def __init__(self, summarizer) -> None:
        self.summarizer = summarizer


class _FakeMemoryMiddleware:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeDeepAgent:
    def __init__(self, *, kwargs: dict) -> None:
        self.kwargs = kwargs

    async def ainvoke(self, payload, config=None):
        self.kwargs["_last_payload"] = payload
        self.kwargs["_last_config"] = config
        assert payload["messages"][0]["role"] == "user"
        name = self.kwargs["name"]
        if name == "research_specialist":
            content = "## Summary\nresearch summary\n\n## Facts\n- grounded by literature agent when needed\n\n## Files\n- reports/research.md"
        elif name == "writing_specialist":
            content = "## Summary\nwriting summary\n\n## Facts\n- manuscript draft updated\n\n## Files\n- drafts/report.md"
        elif name == "litreview_agent":
            content = "## Summary\nliterature review summary\n\n## Facts\n- source-grounded synthesis completed\n\n## Files\n- notes/literature/brief.md"
        else:
            content = "## Summary\nexperiment summary\n\n## Facts\n- bounded execution completed\n\n## Files\n- experiments/out.json"
        return {"messages": [AIMessage(content=content)]}

class _FakeUsageCallback:
    def __init__(self) -> None:
        self.usage_metadata = {
            "task_runner-model": {
                "input_tokens": 123,
                "output_tokens": 17,
                "total_tokens": 140,
                "input_token_details": {"cache_read": 80},
                "output_token_details": {"reasoning": 5},
            }
        }
        self.call_counts_by_model = {"task_runner-model": 2}
        self.usage_metadata_by_role = {
            "experiment_specialist": {
                "task_runner-model": {
                    "input_tokens": 40,
                    "output_tokens": 7,
                    "total_tokens": 47,
                    "input_token_details": {"cache_read": 10},
                }
            }
        }
        self.call_counts_by_role = {"experiment_specialist": 1}


class _FailingToolInput(BaseModel):
    value: str


def _assert_native_skill_groups(agent_kwargs: dict, *groups: str) -> None:
    paths = list(agent_kwargs.get("skills") or [])
    assert len(paths) == len(groups)
    matches = [
        re.fullmatch(
            r"/\.deepagents/snapshots/([0-9a-f]{24})/skills/([^/]+)",
            path,
        )
        for path in paths
    ]
    assert all(match is not None for match in matches)
    assert [match.group(2) for match in matches if match is not None] == list(groups)
    assert len({match.group(1) for match in matches if match is not None}) == 1


def _assert_native_memory(agent_kwargs: dict) -> None:
    memory = list(agent_kwargs["memory"])
    assert len(memory) == 2
    match = re.fullmatch(
        r"/\.deepagents/snapshots/([0-9a-f]{24})/AGENTS\.md",
        memory[0],
    )
    assert match is not None
    assert memory[1] == "/memories/AGENTS.md"
    skill_hashes = {
        skill_match.group(1)
        for path in list(agent_kwargs.get("skills") or [])
        if (
            skill_match := re.fullmatch(
                r"/\.deepagents/snapshots/([0-9a-f]{24})/skills/[^/]+",
                path,
            )
        )
    }
    assert not skill_hashes or skill_hashes == {match.group(1)}


def test_deepagent_context_profile_cap_limits_fraction_summarization_window(tmp_path: Path) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfileWithRuntime(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
    )
    model = SimpleNamespace(
        model="gpt-5.5",
        profile={
            "max_input_tokens": 1_050_000,
            "max_output_tokens": 128_000,
            "tool_calling": True,
        },
    )

    capped = built.runner._apply_deepagent_context_profile_cap(model, role="research_lead")

    assert capped.profile["max_input_tokens"] == 301_176
    assert int(capped.profile["max_input_tokens"] * 0.85) <= 256_000
    assert int(capped.profile["max_input_tokens"] * 0.10) == 30_117
    assert capped.profile["max_output_tokens"] == 128_000
    assert capped.profile["tool_calling"] is True


def test_deepagent_context_profile_cap_can_be_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    class _DisabledProfile(_FakeProfile):
        agent_runtime = SimpleNamespace(
            deepagent_context_trigger_token_cap=None,
        )

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_DisabledProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
    )
    model = SimpleNamespace(model="gpt-5.5", profile={"max_input_tokens": 1_050_000})

    capped = built.runner._apply_deepagent_context_profile_cap(model, role="research_lead")

    assert capped.profile["max_input_tokens"] == 1_050_000


def test_real_registry_covers_specialist_allowlists() -> None:
    registry = get_tool_registry()
    registered = set(registry.tools)

    assert "write_note" not in registered
    assert _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST <= registered
    assert {"mp_search_materials", "mp_download_structure"} <= _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert _RESEARCH_TOOL_ALLOWLIST <= registered
    assert _WRITING_TOOL_ALLOWLIST <= registered
    assert _MATERIALS_WORKER_TOOL_ALLOWLIST <= registered
    assert _DYNAMICS_WORKER_TOOL_ALLOWLIST <= registered
    assert _LITREVIEW_LOCAL_TOOL_ALLOWLIST <= registered
    assert _WRITING_WORKER_TOOL_ALLOWLIST <= registered
    bound_writeback = {
        "record_bound_research_result",
        "mark_bound_research_experiment_failed",
    }
    assert bound_writeback <= _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert bound_writeback <= registered
    assert bound_writeback.isdisjoint(_LITREVIEW_LOCAL_TOOL_ALLOWLIST)
    assert {
        "record_research_result",
        "mark_research_experiment_failed",
        "add_research_hypothesis",
        "add_research_experiment",
    }.isdisjoint(_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST)
    assert {
        "record_research_result",
        "mark_research_experiment_failed",
        "add_research_hypothesis",
        "add_research_experiment",
    }.isdisjoint(_LITREVIEW_LOCAL_TOOL_ALLOWLIST)
    assert "bash" not in _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert "bash" not in _RESEARCH_TOOL_ALLOWLIST
    assert "bash" not in _WRITING_TOOL_ALLOWLIST
    assert "run_literature_research" not in registered


@pytest.mark.parametrize(
    ("provider", "expects_native"),
    [
        ("codex_oauth", True),
        ("openai", True),
        ("langchain", False),
    ],
)
def test_search_surface_follows_each_role_provider(
    tmp_path: Path,
    provider: str,
    expects_native: bool,
) -> None:
    class _ProviderProfile(_FakeProfile):
        def config_for_role(self, role: str) -> SimpleNamespace:
            return SimpleNamespace(model=f"{role}-model", provider=provider, base_url=None)

    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_ProviderProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )

    tools = built.runner._search_tools_for_role("task_runner", audience="materials_worker")
    specialist_tools = built.runner._specialist_tools("experiment")
    search_tools = [
        tool
        for tool in specialist_tools
        if (tool.get("type") if isinstance(tool, dict) else getattr(tool, "name", "")) == "web_search"
    ]

    assert len(tools) == 1
    assert len(search_tools) == 1
    if expects_native:
        assert tools == [{"type": "web_search"}]
        assert search_tools == [{"type": "web_search"}]
    else:
        assert isinstance(tools[0], StructuredTool)
        assert tools[0].name == "web_search"
        assert isinstance(search_tools[0], StructuredTool)


def test_litreview_graph_result_tools_require_a_bound_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    service = ResearchGraphService(workspace=workspace, workspace_id="proj")
    created = service.create_graph(
        GraphCreateRequest(question="Which mechanism controls selectivity?")
    )
    graph_id = created["graph"]["graph_id"]
    experiment = service.add_experiment(
        graph_id,
        ExperimentCreateRequest(
            expected_revision=created["graph"]["revision"],
            objective="Compare the candidate mechanisms.",
            execution_lane="literature_review",
        ),
    )
    experiment_id = experiment["node"]["node_id"]
    direct = service.thread_store.create_thread(
        title="Direct review",
        entrypoint="literature_review",
    )
    bound = service.thread_store.create_thread(
        title="Bound review",
        entrypoint="literature_review",
    )
    service.thread_store.update_thread(
        bound.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id=experiment_id,
    )

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="literature_review",
    )

    direct_names = built.runner._litreview_local_tool_names(direct.thread_id)
    bound_names = built.runner._litreview_local_tool_names(bound.thread_id)
    assert direct_names == _LITREVIEW_LOCAL_TOOL_ALLOWLIST
    assert direct_names.isdisjoint(_BOUND_RESEARCH_EXECUTION_TOOL_ALLOWLIST)
    assert bound_names == (
        _LITREVIEW_LOCAL_TOOL_ALLOWLIST
        | _BOUND_RESEARCH_EXECUTION_TOOL_ALLOWLIST
    )

    monkeypatch.setattr(
        runtime_mod,
        "build_chat_model",
        lambda cfg: {"model": cfg.model},
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_create_deep_agent",
        staticmethod(lambda: lambda **kwargs: _FakeDeepAgent(kwargs=kwargs)),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_subagent",
        staticmethod(lambda: _FakeSubAgent),
    )
    fake_runtime = {
        "checkpointer": object(),
        "store": object(),
        "backend": object(),
    }
    direct_agent = built.runner._build_litreview_agent(
        runtime=fake_runtime,
        thread_id=direct.thread_id,
    )
    bound_agent = built.runner._build_litreview_agent(
        runtime=fake_runtime,
        thread_id=bound.thread_id,
    )
    assert [item.kwargs["name"] for item in direct_agent.kwargs["subagents"]] == [
        "general-purpose"
    ]
    assert [item.kwargs["name"] for item in bound_agent.kwargs["subagents"]] == [
        "general-purpose",
        "evidence_judge",
    ]


def test_specialist_callbacks_include_ui_event_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=SimpleNamespace(emit=lambda event: None),
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )

    callbacks = built.runner._langchain_callbacks(usage_handler=None, default_agent_name="experiment_specialist")
    ui_callbacks = [callback for callback in callbacks if isinstance(callback, UIEventHandler)]
    assert ui_callbacks
    assert ui_callbacks[0].default_agent_name == "experiment_specialist"


def test_specialist_runner_propagates_optional_interrupt_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _FakeDeepAgent(kwargs=kwargs)

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_entry_subagents",
        lambda self, entrypoint, runtime, thread_id="": [],
    )

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
        interrupt_on={"write_file": True, "remote_submission": True},
    )
    agent = asyncio.run(
        built.runner._build_entry_agent(
            entrypoint="experiment",
            runtime={"checkpointer": object(), "store": object(), "backend": object()},
            thread_id="thread-1",
        )
    )

    assert isinstance(agent, _FakeDeepAgent)
    assert captured["agent_kwargs"]["interrupt_on"] == {"write_file": True, "remote_submission": True}
    assert "permissions" not in captured["agent_kwargs"]


def test_specialist_tool_wrapper_returns_nonfatal_error_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="writing",
    )

    def _boom(runtime=None, **kwargs):
        _ = (runtime, kwargs)
        raise RuntimeError("simulated failure")

    failing_tool = StructuredTool.from_function(
        func=_boom,
        name="polish_academic_prose",
        description="fail on purpose",
        args_schema=_FailingToolInput,
        infer_schema=False,
        response_format="content_and_artifact",
    )

    monkeypatch.setattr(
        built.runner.registry,
        "as_langchain_tools",
        lambda allowlist, run_dir, workspace: [failing_tool],
    )
    monkeypatch.setitem(built.runner.registry.tools, "polish_academic_prose", {"function": object()})

    wrapped = built.runner._named_tools({"polish_academic_prose"})
    content, artifact = wrapped[0].func(value="x")
    assert "simulated failure" in content
    assert artifact["tool_name"] == "polish_academic_prose"
    assert artifact["data"]["status"] == "error"
    assert artifact["data"]["tool_name"] == "polish_academic_prose"


def test_specialist_reporting_contract_requires_direct_answer_and_relative_paths() -> None:
    contract = runtime_mod.SpecialistRunner._soft_reporting_contract()
    assert "directly answer the user's actual question" in contract
    assert "answer naturally first" in contract
    assert "do not force fixed `Summary` / `Facts` / `Files` headings" in contract
    assert "optional convention" in contract
    assert "shape the user requested" in contract
    assert "workspace-relative output paths" in contract
    assert "optional `ReviewTarget` section" in contract
    assert "replace or delete stale incorrect reports/notes" in contract


def test_research_reporting_contract_requires_scientific_reasonableness_section() -> None:
    contract = runtime_mod.SpecialistRunner._research_reporting_contract()
    assert "answer naturally first" in contract
    assert "do not force fixed `Summary` / `Facts` / `Files` headings" in contract
    assert "A scientific reasonableness check is required for research closeouts" in contract
    assert "scientifically plausible" in contract
    assert "evidence supports the claim" in contract
    assert "method/QC/literature-context checks" in contract
    assert "state the limitation and minimal recommended next action" in contract
    assert "only when it is required to finish the user's requested stage" in contract


def test_writing_reporting_contract_allows_summary_first_closeout() -> None:
    contract = runtime_mod.SpecialistRunner._writing_reporting_contract()
    assert "shape the user requested" in contract
    assert "not required" in contract
    assert "Include a `Files` section only when" in contract
    assert "optional `ReviewTarget` section" in contract
    assert "Do not add a placeholder `Facts` section" in contract


def test_prose_quality_policy_requires_skill_without_changing_science() -> None:
    policy = runtime_mod.SpecialistRunner._prose_quality_policy()
    assert "read and apply the `humanizer` skill" in policy
    assert "report, literature synthesis, review" in policy
    assert "preserve claim strength" in policy
    assert "numbers, units, equations, citations, uncertainty" in policy
    assert "raw logs" in policy
    assert "machine-readable files" in policy


def test_tool_policy_rejects_hash_and_ad_hoc_contract_ceremony() -> None:
    policy = runtime_mod.SpecialistRunner._tool_policy()
    checksum_rule = "By default, do not calculate or compare hashes/checksums unless the user explicitly requests it."
    contract_rule = "Do not create, freeze, or persist an ad hoc contract, schema, manifest, baseline, lockfile, acceptance checklist, or similar governance artifact merely to formalize a one-off task."
    assert policy.startswith(checksum_rule)
    assert policy.count("hash") == 1
    assert contract_rule in policy
    assert "existing API, tool, reproducibility requirement, or downstream machine consumer actually requires it" in policy
    assert "use targeted inspection or a version-control diff when the question is edit scope" in policy

    shared_prompts = (
        *(runtime_mod.SpecialistRunner._base_system_prompt(entrypoint) for entrypoint in ("research", "experiment", "writing", "peer_review")),
        runtime_mod.SpecialistRunner._litreview_wrapper_prompt(),
        runtime_mod.SpecialistRunner._materials_worker_prompt(),
        runtime_mod.SpecialistRunner._dynamics_worker_prompt(),
        runtime_mod.SpecialistRunner._ml_worker_prompt(),
        runtime_mod.SpecialistRunner._orca_xtb_worker_prompt(),
        runtime_mod.SpecialistRunner._writing_worker_prompt(),
        runtime_mod.SpecialistRunner._writing_polisher_prompt(),
        runtime_mod.SpecialistRunner._peer_review_worker_prompt(),
    )
    for prompt in shared_prompts:
        assert checksum_rule in prompt
        assert contract_rule in prompt


def test_general_purpose_policy_is_context_only_without_lane_or_concurrency_rules() -> None:
    policies = (
        runtime_mod.SpecialistRunner._general_purpose_specialist_policy(),
        runtime_mod.SpecialistRunner._general_purpose_worker_policy(),
    )

    for policy in policies:
        assert "one self-contained, context-heavy branch described by a complete task brief" in policy
        assert "current lane" not in policy
        assert "parallel" not in policy
        assert "sequential" not in policy
        assert "at most one" not in policy


def test_litreview_source_acquisition_is_fused_into_academic_search() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "skills/litreview_agent/nature-academic-search/SKILL.md").read_text(encoding="utf-8")
    assert "acquire_literature_source" in text
    assert "legal non-browser OA routes" in text
    assert "structural, page-count, and identity checks" in text
    assert "Use the returned local path" in text
    assert "Do not reopen the same remote page repeatedly" in text
    assert not (repo_root / "skills/litreview_agent/nature-downloader/SKILL.md").exists()
    assert not (repo_root / "skills/research_specialist/nature-downloader").exists()


def test_litreview_academic_search_scales_breadth_without_overriding_user_scope() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "skills/litreview_agent/nature-academic-search/SKILL.md").read_text(encoding="utf-8")
    assert "until new searches are mostly duplicative" in text
    assert "An explicit brief or focused scope remains controlling" in text
    assert "Keep candidate records shallow" in text
    assert "Do not demand full methods-level extraction for every candidate" in text
    assert "Full-text need is claim-dependent" in text
    assert "An abstract or substantive search summary" in text
    assert "Do not use a fixed paper count or full-text count" in text
    assert "50-60+ candidates" not in text
    assert "at most one" not in text
    assert not (repo_root / "skills/research_specialist/nature-academic-search").exists()


def test_litreview_describes_evidence_by_attributes_not_source_tiers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skill_root = repo_root / "skills/litreview_agent/nature-academic-search"
    skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    attributes = (skill_root / "references/evidence-attributes.md").read_text(
        encoding="utf-8"
    )

    assert "Describe evidence by attributes" in skill_text
    assert "access-depth attributes, not reliability grades" in skill_text
    assert "These are descriptive attributes, not ordered labels" in attributes
    assert not (skill_root / "references/source-tiers.md").exists()
    assert "T1" not in attributes
    assert "T2" not in attributes
    assert "T3" not in attributes


def test_litreview_selection_uses_states_and_reasons_without_scores() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_root = repo_root / "skills/litreview_agent/nature-literature-pipeline"
    selection = (pipeline_root / "references/selection-policy.md").read_text(
        encoding="utf-8"
    )
    template = (pipeline_root / "templates/literature-push-template.md").read_text(
        encoding="utf-8"
    )
    attributes = (
        repo_root
        / "skills/litreview_agent/nature-academic-search/references/evidence-attributes.md"
    ).read_text(encoding="utf-8")

    assert "`selected`" in selection
    assert "`deferred`" in selection
    assert "`excluded`" in selection
    assert "one concise reason" in selection
    assert "component values" in selection
    assert "selected, deferred, or excluded" in template
    assert "status has a reason" in template
    assert not (pipeline_root / "references/scoring-system.md").exists()
    assert "LATS" not in selection
    assert "score" not in template.casefold()
    assert "LATS" not in attributes
    assert "descriptive attributes, not ordered labels" in attributes
    assert "not a reliability grade" in attributes


def test_active_literature_skills_do_not_restore_retired_grade_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    roots = [
        repo_root / "skills/litreview_agent/nature-literature-pipeline",
        repo_root / "skills/research_specialist/nature-literature-pipeline",
        repo_root / "skills/litreview_agent/nature-citation",
        repo_root / "skills/research_specialist/nature-citation",
        repo_root / "skills/writing_specialist/nature-citation",
    ]
    retired_terms = (
        "support_grade",
        "support-grade",
        "support grades",
        "classify hits into three tiers",
        "按相关性分级",
    )

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".py", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8").casefold()
            for retired in retired_terms:
                assert retired.casefold() not in text, (path, retired)


def test_writing_task_scale_numbers_live_in_skills_not_system_prompts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scientific_writing = (repo_root / "skills/writing_specialist/scientific-writing/SKILL.md").read_text(
        encoding="utf-8"
    )
    achemso = (repo_root / "skills/writing_specialist/achemso-latex-manuscript/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "2-4 core claims can be a useful planning reference" in scientific_writing
    assert "do not run a preset number of review or polishing rounds" in scientific_writing
    assert "do not target a preset number of passes" in achemso

    prompts = (
        runtime_mod.SpecialistRunner._base_system_prompt("research"),
        runtime_mod.SpecialistRunner._base_system_prompt("writing"),
        runtime_mod.SpecialistRunner._writing_worker_prompt(),
    )
    for prompt in prompts:
        assert "2-4 bullets" not in prompt
        assert "at least one direct compile pass" not in prompt
        assert "manuscript-review capability once" not in prompt
        assert "one more bounded polishing/revision pass" not in prompt


def test_report_parser_supports_review_target() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )
    summary, facts, files, review_target = runner._parse_summary_and_files(
        "## Summary\nok\n\n## Facts\n- a\n\n## Files\n- `manuscript/paper.pdf`\n\n## ReviewTarget\n- `manuscript/paper.pdf`"
    )
    assert summary == "ok"
    assert facts == ["a"]
    assert files == ["manuscript/paper.pdf"]
    assert review_target == "manuscript/paper.pdf"


def test_materials_worker_prompt_includes_workspace_path_discipline() -> None:
    prompt = runtime_mod.SpecialistRunner._materials_worker_prompt()
    assert "Workspace path discipline" in prompt
    assert "Treat `/` only as the workspace virtual root" in prompt
    assert "Do not pass guessed input paths into tools" in prompt
    assert "never use leading-slash workspace paths like `/writing/...`" in prompt
    assert "Only persist key constraints, decisive results" in prompt
    assert "literature/" in prompt
    assert "structures/" in prompt
    assert "calculations/" in prompt
    assert "notes/" in prompt
    assert "writing/" in prompt
    assert "Workspace script header policy" in prompt
    assert "`Code writing date: YYYY-MM-DD`" in prompt
    assert "`Responsible/related agent: <agent name>`" in prompt
    assert "`Implementation principle: <how it works>`" in prompt
    assert "`Purpose: <what it is for>`" in prompt


def test_orca_xtb_worker_prompt_includes_workspace_path_discipline() -> None:
    prompt = runtime_mod.SpecialistRunner._orca_xtb_worker_prompt()
    assert "Workspace path discipline" in prompt
    assert "Treat `/` only as the workspace virtual root" in prompt
    assert "molecular quantum-chemistry subtask" in prompt
    assert "first create the structure under `<topic>/structures/`" in prompt
    assert "Do not guess that a path like `<topic>/structures/<name>.xyz` already exists" in prompt
    assert "do not choose ORCA-XTB as the default fallback for routine preopt steps" in prompt


def test_common_worker_prompts_require_relevant_skill_check() -> None:
    expected = "Tool discipline: if a relevant skill is available to the current agent, read it before acting."

    assert expected in runtime_mod.SpecialistRunner._materials_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._ml_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._orca_xtb_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._writing_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._writing_polisher_prompt()
    assert expected in runtime_mod.SpecialistRunner._peer_review_worker_prompt()


def test_delegating_worker_prompts_do_not_gain_blanket_gp_serialization() -> None:
    prompts = (
        runtime_mod.SpecialistRunner._materials_worker_prompt(),
        runtime_mod.SpecialistRunner._ml_worker_prompt(),
        runtime_mod.SpecialistRunner._dynamics_worker_prompt(),
        runtime_mod.SpecialistRunner._orca_xtb_worker_prompt(),
        runtime_mod.SpecialistRunner._writing_worker_prompt(),
        runtime_mod.SpecialistRunner._writing_polisher_prompt(),
        runtime_mod.SpecialistRunner._peer_review_worker_prompt(),
    )

    for prompt in prompts:
        assert "current shared workspace makes parallel subagents unsafe" not in prompt


def test_experiment_specialist_can_use_materials_project_tools_directly() -> None:
    experiment_prompt = runtime_mod.SpecialistRunner._base_system_prompt("experiment")
    materials_prompt = runtime_mod.SpecialistRunner._materials_worker_prompt()

    assert {"mp_search_materials", "mp_download_structure"} <= runtime_mod._EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert "direct Materials Project lookup/download tools" in experiment_prompt
    assert "if you cannot see MP tools" not in experiment_prompt
    assert "For Materials Project search or structure download steps" in materials_prompt
    assert "report precise API-key" in materials_prompt


def test_specialist_prompts_default_to_on_demand_delegation() -> None:
    research_prompt = runtime_mod.SpecialistRunner._base_system_prompt("research", thread_id="thread-1")
    experiment_prompt = runtime_mod.SpecialistRunner._base_system_prompt("experiment")
    writing_prompt = runtime_mod.SpecialistRunner._base_system_prompt("writing")
    peer_review_prompt = runtime_mod.SpecialistRunner._base_system_prompt("peer_review")
    litreview_prompt = runtime_mod.SpecialistRunner._litreview_wrapper_prompt()

    assert "requested deliverable or explicitly approved stage as the stop condition" in research_prompt
    assert "Default to on-demand closeout, not autonomous research expansion" in research_prompt
    assert "condition mismatch, incomplete provenance, unresolved alternatives" in research_prompt
    assert "weak evidence" not in research_prompt
    assert "issue a bounded probe to `experiment_specialist` rather than deciding from absence in the research thread" in research_prompt
    assert "Research Graph contract" in research_prompt
    assert "Never guess among multiple graphs" in research_prompt
    assert "scientific reasonableness check" in research_prompt.lower()
    assert "A scientific reasonableness check is required for research closeouts" in research_prompt
    assert "do not force fixed `Summary` / `Facts` / `Files` headings" in research_prompt
    for prompt in (research_prompt, experiment_prompt, writing_prompt):
        assert "current shared workspace makes parallel subagents unsafe" not in prompt
    assert "Run delegated review episodes sequentially" in peer_review_prompt
    assert "general-purpose" not in litreview_prompt
    assert "acquire_literature_source" not in litreview_prompt
    assert "finalize_citations" not in litreview_prompt
    assert "treat its execution and domain QC as authoritative" in experiment_prompt
    assert "delegate a bounded probe to the matching worker instead of concluding the capability is absent" in experiment_prompt
    assert "Experiment closeout discipline: use worker/tool returns as the QC source of record" in experiment_prompt
    assert "Do not rerun or reparse calculation outputs just to repeat domain QC" in experiment_prompt
    assert "When one writing-worker pass returns, actively decide whether another bounded delegate pass is needed" in writing_prompt
    assert "When one worker review episode returns, actively decide whether another bounded delegate pass is needed" in peer_review_prompt


def test_specialist_prompts_integrate_property_lookup_and_delegated_compute_rules() -> None:
    research_prompt = runtime_mod.SpecialistRunner._base_system_prompt("research", thread_id="thread-1")
    experiment_prompt = runtime_mod.SpecialistRunner._base_system_prompt("experiment")

    for prompt in (research_prompt, experiment_prompt):
        assert "Physical/chemical property lookup policy" in prompt
        assert "treat it first as a literature-grounded or existing-evidence lookup" in prompt
        assert "do not launch new DFT" in prompt
        assert "explicitly request a calculation" in prompt
        assert "Delegated computation role policy" in prompt
        assert "delegate a bounded calculation/probe" in prompt
        assert "concrete missing input, task registration, resource configuration, stage layout, or user approval" in prompt


def test_execution_capability_contract_is_worker_scoped_and_tool_surface_bound(tmp_path: Path) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
    )

    assert set(_MATERIALS_WORKER_TOOL_ALLOWLIST).issubset(set(built.runner.registry.tools))
    assert set(_DYNAMICS_WORKER_TOOL_ALLOWLIST).issubset(set(built.runner.registry.tools))
    assert set(_ML_WORKER_TOOL_ALLOWLIST).issubset(set(built.runner.registry.tools))
    assert set(_ORCA_XTB_WORKER_TOOL_ALLOWLIST).issubset(set(built.runner.registry.tools))
    assert "remote_submission" in _MATERIALS_WORKER_TOOL_ALLOWLIST
    assert "remote_submission" in _DYNAMICS_WORKER_TOOL_ALLOWLIST
    assert "remote_submission" in _ML_WORKER_TOOL_ALLOWLIST
    assert "remote_submission" in _ORCA_XTB_WORKER_TOOL_ALLOWLIST
    assert "xtb_prepare" in _ORCA_XTB_WORKER_TOOL_ALLOWLIST
    assert {name for name in _MATERIALS_WORKER_TOOL_ALLOWLIST if name.startswith("cp2k_")} == {"cp2k_prepare"}
    assert {"cp2k_aimd_prepare", "cp2k_output_summary", "lammps_prepare", "lammps_log_summary"} <= _DYNAMICS_WORKER_TOOL_ALLOWLIST
    assert "mace_neb_batch" not in _MATERIALS_WORKER_TOOL_ALLOWLIST
    assert "mace_train" not in _ML_WORKER_TOOL_ALLOWLIST
    assert "orca_execute_batch" not in _ORCA_XTB_WORKER_TOOL_ALLOWLIST
    assert "mace_neb_batch" not in _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert "mace_train" not in _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert "orca_execute_batch" not in _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST


def test_writing_worker_and_proposal_prompts_include_workspace_layout_guidance() -> None:
    writing_prompt = runtime_mod.SpecialistRunner._writing_worker_prompt()
    proposal_prompt = runtime_mod.SpecialistRunner._proposal_system_prompt("experiment")
    assert "Persistent project memory" in writing_prompt
    assert "Only persist key constraints, decisive results" in writing_prompt
    assert "structures/" in writing_prompt
    assert "calculations/" in writing_prompt
    assert "notes/" in writing_prompt
    assert "writing/" in writing_prompt
    assert "Workspace path discipline" in proposal_prompt
    assert "literature/" in proposal_prompt
    assert "writing/" in proposal_prompt


def test_default_tool_error_middleware_returns_tool_message() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    handler_mw = middleware[-1]

    class _Request:
        tool_call = {
            "id": "call-1",
            "name": "create_molecule_from_smiles",
            "args": {"smiles": "C#O"},
        }

    async def _handler(_request):
        raise runtime_mod.CatMasterToolExecutionError(
            tool_name="create_molecule_from_smiles",
            public_message="Failed to build molecule from SMILES: Invalid SMILES: C#O",
            artifact={"tool_name": "create_molecule_from_smiles", "data": {"smiles": "C#O"}},
            error_code="molecule_build_failed",
        )

    async def _run():
        return await handler_mw.awrap_tool_call(_Request(), _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "Invalid SMILES" in str(result.content)
    assert result.tool_call_id == "call-1"


def test_tool_result_middleware_preserves_multimodal_tool_messages() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    assert [type(item).__name__ for item in middleware] == ["catmaster_nonfatal_tool_errors"]
    tool_mw = middleware[-1]

    class _Request:
        tool_call = {
            "id": "call-1",
            "name": "read_file",
            "args": {"file_path": "/paper/page.png"},
        }

    async def _handler(_request):
        return ToolMessage(
            content_blocks=[
                {
                    "type": "image",
                    "id": "img-1",
                    "base64": "not-for-history",
                    "mime_type": "image/png",
                }
            ],
            additional_kwargs={
                "read_file_path": "/paper/page.png",
                "read_file_media_type": "image/png",
            },
            tool_call_id="call-1",
            name="read_file",
            status="success",
        )

    async def _run():
        return await tool_mw.awrap_tool_call(_Request(), _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "image"
    assert result.content[0]["base64"] == "not-for-history"
    assert result.additional_kwargs["read_file_path"] == "/paper/page.png"
    assert result.tool_call_id == "call-1"


def test_codex_stream_overload_retry_is_narrow_and_centrally_configured() -> None:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    overload = openai.APIError(
        "Our servers are currently overloaded. Please try again later.",
        request=request,
        body={
            "type": "service_unavailable_error",
            "code": "server_is_overloaded",
        },
    )
    unrelated = openai.APIError(
        "A different stream failure",
        request=request,
        body={"type": "server_error", "code": "different_error"},
    )
    overload_without_code = openai.APIError(
        "Our servers are currently overloaded. Please try again later.",
        request=request,
        body={"type": "server_error"},
    )
    retryable_request_error = openai.APIError(
        "An error occurred while processing your request. You can retry your "
        "request, or contact support. Please include the request ID request-123.",
        request=request,
        body={"type": "server_error"},
    )

    assert runtime_mod._is_codex_stream_overload_error(overload)
    assert runtime_mod._is_codex_stream_overload_error(overload_without_code)
    assert runtime_mod._is_codex_stream_overload_error(retryable_request_error)
    assert not runtime_mod._is_codex_stream_overload_error(unrelated)
    assert not runtime_mod._is_codex_stream_overload_error(RuntimeError("overloaded"))

    retry = runtime_mod._build_codex_overload_retry_middleware()[0]
    assert type(retry).__name__ == "ModelRetryMiddleware"
    assert retry.max_retries == 6
    assert retry.initial_delay == 30.0
    assert retry.backoff_factor == 2.0
    assert retry.max_delay == 600.0
    assert retry.jitter is False
    assert retry.on_failure == "error"


def test_codex_incomplete_stream_retry_is_narrow_and_centrally_configured() -> None:
    dropped_body = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body "
        "(incomplete chunked read)"
    )
    wrapped = openai.APIConnectionError(
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    )
    wrapped.__cause__ = dropped_body

    assert runtime_mod._is_codex_incomplete_stream_error(dropped_body)
    assert runtime_mod._is_codex_incomplete_stream_error(wrapped)
    assert not runtime_mod._is_codex_incomplete_stream_error(
        httpx.RemoteProtocolError("Server disconnected without sending a response")
    )
    assert not runtime_mod._is_codex_incomplete_stream_error(
        openai.APIError(
            "Our servers are currently overloaded. Please try again later.",
            request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
            body={"code": "server_is_overloaded"},
        )
    )

    retry = runtime_mod._build_codex_incomplete_stream_retry_middleware()[0]
    assert type(retry).__name__ == "_CodexIncompleteStreamRetryMiddleware"
    assert retry.max_retries == 2
    assert retry.initial_delay == 2.0
    assert retry.backoff_factor == 2.0
    assert retry.max_delay == 10.0
    assert retry.jitter is False
    assert retry.on_failure == "error"


def test_codex_incomplete_stream_retry_replays_only_the_model_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retry = runtime_mod._build_codex_incomplete_stream_retry_middleware()[0]
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def handler(_request: Any) -> AIMessage:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body "
                "(incomplete chunked read)"
            )
        return AIMessage(content="recovered")

    monkeypatch.setattr(
        "langchain.agents.middleware.model_retry.asyncio.sleep",
        fake_sleep,
    )
    result = asyncio.run(retry.awrap_model_call(SimpleNamespace(), handler))

    assert isinstance(result, AIMessage)
    assert result.content == "recovered"
    assert attempts == 3
    assert delays == [2.0, 4.0]


def test_deepagent_loader_registers_codex_retry_as_provider_middleware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepagents

    registrations: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        deepagents,
        "register_harness_profile",
        lambda key, profile: registrations.append((key, profile)),
    )
    runtime_mod.SpecialistRunner._load_create_deep_agent.cache_clear()
    try:
        assert runtime_mod.SpecialistRunner._load_create_deep_agent() is deepagents.create_deep_agent
    finally:
        runtime_mod.SpecialistRunner._load_create_deep_agent.cache_clear()

    assert len(registrations) == 1
    key, profile = registrations[0]
    assert key == "openai-codex"
    retries = profile.materialize_extra_middleware()
    assert [type(retry).__name__ for retry in retries] == [
        "ModelRetryMiddleware",
        "_CodexIncompleteStreamRetryMiddleware",
    ]
    assert [retry.retry_on for retry in retries] == [
        runtime_mod._is_codex_stream_overload_error,
        runtime_mod._is_codex_incomplete_stream_error,
    ]


def test_codex_retry_profile_reaches_native_general_purpose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents import HarnessProfile, create_deep_agent
    from deepagents.profiles.harness import harness_profiles
    from langchain_openai.chat_models.codex import _ChatOpenAICodex

    built_retry_groups: list[list[Any]] = []

    def build_retry() -> list[Any]:
        retries = runtime_mod._build_codex_retry_middleware()
        built_retry_groups.append(retries)
        return retries

    monkeypatch.setitem(
        harness_profiles._HARNESS_PROFILES,
        "openai-codex",
        HarnessProfile(extra_middleware=build_retry),
    )
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    token_provider = SimpleNamespace(
        get_token=lambda: None,
        aget_token=lambda: None,
        get_access_token=lambda: "unused",
        aget_access_token=lambda: None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _ChatOpenAICodex(model="gpt-5.6-sol", token_provider=token_provider)

    assert model._get_ls_params()["ls_provider"] == "openai-codex"
    create_deep_agent(model=model, tools=[])

    assert len(built_retry_groups) == 2
    assert all(len(group) == 2 for group in built_retry_groups)
    assert built_retry_groups[0][0] is not built_retry_groups[1][0]
    assert built_retry_groups[0][1] is not built_retry_groups[1][1]


def test_extract_final_text_ignores_user_message_fallback() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )
    raw = {
        "messages": [
            {"role": "user", "content": "please do the calculation"},
            {"role": "assistant", "content": ""},
        ]
    }
    assert runner._extract_final_text(raw) == ""


def test_message_text_ignores_reasoning_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "text": "hidden chain"},
            {"type": "text", "text": "## Summary\nusable"},
        ]
    )
    assert runtime_mod.SpecialistRunner._message_text(message) == "## Summary\nusable"


def test_coerce_report_accepts_plain_text_without_summary_heading() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )
    parsed = runner._coerce_report(raw={"messages": [AIMessage(content="plain echo without headings")]})

    assert parsed["text"] == "plain echo without headings"
    assert parsed["summary"] == "plain echo without headings"
    assert parsed["facts"] == []
    assert parsed["files"] == []
    assert parsed["structured_report"] is False


def test_specialist_usage_callback_tracks_agent_scoped_usage() -> None:
    handler = runtime_mod.SpecialistUsageCallbackHandler(default_agent_name="writing_specialist")
    message = AIMessage(
        content="done",
        response_metadata={"model_name": "openai/gpt-5.4-20260305"},
        usage_metadata={
            "input_tokens": 25,
            "output_tokens": 4,
            "total_tokens": 29,
            "input_token_details": {"cache_read": 6},
            "output_token_details": {"reasoning": 2},
        },
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    handler.on_chat_model_start({}, [[]], run_id="run-1", metadata={"agent_name": "writing_specialist"})
    handler.on_llm_end(result, run_id="run-1")

    assert handler.call_counts_by_model["openai/gpt-5.4-20260305"] == 1
    assert handler.call_counts_by_role["writing_specialist"] == 1
    assert handler.usage_metadata_by_role["writing_specialist"]["openai/gpt-5.4-20260305"]["input_tokens"] == 25


def test_specialist_usage_callback_falls_back_to_default_agent_name() -> None:
    handler = runtime_mod.SpecialistUsageCallbackHandler(default_agent_name="experiment_specialist")
    message = AIMessage(
        content="done",
        response_metadata={"model_name": "openai/gpt-5.4-20260305"},
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    handler.on_chat_model_start({}, [[]], run_id="run-2")
    handler.on_llm_end(result, run_id="run-2")

    assert handler.call_counts_by_role["experiment_specialist"] == 1
    assert handler.usage_metadata_by_role["experiment_specialist"]["openai/gpt-5.4-20260305"]["total_tokens"] == 13


def test_specialist_usage_callback_deduplicates_callback_and_stream_message() -> None:
    handler = runtime_mod.SpecialistUsageCallbackHandler(default_agent_name="research_specialist")
    updates: list[dict[str, dict[str, object]]] = []
    handler.set_usage_update_callback(lambda: updates.append(dict(handler.usage_metadata)))
    message = AIMessage(
        id="resp_usage_1",
        content="done",
        response_metadata={"model_name": "gpt-5.6-sol"},
        usage_metadata={
            "input_tokens": 2580,
            "output_tokens": 120,
            "total_tokens": 2700,
            "input_token_details": {"cache_read": 1024},
            "output_token_details": {"reasoning": 80},
        },
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    handler.on_chat_model_start({}, [[]], run_id="llm-run-1")
    handler.on_llm_end(result, run_id="llm-run-1")
    ingested_again = handler.ingest_ai_message(
        message,
        call_id="llm-run-1",
        agent_name="research_specialist",
    )

    assert ingested_again is False
    assert handler.usage_metadata["gpt-5.6-sol"]["total_tokens"] == 2700
    assert handler.call_counts_by_model == {"gpt-5.6-sol": 1}
    assert handler.call_counts_by_role == {"research_specialist": 1}
    assert len(updates) == 1


def test_specialist_usage_callback_persists_after_each_completed_call(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_usage_live"
    run_dir.mkdir()
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(
            workspace=tmp_path,
            run_dir=run_dir,
            run_id="run_usage_live",
            project_id="proj",
        ),
        reporter=None,
        run_control=None,
    )
    handler = runner._new_usage_callback()
    handler.set_usage_update_callback(lambda: runner._write_usage_summary(handler))

    for index, input_tokens in enumerate((20, 30), start=1):
        message = AIMessage(
            id=f"resp_usage_{index}",
            content="done",
            response_metadata={"model_name": "gpt-5.6-sol"},
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": 5,
                "total_tokens": input_tokens + 5,
                "input_token_details": {"cache_read": index},
                "output_token_details": {"reasoning": 2},
            },
        )
        model = FakeMessagesListChatModel(responses=[message])
        model.invoke(
            "count this call",
            config={
                "callbacks": [handler],
                "metadata": {"lc_agent_name": "research_specialist"},
            },
        )

        persisted = load_usage_summary(run_dir)
        assert persisted["calls"] == index
        assert persisted["input_tokens"] == sum((20, 30)[:index])
        assert persisted["output_tokens"] == 5 * index

    assert persisted["total_tokens"] == 60
    assert persisted["input_cached_tokens"] == 3
    assert persisted["reasoning_tokens"] == 4


def test_finalize_report_runs_compile_guard_for_tex_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    (workspace / "files" / "writeup").mkdir(parents=True)
    tex_path = workspace / "files" / "writeup" / "note.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}Hi\\end{document}\n", encoding="utf-8")

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="writing",
    )

    def _fake_compile(payload):
        assert payload == {"source_path": "writeup/note.tex"}
        return (
            "compiled",
            {
                "tool_name": "compile_text",
                "data": {
                    "compiled_ok": True,
                    "pdf_path": "writeup/note.pdf",
                    "bib_paths": ["writeup/references.bib"],
                    "inspected_files": ["writeup/note.tex", "writeup/references.bib"],
                    "remaining_diagnostics": [],
                },
            },
        )

    monkeypatch.setattr(built.runner.registry, "get_tool_function", lambda name: _fake_compile if name == "compile_text" else None)

    finalized = built.runner._finalize_report(
        {
            "text": "## Summary\nshort\n\n## Facts\n- one\n\n## Files\n- `writeup/note.tex`",
            "summary": "short",
            "facts": ["one"],
            "files": ["writeup/note.tex"],
        }
    )

    assert finalized["files"] == ["writeup/note.tex", "writeup/note.pdf", "writeup/references.bib"]
    assert any("Compile guard produced `writeup/note.pdf`" in fact for fact in finalized["facts"])
    assert "`writeup/note.pdf`" in finalized["text"]
    assert "`writeup/references.bib`" in finalized["text"]


def test_render_compact_report_omits_empty_sections() -> None:
    rendered = runtime_mod.SpecialistRunner._render_compact_report(
        summary="draft revised",
        facts=[],
        files=[],
    )

    assert rendered == "## Summary\ndraft revised"


def test_finalize_report_preserves_unstructured_plain_text() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )

    finalized = runner._finalize_report(
        {
            "text": "## 1. Initial model interpretation\nPlain requested shape.",
            "summary": "## 1. Initial model interpretation\nPlain requested shape.",
            "facts": [],
            "files": [],
            "structured_report": False,
        }
    )

    assert finalized["text"] == "## 1. Initial model interpretation\nPlain requested shape."


def test_run_impl_retries_invalid_final_report_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    created_agents: list[dict] = []
    sleeps: list[float] = []

    class _RetryAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            _ = (payload, config)
            self.calls += 1
            if self.calls == 1:
                return {"messages": [{"role": "user", "content": "echoed prompt"}]}
            return {"messages": [AIMessage(content="## Summary\nrecovered\n\n## Facts\n- ok")]}

    retry_agent = _RetryAgent()

    def _fake_create_deep_agent(**kwargs):
        created_agents.append(kwargs)
        return retry_agent

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_entry_subagents",
        lambda self, entrypoint, runtime, thread_id="": [],
    )
    monkeypatch.setattr(runtime_mod.asyncio, "sleep", _fake_sleep)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Design the stage-2/3 plan.",
            entrypoint="experiment",
            proposal_review=False,
        )
    )

    assert result["status"] == "done"
    assert result["summary"] == "recovered"
    assert retry_agent.calls == 2
    assert sleeps == [30.0]


def test_run_impl_does_not_restart_episode_after_model_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    provider_error = openai.APIError(
        "An error occurred while processing your request. You can retry your request.",
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
        body=None,
    )

    class _FailedAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            _ = (payload, config)
            self.calls += 1
            raise provider_error

    failed_agent = _FailedAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_create_deep_agent",
        staticmethod(lambda: lambda **kwargs: failed_agent),
    )
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_entry_subagents",
        lambda self, entrypoint, runtime, thread_id="": [],
    )

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )

    with pytest.raises(openai.APIError) as caught:
        asyncio.run(
            built.runner.arun(
                "Run one bounded experiment.",
                entrypoint="experiment",
                proposal_review=False,
            )
        )

    assert caught.value is provider_error
    assert failed_agent.calls == 1


def test_research_reasoning_boundary_preserves_read_tools_and_hides_mutations() -> None:
    class _Request:
        def __init__(self, tools: list[Any]) -> None:
            self.tools = tools

        def override(self, **kwargs: Any) -> "_Request":
            return _Request(list(kwargs.get("tools", self.tools)))

    tools = [
        SimpleNamespace(name="query_research_graph_sql"),
        SimpleNamespace(name="query_literature_corpus"),
        SimpleNamespace(name="stage_research_plan"),
        {"type": "web_search"},
        SimpleNamespace(name="write_todos"),
        SimpleNamespace(name="read_file"),
        SimpleNamespace(name="read_document"),
        SimpleNamespace(name="ls"),
        SimpleNamespace(name="glob"),
        SimpleNamespace(name="grep"),
        SimpleNamespace(name="write_file"),
        SimpleNamespace(name="edit_file"),
        SimpleNamespace(name="execute"),
        SimpleNamespace(name="apply_patch"),
    ]
    boundary = runtime_mod._ResearchReasoningToolBoundaryMiddleware()

    async def _handler(request: _Request) -> _Request:
        return request

    bounded = asyncio.run(boundary.awrap_model_call(_Request(tools), _handler))
    visible_names = {runtime_mod._agent_tool_name(tool) for tool in bounded.tools}

    assert visible_names == {
        "query_research_graph_sql",
        "query_literature_corpus",
        "stage_research_plan",
        "web_search",
        "write_todos",
        "read_file",
        "read_document",
        "ls",
        "glob",
        "grep",
    }

    blocked_handler_called = False

    async def _blocked_handler(_request: Any) -> ToolMessage:
        nonlocal blocked_handler_called
        blocked_handler_called = True
        return ToolMessage(content="executed", tool_call_id="call-execute")

    blocked = asyncio.run(
        boundary.awrap_tool_call(
            SimpleNamespace(
                tool_call={
                    "name": "execute",
                    "args": {"command": "printenv"},
                    "id": "call-execute",
                }
            ),
            _blocked_handler,
        )
    )

    assert blocked_handler_called is False
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert blocked.tool_call_id == "call-execute"


@pytest.mark.parametrize(
    ("role", "skill_name", "extra_names"),
    [
        (
            "hypothesis_proposer",
            "research-graph-query",
            {"stage_research_plan"},
        ),
        ("evidence_judge", "research-evidence-reconciliation", set()),
    ],
)
def test_research_reasoning_final_model_surface_reads_only_scoped_skill(
    tmp_path: Path,
    role: str,
    skill_name: str,
    extra_names: set[str],
) -> None:
    from deepagents.backends import LocalShellBackend

    class _CapturingModel(FakeMessagesListChatModel):
        bound_tool_names: ClassVar[list[list[str]]] = []
        observed_messages: ClassVar[list[list[Any]]] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tool_choice, kwargs)
            self.bound_tool_names.append(
                [runtime_mod._agent_tool_name(tool) for tool in tools]
            )
            return self

        def _generate(self, messages, *args, **kwargs):
            self.observed_messages.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    _CapturingModel.bound_tool_names = []
    _CapturingModel.observed_messages = []
    workspace = tmp_path / "project_space"
    files_root = workspace / "files"
    files_root.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
    )
    built.runner._stage_deepagent_assets(files_root, thread_id="thread-1")
    reasoning_root = built.runner._skill_roots_for_group("research_reasoning")[0]
    skill_path = f"{reasoning_root}/{skill_name}/SKILL.md"

    child_model = _CapturingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": skill_path},
                        "id": "call-read-skill",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Scoped skill applied."),
        ]
    )
    parent_model = _CapturingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Open the scoped skill and report readiness.",
                            "subagent_type": role,
                        },
                        "id": "call-task",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Parent received the scoped result."),
        ]
    )
    SubAgent = built.runner._load_subagent()
    subagent = SubAgent(
        name=role,
        description="Exercise the final research reasoning surface.",
        system_prompt="Read the applicable skill before acting.",
        tools=built.runner._research_reasoning_tools(
            role=role,
            thread_id="thread-1",
            extra_names=extra_names,
        ),
        middleware=built.runner._research_reasoning_middleware(),
        skills=[reasoning_root],
        model=child_model,
    )
    agent = built.runner._load_create_deep_agent()(
        model=parent_model,
        tools=[],
        subagents=[subagent],
        backend=LocalShellBackend(root_dir=files_root, virtual_mode=True),
    )

    result = asyncio.run(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": "Delegate the check."}]}
        )
    )

    assert result["messages"][-1].content == "Parent received the scoped result."
    child_surfaces = [
        set(names)
        for names in _CapturingModel.bound_tool_names
        if "query_research_graph_sql" in names
    ]
    assert child_surfaces
    for surface in child_surfaces:
        assert {
            "ls",
            "glob",
            "grep",
            "read_file",
            "read_document",
            "query_research_graph_sql",
            "query_literature_corpus",
            "acquire_literature_source",
            "web_search",
        } <= surface
        assert {
            "write_file",
            "edit_file",
            "execute",
            "apply_patch",
        }.isdisjoint(surface)
        assert extra_names <= surface
        if role == "evidence_judge":
            assert "stage_research_plan" not in surface
    read_results = [
        str(message.content)
        for batch in _CapturingModel.observed_messages
        for message in batch
        if getattr(message, "name", None) == "read_file"
    ]
    assert any(f"# {skill_name}" in content for content in read_results)


def test_explicit_general_purpose_runtime_is_context_only_and_non_delegating(tmp_path: Path) -> None:
    class _BindableFakeModel(FakeMessagesListChatModel):
        bound_tool_names: list[list[str]] = []
        observed_system_prompts: list[str] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tool_choice, kwargs)
            self.bound_tool_names.append([getattr(tool, "name", str(tool)) for tool in tools])
            return self

        def _generate(self, messages, *args, **kwargs):
            system_text = "\n".join(
                str(message.content)
                for message in messages
                if getattr(message, "type", "") == "system"
            )
            self.observed_system_prompts.append(system_text)
            return super()._generate(messages, *args, **kwargs)

    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )
    general_purpose = built.runner._general_purpose_subagent(skills=[])

    def _caller_tool(value: str) -> str:
        return value

    caller_tool = StructuredTool.from_function(
        func=_caller_tool,
        name="caller_tool",
        description="Return one caller-layer value.",
    )
    model = _BindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Return the bounded result.",
                            "subagent_type": "general-purpose",
                        },
                        "id": "call-task",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="child result"),
            AIMessage(content="parent result"),
        ]
    )
    agent = built.runner._load_create_deep_agent()(
        model=model,
        tools=[caller_tool],
        subagents=[general_purpose],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Delegate this bounded branch."}]}
    )

    assert result["messages"][-1].content == "parent result"
    assert [
        message.content
        for message in result["messages"]
        if getattr(message, "name", None) == "task"
    ] == ["child result"]
    child_tool_sets = [
        set(names)
        for names in model.bound_tool_names
        if "read_document" in names and "task" not in names
    ]
    assert len(child_tool_sets) == 1
    assert {"caller_tool", "read_document"} <= child_tool_sets[0]
    assert "task" not in child_tool_sets[0]
    child_system_prompts = [
        prompt
        for prompt in model.observed_system_prompts
        if "CatMaster's general-purpose context worker" in prompt
    ]
    assert len(child_system_prompts) == 1
    assert "You have no subagents and must not transfer the task onward" in child_system_prompts[0]
    assert "current lane" not in child_system_prompts[0]
    assert "another lane" not in child_system_prompts[0]
    assert "Research Graph contract" not in child_system_prompts[0]


@pytest.mark.parametrize(
    ("entrypoint", "expected_subagent_names"),
    [
        (
            "research",
            [
                "general-purpose",
                "hypothesis_proposer",
                "evidence_judge",
                "experiment_evaluator",
                "experiment_specialist",
                "writing_specialist",
                "peer_review_specialist",
                "litreview_agent",
            ],
        ),
        (
            "experiment",
            [
                "general-purpose",
                "evidence_judge",
                "materials_worker",
                "ml_worker",
                "dynamics_worker",
                "orca_xtb_worker",
            ],
        ),
        ("literature_review", ["general-purpose"]),
        ("writing", ["general-purpose", "writing_worker_agent", "writing_polisher_agent"]),
        ("peer_review", ["general-purpose", "peer_review_worker_agent"]),
    ],
)
def test_specialist_lanes_start_with_staged_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    expected_subagent_names: list[str],
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("Project-level instructions.", encoding="utf-8")
    override = workspace / "metadata" / "self_evolution" / "self_develop_skills" / "materials_worker" / "workspace-demo"
    override.mkdir(parents=True)
    (override / "SKILL.md").write_text(
        "---\nname: workspace-demo\ndescription: Workspace override used for runtime staging tests.\n---\n# workspace-demo\n",
        encoding="utf-8",
    )
    bound_thread_id = ""
    bound_graph_id = ""
    if entrypoint == "research":
        graph_service = ResearchGraphService(workspace=workspace, workspace_id="proj")
        graph = graph_service.create_graph(
            GraphCreateRequest(question="Which evidence should Writing use?")
        )
        bound_graph_id = graph["graph"]["graph_id"]
        bound_thread = graph_service.thread_store.create_thread(
            title="Bound Research",
            entrypoint="research",
        )
        graph_service.thread_store.update_thread(
            bound_thread.thread_id,
            active_research_graph_id=bound_graph_id,
        )
        bound_thread_id = bound_thread.thread_id

    created_agents: list[dict] = []

    def _fake_create_deep_agent(**kwargs):
        created_agents.append(kwargs)
        return _FakeDeepAgent(kwargs=kwargs)

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_tool_strategy", staticmethod(lambda: _FakeToolStrategy))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_summarization_middleware",
        staticmethod(lambda: _FakeSummarizationMiddleware),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_memory_middleware",
        staticmethod(lambda: _FakeMemoryMiddleware),
    )
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint=entrypoint,
    )

    result = asyncio.run(
        built.runner.arun(
            "Run the lane smoke test.",
            entrypoint=entrypoint,
            proposal_review=False,
            thread_id=bound_thread_id,
        )
    )

    assert result["status"] == "done"
    assert created_agents, "expected create_deep_agent to be called"
    agent_kwargs = created_agents[-1]
    expected_agent_name = "litreview_agent" if entrypoint == "literature_review" else f"{entrypoint}_specialist"
    assert agent_kwargs["name"] == expected_agent_name
    expected_entry_model_role = {
        "research": "research_lead",
        "experiment": "director",
        "literature_review": "literature_deep_research",
        "writing": "write_director",
        "peer_review": "write_reviewer",
    }[entrypoint]
    assert agent_kwargs["model"] == {"model": f"{expected_entry_model_role}-model"}
    expected_entry_groups = {
        "research": ("research_specialist", "research_reasoning"),
        "experiment": ("writing_quality",),
        "literature_review": ("litreview_agent", "writing_quality"),
        "writing": ("writing_specialist", "writing_quality"),
        "peer_review": ("writing_specialist", "writing_quality"),
    }[entrypoint]
    _assert_native_skill_groups(agent_kwargs, *expected_entry_groups)
    _assert_native_memory(agent_kwargs)
    assert any(
        type(item).__name__ == "ReloadDeepAgentContextMiddleware"
        for item in agent_kwargs["middleware"]
    )
    internal_thread_id = agent_kwargs["_last_config"]["configurable"]["thread_id"]
    assert internal_thread_id.endswith(f"::run::{built.run_context.run_id}")
    assert "search_memory" not in {tool.name for tool in agent_kwargs["tools"]}
    assert "manage_memory" not in {tool.name for tool in agent_kwargs["tools"]}
    assert "Persistent project memory" in agent_kwargs["system_prompt"]
    assert "Do not store transient requests" in agent_kwargs["system_prompt"]
    assert all(getattr(tool, "name", None) != "bash" for tool in agent_kwargs["tools"])
    assert all(getattr(tool, "name", None) != "run_literature_research" for tool in agent_kwargs["tools"])
    top_subagents = list(agent_kwargs.get("subagents") or [])
    assert [subagent.kwargs["name"] for subagent in top_subagents] == expected_subagent_names
    for subagent in top_subagents:
        if "tools" in subagent.kwargs:
            assert all(getattr(tool, "name", None) != "bash" for tool in subagent.kwargs["tools"])
        if "middleware" in subagent.kwargs:
            middleware_names = {type(item).__name__ for item in (subagent.kwargs.get("middleware") or [])}
            if subagent.kwargs["name"] in {
                "hypothesis_proposer",
                "evidence_judge",
                "experiment_evaluator",
            }:
                assert {
                    "_ResearchReasoningToolBoundaryMiddleware",
                    "catmaster_nonfatal_tool_errors",
                } <= middleware_names
            else:
                assert "catmaster_nonfatal_tool_errors" in middleware_names

    subagents_by_name = {subagent.kwargs["name"]: subagent.kwargs for subagent in top_subagents}

    def _assert_explicit_general_purpose(owner_kwargs: dict[str, Any]) -> None:
        specs = [subagent.kwargs for subagent in owner_kwargs.get("subagents") or []]
        general_purpose = [spec for spec in specs if spec["name"] == "general-purpose"]
        assert len(general_purpose) == 1
        spec = general_purpose[0]
        assert "tools" not in spec
        assert "model" not in spec
        assert spec["skills"] == list(owner_kwargs.get("skills") or [])
        assert "CatMaster's general-purpose context worker" in spec["system_prompt"]
        assert "The brief is the source of scope" in spec["system_prompt"]
        assert "You have no subagents and must not transfer the task onward" in spec["system_prompt"]
        assert "Use workspace-relative paths and the paths supplied in the brief" in spec["system_prompt"]
        assert "Treat `/` only as the workspace virtual root" not in spec["system_prompt"]
        assert "never use leading-slash workspace paths" not in spec["system_prompt"]
        assert "do not calculate or compare hashes/checksums" in spec["system_prompt"]
        assert "Do not create, freeze, or persist an ad hoc contract" in spec["system_prompt"]
        assert "current lane" not in spec["description"]
        assert "current lane" not in spec["system_prompt"]
        assert "another lane" not in spec["system_prompt"]
        assert "Workspace script header policy" not in spec["system_prompt"]
        assert "topic-centric layout" not in spec["system_prompt"]
        assert "Research Graph contract" not in spec["system_prompt"]
        assert "browser use and full-text acquisition" not in spec["system_prompt"]
        assert "registered managed execution" not in spec["system_prompt"]
        middleware_names = {type(item).__name__ for item in spec["middleware"]}
        assert {"DocumentAccessMiddleware", "catmaster_nonfatal_tool_errors"} <= middleware_names
        document_access = next(item for item in spec["middleware"] if type(item).__name__ == "DocumentAccessMiddleware")
        assert {tool.name for tool in document_access.tools} == {"read_document"}

    for created_agent_kwargs in created_agents:
        _assert_explicit_general_purpose(created_agent_kwargs)

    def _created_agents_named(name: str) -> list[dict]:
        return [kwargs for kwargs in created_agents if kwargs["name"] == name]

    def _find_created_agent(
        name: str,
        *,
        tool_names: set[str] | None = None,
        prompt_contains: str | None = None,
    ) -> dict:
        matches = _created_agents_named(name)
        if tool_names is not None:
            matches = [kwargs for kwargs in matches if {tool.name for tool in kwargs["tools"]} == tool_names]
        if prompt_contains is not None:
            matches = [kwargs for kwargs in matches if prompt_contains in kwargs["system_prompt"]]
        assert matches, f"expected created agent {name!r}"
        return matches[0]

    if entrypoint == "research":
        assert {tool.name for tool in agent_kwargs["tools"]} == (
            _RESEARCH_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        assert "Research Graph contract" in agent_kwargs["system_prompt"]
        assert "Never guess among multiple graphs" in agent_kwargs["system_prompt"]
        assert "Research Kernel" not in agent_kwargs["system_prompt"]
        assert "Default to on-demand closeout, not autonomous research expansion" in agent_kwargs["system_prompt"]
        assert "scientific reasonableness check" in agent_kwargs["system_prompt"].lower()
        assert "A scientific reasonableness check is required for research closeouts" in agent_kwargs["system_prompt"]
        assert "do not force fixed `Summary` / `Facts` / `Files` headings" in agent_kwargs["system_prompt"]
        assert "litreview_agent" in agent_kwargs["system_prompt"]
        assert "hypothesis_proposer" in agent_kwargs["system_prompt"]
        assert "evidence_judge" in agent_kwargs["system_prompt"]
        assert "experiment_evaluator" in agent_kwargs["system_prompt"]
        assert "do not invent graph hypotheses" in agent_kwargs["system_prompt"]
        assert (
            "record only the hypothesis effects that the evidence actually addresses"
            in agent_kwargs["system_prompt"]
        )
        assert "metadata_agent" not in agent_kwargs["system_prompt"]
        assert "paper, manuscript, journal-style LaTeX draft" in agent_kwargs["system_prompt"]
        assert "experiment report, validation summary, QC note" in agent_kwargs["system_prompt"]
        assert "compact inline author packet" in agent_kwargs["system_prompt"]
        assert "compact inline report packet" in agent_kwargs["system_prompt"]
        assert "Default to not launching `peer_review_specialist`" in agent_kwargs["system_prompt"]
        assert "publication-level paper quality" in agent_kwargs["system_prompt"]
        assert "formal submission requirements" in agent_kwargs["system_prompt"]
        assert "explicitly hand it the canonical workspace-relative manuscript PDF path" in agent_kwargs["system_prompt"]
        assert "Do not rely on a graph node to preserve full editor/reviewer comment text" in agent_kwargs["system_prompt"]
        assert "If `peer_review_specialist` gives you a saved review memo path, read that memo directly" in agent_kwargs["system_prompt"]
        assert "You remain the sole coordinator and final decision-maker" in agent_kwargs["system_prompt"]
        assert "condition mismatch, incomplete provenance, unresolved alternatives" in agent_kwargs["system_prompt"]
        assert "weak evidence" not in agent_kwargs["system_prompt"]
        assert "current shared workspace makes parallel subagents unsafe" not in agent_kwargs["system_prompt"]
        assert "runnable" in subagents_by_name["experiment_specialist"]
        assert "runnable" in subagents_by_name["writing_specialist"]
        assert "runnable" in subagents_by_name["peer_review_specialist"]
        proposer = subagents_by_name["hypothesis_proposer"]
        judge = subagents_by_name["evidence_judge"]
        evaluator = subagents_by_name["experiment_evaluator"]
        assert {tool.name for tool in proposer["tools"]} == {
            "acquire_literature_source",
            "query_literature_corpus",
            "query_research_graph_sql",
            "stage_research_plan",
            "web_search",
        }
        assert {tool.name for tool in judge["tools"]} == {
            "acquire_literature_source",
            "query_literature_corpus",
            "query_research_graph_sql",
            "web_search",
        }
        assert {tool.name for tool in evaluator["tools"]} == {
            "acquire_literature_source",
            "evaluate_research_experiments",
            "query_literature_corpus",
            "query_research_graph_sql",
            "web_search",
        }
        assert {
            "_ResearchReasoningToolBoundaryMiddleware",
            "catmaster_nonfatal_tool_errors",
        } <= {
            type(item).__name__ for item in proposer["middleware"]
        }
        for reasoning in (proposer, judge, evaluator):
            _assert_native_skill_groups(reasoning, "research_reasoning")
        assert proposer["model"] == {"model": "hypothesis_proposer-model"}
        assert judge["model"] == {"model": "evidence_judge-model"}
        assert evaluator["model"] == {"model": "hypothesis_proposer-model"}
        assert "evidence attributes, not a global strength grade" in judge["system_prompt"]
        assert "response_format" not in proposer
        assert "response_format" not in judge
        assert "response_format" not in evaluator
        assert "does not execute scientific experiments" in proposer["description"]
        assert "does not propose branches or schedule work" in judge["description"]

        experiment_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "experiment_specialist"]
        assert experiment_agents, "expected nested experiment specialist to be created"
        experiment_agent_kwargs = experiment_agents[0]
        assert experiment_agent_kwargs["model"] == {"model": "director-model"}
        assert {tool.name for tool in experiment_agent_kwargs["tools"]} == (_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        assert not (
            _RESEARCH_TOOL_ALLOWLIST
            & {tool.name for tool in experiment_agent_kwargs["tools"]}
        )
        assert "mace_neb_batch" not in {tool.name for tool in experiment_agent_kwargs["tools"]}
        _assert_native_skill_groups(experiment_agent_kwargs, "writing_quality")
        assert "read and apply the `humanizer` skill" in experiment_agent_kwargs["system_prompt"]
        _assert_native_memory(experiment_agent_kwargs)
        assert [subagent.kwargs["name"] for subagent in experiment_agent_kwargs["subagents"]] == [
            "general-purpose",
            "evidence_judge",
            "materials_worker",
            "ml_worker",
            "dynamics_worker",
            "orca_xtb_worker",
        ]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in experiment_agent_kwargs["middleware"])

        writing_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "writing_specialist"]
        assert writing_agents, "expected nested writing specialist to be created"
        writing_agent_kwargs = writing_agents[0]
        assert writing_agent_kwargs["model"] == {"model": "write_director-model"}
        assert {tool.name for tool in writing_agent_kwargs["tools"]} == (_WRITING_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        graph_query = next(
            tool
            for tool in writing_agent_kwargs["tools"]
            if tool.name == "query_research_graph_sql"
        )
        query_result = json.loads(
            graph_query.invoke(
                {"sql": "SELECT graph_id FROM research_graphs"}
            )
        )
        assert query_result["graph_id"] == bound_graph_id
        assert query_result["rows"] == [{"graph_id": bound_graph_id}]
        _assert_native_skill_groups(writing_agent_kwargs, "writing_specialist", "writing_quality")
        _assert_native_memory(writing_agent_kwargs)
        assert [subagent.kwargs["name"] for subagent in writing_agent_kwargs["subagents"]] == [
            "general-purpose",
            "writing_worker_agent",
            "writing_polisher_agent",
        ]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in writing_agent_kwargs["middleware"])

        peer_review_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "peer_review_specialist"]
        assert peer_review_agents, "expected nested peer-review specialist to be created"
        peer_review_agent_kwargs = peer_review_agents[0]
        assert peer_review_agent_kwargs["model"] == {"model": "write_reviewer-model"}
        assert {tool.name for tool in peer_review_agent_kwargs["tools"]} == ({"peer_review_request"} | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(peer_review_agent_kwargs, "writing_specialist", "writing_quality")
        _assert_native_memory(peer_review_agent_kwargs)
        assert "Act like a journal editor coordinating external peer review" in peer_review_agent_kwargs["system_prompt"]
        assert "explicit `ReviewTarget` or manuscript PDF path" in peer_review_agent_kwargs["system_prompt"]
        assert "delegate the bounded review episode to `peer_review_worker_agent`" in peer_review_agent_kwargs["system_prompt"]
        assert [subagent.kwargs["name"] for subagent in peer_review_agent_kwargs["subagents"]] == [
            "general-purpose",
            "peer_review_worker_agent",
        ]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in peer_review_agent_kwargs["middleware"])
        litreview_compiled = subagents_by_name["litreview_agent"]
        assert "runnable" in litreview_compiled
        litreview_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "litreview_agent"]
        assert litreview_agents, "expected nested litreview agent to be created"
        litreview_agent_kwargs = litreview_agents[0]
        assert litreview_agent_kwargs["model"] == {"model": "literature_deep_research-model"}
        assert {tool.name for tool in litreview_agent_kwargs["tools"]} == (
            _LITREVIEW_LOCAL_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        assert not (
            _RESEARCH_TOOL_ALLOWLIST
            & {tool.name for tool in litreview_agent_kwargs["tools"]}
        )
        _assert_native_skill_groups(litreview_agent_kwargs, "litreview_agent", "writing_quality")
        _assert_native_memory(litreview_agent_kwargs)
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in litreview_agent_kwargs["middleware"])
        assert [subagent.kwargs["name"] for subagent in litreview_agent_kwargs["subagents"]] == ["general-purpose"]
        assert "requested scope" in litreview_agent_kwargs["system_prompt"]
        assert "fixed paper count or full-text count" in litreview_agent_kwargs["system_prompt"]
        assert "50-60" not in litreview_agent_kwargs["system_prompt"]
        assert "read and apply the `humanizer` skill" in litreview_agent_kwargs["system_prompt"]
        assert "metadata_agent" not in litreview_agent_kwargs["system_prompt"]
        assert "literature_agent" not in litreview_agent_kwargs["system_prompt"]
    elif entrypoint == "literature_review":
        assert {tool.name for tool in agent_kwargs["tools"]} == (
            _LITREVIEW_LOCAL_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        _assert_native_skill_groups(agent_kwargs, "litreview_agent", "writing_quality")
        assert "Own the review question" in agent_kwargs["system_prompt"]
        assert "Use each source only for what it supports" in agent_kwargs["system_prompt"]
        assert "methods, conditions, quantitative comparisons" in agent_kwargs["system_prompt"]
        assert "Distinguish reported results from your synthesis" in agent_kwargs["system_prompt"]
        assert "at most one" not in agent_kwargs["system_prompt"]
        assert "full-text access as unknown until tested" not in agent_kwargs["system_prompt"]
        assert "acquire_literature_source" not in agent_kwargs["system_prompt"]
        assert "finalize_citations" not in agent_kwargs["system_prompt"]
        assert "general-purpose" not in agent_kwargs["system_prompt"]
        assert "50-60" not in agent_kwargs["system_prompt"]
        assert "read and apply the `humanizer` skill" in agent_kwargs["system_prompt"]
        assert "Do not perform computational execution" in agent_kwargs["system_prompt"]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in agent_kwargs["middleware"])
        assert [subagent.kwargs["name"] for subagent in agent_kwargs["subagents"]] == ["general-purpose"]
        assert not _created_agents_named("literature_agent")
        assert not _created_agents_named("metadata_agent")
    elif entrypoint == "experiment":
        materials_worker_kwargs = _find_created_agent("materials_worker")
        ml_worker_kwargs = _find_created_agent("ml_worker")
        dynamics_worker_kwargs = _find_created_agent("dynamics_worker")
        orca_worker_kwargs = _find_created_agent("orca_xtb_worker")
        for worker_kwargs in (materials_worker_kwargs, ml_worker_kwargs, dynamics_worker_kwargs, orca_worker_kwargs):
            assert worker_kwargs["model"] == {"model": "task_runner-model"}
        assert "runnable" in subagents_by_name["materials_worker"]
        assert "runnable" in subagents_by_name["ml_worker"]
        assert "runnable" in subagents_by_name["dynamics_worker"]
        assert "runnable" in subagents_by_name["orca_xtb_worker"]
        assert {tool.name for tool in materials_worker_kwargs["tools"]} == (_MATERIALS_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(materials_worker_kwargs, "materials_worker", "execution")
        assert {tool.name for tool in dynamics_worker_kwargs["tools"]} == (_DYNAMICS_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(dynamics_worker_kwargs, "dynamics_worker", "execution")
        assert {tool.name for tool in ml_worker_kwargs["tools"]} == (_ML_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(ml_worker_kwargs, "ml_worker", "execution")
        assert {tool.name for tool in orca_worker_kwargs["tools"]} == (_ORCA_XTB_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(orca_worker_kwargs, "orca_xtb_worker", "execution")
        assert {tool.name for tool in agent_kwargs["tools"]} == (_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(agent_kwargs, "writing_quality")
        assert {"mp_search_materials", "mp_download_structure"} <= {tool.name for tool in agent_kwargs["tools"]}
        assert "mace_neb_batch" not in {tool.name for tool in agent_kwargs["tools"]}
        assert "parent-maintained project memory" in materials_worker_kwargs["system_prompt"]
        assert "Instruction context files" not in materials_worker_kwargs["system_prompt"]
        assert "dataset/model lifecycle tasks" in ml_worker_kwargs["system_prompt"]
        assert "default role is coordination, dispatch, and decision-making across the experiment lane" in agent_kwargs["system_prompt"]
        assert "read and apply the `humanizer` skill" in agent_kwargs["system_prompt"]
        assert "Keep direct work in the specialist thread minimal and coordination-oriented" in agent_kwargs["system_prompt"]
        assert "Route by the current working artifact" in agent_kwargs["system_prompt"]
        assert "When a request clearly falls into one of those worker-owned domains, delegate first instead of doing the domain work yourself." in agent_kwargs["system_prompt"]
        assert "route every MLFF MD, restart, and trajectory-QC task to `dynamics_worker`" in agent_kwargs["system_prompt"]
        assert "Model fine-tuning, training, evaluation, feature/data pipelines, and ML algorithm development belong to `ml_worker`" in agent_kwargs["system_prompt"]
        assert "use `orca_xtb_worker` for molecular or cluster quantum-chemistry work" in agent_kwargs["system_prompt"]
        assert "purely report writing from already completed evidence" in agent_kwargs["system_prompt"]
        assert "stays in `ExperimentSpecialist`" in agent_kwargs["system_prompt"]
        assert "Each worker should receive only one bounded execution episode around one primary artifact" in agent_kwargs["system_prompt"]
        assert "Do not hand an entire high-throughput campaign to one worker" in agent_kwargs["system_prompt"]
        assert "Do not personally absorb worker-owned tasks just because your own direct tool surface appears sufficient" in agent_kwargs["system_prompt"]
        assert "Only do the implementation directly in the specialist thread when no available worker matches the task" in agent_kwargs["system_prompt"]
        assert "Delegate domain-owned work to the proper specialized subagent first." in agent_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in agent_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in agent_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in agent_kwargs["system_prompt"]
        assert "Do not use `read_file` directly on PDF, DOCX, XLSX, or PPTX files." in agent_kwargs["system_prompt"]
        assert "It inherits the caller's direct tools and staged skills, cannot delegate, and returns one handoff." in agent_kwargs["system_prompt"]
        assert "do not stop at that boundary alone" in agent_kwargs["system_prompt"]
        assert "prefer materializing it as a reusable workspace script under `scripts/`" in agent_kwargs["system_prompt"]
        assert "If a worker needs a handy Python package for a bounded local step and it is missing" in agent_kwargs["system_prompt"]
        assert "Experiment closeout discipline: use worker/tool returns as the QC source of record" in agent_kwargs["system_prompt"]
        assert "Do not rerun or reparse calculation outputs just to repeat domain QC" in agent_kwargs["system_prompt"]
        assert "If the scope is complete, state the executed scope, key evidence paths, and residual limitations" in agent_kwargs["system_prompt"]
        assert "remote_submission" in {tool.name for tool in materials_worker_kwargs["tools"]}
        assert "mace_neb_batch" not in {tool.name for tool in materials_worker_kwargs["tools"]}
        assert "Typical managed MLFF work here includes surrogate screening, relaxation, single-point ranking, and path optimization" in materials_worker_kwargs["system_prompt"]
        assert "All MLFF MD, restart, and trajectory-health tasks belong to `dynamics_worker`" in materials_worker_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in materials_worker_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in materials_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in materials_worker_kwargs["system_prompt"]
        assert "Do not use `read_file` directly on PDF, DOCX, XLSX, or PPTX files." in materials_worker_kwargs["system_prompt"]
        assert "It inherits the caller's direct tools and staged skills, cannot delegate, and returns one handoff." in materials_worker_kwargs["system_prompt"]
        assert "obtain POTCARs through the pymatgen interface" in materials_worker_kwargs["system_prompt"]
        assert "If a handy Python package is missing for a bounded local step" in materials_worker_kwargs["system_prompt"]
        assert "write a reusable workspace script under `scripts/`" in materials_worker_kwargs["system_prompt"]
        assert "registered managed execution in this worker is authoritative" in materials_worker_kwargs["system_prompt"]
        assert "Before low-level managed remote submission, read the task catalog or mounted execution skill" in materials_worker_kwargs["system_prompt"]
        assert "If managed submission fails with receipt/context fields" in materials_worker_kwargs["system_prompt"]
        assert "CP2K AIMD preparation/execution handoff, managed MLFF MD sampling" in dynamics_worker_kwargs["system_prompt"]
        assert "Do not invent force-field parameters" in dynamics_worker_kwargs["system_prompt"]
        assert "registered managed execution in this worker is authoritative" in dynamics_worker_kwargs["system_prompt"]
        assert "Start here when the primary artifact is a curated dataset" in ml_worker_kwargs["system_prompt"]
        assert "When a registered managed ML tool fits the task, prefer that managed path first." in ml_worker_kwargs["system_prompt"]
        assert "Prefer using libraries already available in the environment and reusable workspace code" in ml_worker_kwargs["system_prompt"]
        assert "Common libraries already available here include `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `joblib`, and `matminer`" in ml_worker_kwargs["system_prompt"]
        assert "If a handy Python package is still missing for a bounded local step" in ml_worker_kwargs["system_prompt"]
        assert "If the ML logic is longer than a short throwaway snippet and no managed tool covers it" in ml_worker_kwargs["system_prompt"]
        assert "Prefer organizing topic-specific ML scripts under `scripts/<topic>/`" in ml_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in ml_worker_kwargs["system_prompt"]
        assert "Do not use `read_file` directly on PDF, DOCX, XLSX, or PPTX files." in ml_worker_kwargs["system_prompt"]
        assert "It inherits the caller's direct tools and staged skills, cannot delegate, and returns one handoff." in ml_worker_kwargs["system_prompt"]
        assert "Prefer materializing training pipelines, feature generation, sweeps, evaluation harnesses, embedding workflows, and data-processing logic as reusable scripts" in ml_worker_kwargs["system_prompt"]
        assert "Treat the managed ML tools as preferred paths when they fit, not as an exclusive gate" in ml_worker_kwargs["system_prompt"]
        assert "keep going locally with reusable scripts under `scripts/` instead of stopping" in ml_worker_kwargs["system_prompt"]
        assert "registered managed execution in this worker is authoritative" in ml_worker_kwargs["system_prompt"]
        assert "write a reusable workspace script under `scripts/`" in ml_worker_kwargs["system_prompt"]
        assert "molecular quantum-chemistry subtask" in orca_worker_kwargs["system_prompt"]
        assert "Treat xTB/CREST as the fast exploration layer" in orca_worker_kwargs["system_prompt"]
        assert "If a handy Python package is missing for a bounded local step" in orca_worker_kwargs["system_prompt"]
        assert "registered managed execution in this worker is authoritative" in orca_worker_kwargs["system_prompt"]
        assert "treat its execution and domain QC as authoritative" in agent_kwargs["system_prompt"]
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in materials_worker_kwargs["middleware"]
        )
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in orca_worker_kwargs["middleware"]
        )
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in dynamics_worker_kwargs["middleware"]
        )
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in ml_worker_kwargs["middleware"]
        )
    elif entrypoint == "writing":
        assert {tool.name for tool in agent_kwargs["tools"]} == (_WRITING_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(agent_kwargs, "writing_specialist", "writing_quality")
        assert "compile_text" not in {tool.name for tool in agent_kwargs["tools"]}
        writing_worker_kwargs = _find_created_agent("writing_worker_agent")
        writing_polisher_kwargs = _find_created_agent("writing_polisher_agent")
        assert writing_worker_kwargs["model"] == {"model": "section_writer-model"}
        assert writing_polisher_kwargs["model"] == {"model": "academic_polisher-model"}
        assert "runnable" in subagents_by_name["writing_worker_agent"]
        assert "runnable" in subagents_by_name["writing_polisher_agent"]
        assert {tool.name for tool in writing_worker_kwargs["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(writing_worker_kwargs, "writing_specialist", "writing_quality")
        assert {tool.name for tool in writing_polisher_kwargs["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(writing_polisher_kwargs, "writing_specialist", "writing_quality")
        assert "This lane owns paper, manuscript, and author-facing scientific writing" in agent_kwargs["system_prompt"]
        assert "compact inline author packet" in agent_kwargs["system_prompt"]
        assert "Use `writing_polisher_agent` only for local prose cleanup" in agent_kwargs["system_prompt"]
        assert "read and apply the `humanizer` skill" in agent_kwargs["system_prompt"]
        assert "narrow background supplementation" in agent_kwargs["system_prompt"]
        assert "Each writing-worker handoff should cover only one section or one bounded organization/integration task" in agent_kwargs["system_prompt"]
        assert "figures, tables, and concise explanatory schematics as part of the default deliverable" in agent_kwargs["system_prompt"]
        assert "Supporting Information / Supporting Data package" in agent_kwargs["system_prompt"]
        assert "keep Supporting Information in the same manuscript file" in agent_kwargs["system_prompt"]
        assert "place it after the references" in agent_kwargs["system_prompt"]
        assert "journal-style title centered on the chemical system and principal scientific finding" in agent_kwargs["system_prompt"]
        assert "figures to be inserted near their first substantive discussion rather than batched at the end" in agent_kwargs["system_prompt"]
        assert "manuscript-review capability perform a comment-only publication-readiness review" in agent_kwargs["system_prompt"]
        assert "complete the bounded polishing or revision work still needed" in agent_kwargs["system_prompt"]
        assert "clearly exposed as `ReviewTarget`" in agent_kwargs["system_prompt"]
        assert "publishable paper ready to enter peer review" in agent_kwargs["system_prompt"]
        assert "Do not mention the workspace, files, runs, prompts, tools, agents, interruptions" in agent_kwargs["system_prompt"]
        assert "Delegate domain-owned work to the proper specialized subagent first." in agent_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in agent_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in agent_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in agent_kwargs["system_prompt"]
        assert "Do not use `read_file` directly on PDF, DOCX, XLSX, or PPTX files." in agent_kwargs["system_prompt"]
        assert "It inherits the caller's direct tools and staged skills, cannot delegate, and returns one handoff." in agent_kwargs["system_prompt"]
        assert "Handle only one section or one bounded organization/integration task at a time" in writing_worker_kwargs["system_prompt"]
        assert "compact author packet" in writing_worker_kwargs["system_prompt"]
        assert "organize what belongs in the main text versus Supporting Information / Supporting Data" in writing_worker_kwargs["system_prompt"]
        assert "keep Supporting Information in the same manuscript file" in writing_worker_kwargs["system_prompt"]
        assert "For short notes or compact summaries, do not manufacture extra visuals" in writing_worker_kwargs["system_prompt"]
        assert "Use `generate_nanobanana_figure` for conceptual, mechanistic, or workflow figures" in writing_worker_kwargs["system_prompt"]
        assert "produce a compact journal-style title" in writing_worker_kwargs["system_prompt"]
        assert "do not batch figures into a later block" in writing_worker_kwargs["system_prompt"]
        assert "Do not treat a successful TeX compile as sufficient" in writing_worker_kwargs["system_prompt"]
        assert "publishable paper ready to enter peer review" in writing_worker_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in writing_worker_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in writing_worker_kwargs["system_prompt"]
        assert "Perform conservative section-level prose polish" in writing_polisher_kwargs["system_prompt"]
        assert "without changing claim strength, scientific scope, evidence selection" in writing_polisher_kwargs["system_prompt"]
        assert "For journal-facing citations and BibTeX, use publication-style metadata only" in writing_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in writing_worker_kwargs["system_prompt"]
        assert "Do not use `read_file` directly on PDF, DOCX, XLSX, or PPTX files." in writing_worker_kwargs["system_prompt"]
        assert "It inherits the caller's direct tools and staged skills, cannot delegate, and returns one handoff." in writing_worker_kwargs["system_prompt"]
    else:
        assert {tool.name for tool in agent_kwargs["tools"]} == ({"peer_review_request"} | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(agent_kwargs, "writing_specialist", "writing_quality")
        assert "Act like a journal editor coordinating external peer review" in agent_kwargs["system_prompt"]
        assert "Reviewer Comments" in agent_kwargs["system_prompt"]
        assert "save the full review as one durable workspace markdown memo" in agent_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in agent_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in agent_kwargs["system_prompt"]
        assert "do not compress away the editor comment or reviewer comment sections" in agent_kwargs["system_prompt"]
        assert "peer_review_worker_agent" in subagents_by_name
        peer_review_worker_kwargs = _find_created_agent("peer_review_worker_agent")
        assert peer_review_worker_kwargs["model"] == {"model": "task_runner-model"}
        assert "runnable" in subagents_by_name["peer_review_worker_agent"]
        assert {tool.name for tool in peer_review_worker_kwargs["tools"]} == ({"peer_review_request"} | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(peer_review_worker_kwargs, "writing_specialist", "writing_quality")
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in peer_review_worker_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in peer_review_worker_kwargs["system_prompt"]
        assert "dedicated peer-review request capability on that PDF exactly once" in peer_review_worker_kwargs["system_prompt"]

    deepagents_root = workspace / "files" / ".deepagents"
    snapshot_match = re.fullmatch(
        r"/\.deepagents/snapshots/([0-9a-f]{24})/AGENTS\.md",
        agent_kwargs["memory"][0],
    )
    assert snapshot_match is not None
    snapshot_root = deepagents_root / "snapshots" / snapshot_match.group(1)
    staged_agents = snapshot_root / "AGENTS.md"
    staged_materials = snapshot_root / "skills" / "materials_worker"
    staged_writing = snapshot_root / "skills" / "writing_specialist"
    staged_researcher = snapshot_root / "skills" / "research_specialist"
    staged_reasoning = snapshot_root / "skills" / "research_reasoning"
    staged_literature = snapshot_root / "skills" / "litreview_agent"
    staged_writing_quality = snapshot_root / "skills" / "writing_quality"
    staged_quantum_chemistry = snapshot_root / "skills" / "orca_xtb_worker"
    staged_execution = snapshot_root / "skills" / "execution"
    assert staged_agents.read_text(encoding="utf-8") == "Project-level instructions."
    assert staged_materials.is_dir()
    assert staged_writing.is_dir()
    assert staged_researcher.is_dir()
    assert staged_reasoning.is_dir()
    assert staged_literature.is_dir()
    assert staged_writing_quality.is_dir()
    assert staged_quantum_chemistry.is_dir()
    assert staged_execution.is_dir()
    staged_workspace_override = (
        snapshot_root
        / "skills"
        / "materials_worker"
        / "workspace-demo"
        / "SKILL.md"
    )
    assert not staged_workspace_override.exists()
    assert (override / "SKILL.md").is_file()
    staged_machine_learning = snapshot_root / "skills" / "ml_worker"
    assert staged_machine_learning.is_dir()
    repo_root = Path(runtime_mod.__file__).resolve().parents[2]

    def _skill_names(root: Path) -> set[str]:
        return {path.parent.name for path in root.glob("*/SKILL.md") if path.is_file()}

    assert _skill_names(staged_materials) == _skill_names(repo_root / "skills" / "materials_worker")
    assert _skill_names(staged_machine_learning) == _skill_names(repo_root / "skills" / "ml_worker")
    assert _skill_names(staged_quantum_chemistry) == _skill_names(repo_root / "skills" / "orca_xtb_worker")
    assert _skill_names(staged_execution) == _skill_names(repo_root / "skills" / "execution")
    assert _skill_names(staged_researcher) == _skill_names(repo_root / "skills" / "research_specialist")
    assert _skill_names(staged_reasoning) == {
        "research-evidence-reconciliation",
        "research-graph-query",
    }
    assert _skill_names(staged_literature) == _skill_names(repo_root / "skills" / "litreview_agent")
    assert _skill_names(staged_writing) == _skill_names(repo_root / "skills" / "writing_specialist")
    assert _skill_names(staged_writing_quality) == {"humanizer"}
    assert _skill_names(staged_writing)
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["entrypoint"] == entrypoint
    assert run_state["status"] == "done"
    assert run_state["summary"]
    assert "skill_projection" not in run_state
    assert isinstance(run_state.get("facts"), list)
    if entrypoint == "research":
        assert "research_kernel_path" not in run_state
        assert "research_kernel" not in run_state
        assert "hypothesis_engine" not in run_state
        assert "research_goal_path" not in run_state
        assert "research_goal" not in run_state
    usage_summary = load_usage_summary(built.run_context.run_dir)
    assert usage_summary["source"] == "langchain_usage_metadata"
    assert usage_summary["input_tokens"] == 123
    assert usage_summary["input_cached_tokens"] == 80
    assert usage_summary["output_tokens"] == 17
    assert usage_summary["reasoning_tokens"] == 5
    assert usage_summary["calls"] == 2
    assert usage_summary["by_role"][0]["name"] == "experiment_specialist"
    assert usage_summary["by_role"][0]["calls"] == 1

def test_specialist_run_passes_project_id_to_runtime_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- stored\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_tool_strategy", staticmethod(lambda: _FakeToolStrategy))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_memory_ns",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Remember durable project facts when justified.",
            entrypoint="experiment",
            proposal_review=False,
            thread_id="thread-123",
        )
    )

    assert result["status"] == "done"
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["configurable"]["thread_id"] == f"thread-123::run::{built.run_context.run_id}"
    assert config["configurable"]["project_id"] == "proj_memory_ns"
    assert config["metadata"]["catmaster_thread_id"] == "thread-123"


def test_proposal_review_flag_is_ignored_and_run_executes_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- executed directly\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_proposal_gate",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Run the experiment lane directly.",
            entrypoint="experiment",
            proposal_review=True,
        )
    )

    assert result["status"] == "done"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"] == "Run the experiment lane directly."
    assert "Human review feedback" not in payload["messages"][0]["content"]
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "done"
    assert run_state["proposal_review"] is False
    assert run_state["proposal_revision_count"] == 0


def test_interrupted_run_can_resume_into_normal_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- resumed legacy proposal run\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_resume_legacy_proposal",
        preferred_entrypoint="experiment",
    )
    (built.run_context.run_dir / RUN_STATE_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoint": "experiment",
                "status": "interrupted_paused",
                "phase": "interrupted",
                "active_specialist": "experiment",
                "thread_id": "thread-legacy",
                "proposal_review": False,
                "proposal_revision_count": 0,
                "pending_human_input": None,
                "todo_items": [],
                "artifacts": [],
                "delegation_log": [],
                "user_prompt": "Resume this old stuck run.",
                "chat_session_id": "chat-legacy",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = asyncio.run(built.runner.aresume(""))

    assert result["status"] == "done"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"] == "Continue the previous interrupted request."
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "done"
    assert run_state["proposal_review"] is False
    assert run_state["proposal_revision_count"] == 0


def test_research_resume_preserves_original_prompt_without_goal_shadow_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- resumed original objective\n\n## Files\n- notes/research/resume.md"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured.setdefault("agent_kwargs", kwargs)
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_resume_research_prompt",
        preferred_entrypoint="research",
    )
    original_prompt = (
        "Use MACE to compute the O2 bond length and report the evidence path."
    )
    (built.run_context.run_dir / RUN_STATE_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoint": "research",
                "status": "interrupted_paused",
                "phase": "interrupted",
                "active_specialist": "research",
                "thread_id": "thread-research-prompt",
                "proposal_review": False,
                "proposal_revision_count": 0,
                "pending_human_input": None,
                "todo_items": [],
                "artifacts": [],
                "delegation_log": [],
                "user_prompt": original_prompt,
                "chat_session_id": "chat-research-prompt",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = asyncio.run(built.runner.aresume("also include a short caveat"))

    assert result["status"] == "done"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    resume_message = payload["messages"][0]["content"]
    assert original_prompt in resume_message
    assert "also include a short caveat" in resume_message
    assert "formal completion audit" in resume_message
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["user_prompt"] == original_prompt
    assert "research_goal" not in run_state


def test_conversation_messages_are_replayed_only_for_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- replayed chat history\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_history",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Current request.",
            entrypoint="experiment",
            proposal_review=False,
            conversation_messages=[
                {"role": "user", "content": "Older request."},
                {"role": "assistant", "content": "Older answer."},
            ],
        )
    )

    assert result["status"] == "done"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"] == [
        {"role": "user", "content": "Older request."},
        {"role": "assistant", "content": "Older answer."},
        {"role": "user", "content": "Current request."},
    ]


def test_specialist_runner_returns_interrupted_paused_when_interrupt_requested_before_start(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    run_control = RunControl(run_id="run_interrupt")
    run_control.request_interrupt(source="ui", note="stop")

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=run_control,
        project_id="proj_interrupt_before_start",
        preferred_entrypoint="research",
    )

    result = asyncio.run(
        built.runner.arun(
            "Stop before any deepagent work starts.",
            entrypoint="research",
            proposal_review=False,
        )
    )

    assert result["status"] == "interrupted_paused"
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "interrupted_paused"
    assert run_state["summary"] == "Run interrupted by user."
    assert run_state["user_prompt"] == "Stop before any deepagent work starts."
    assert "research_goal" not in run_state


def test_specialist_resume_clears_stale_interrupt_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    run_control = RunControl(run_id="run_interrupt_resume")
    run_control.request_interrupt(source="ui", note="stop")
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nresumed\n\n## Facts\n- interrupt flag cleared\n\n## Files\n- notes/resumed.md"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=run_control,
        project_id="proj_interrupt_resume",
        preferred_entrypoint="research",
    )

    interrupted = asyncio.run(
        built.runner.arun(
            "Interrupt this run first.",
            entrypoint="research",
            proposal_review=False,
        )
    )
    assert interrupted["status"] == "interrupted_paused"
    assert run_control.snapshot()["requested"] is True

    result = asyncio.run(built.runner.aresume("continue now"))

    assert result["status"] == "done"
    assert run_control.snapshot()["requested"] is False
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert "continue now" in payload["messages"][0]["content"]
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "done"
