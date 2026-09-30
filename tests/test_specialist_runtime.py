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
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
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
    _EXPERIMENT_SPECIALIST_BASE_TOOL_ALLOWLIST,
    _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST,
    _LITREVIEW_LOCAL_TOOL_ALLOWLIST,
    _MATERIALS_WORKER_TOOL_ALLOWLIST,
    _ML_WORKER_TOOL_ALLOWLIST,
    _ORCA_XTB_WORKER_TOOL_ALLOWLIST,
    _PLOT_WORKER_TOOL_ALLOWLIST,
    _PRESENTATION_WORKER_TOOL_ALLOWLIST,
    _RESEARCH_COMPARISON_TOOL_ALLOWLIST,
    _RESEARCH_TOOL_ALLOWLIST,
    _WRITING_WORKER_TOOL_ALLOWLIST,
    _WRITING_TOOL_ALLOWLIST,
    build_specialist_runner,
)
from catmaster.runtime.usage_stats import load_usage_summary
from catmaster.runtime.artifact_callback import UIEventHandler
from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
    ResearchGraphPlanningProposal,
    ResultCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.tools.registry import get_tool_registry
from catmaster.llm.config import AgentRuntimeConfig


class _FakeProfile:
    agent_runtime = AgentRuntimeConfig()

    def config_for_role(self, role: str) -> SimpleNamespace:
        return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)


class _FakeToolStrategy:
    def __init__(self, schema, handle_errors: bool = False) -> None:
        self.schema = schema
        self.handle_errors = handle_errors


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
    assert paths == [f"/.deepagents/skills/{group}" for group in groups]


def _assert_native_memory(agent_kwargs: dict) -> None:
    memory = list(agent_kwargs["memory"])
    assert memory == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]


def _backend_for_staged_skills(runner: Any, files_root: Path) -> Any:
    from deepagents.backends import CompositeBackend, FilesystemBackend

    assert runner._skill_snapshot_root is not None
    return CompositeBackend(
        default=runtime_mod.CatMasterLocalShellBackend(
            root_dir=files_root,
            virtual_mode=True,
        ),
        routes={
            "/.deepagents/": FilesystemBackend(
                root_dir=runner._skill_snapshot_root,
                virtual_mode=True,
            )
        },
    )


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
    assert {
        "search_openalex", "search_semantic_scholar", "get_openalex_record",
        "get_semantic_scholar_record", "recommend_semantic_scholar",
        "acquire_literature_source", "batch_acquire_literature_sources",
        "finalize_citations",
    } <= _LITREVIEW_LOCAL_TOOL_ALLOWLIST
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

    native = {"type": "web_search", "search_context_size": "low"}
    custom = StructuredTool.from_function(
        lambda query: query, name="web_search", description="Search the web"
    )
    unrelated = StructuredTool.from_function(
        lambda value: value, name="echo", description="Echo input"
    )
    # Both Responses and Chat Completions custom-function schemas occur at
    # provider boundaries. Neither may hide or duplicate the hosted tool.
    for conflicting in [
        custom,
        {"type": "function", "name": "web_search", "parameters": {"type": "object"}},
        {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}},
    ]:
        result = built.runner._augment_with_default_autonomous_tools(
            [unrelated, conflicting, native, native], model_role="task_runner"
        )
        assert result[0] is unrelated
        assert result[1] is (native if expects_native else conflicting)
        assert len(result) == (3 if provider == "codex_oauth" else 2)
        assert built.runner._augment_with_default_autonomous_tools(
            result, model_role="task_runner"
        ) == result

    result = built.runner._augment_with_default_autonomous_tools(
        [custom], model_role="task_runner"
    )
    assert result[0] == ({"type": "web_search"} if expects_native else custom)


def test_litreview_graph_tools_follow_the_turn_snapshot(
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

    unbound_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="literature_review",
        runtime_context={
            "research_graph_id": "",
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )
    bound_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="literature_review",
        runtime_context={
            "research_graph_id": graph_id,
            "research_focus_node_id": experiment_id,
            "research_launch_id": "",
        },
    )
    standalone_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="literature_review",
        runtime_context={
            "research_graph_id": graph_id,
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )

    direct_names = unbound_runner.runner._litreview_local_tool_names(
        direct.thread_id,
        top_level=True,
    )
    bound_names = bound_runner.runner._litreview_local_tool_names(
        bound.thread_id,
        top_level=True,
    )
    standalone_names = standalone_runner.runner._litreview_local_tool_names(
        bound.thread_id,
        top_level=True,
    )
    assert direct_names == (_LITREVIEW_LOCAL_TOOL_ALLOWLIST | {"notify_progress"})
    assert direct_names.isdisjoint(_BOUND_RESEARCH_EXECUTION_TOOL_ALLOWLIST)
    expected_graph_bound_names = (
        _LITREVIEW_LOCAL_TOOL_ALLOWLIST
        | (_BOUND_RESEARCH_EXECUTION_TOOL_ALLOWLIST - {"create_bound_research_experiment"})
        | {"notify_progress", "query_research_graph_sql"}
    )
    assert bound_names == expected_graph_bound_names
    assert standalone_names == expected_graph_bound_names
    assert bound_runner.runner._litreview_local_tool_names(
        bound.thread_id,
        top_level=False,
    ) == _LITREVIEW_LOCAL_TOOL_ALLOWLIST

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
        staticmethod(lambda: SubAgent),
    )
    fake_runtime = {
        "checkpointer": object(),
        "store": object(),
        "backend": object(),
    }
    direct_agent = unbound_runner.runner._build_litreview_agent(
        runtime=fake_runtime,
        thread_id=direct.thread_id,
        top_level=True,
    )
    bound_agent = bound_runner.runner._build_litreview_agent(
        runtime=fake_runtime,
        thread_id=bound.thread_id,
        top_level=True,
    )
    assert [item["name"] for item in direct_agent.kwargs["subagents"]] == [
        "general-purpose",
        "litreview_worker_agent",
    ]
    assert [item["name"] for item in bound_agent.kwargs["subagents"]] == [
        "general-purpose",
        "litreview_worker_agent",
    ]


def test_experiment_graph_tools_exist_only_on_a_bound_top_level_turn(
    tmp_path: Path,
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
            execution_lane="experiment",
        ),
    )
    experiment_id = experiment["node"]["node_id"]

    unbound = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )
    bound = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
        runtime_context={
            "research_graph_id": graph_id,
            "research_focus_node_id": experiment_id,
            "research_launch_id": "",
        },
    )

    unbound_names = {
        tool.name for tool in unbound.runner._specialist_tools("experiment")
    }
    bound_names = {
        tool.name for tool in bound.runner._specialist_tools("experiment")
    }
    nested_names = {
        tool.name
        for tool in bound.runner._specialist_tools(
            "experiment",
            top_level=False,
        )
    }
    assert unbound_names == (
        _EXPERIMENT_SPECIALIST_BASE_TOOL_ALLOWLIST
        | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        | {"manage_effective_skills"}
    )
    assert bound_names == (
        _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
        | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        | {"manage_effective_skills"}
    )
    assert nested_names == (
        _EXPERIMENT_SPECIALIST_BASE_TOOL_ALLOWLIST
        | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
    )


def test_research_tools_and_reasoning_roles_follow_turn_planning_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    service = ResearchGraphService(workspace=workspace, workspace_id="proj")
    created = service.create_graph(
        GraphCreateRequest(question="Which mechanism controls selectivity?")
    )
    graph_id = created["graph"]["graph_id"]
    revision = created["graph"]["revision"]
    bound_thread = service.thread_store.create_thread(
        title="Ordinary bound Research",
        entrypoint="research",
    )
    service.thread_store.update_thread(
        bound_thread.thread_id,
        active_research_graph_id=graph_id,
    )
    planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
    )
    assert claimed is True
    planning_thread = service.thread_store.create_thread(
        title="Internal planning",
        entrypoint="research",
        thread_role="research_planning",
    )
    service.thread_store.update_thread(
        planning_thread.thread_id,
        active_research_graph_id=graph_id,
    )
    service.store.update_planning(
        graph_id,
        planning["planning_id"],
        start_revision=revision,
        status="attached",
        thread_id=planning_thread.thread_id,
    )

    unbound = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
        runtime_context={
            "research_graph_id": "",
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )
    bound = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
        runtime_context={
            "research_graph_id": graph_id,
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )
    planning_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
        runtime_context={
            "research_graph_id": graph_id,
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )
    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_subagent",
        staticmethod(lambda: SubAgent),
    )

    def _tool_names(runner: Any, thread_id: str) -> set[str]:
        return {
            tool.name
            for tool in runner._specialist_tools(
                "research",
                thread_id=thread_id,
            )
        }

    unbound_tools = _tool_names(unbound.runner, "thread_unbound")
    bound_tools = _tool_names(bound.runner, bound_thread.thread_id)
    planning_tools = _tool_names(
        planning_runner.runner,
        planning_thread.thread_id,
    )
    assert unbound_tools == {
        "manage_effective_skills",
        "notify_progress",
        "web_search",
    }
    assert "query_research_graph_sql" in bound_tools
    for runner in (unbound.runner, bound.runner):
        for entrypoint in ("research", "persistent_research"):
            for tools in (
                runner._specialist_tools(entrypoint, thread_id="thread_unbound"),
                runner._specialist_subagent_tools(entrypoint),
            ):
                assert not {"list_research_graphs", "create_research_graph"} & {t.name for t in tools}
    assert "stage_research_plan" not in bound_tools
    assert planning_tools == {
        "mark_research_planning_no_change",
        "query_research_graph_sql",
        "set_research_graph_completion",
    }

    from deepagents.backends import LocalShellBackend

    files_root = workspace / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    fake_runtime: dict[str, object] = {
        "backend": LocalShellBackend(root_dir=files_root, virtual_mode=True)
    }
    unbound_reasoning = unbound.runner._scientific_reasoning_subagents(
        runtime=fake_runtime,
        thread_id="thread_unbound",
    )
    bound_reasoning = bound.runner._scientific_reasoning_subagents(
        runtime=fake_runtime,
        thread_id=bound_thread.thread_id,
    )
    planning_reasoning = planning_runner.runner._scientific_reasoning_subagents(
        runtime=fake_runtime,
        thread_id=planning_thread.thread_id,
    )
    assert [item["name"] for item in unbound_reasoning] == [
        "hypothesis_proposer"
    ]
    assert [item["name"] for item in bound_reasoning] == [
        "hypothesis_proposer",
    ]
    assert [item["name"] for item in planning_reasoning] == [
        "hypothesis_proposer",
    ]
    assert "query_research_graph_sql" not in {
        tool.name for tool in unbound_reasoning[0]["tools"]
    }
    assert "stage_research_plan" not in {
        tool.name for tool in bound_reasoning[0]["tools"]
    }
    assert "stage_research_plan" in {
        tool.name for tool in planning_reasoning[0]["tools"]
    }
    captured_agent: dict[str, Any] = {}

    def _capture_planning_agent(**kwargs):
        captured_agent.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_create_deep_agent",
        staticmethod(lambda: _capture_planning_agent),
    )
    asyncio.run(
        planning_runner.runner._build_entry_agent(
            entrypoint="research",
            runtime={
                **fake_runtime,
                "checkpointer": object(),
                "store": object(),
            },
            thread_id=planning_thread.thread_id,
        )
    )
    assert [item["name"] for item in captured_agent["subagents"]] == [
        "hypothesis_proposer",
    ]
    assert {tool.name for tool in captured_agent["tools"]} == planning_tools
    assert "This turn is for Research Graph planning" in captured_agent["system_prompt"]


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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
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
        name="simulated_failure_tool",
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
    monkeypatch.setitem(built.runner.registry.tools, "simulated_failure_tool", {"function": object()})

    wrapped = built.runner._named_tools({"simulated_failure_tool"})
    content, artifact = wrapped[0].func(value="x")
    assert "simulated failure" in content
    assert artifact["tool_name"] == "simulated_failure_tool"
    assert artifact["data"]["status"] == "error"
    assert artifact["data"]["tool_name"] == "simulated_failure_tool"


def test_specialist_reporting_contract_reaches_execution_workers(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "handoff-policy-sentinel"
    monkeypatch.setattr(runner, "_soft_reporting_contract", staticmethod(lambda: marker))
    for prompt in (runner._materials_worker_prompt(), runner._ml_worker_prompt(),
                   runner._orca_xtb_worker_prompt(), runner._dynamics_worker_prompt()):
        assert prompt.count(marker) == 1


def test_research_reporting_policy_reaches_both_research_entrypoints(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "research-reporting-policy-sentinel"
    monkeypatch.setattr(runner, "_research_reporting_contract", staticmethod(lambda: marker))

    for entrypoint in ("research", "persistent_research"):
        assert runner._base_system_prompt(entrypoint).count(marker) == 1


def test_literature_reporting_policy_is_local_to_literature_roles(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "literature-reporting-policy-sentinel"
    monkeypatch.setattr(runner, "_literature_reporting_contract", staticmethod(lambda: marker))
    for prompt in (runner._litreview_wrapper_prompt(), runner._litreview_worker_prompt()):
        assert prompt.count(marker) == 1
    for prompt in (
        runner._base_system_prompt("experiment"),
        runner._materials_worker_prompt(), runner._ml_worker_prompt(),
        runner._orca_xtb_worker_prompt(), runner._dynamics_worker_prompt(),
        runner._writing_worker_prompt(),
    ):
        assert marker not in prompt


def test_writing_reporting_contract_allows_summary_first_closeout() -> None:
    contract = runtime_mod.SpecialistRunner._writing_reporting_contract()
    assert "shape the user requested" in contract
    assert "not required" in contract
    assert "Include a `Files` section only when" in contract
    assert "optional `ReviewTarget` section" in contract
    assert "Do not add a placeholder `Facts` section" in contract


@pytest.mark.parametrize("entrypoint", ["research", "persistent_research", "experiment", "writing"])
def test_report_handoff_policy_reaches_writing_delegators(monkeypatch, entrypoint) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "report-handoff-policy-sentinel"
    monkeypatch.setattr(runner, "_report_packet_policy", staticmethod(lambda: marker))
    assert runner._base_system_prompt(entrypoint).count(marker) == 1


def test_scientific_communication_policy_reaches_narrative_roles(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "scientific-communication-policy-sentinel"
    monkeypatch.setattr(
        runner, "_scientific_communication_policy", staticmethod(lambda: marker)
    )
    prompts = [
        *(
            runner._base_system_prompt(entry)
            for entry in ("research", "persistent_research", "experiment", "writing", "peer_review")
        ),
        runner._litreview_wrapper_prompt(),
        runner._litreview_worker_prompt(),
        runner._peer_review_worker_prompt(),
        runner._writing_worker_prompt(),
        runner._presentation_worker_prompt(),
        runner._general_purpose_child_prompt(),
    ]
    assert all(prompt.count(marker) == 1 for prompt in prompts)


def test_writing_acceptance_policy_is_local_to_writing_coordinator(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "writing-acceptance-policy-sentinel"
    monkeypatch.setattr(runner, "_writing_acceptance_policy", staticmethod(lambda: marker))
    assert runner._base_system_prompt("writing").count(marker) == 1
    other_prompts = [
        *(runner._base_system_prompt(entry) for entry in (
            "research", "persistent_research", "experiment", "peer_review", "literature_review",
        )),
        runner._writing_worker_prompt(),
        runner._presentation_worker_prompt(),
        runner._plot_worker_prompt(),
    ]
    assert all(marker not in prompt for prompt in other_prompts)


def test_leaf_system_prompts_do_not_describe_invisible_agent_choreography() -> None:
    cases = {
        "general-purpose": runtime_mod.SpecialistRunner._general_purpose_child_prompt(),
        "hypothesis_proposer": runtime_mod.SpecialistRunner._hypothesis_proposer_prompt(),
        "experiment_pair_comparator": runtime_mod.SpecialistRunner._experiment_pair_comparator_prompt(),
        "materials_worker": runtime_mod.SpecialistRunner._materials_worker_prompt(),
        "ml_worker": runtime_mod.SpecialistRunner._ml_worker_prompt(),
        "dynamics_worker": runtime_mod.SpecialistRunner._dynamics_worker_prompt(),
        "orca_xtb_worker": runtime_mod.SpecialistRunner._orca_xtb_worker_prompt(),
        "litreview_worker_agent": runtime_mod.SpecialistRunner._litreview_worker_prompt(),
        "writing_worker_agent": runtime_mod.SpecialistRunner._writing_worker_prompt(),
        "plot_worker": runtime_mod.SpecialistRunner._plot_worker_prompt(),
        "peer_review_worker_agent": runtime_mod.SpecialistRunner._peer_review_worker_prompt(),
    }
    role_names = {
        "ResearchSpecialist",
        "ExperimentSpecialist",
        "WritingSpecialist",
        "PeerReviewSpecialist",
        "hypothesis_proposer",
        "experiment_pair_comparator",
        "materials_worker",
        "ml_worker",
        "dynamics_worker",
        "orca_xtb_worker",
        "litreview_agent",
        "litreview_worker_agent",
        "writing_worker_agent",
        "plot_worker",
        "peer_review_worker_agent",
    }

    for own_name, prompt in cases.items():
        assert "parent" not in prompt.lower()
        assert "coordinator" not in prompt.lower()
        assert "host-built" not in prompt.lower()
        assert "host routing" not in prompt.lower()
        assert "resume_parent" not in prompt
        for other_name in role_names - {own_name}:
            assert other_name not in prompt


def test_plot_worker_prompt_requires_direct_origin_style_rendered_qa() -> None:
    prompt = runtime_mod.SpecialistRunner._plot_worker_prompt()

    assert "Complete one bounded quantitative or data-native publication-figure job directly" in prompt
    assert "Do not delegate" in prompt
    assert "Origin-like scientific style" in prompt
    assert "Never deliver matplotlib's default style or default color cycle" in prompt
    for color in ("#E64B35", "#4DBBD5", "#F39B7F", "#8491B4"):
        assert color in prompt
    assert "inspect the final PNG itself with `read_file`" in prompt
    assert "persist exactly one final figure file" in prompt
    assert "disposable raster QA preview under `/tmp/`" in prompt
    assert "overlap between text and the visual signal" in prompt
    assert "Do not report irrelevant hardware, platform, build, launcher" in prompt
    assert "publication-data-plotting" in prompt


def test_plot_palette_preserves_customer_color_values() -> None:
    import runpy

    root = Path(__file__).resolve().parents[1]
    palette = runpy.run_path(str(
        root / "skills/plot_worker/publication-data-plotting/scripts/palette.py"
    ))
    assert palette["DEFAULT_COLORS"] == [
        "#E64B35", "#4DBBD5", "#F39B7F", "#8491B4", "#00A087",
        "#3C5488", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    ]
    assert set(palette["PALETTE"].values()) == set(palette["DEFAULT_COLORS"])

def test_shared_tool_policy_reaches_specialists_and_workers(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    rules = ("_scientific_provenance_policy", "_hash_policy", "_contract_policy")
    for rule in rules:
        monkeypatch.setattr(runner, rule, staticmethod(lambda name=rule: name + "-sentinel"))
    prompts = [runner._tool_policy(), runner._materials_worker_prompt(),
               runner._litreview_wrapper_prompt(), runner._general_purpose_child_prompt(),
               *(runner._base_system_prompt(role) for role in ("research", "experiment", "writing", "peer_review"))]
    for prompt in prompts:
        for rule in rules:
            assert prompt.count(rule + "-sentinel") == 1


def test_atomic_geometry_integrity_policy_reaches_coordinate_owning_lanes() -> None:
    policy = runtime_mod.SpecialistRunner._atomic_geometry_integrity_policy()
    assert policy.startswith(
        "Atomic-geometry discipline: treat initial structure construction and physical optimization as distinct stages."
    )
    assert "localize force outliers to atoms and contacts" in policy
    prompts = (
        runtime_mod.SpecialistRunner._base_system_prompt("experiment"),
        runtime_mod.SpecialistRunner._materials_worker_prompt(),
        runtime_mod.SpecialistRunner._dynamics_worker_prompt(),
        runtime_mod.SpecialistRunner._orca_xtb_worker_prompt(),
    )
    for prompt in prompts:
        assert policy in prompt

    assert policy not in runtime_mod.SpecialistRunner._base_system_prompt("research")
    assert policy not in runtime_mod.SpecialistRunner._ml_worker_prompt()
    assert (
        "return the exact structure evidence and required reconstruction"
        in runtime_mod.SpecialistRunner._dynamics_worker_prompt()
    )


def test_audit_reuse_policy_is_shared_once(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "audit-reuse-sentinel"
    monkeypatch.setattr(runner, "_audit_reuse_policy", staticmethod(lambda: marker))
    assert runner._deepagent_execution_policy().count(marker) == 1
    assert runner._proposal_system_prompt("research").count(marker) == 1
    assert runner._research_challenger_prompt().count(marker) == 1


def test_computation_guidance_reaches_its_owning_layer(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    for name in ("_cross_layer_computation_brief_policy", "_worker_execution_adaptation_policy"):
        monkeypatch.setattr(runner, name, staticmethod(lambda name=name: name + "-sentinel"))
    for role in ("research", "experiment"):
        prompt = runner._base_system_prompt(role)
        assert prompt.count("_cross_layer_computation_brief_policy-sentinel") == 1
        assert "_worker_execution_adaptation_policy-sentinel" not in prompt
    assert "_cross_layer_computation_brief_policy-sentinel" not in runner._materials_worker_prompt()


def test_persistent_research_reuses_research_model_role() -> None:
    assert runtime_mod._ENTRYPOINT_TO_MODEL_ROLE["persistent_research"] == runtime_mod._ENTRYPOINT_TO_MODEL_ROLE["research"] == "research_lead"


def test_skill_output_contracts_do_not_require_operational_qc_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skill_paths = sorted((repo_root / "skills").glob("**/SKILL.md"))
    recovery_skill = repo_root / "skills/execution/dpdispatcher-remote-receipts/SKILL.md"
    operational_field = re.compile(
        r"receipt|remote_context_id|submission_hash|task_state_counts|hardware|"
        r"launcher|\bmpi\b|openmp|software build|provider version|device identity|"
        r"performance telemetry",
        re.IGNORECASE,
    )

    assert skill_paths
    for path in skill_paths:
        if path == recovery_skill:
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^## Output [Cc]ontract\s*$", text, re.MULTILINE)
        if match is None:
            continue
        output_contract = re.split(r"^## ", text[match.end() :], maxsplit=1, flags=re.MULTILINE)[0]
        operational_bullets = [
            line
            for line in output_contract.splitlines()
            if line.startswith("- ") and operational_field.search(line)
        ]
        assert not operational_bullets, f"{path}: {operational_bullets}"

    recovery_text = recovery_skill.read_text(encoding="utf-8")
    assert "narrow operational-recovery exception" in recovery_text
    assert "they are not scientific QC" in recovery_text
    assert "not a scientific `NO_GO`" in recovery_text

    lammps_preparation_text = (
        repo_root / "skills/dynamics_worker/lammps-preparation/SKILL.md"
    ).read_text(encoding="utf-8")
    lammps_md_text = (
        repo_root / "skills/dynamics_worker/lammps-md-execution/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "same-coordinate CPU/KOKKOS comparison" in lammps_preparation_text
    assert "is not required before production" in lammps_preparation_text
    assert "Cross-backend diagnosis is exceptional" in lammps_md_text

    writeback_text = (
        repo_root / "skills/research_execution/research-graph-writeback/SKILL.md"
    ).read_text(encoding="utf-8")
    reconciliation_text = (
        repo_root / "skills/research_reasoning/research-evidence-reconciliation/SKILL.md"
    ).read_text(encoding="utf-8")
    authoring_text = (repo_root / "skills/AGENTS.MD").read_text(encoding="utf-8")
    assert "Do not calculate or compare hashes/checksums merely because work uses transfer" in authoring_text
    assert "Existing tool/API/file-format contracts remain binding" in authoring_text
    assert "do not create an ad hoc schema, secure manifest" in authoring_text
    assert "An explicit user request overrides this default" in authoring_text


def test_skill_bundles_do_not_mandate_proactive_operational_audits() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    recovery_skill = repo_root / "skills/execution/dpdispatcher-remote-receipts/SKILL.md"
    skill_docs = sorted((repo_root / "skills").glob("**/*.md"))
    prohibited = {
        "receipt IDs in reports": re.compile(
            r"\b(?:report|record|preserve|retain)\b[^\n]{0,120}"
            r"\b(?:remote )?receipt/context IDs?\b",
            re.IGNORECASE,
        ),
        "model-file SHA collection": re.compile(
            r"submission validator records[^\n]{0,80}\bSHA-?256\b",
            re.IGNORECASE,
        ),
        "hardware-stack benchmark gate": re.compile(
            r"benchmark[^\n]{0,120}\bGPU class\b[^\n]{0,120}\bruntime stack\b",
            re.IGNORECASE,
        ),
        "default software-environment inventory": re.compile(
            r"Software, package versions[^\n]{0,120}\boperating system\b",
            re.IGNORECASE,
        ),
        "success-path receipt retention": re.compile(
            r"After dispatch, retain[^\n]{0,120}\breceipt/context\b",
            re.IGNORECASE,
        ),
    }

    for path in skill_docs:
        if path == recovery_skill:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in prohibited.items():
            assert pattern.search(text) is None, f"{path}: {label}"


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


def test_shared_literature_assets_are_reachable_from_staged_backend(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    files_root = workspace / "files"
    files_root.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace, llm_profile=_FakeProfile(), reporter=None,
        run_control=None, project_id="proj", preferred_entrypoint="writing",
    )
    built.runner._stage_deepagent_assets(files_root, thread_id="evidence")
    backend = _backend_for_staged_skills(built.runner, files_root)
    paths = (
        "litreview_agent/literature-evidence-use/references/evidence-attributes.md",
        "writing_specialist/citation-management/scripts/citation_records.py",
        "writing_specialist/citation-management/scripts/academic_search.py",
        "writing_specialist/publication-launch-writing/references/reporting-standards.md",
    )
    for path in paths:
        result = backend.read("/.deepagents/skills/" + path)
        assert result.error is None, path
        assert result.file_data is not None, path
        assert result.file_data["content"], path


def test_writing_system_prompts_do_not_prescribe_task_scale() -> None:
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
    assert "literature/" in prompt
    assert "structures/" in prompt
    assert "calculations/" in prompt
    assert "notes/" in prompt
    assert "writing/" in prompt


def test_every_filesystem_middleware_mounts_shared_artifact_policy(tmp_path: Path) -> None:
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
    files_root = workspace / "files"
    backend = runtime_mod.CatMasterLocalShellBackend(
        root_dir=files_root,
        virtual_mode=True,
    )
    policy = built.runner._artifact_persistence_policy()

    for read_only in (False, True):
        middleware = built.runner._build_filesystem_middleware(
            backend=backend,
            read_only=read_only,
        )
        prompt = middleware._custom_system_prompt
        assert prompt.count(policy) == 1

    result = backend.write("/tmp/ocr.txt", "scratch")
    assert result.error is None
    assert (files_root / "tmp" / "ocr.txt").read_text(encoding="utf-8") == "scratch"


def test_orca_xtb_worker_prompt_includes_workspace_path_discipline() -> None:
    prompt = runtime_mod.SpecialistRunner._orca_xtb_worker_prompt()
    assert "Workspace path discipline" in prompt
    assert "Treat `/` only as the workspace virtual root" in prompt
    assert "molecular quantum-chemistry subtask" in prompt
    assert "first create the structure under `<topic>/structures/`" in prompt
    assert "Do not guess that a path like `<topic>/structures/<name>.xyz` already exists" in prompt
    assert "do not choose ORCA-XTB as the default fallback for routine preopt steps" in prompt


def test_common_worker_prompts_share_tool_guidance(monkeypatch) -> None:
    runner = runtime_mod.SpecialistRunner
    marker = "tool-guidance-sentinel"
    monkeypatch.setattr(runner, "_tool_policy", classmethod(lambda cls: marker))
    for prompt in (runner._materials_worker_prompt(), runner._ml_worker_prompt(),
                   runner._orca_xtb_worker_prompt(), runner._writing_worker_prompt(),
                   runner._plot_worker_prompt(), runner._peer_review_worker_prompt()):
        assert prompt.count(marker) == 1


def test_scientific_skill_examples_separate_tasks_from_unrelated_method_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    references = {
        "skills/dynamics_worker/lammps-preparation/references/lammps_native_examples.md": (
            "pair_style",
            "pair_coeff",
        ),
        "skills/materials_worker/cp2k-dft-preparation/references/cp2k_native_examples.md": (
            "XC_FUNCTIONAL",
            "BASIS_SET",
            "CUTOFF 600",
        ),
        "skills/dynamics_worker/cp2k-aimd-preparation/references/cp2k_aimd_native_examples.md": (
            "XC_FUNCTIONAL",
            "BASIS_SET",
            "CUTOFF 600",
        ),
        "skills/orca_xtb_worker/xtb-screen-and-prune/references/xtb_native_examples.md": (
            '\"--gfn\", \"2\"',
            '\"water\"',
        ),
        "skills/orca_xtb_worker/conformer-search-and-preopt/references/crest_native_examples.md": (
            '\"--gfn2\"',
            '\"water\"',
        ),
    }

    for relative_path, forbidden_tokens in references.items():
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "fragment" in text.lower()
        for token in forbidden_tokens:
            assert token not in text


def test_delegating_worker_prompts_do_not_gain_blanket_gp_serialization() -> None:
    prompts = (
        runtime_mod.SpecialistRunner._materials_worker_prompt(),
        runtime_mod.SpecialistRunner._ml_worker_prompt(),
        runtime_mod.SpecialistRunner._dynamics_worker_prompt(),
        runtime_mod.SpecialistRunner._orca_xtb_worker_prompt(),
        runtime_mod.SpecialistRunner._writing_worker_prompt(),
        runtime_mod.SpecialistRunner._plot_worker_prompt(),
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

    assert "requested deliverable or explicitly approved scientific stage as the stop condition" in research_prompt
    assert "delegate owns implementation-equivalent corrections inside that stage" in research_prompt
    assert "distinguish an internal delegation mismatch from a human blocker" in research_prompt
    assert "revise the delegation and continue within the existing authorization" in research_prompt
    assert "weak evidence" not in research_prompt
    assert "Pass execution needs through an exposed experiment capability" in research_prompt
    assert "request a standalone capability check only" in research_prompt
    assert "Research Graph contract" in research_prompt
    assert runtime_mod.SpecialistRunner._research_reporting_contract() in research_prompt
    for prompt in (research_prompt, experiment_prompt, writing_prompt):
        assert "current shared workspace makes parallel subagents unsafe" not in prompt
    assert "Run delegated review episodes sequentially" in peer_review_prompt
    assert "general-purpose" not in litreview_prompt
    assert "litreview_worker_agent" in litreview_prompt
    assert "acquire_literature_source" not in litreview_prompt
    assert "finalize_citations" not in litreview_prompt
    assert "treat its execution and domain QC as authoritative" in experiment_prompt
    assert experiment_prompt.count(runtime_mod.SpecialistRunner._delegated_computation_role_policy()) == 1
    assert experiment_prompt.count(runtime_mod.SpecialistRunner._experiment_layered_capability_visibility_policy()) == 1
    assert "Experiment closeout discipline: use worker/tool returns as the QC source of record" in experiment_prompt
    assert "Do not rerun or reparse calculation outputs just to repeat domain QC" in experiment_prompt
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
        assert "delegate the bounded scientific calculation" in prompt
        assert "rather than substituting a capability probe for the requested work" in prompt
        assert "specific missing input, task registration, resource configuration, stage layout, or user-controlled decision" in prompt


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
    assert {name for name in _MATERIALS_WORKER_TOOL_ALLOWLIST if name.startswith("cp2k_")} == {
        "cp2k_prepare",
        "cp2k_output_summary",
    }
    assert {"cp2k_prepare", "cp2k_output_summary", "lammps_prepare", "lammps_log_summary"} <= _DYNAMICS_WORKER_TOOL_ALLOWLIST
    assert "cp2k_aimd_prepare" not in _DYNAMICS_WORKER_TOOL_ALLOWLIST
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
    assert "Prefer a topic-centric layout" in writing_prompt
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


def test_tool_error_middleware_preserves_native_graph_control() -> None:
    from langgraph.errors import GraphBubbleUp, GraphInterrupt
    from langgraph.types import Interrupt

    middleware = runtime_mod.SpecialistRunner._build_default_middleware()[-1]
    request = SimpleNamespace(tool_call={"id": "cost", "name": "set_research_task_cost"})
    signals = [
        GraphInterrupt((Interrupt(value={"kind": "research_capacity", "task_cost": "high"}),)),
        GraphBubbleUp("graph control"),
    ]
    for signal in signals:
        async def handler(_request):
            raise signal

        with pytest.raises(type(signal)) as raised:
            asyncio.run(middleware.awrap_tool_call(request, handler))
        assert raised.value is signal


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


def test_specialist_usage_callback_groups_by_configured_model_label() -> None:
    handler = runtime_mod.SpecialistUsageCallbackHandler(
        default_agent_name="writing_specialist"
    )
    message = AIMessage(
        content="done",
        response_metadata={"model_name": "gpt-5.6-luna"},
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 12,
            "total_tokens": 112,
            "input_token_details": {"cache_read": 60},
        },
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    handler.on_chat_model_start(
        {},
        [[]],
        run_id="run-labeled",
        metadata={
            "lc_agent_name": "writing_worker_agent",
            "catmaster_model_label": "codex-oauth-luna-worker",
        },
    )
    handler.on_llm_end(result, run_id="run-labeled")

    assert handler.call_counts_by_model == {"codex-oauth-luna-worker": 1}
    usage = handler.usage_metadata["codex-oauth-luna-worker"]
    assert usage["_catmaster_model_name"] == "gpt-5.6-luna"
    assert usage["input_tokens"] == 100
    assert handler.usage_metadata_by_role["writing_worker_agent"][
        "codex-oauth-luna-worker"
    ]["output_tokens"] == 12


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


def test_structured_final_report_preserves_authored_text() -> None:
    runner = runtime_mod.SpecialistRunner.__new__(runtime_mod.SpecialistRunner)
    text = (
        "这里是重要的前提限制。\n\n## Summary\n研究结论。\n\n"
        "## Facts\n- 证据一\n\n## Files\n- `result.md`：必须保留的文件说明\n\n"
        "## 尚未解决的问题\n样本量不足。"
    )
    finalized = runner._finalize_report(runner._coerce_report(raw=text))
    assert finalized["text"] == text


@pytest.mark.parametrize(
    "text",
    [
        "## 1. Initial model interpretation\nPlain requested shape.",
        "The tool-using agents were evaluated across independent runs with fixed prompts.",
        "已更新报告：`writing/report.md`。",
    ],
)
def test_finalize_report_preserves_unstructured_plain_text(text: str) -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )

    finalized = runner._finalize_report(
        {
            "text": text,
            "summary": text,
            "facts": [],
            "files": [],
            "structured_report": False,
        }
    )

    assert finalized["text"] == text


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
        ("hypothesis_proposer", "research-evidence-reconciliation", {"record_research_review"}),
    ],
)
def test_research_reasoning_final_model_surface_reads_only_scoped_skill(
    tmp_path: Path,
    role: str,
    skill_name: str,
    extra_names: set[str],
) -> None:
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
    graph_service = ResearchGraphService(workspace=workspace, workspace_id="proj")
    graph = graph_service.create_graph(
        GraphCreateRequest(question="Which route should be planned next?")
    )
    graph_id = graph["graph"]["graph_id"]
    revision = graph["graph"]["revision"]
    planning, claimed = graph_service.store.claim_planning(
        graph_id,
        expected_revision=revision,
    )
    assert claimed is True
    planning_thread = graph_service.thread_store.create_thread(
        title="Internal planning",
        entrypoint="research",
    )
    graph_service.thread_store.update_thread(
        planning_thread.thread_id,
        active_research_graph_id=graph_id,
    )
    graph_service.store.update_planning(
        graph_id,
        planning["planning_id"],
        start_revision=revision,
        status="attached",
        thread_id=planning_thread.thread_id,
    )
    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
        runtime_context={
            "research_graph_id": graph_id,
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )
    built.runner._stage_deepagent_assets(
        files_root,
        thread_id=planning_thread.thread_id,
    )
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
    backend = _backend_for_staged_skills(built.runner, files_root)
    runtime = {"backend": backend}
    subagent = SubAgent(
        name=role,
        description="Exercise the final research reasoning surface.",
        system_prompt="Read the applicable skill before acting.",
        tools=built.runner._research_reasoning_tools(
            role=role,
            thread_id=planning_thread.thread_id,
            extra_names=extra_names,
        ),
        middleware=built.runner._research_reasoning_middleware(runtime=runtime),
        skills=[reasoning_root],
        model=child_model,
    )
    agent = built.runner._load_create_deep_agent()(
        model=parent_model,
        tools=[],
        subagents=[subagent],
        backend=backend,
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
            "write_todos",
            "ls",
            "glob",
            "grep",
            "read_file",
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
            "delete",
        }.isdisjoint(surface)
        assert extra_names <= surface
    read_results = [
        str(message.content)
        for batch in _CapturingModel.observed_messages
        for message in batch
        if getattr(message, "name", None) == "read_file"
    ]
    assert any(f"# {skill_name}" in content for content in read_results)


@pytest.mark.parametrize("worker", ["plot_worker", "presentation_worker"])
def test_writing_workers_final_model_surface_has_direct_file_and_execution_tools(tmp_path: Path, worker: str) -> None:
    class _BindablePlotModel(FakeMessagesListChatModel):
        bound_tool_names: ClassVar[list[list[str]]] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tool_choice, kwargs)
            self.bound_tool_names.append(
                [runtime_mod._agent_tool_name(tool) for tool in tools]
            )
            return self

    _BindablePlotModel.bound_tool_names = []
    workspace = tmp_path / "project_space"
    files_root = workspace / "files"
    files_root.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="writing",
    )
    built.runner._stage_deepagent_assets(files_root, thread_id="plot-surface")
    plot_root = built.runner._skill_roots_for_group(worker)[0]
    model = _BindablePlotModel(responses=[AIMessage(content="Figure job ready.")])
    backend = _backend_for_staged_skills(built.runner, files_root)
    runtime = {"backend": backend}
    agent = built.runner._load_create_deep_agent()(
        model=model,
        tools=built.runner._named_tools(_PRESENTATION_WORKER_TOOL_ALLOWLIST) if worker == "presentation_worker" else [],
        system_prompt=getattr(built.runner, f"_{worker}_prompt")(),
        skills=[plot_root],
        subagents=[
            built.runner._general_purpose_subagent(
                runtime=runtime,
                skills=[plot_root],
            )
        ],
        middleware=built.runner._catmaster_agent_middleware(
            runtime=runtime,
            skills=[plot_root],
            extra=[runtime_mod._NoDelegationToolBoundaryMiddleware()] if worker == "plot_worker" else [],
        ),
        backend=backend,
    )

    result = asyncio.run(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": "Prepare one bounded plot."}]}
        )
    )

    assert result["messages"][-1].content == "Figure job ready."
    assert _BindablePlotModel.bound_tool_names
    parent_surface = set(_BindablePlotModel.bound_tool_names[0])
    assert {
        "write_todos",
        "ls",
        "glob",
        "grep",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "execute",
    } <= parent_surface
    if worker == "plot_worker":
        assert "task" not in parent_surface
    else:
        assert {"task", "generate_figure"} <= parent_surface
    assert "web_search" not in parent_surface


def test_presentation_worker_discovers_and_reads_shared_quality_assets(
    tmp_path: Path, monkeypatch
) -> None:
    class _CapturingModel(FakeMessagesListChatModel):
        observed_messages: ClassVar[list[list[Any]]] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, messages, *args, **kwargs):
            self.observed_messages.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    built = build_specialist_runner(
        workspace=tmp_path / "workspace",
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="quality-surface",
        preferred_entrypoint="writing",
    )
    files_root = tmp_path / "workspace/files"
    built.runner._stage_deepagent_assets(files_root, thread_id="quality-surface")
    backend = _backend_for_staged_skills(built.runner, files_root)
    runtime = {"backend": backend}
    # Capture the production worker specification rather than recreating its roots.
    monkeypatch.setattr(
        built.runner, "_compiled_worker_subagent", lambda **kwargs: kwargs
    )
    spec = next(
        item for item in built.runner._writing_subagents(runtime=runtime)
        if item["name"] == "presentation_worker"
    )
    quality_root = built.runner._skill_roots_for_group("writing_quality")[0]
    assert quality_root in spec["skills"]
    assert set(built.runner._skill_roots_for_group("plot_worker")) <= set(spec["skills"])
    source_root = built.runner._skill_snapshot_root / "skills/writing_quality"
    # Skills and on-demand reference files stay readable as the source tree evolves.
    sources = sorted(source_root.rglob("*.md"))
    calls = [
        {
            "name": "read_file",
            "args": {
                "file_path": f"{quality_root}/{source.relative_to(source_root).as_posix()}",
                "limit": 1000,
            },
            "id": f"read-quality-{index}",
            "type": "tool_call",
        }
        for index, source in enumerate(sources)
    ]
    model = _CapturingModel(responses=[
        AIMessage(content="", tool_calls=calls),
        AIMessage(content="Read complete."),
    ])
    agent = built.runner._load_create_deep_agent()(
        model=model,
        tools=[],
        backend=backend,
        system_prompt=spec["system_prompt"],
        skills=spec["skills"],
        middleware=built.runner._catmaster_agent_middleware(
            runtime=runtime, skills=spec["skills"]
        ),
    )
    result = asyncio.run(agent.ainvoke({"messages": [
        {"role": "user", "content": "Read the presentation quality guidance."}
    ]}))
    assert result["messages"][-1].content == "Read complete."
    first_context = "\n".join(
        str(message.content) for message in model.observed_messages[0]
    )
    for call, source in zip(calls, sources):
        if source.name == "SKILL.md":
            assert call["args"]["file_path"] in first_context
        reply = next(
            message for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == call["id"]
        )
        assert reply.status == "success"
        first_line = next(
            (line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()),
            "",
        )
        assert first_line in str(reply.content)


def test_staged_support_assets_are_reachable_without_becoming_skills(
    tmp_path: Path, monkeypatch,
) -> None:
    workspace = tmp_path / "project_space"
    files_root = workspace / "files"
    files_root.mkdir(parents=True)
    built = build_specialist_runner(
        workspace=workspace, llm_profile=_FakeProfile(), reporter=None,
        run_control=None, project_id="proj", preferred_entrypoint="writing",
    )
    # Support assets are generic files, not a particular optional skill contract.
    repo = tmp_path / "source"
    support = repo / "skills/writing_specialist/support/core"
    support.mkdir(parents=True)
    (support / "notes.md").write_text("Shared reference content.\n", encoding="utf-8")
    monkeypatch.setattr(runtime_mod, "__file__", str(repo / "catmaster/specialists/runtime.py"))
    built.runner._stage_deepagent_assets(files_root, thread_id="support")
    backend = _backend_for_staged_skills(built.runner, files_root)
    result = backend.read("/.deepagents/skills/writing_specialist/support/core/notes.md")
    assert result.error is None
    assert result.file_data is not None
    assert "Shared reference content." in result.file_data["content"]
    staged_support = built.runner._skill_snapshot_root / "skills/writing_specialist/support"
    assert not (staged_support / "SKILL.md").exists()
    assert not (staged_support / "manifest.yaml").exists()

@pytest.mark.parametrize("native_search", [False, True])
def test_explicit_general_purpose_runtime_is_context_only_and_non_delegating(
    tmp_path: Path, native_search: bool,
) -> None:
    class _BindableFakeModel(FakeMessagesListChatModel):
        bound_tool_names: list[list[str]] = []
        observed_system_prompts: list[str] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tool_choice, kwargs)
            self.bound_tool_names.append([runtime_mod._agent_tool_name(tool) for tool in tools])
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
    from deepagents.backends import StateBackend

    backend = StateBackend()
    runtime = {"backend": backend}
    general_purpose = built.runner._general_purpose_subagent(
        runtime=runtime,
        skills=[],
    )

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
        tools=[caller_tool, *([{"type": "web_search"}] if native_search else [])],
        subagents=[general_purpose],
        backend=backend,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Delegate this bounded branch."}]}
    )

    assert result["messages"][-1].content == "parent result"
    if native_search:
        assert all(names.count("web_search") == 1 for names in model.bound_tool_names)
    assert [
        message.content
        for message in result["messages"]
        if getattr(message, "name", None) == "task"
    ] == ["child result"]
    child_tool_sets = [
        set(names)
        for names in model.bound_tool_names
        if "caller_tool" in names and "task" not in names
    ]
    assert len(child_tool_sets) == 1
    assert {"caller_tool", "read_file"} <= child_tool_sets[0]
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
                        "materials_worker",
                "ml_worker",
                "dynamics_worker",
                "orca_xtb_worker",
            ],
        ),
        ("literature_review", ["general-purpose", "litreview_worker_agent"]),
        ("writing", ["general-purpose", "writing_worker_agent", "presentation_worker", "plot_worker"]),
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: SubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
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
        runtime_context=(
            {
                "research_graph_id": bound_graph_id,
                "research_focus_node_id": "",
                "research_launch_id": "",
            }
            if entrypoint == "research"
            else None
        ),
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
        "research": ("research_specialist", "research_reasoning", "writing_quality"),
        "experiment": ("atomistic", "research_execution", "writing_quality"),
        "literature_review": (
            "litreview_agent",
            "research_execution",
            "writing_quality",
        ),
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
    assert [subagent["name"] for subagent in top_subagents] == expected_subagent_names
    for subagent in top_subagents:
        if "tools" in subagent:
            assert all(getattr(tool, "name", None) != "bash" for tool in subagent["tools"])
        if "middleware" in subagent:
            middleware_names = {type(item).__name__ for item in (subagent.get("middleware") or [])}
            if subagent["name"] in {
                "hypothesis_proposer",
                    }:
                assert {
                    "_ResearchReasoningToolBoundaryMiddleware",
                    "catmaster_nonfatal_tool_errors",
                } <= middleware_names
            else:
                assert "catmaster_nonfatal_tool_errors" in middleware_names

    subagents_by_name = {subagent["name"]: subagent for subagent in top_subagents}

    def _assert_explicit_general_purpose(owner_kwargs: dict[str, Any]) -> None:
        specs = [subagent for subagent in owner_kwargs.get("subagents") or []]
        general_purpose = [spec for spec in specs if spec["name"] == "general-purpose"]
        assert len(general_purpose) == 1
        spec = general_purpose[0]
        if owner_kwargs["name"] == "litreview_agent":
            assert spec["model"] == {"model": "literature_worker-model"}
            assert {tool.name for tool in spec["tools"]} == (
                _LITREVIEW_LOCAL_TOOL_ALLOWLIST
                | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
            )
        else:
            assert "tools" not in spec
            assert "model" not in spec
        assert spec["skills"] == list(owner_kwargs.get("skills") or [])
        assert "CatMaster's general-purpose context worker" in spec["system_prompt"]
        assert "You have no subagents and must not transfer the task onward" in spec["system_prompt"]
        assert "Use workspace-relative paths and the paths supplied in the brief" in spec["system_prompt"]
        assert "Treat `/` only as the workspace virtual root" not in spec["system_prompt"]
        assert "never use leading-slash workspace paths" not in spec["system_prompt"]
        assert runtime_mod.SpecialistRunner._hash_policy() in spec["system_prompt"]
        assert runtime_mod.SpecialistRunner._contract_policy() in spec["system_prompt"]
        assert "current lane" not in spec["description"]
        assert "current lane" not in spec["system_prompt"]
        assert "another lane" not in spec["system_prompt"]
        assert "Workspace script header policy" not in spec["system_prompt"]
        assert "topic-centric layout" not in spec["system_prompt"]
        assert "Research Graph contract" not in spec["system_prompt"]
        assert "browser use and full-text acquisition" not in spec["system_prompt"]
        assert "registered managed execution" not in spec["system_prompt"]
        middleware_names = {type(item).__name__ for item in spec["middleware"]}
        assert {"FilesystemMiddleware", "catmaster_nonfatal_tool_errors"} <= middleware_names
        assert "BoundedDocumentReadMiddleware" in middleware_names
        assert "DocumentAccessMiddleware" not in middleware_names
        filesystem_middleware = next(
            item for item in spec["middleware"] if type(item).__name__ == "FilesystemMiddleware"
        )
        assert (
            runtime_mod.SpecialistRunner._artifact_persistence_policy()
            in filesystem_middleware._custom_system_prompt
        )

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
            (_RESEARCH_TOOL_ALLOWLIST - {"stage_research_plan"})
            | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
            | {"manage_effective_skills", "notify_progress"}
        )
        assert "Research Graph contract" in agent_kwargs["system_prompt"]
        assert "Research Kernel" not in agent_kwargs["system_prompt"]
        assert runtime_mod.SpecialistRunner._research_reporting_contract() in agent_kwargs["system_prompt"]
        assert "current task interface is the source of truth" in agent_kwargs["system_prompt"]
        assert "exposed literature capability" in agent_kwargs["system_prompt"]
        for invisible_name in (
            "litreview_agent",
            "hypothesis_proposer",
                "experiment_specialist",
            "writing_specialist",
            "peer_review_specialist",
        ):
            assert invisible_name not in agent_kwargs["system_prompt"]
        assert "experiment_evaluator" not in agent_kwargs["system_prompt"]
        assert "Record only hypothesis effects that the evidence actually addresses" in agent_kwargs["system_prompt"]
        assert "metadata_agent" not in agent_kwargs["system_prompt"]
        assert runtime_mod.SpecialistRunner._report_packet_policy() in agent_kwargs["system_prompt"]
        assert "Cross-layer computation brief discipline" in agent_kwargs["system_prompt"]
        assert "do not by themselves authorize a tighter numerical standard" in agent_kwargs["system_prompt"]
        assert "delegate owns implementation-equivalent corrections inside that stage" in agent_kwargs["system_prompt"]
        assert "Delegation-failure routing contract" in agent_kwargs["system_prompt"]
        assert "Do not start a formal peer-review process by default" in agent_kwargs["system_prompt"]
        assert "publication-level, submission-ready, peer-review-ready" in agent_kwargs["system_prompt"]
        assert "Supply the canonical workspace-relative manuscript PDF" in agent_kwargs["system_prompt"]
        assert "read any returned full review memo" in agent_kwargs["system_prompt"]
        # Role wording is editable; the concrete specialist bindings below
        # establish the coordinator's capability surface.
        assert "weak evidence" not in agent_kwargs["system_prompt"]
        assert "current shared workspace makes parallel subagents unsafe" not in agent_kwargs["system_prompt"]
        assert "runnable" in subagents_by_name["experiment_specialist"]
        assert "runnable" in subagents_by_name["writing_specialist"]
        assert "runnable" in subagents_by_name["peer_review_specialist"]
        assert "hypothesis_proposer" not in subagents_by_name
        assert "research_challenger" not in subagents_by_name
        assert "evidence_judge" not in subagents_by_name

        experiment_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "experiment_specialist"]
        assert experiment_agents, "expected nested experiment specialist to be created"
        experiment_agent_kwargs = experiment_agents[0]
        assert experiment_agent_kwargs["model"] == {"model": "director-model"}
        assert {tool.name for tool in experiment_agent_kwargs["tools"]} == (
            _EXPERIMENT_SPECIALIST_BASE_TOOL_ALLOWLIST
            | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        assert not (
            _RESEARCH_TOOL_ALLOWLIST
            & {tool.name for tool in experiment_agent_kwargs["tools"]}
        )
        assert "mace_neb_batch" not in {tool.name for tool in experiment_agent_kwargs["tools"]}
        _assert_native_skill_groups(
            experiment_agent_kwargs,
            "atomistic",
            "research_execution",
            "writing_quality",
        )
        _assert_native_memory(experiment_agent_kwargs)
        assert [subagent["name"] for subagent in experiment_agent_kwargs["subagents"]] == [
            "general-purpose",
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
        assert [subagent["name"] for subagent in writing_agent_kwargs["subagents"]] == [
            "general-purpose",
            "writing_worker_agent",
            "presentation_worker",
            "plot_worker",
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
        assert [subagent["name"] for subagent in peer_review_agent_kwargs["subagents"]] == [
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
        _assert_native_skill_groups(
            litreview_agent_kwargs,
            "litreview_agent",
            "research_execution",
            "writing_quality",
        )
        _assert_native_memory(litreview_agent_kwargs)
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in litreview_agent_kwargs["middleware"])
        assert [subagent["name"] for subagent in litreview_agent_kwargs["subagents"]] == [
            "general-purpose",
            "litreview_worker_agent",
        ]
        litreview_worker_kwargs = next(
            subagent
            for subagent in litreview_agent_kwargs["subagents"]
            if subagent["name"] == "litreview_worker_agent"
        )
        assert litreview_worker_kwargs["model"] == {"model": "literature_worker-model"}
        assert {tool.name for tool in litreview_worker_kwargs["tools"]} == (
            _LITREVIEW_LOCAL_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        assert "runnable" not in litreview_worker_kwargs
        assert "subagents" not in litreview_worker_kwargs
        assert "Do not delegate or broaden it into a full review" in litreview_worker_kwargs["system_prompt"]
        assert "requested scope" in litreview_agent_kwargs["system_prompt"]
        assert "50-60" not in litreview_agent_kwargs["system_prompt"]
        assert "metadata_agent" not in litreview_agent_kwargs["system_prompt"]
        assert "literature_agent" not in litreview_agent_kwargs["system_prompt"]
    elif entrypoint == "literature_review":
        assert {tool.name for tool in agent_kwargs["tools"]} == (
            _LITREVIEW_LOCAL_TOOL_ALLOWLIST
            | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
            | {"notify_progress"}
        )
        _assert_native_skill_groups(
            agent_kwargs,
            "litreview_agent",
            "research_execution",
            "writing_quality",
        )
        assert "Own the review question" in agent_kwargs["system_prompt"]
        assert "Use each source only for what it supports" in agent_kwargs["system_prompt"]
        assert "methods, conditions, quantitative comparisons" in agent_kwargs["system_prompt"]
        assert "Distinguish reported results from your synthesis" in agent_kwargs["system_prompt"]
        assert "at most one" not in agent_kwargs["system_prompt"]
        assert "full-text access as unknown until tested" not in agent_kwargs["system_prompt"]
        assert "acquire_literature_source" not in agent_kwargs["system_prompt"]
        assert "finalize_citations" not in agent_kwargs["system_prompt"]
        assert "general-purpose" not in agent_kwargs["system_prompt"]
        assert "litreview_worker_agent" in agent_kwargs["system_prompt"]
        assert "50-60" not in agent_kwargs["system_prompt"]
        assert "Do not perform computational execution" in agent_kwargs["system_prompt"]
        assert "Research Graph writeback is a post-result decision" in agent_kwargs["system_prompt"]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in agent_kwargs["middleware"])
        assert [subagent["name"] for subagent in agent_kwargs["subagents"]] == [
            "general-purpose",
            "litreview_worker_agent",
        ]
        litreview_worker_kwargs = subagents_by_name["litreview_worker_agent"]
        assert litreview_worker_kwargs["model"] == {"model": "literature_worker-model"}
        assert {tool.name for tool in litreview_worker_kwargs["tools"]} == (
            _LITREVIEW_LOCAL_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        assert "runnable" not in litreview_worker_kwargs
        assert "subagents" not in litreview_worker_kwargs
        assert "Do not delegate or broaden it into a full review" in litreview_worker_kwargs["system_prompt"]
        assert not _created_agents_named("literature_agent")
        assert not _created_agents_named("metadata_agent")
    elif entrypoint == "experiment":
        materials_worker_kwargs = _find_created_agent("materials_worker")
        ml_worker_kwargs = _find_created_agent("ml_worker")
        dynamics_worker_kwargs = _find_created_agent("dynamics_worker")
        orca_worker_kwargs = _find_created_agent("orca_xtb_worker")
        for worker_kwargs in (materials_worker_kwargs, ml_worker_kwargs, dynamics_worker_kwargs, orca_worker_kwargs):
            assert worker_kwargs["model"] == {"model": "task_runner-model"}
            assert runtime_mod.SpecialistRunner._worker_execution_adaptation_policy() in worker_kwargs["system_prompt"]
            assert "Delegation-failure routing contract" in worker_kwargs["system_prompt"]
            assert "Report an internal capability mismatch in the final result" in worker_kwargs["system_prompt"]
        assert "runnable" in subagents_by_name["materials_worker"]
        assert "runnable" in subagents_by_name["ml_worker"]
        assert "runnable" in subagents_by_name["dynamics_worker"]
        assert "runnable" in subagents_by_name["orca_xtb_worker"]
        assert {tool.name for tool in materials_worker_kwargs["tools"]} == (_MATERIALS_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(materials_worker_kwargs, "materials_worker", "atomistic", "execution")
        materials_general_purpose_kwargs = next(
            subagent
            for subagent in materials_worker_kwargs["subagents"]
            if subagent["name"] == "general-purpose"
        )
        _assert_native_skill_groups(
            materials_general_purpose_kwargs,
            "materials_worker",
            "atomistic",
            "execution",
        )
        assert {tool.name for tool in dynamics_worker_kwargs["tools"]} == (_DYNAMICS_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(dynamics_worker_kwargs, "dynamics_worker", "atomistic", "execution")
        assert {tool.name for tool in ml_worker_kwargs["tools"]} == (_ML_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(ml_worker_kwargs, "ml_worker", "execution")
        assert {tool.name for tool in orca_worker_kwargs["tools"]} == (_ORCA_XTB_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(orca_worker_kwargs, "orca_xtb_worker", "atomistic", "execution")
        assert {tool.name for tool in agent_kwargs["tools"]} == (
            _EXPERIMENT_SPECIALIST_BASE_TOOL_ALLOWLIST
            | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
            | {"manage_effective_skills"}
        )
        _assert_native_skill_groups(
            agent_kwargs,
            "atomistic",
            "research_execution",
            "writing_quality",
        )
        assert {"mp_search_materials", "mp_download_structure"} <= {tool.name for tool in agent_kwargs["tools"]}
        assert "mace_neb_batch" not in {tool.name for tool in agent_kwargs["tools"]}
        assert "Memory is read-only in this context" in materials_worker_kwargs["system_prompt"]
        assert "Instruction context files" not in materials_worker_kwargs["system_prompt"]
        assert "dataset/model lifecycle tasks" in ml_worker_kwargs["system_prompt"]
        assert "default role is coordination, dispatch, and decision-making across the experiment lane" in agent_kwargs["system_prompt"]
        assert "Research Graph writeback is a post-result decision" in agent_kwargs["system_prompt"]
        assert "Keep direct work in the specialist thread minimal and coordination-oriented" in agent_kwargs["system_prompt"]
        assert "Route by the current working artifact" in agent_kwargs["system_prompt"]
        assert "When a request clearly falls into one of those worker-owned domains, delegate first instead of doing the domain work yourself." in agent_kwargs["system_prompt"]
        assert "use `orca_xtb_worker` for molecular or cluster quantum-chemistry work" in agent_kwargs["system_prompt"]
        assert "bounded by scientific scope, authorization, and cost" in agent_kwargs["system_prompt"]
        assert "not by a fixed tool sequence or a single submission" in agent_kwargs["system_prompt"]
        assert "split it at genuine scientific decision boundaries" in agent_kwargs["system_prompt"]
        assert "Cross-layer computation brief discipline" in agent_kwargs["system_prompt"]
        assert "Delegation-failure routing contract" in agent_kwargs["system_prompt"]
        assert agent_kwargs["system_prompt"].count(runtime_mod.SpecialistRunner._delegation_failure_routing_policy()) == 1
        assert "Do not personally absorb worker-owned tasks just because your own direct tool surface appears sufficient" in agent_kwargs["system_prompt"]
        assert "Only do the implementation directly in the specialist thread when no available worker matches the task" in agent_kwargs["system_prompt"]
        assert runtime_mod.SpecialistRunner._experiment_layered_capability_visibility_policy() in agent_kwargs["system_prompt"]
        assert "Delegate domain-owned work to the proper specialized subagent first." in agent_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in agent_kwargs["system_prompt"]
        assert "Use the built-in `read_file(file_path=...)` tool for stored PDF, DOCX, XLSX, PPTX" in agent_kwargs["system_prompt"]
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
        assert "MLFF MD, restart, and trajectory-health execution are outside this task's ownership" in materials_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in materials_worker_kwargs["system_prompt"]
        assert "Use the built-in `read_file(file_path=...)` tool for stored PDF, DOCX, XLSX, PPTX" in materials_worker_kwargs["system_prompt"]
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
        assert "do not require a CPU reference run" in dynamics_worker_kwargs["system_prompt"]
        assert "not a scientific NO-GO or a global experiment-wide recovery quota" in dynamics_worker_kwargs["system_prompt"]
        assert "Start here when the primary artifact is a curated dataset" in ml_worker_kwargs["system_prompt"]
        assert "Prefer using libraries already available in the environment and reusable workspace code" in ml_worker_kwargs["system_prompt"]
        assert "Common libraries already available here include `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `joblib`, and `matminer`" in ml_worker_kwargs["system_prompt"]
        assert "If a handy Python package is still missing for a bounded local step" in ml_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only to isolate one self-contained, context-heavy branch described by a complete task brief." in ml_worker_kwargs["system_prompt"]
        assert "Use the built-in `read_file(file_path=...)` tool for stored PDF, DOCX, XLSX, PPTX" in ml_worker_kwargs["system_prompt"]
        assert "It inherits the caller's direct tools and staged skills, cannot delegate, and returns one handoff." in ml_worker_kwargs["system_prompt"]
        assert "registered managed execution in this worker is authoritative" in ml_worker_kwargs["system_prompt"]
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
        assert {tool.name for tool in agent_kwargs["tools"]} == (
            (
                _WRITING_TOOL_ALLOWLIST
                - {"query_research_graph_sql", "set_research_graph_focus"}
            )
            | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
            | {"manage_effective_skills"}
        )
        _assert_native_skill_groups(agent_kwargs, "writing_specialist", "writing_quality")
        assert "compile_text" not in {tool.name for tool in agent_kwargs["tools"]}
        writing_worker_kwargs = _find_created_agent("writing_worker_agent")
        plot_worker_kwargs = _find_created_agent("plot_worker")
        presentation_kwargs = _find_created_agent("presentation_worker")
        assert presentation_kwargs["model"] == {"model": "presentation_worker-model"}
        assert "runnable" in subagents_by_name["presentation_worker"]
        assert {tool.name for tool in presentation_kwargs["tools"]} == (
            _PRESENTATION_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        _assert_native_skill_groups(presentation_kwargs, "presentation_worker", "writing_quality", "plot_worker")
        assert writing_worker_kwargs["model"] == {"model": "section_writer-model"}
        assert plot_worker_kwargs["model"] == {"model": "plot_worker-model"}
        assert "runnable" in subagents_by_name["writing_worker_agent"]
        assert "runnable" in subagents_by_name["plot_worker"]
        assert {tool.name for tool in writing_worker_kwargs["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(writing_worker_kwargs, "writing_specialist", "writing_quality")
        assert {tool.name for tool in plot_worker_kwargs["tools"]} == _PLOT_WORKER_TOOL_ALLOWLIST
        _assert_native_skill_groups(plot_worker_kwargs, "plot_worker")
        assert "generate_figure" in {tool.name for tool in writing_worker_kwargs["tools"]}
        assert "generate_figure" not in {tool.name for tool in plot_worker_kwargs["tools"]}
    else:
        assert {tool.name for tool in agent_kwargs["tools"]} == (
            {"peer_review_request", "manage_effective_skills"}
            | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES
        )
        _assert_native_skill_groups(agent_kwargs, "writing_specialist", "writing_quality")
        assert "Act like a journal editor coordinating external peer review" in agent_kwargs["system_prompt"]
        assert "Reviewer Comments" in agent_kwargs["system_prompt"]
        assert "save the full review as one durable workspace markdown memo" in agent_kwargs["system_prompt"]
        assert "do not compress away the editor comment or reviewer comment sections" in agent_kwargs["system_prompt"]
        assert "peer_review_worker_agent" in subagents_by_name
        peer_review_worker_kwargs = _find_created_agent("peer_review_worker_agent")
        assert peer_review_worker_kwargs["model"] == {"model": "task_runner-model"}
        assert "runnable" in subagents_by_name["peer_review_worker_agent"]
        assert {tool.name for tool in peer_review_worker_kwargs["tools"]} == ({"peer_review_request"} | _DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)
        _assert_native_skill_groups(peer_review_worker_kwargs, "writing_specialist", "writing_quality")
        assert "dedicated peer-review request capability on that PDF exactly once" in peer_review_worker_kwargs["system_prompt"]

    assert agent_kwargs["memory"][0] == "/.deepagents/AGENTS.md"
    snapshot_root = built.runner._skill_snapshot_root
    assert snapshot_root is not None
    staged_agents = snapshot_root / "AGENTS.md"
    staged_materials = snapshot_root / "skills" / "materials_worker"
    staged_atomistic = snapshot_root / "skills" / "atomistic"
    staged_writing = snapshot_root / "skills" / "writing_specialist"
    staged_researcher = snapshot_root / "skills" / "research_specialist"
    staged_reasoning = snapshot_root / "skills" / "research_reasoning"
    staged_literature = snapshot_root / "skills" / "litreview_agent"
    staged_research_execution = snapshot_root / "skills" / "research_execution"
    staged_writing_quality = snapshot_root / "skills" / "writing_quality"
    staged_plot_worker = snapshot_root / "skills" / "plot_worker"
    staged_quantum_chemistry = snapshot_root / "skills" / "orca_xtb_worker"
    staged_execution = snapshot_root / "skills" / "execution"
    assert staged_agents.read_text(encoding="utf-8") == "Project-level instructions."
    assert staged_materials.is_dir()
    assert staged_atomistic.is_dir()
    assert staged_writing.is_dir()
    assert staged_researcher.is_dir()
    assert staged_reasoning.is_dir()
    assert staged_literature.is_dir()
    assert staged_writing_quality.is_dir()
    assert staged_plot_worker.is_dir()
    assert staged_research_execution.is_dir()
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
    assert _skill_names(staged_atomistic) == {
        "atomic-structure-validation-and-recovery",
        "constraint-guided-atomic-assembly",
        "literature-figure-guided-structure-reconstruction",
    }
    assert _skill_names(staged_machine_learning) == _skill_names(repo_root / "skills" / "ml_worker")
    assert _skill_names(staged_quantum_chemistry) == _skill_names(repo_root / "skills" / "orca_xtb_worker")
    assert _skill_names(staged_execution) == _skill_names(repo_root / "skills" / "execution")
    staged_research_names = _skill_names(staged_researcher)
    assert staged_research_names == _skill_names(repo_root / "skills" / "research_specialist")
    assert _skill_names(staged_reasoning) == _skill_names(repo_root / "skills" / "research_reasoning")
    assert _skill_names(staged_literature) == _skill_names(repo_root / "skills" / "litreview_agent")
    assert _skill_names(staged_research_execution) == {
        "research-graph-writeback"
    }
    staged_writing_names = _skill_names(staged_writing)
    assert staged_writing_names == _skill_names(repo_root / "skills" / "writing_specialist")
    assert _skill_names(staged_writing_quality) == _skill_names(repo_root / "skills" / "writing_quality")
    for source in (repo_root / "skills" / "writing_quality").rglob("*"):
        if source.is_file():
            relative = source.relative_to(repo_root / "skills" / "writing_quality")
            assert (staged_writing_quality / relative).read_bytes() == source.read_bytes()
    assert _skill_names(staged_plot_worker) == {"publication-data-plotting"}
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: SubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: SubAgent))
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: SubAgent))
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: SubAgent))
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: CompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: SubAgent))
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
