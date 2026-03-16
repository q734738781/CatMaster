from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import catmaster.specialists.runtime as runtime_mod
from catmaster.specialists.runtime import (
    RUN_STATE_FILE,
    _COMPILE_AGENT_TOOL_ALLOWLIST,
    _EXPERIMENT_TOOL_ALLOWLIST,
    _LITERATURE_AGENT_TOOL_ALLOWLIST,
    _RESEARCH_TOOL_ALLOWLIST,
    _TASK_WORKER_TOOL_ALLOWLIST,
    _WRITING_WORKER_TOOL_ALLOWLIST,
    _WRITING_TOOL_ALLOWLIST,
    build_specialist_runner,
)
from catmaster.runtime.usage_stats import load_usage_summary
from catmaster.runtime.artifact_callback import UIEventHandler
from catmaster.tools.registry import get_tool_registry


class _FakeProfile:
    def config_for_role(self, role: str) -> SimpleNamespace:
        return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)


class _FakeToolStrategy:
    def __init__(self, schema, handle_errors: bool = False) -> None:
        self.schema = schema
        self.handle_errors = handle_errors


class _FakeSubAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeDeepAgent:
    def __init__(self, *, kwargs: dict) -> None:
        self.kwargs = kwargs

    async def ainvoke(self, payload, config=None):
        _ = config
        assert payload["messages"][0]["role"] == "user"
        name = self.kwargs["name"]
        if name == "research_specialist":
            content = "## Summary\nresearch summary\n\n## Facts\n- grounded by literature agent when needed\n\n## Files\n- reports/research.md"
        elif name == "writing_specialist":
            content = "## Summary\nwriting summary\n\n## Facts\n- manuscript draft updated\n\n## Files\n- drafts/report.md"
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


def test_real_registry_covers_specialist_allowlists() -> None:
    registry = get_tool_registry()
    registered = set(registry.tools)

    assert "write_note" not in registered
    assert _EXPERIMENT_TOOL_ALLOWLIST <= registered
    assert _RESEARCH_TOOL_ALLOWLIST <= registered
    assert _WRITING_TOOL_ALLOWLIST <= registered
    assert _TASK_WORKER_TOOL_ALLOWLIST <= registered
    assert _LITERATURE_AGENT_TOOL_ALLOWLIST <= registered
    assert _WRITING_WORKER_TOOL_ALLOWLIST <= registered
    assert _COMPILE_AGENT_TOOL_ALLOWLIST <= registered
    assert "bash" not in _EXPERIMENT_TOOL_ALLOWLIST
    assert "bash" not in _RESEARCH_TOOL_ALLOWLIST
    assert "bash" not in _WRITING_TOOL_ALLOWLIST


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

    callbacks = built.runner._langchain_callbacks(usage_handler=None)
    assert any(isinstance(callback, UIEventHandler) for callback in callbacks)


@pytest.mark.parametrize(
    ("entrypoint", "expected_skills", "expected_subagent_names"),
    [
        ("research", ["/.deepagents/skills/experiment", "/.deepagents/skills/writing"], ["experiment_specialist", "writing_specialist", "literature_agent"]),
        ("experiment", ["/.deepagents/skills/experiment"], ["task_worker_agent", "literature_agent"]),
        ("writing", ["/.deepagents/skills/writing"], ["writing_worker_agent", "compile_agent"]),
    ],
)
def test_three_specialist_lanes_start_with_staged_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    expected_skills: list[str],
    expected_subagent_names: list[str],
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("Project-level instructions.", encoding="utf-8")

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
        )
    )

    assert result["status"] == "done"
    assert created_agents, "expected create_deep_agent to be called"
    agent_kwargs = created_agents[-1]
    assert agent_kwargs["name"] == f"{entrypoint}_specialist"
    assert agent_kwargs["skills"] == expected_skills
    assert agent_kwargs["memory"] == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]
    assert "Never store transient task requests" in agent_kwargs["system_prompt"]
    assert all(getattr(tool, "name", None) != "bash" for tool in agent_kwargs["tools"])
    assert [subagent.kwargs["name"] for subagent in agent_kwargs["subagents"]] == expected_subagent_names
    for subagent in agent_kwargs["subagents"]:
        assert all(getattr(tool, "name", None) != "bash" for tool in subagent.kwargs["tools"])
        assert "middleware" in subagent.kwargs

    subagents_by_name = {subagent.kwargs["name"]: subagent.kwargs for subagent in agent_kwargs["subagents"]}
    if entrypoint == "research":
        assert {tool.name for tool in subagents_by_name["literature_agent"]["tools"]} == _LITERATURE_AGENT_TOOL_ALLOWLIST
    elif entrypoint == "experiment":
        assert {tool.name for tool in subagents_by_name["task_worker_agent"]["tools"]} == _TASK_WORKER_TOOL_ALLOWLIST
        assert {tool.name for tool in subagents_by_name["literature_agent"]["tools"]} == _LITERATURE_AGENT_TOOL_ALLOWLIST
    else:
        assert {tool.name for tool in subagents_by_name["writing_worker_agent"]["tools"]} == _WRITING_WORKER_TOOL_ALLOWLIST
        assert {tool.name for tool in subagents_by_name["compile_agent"]["tools"]} == _COMPILE_AGENT_TOOL_ALLOWLIST

    staged_agents = workspace / "files" / ".deepagents" / "AGENTS.md"
    staged_experiment = workspace / "files" / ".deepagents" / "skills" / "experiment"
    staged_writing = workspace / "files" / ".deepagents" / "skills" / "writing"
    assert staged_agents.read_text(encoding="utf-8") == "Project-level instructions."
    assert staged_experiment.is_dir()
    assert staged_writing.is_dir()

    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["entrypoint"] == entrypoint
    assert run_state["status"] == "done"
    assert run_state["summary"]
    assert isinstance(run_state.get("facts"), list)
    usage_summary = load_usage_summary(built.run_context.run_dir)
    assert usage_summary["source"] == "langchain_usage_metadata"
    assert usage_summary["input_tokens"] == 123
    assert usage_summary["input_cached_tokens"] == 80
    assert usage_summary["output_tokens"] == 17
    assert usage_summary["reasoning_tokens"] == 5
    assert usage_summary["calls"] == 2
