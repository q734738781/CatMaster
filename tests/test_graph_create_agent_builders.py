from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents import graph
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.run_context import RunContext


class _DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyModel:
    pass


def _memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    return store


def test_builders_use_create_agent_and_system_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[dict] = []

    def fake_create_agent(**kwargs):
        captured.append(dict(kwargs))
        return {"ok": True}

    class _FakeAgentMiddleware:
        pass

    class _FakeToolStrategy:
        def __init__(self, schema, handle_errors=False):
            self.schema = schema
            self.handle_errors = handle_errors

    monkeypatch.setitem(sys.modules, "langchain.agents", types.SimpleNamespace(create_agent=fake_create_agent))
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents.structured_output",
        types.SimpleNamespace(ToolStrategy=_FakeToolStrategy),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents.middleware",
        types.SimpleNamespace(
            AgentMiddleware=_FakeAgentMiddleware,
        ),
    )

    model = _DummyModel()
    bash = _DummyTool("bash")

    graph._build_proposal_agent(model, [bash])
    graph._build_director_agent(model, [bash])
    graph._build_fast_director_agent(model, [bash])
    graph._build_task_runner_agent(model, [bash], _memory_store(tmp_path))
    graph._build_memory_patcher_agent(model, [bash])

    assert len(captured) == 5

    proposal_call = captured[0]
    director_call = captured[1]
    fast_director_call = captured[2]
    task_call = captured[3]
    memory_call = captured[4]

    proposal_format = proposal_call.get("response_format")
    director_format = director_call.get("response_format")
    fast_director_format = fast_director_call.get("response_format")
    task_format = task_call.get("response_format")
    memory_format = memory_call.get("response_format")

    assert proposal_call.get("system_prompt") == graph.PROPOSAL_SYSTEM_PROMPT
    assert isinstance(proposal_format, _FakeToolStrategy)
    assert proposal_format.schema.__name__ == "ProposalOutput"
    assert proposal_format.handle_errors is False
    assert "prompt" not in proposal_call
    assert isinstance(proposal_call.get("middleware"), list)
    assert len(proposal_call.get("middleware") or []) == 2

    assert director_call.get("system_prompt") == graph.DIRECTOR_SYSTEM_PROMPT
    assert isinstance(director_format, _FakeToolStrategy)
    assert director_format.schema.__name__ == "DirectorOutput"
    assert director_format.handle_errors is False
    assert isinstance(director_call.get("middleware"), list)
    assert len(director_call.get("middleware") or []) == 2

    assert fast_director_call.get("system_prompt") == graph.FAST_DIRECTOR_SYSTEM_PROMPT
    assert isinstance(fast_director_format, _FakeToolStrategy)
    assert fast_director_format.schema.__name__ == "FastDirectorOutput"
    assert fast_director_format.handle_errors is False
    assert isinstance(fast_director_call.get("middleware"), list)
    assert len(fast_director_call.get("middleware") or []) == 2

    assert task_call.get("system_prompt") == graph.TASK_RUNNER_SYSTEM_PROMPT
    assert isinstance(task_format, _FakeToolStrategy)
    assert task_format.schema.__name__ == "TaskOutput"
    assert task_format.handle_errors is False
    assert isinstance(task_call.get("middleware"), list)
    assert len(task_call.get("middleware") or []) == 2

    assert memory_call.get("system_prompt") == graph.MEMORY_PATCHER_SYSTEM_PROMPT
    assert isinstance(memory_format, _FakeToolStrategy)
    assert memory_format.schema.__name__ == "MemoryPatchOutput"
    assert memory_format.handle_errors is False
    assert isinstance(memory_call.get("middleware"), list)
    assert len(memory_call.get("middleware") or []) == 2


def test_graph_runner_uses_fallback_skill_guides_when_skills_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_build_runtime_tool_surface(**kwargs):
        _ = kwargs
        return graph.RuntimeToolSurface(
            proposal_tools=[],
            director_tools=[],
            fast_director_tools=[],
            task_tools=[],
            task_runner_capability_guide_full="full guide",
            task_runner_capability_guide_short="short guide",
        )

    def fake_build_standard_graph(**kwargs):
        captured["proposal"] = kwargs["proposal_execution_context_guide"]
        captured["director"] = kwargs["director_execution_context_guide"]
        return object()

    async def fake_ainvoke_loop(self, compiled, initial_state, config, workspace, lane="standard"):
        _ = (self, compiled, initial_state, config, workspace, lane)
        return {"tasks": [], "observations": [], "summary": "", "final_answer": "", "status": "done"}

    monkeypatch.setattr(graph, "build_runtime_tool_surface", fake_build_runtime_tool_surface)
    monkeypatch.setattr(graph, "build_standard_graph", fake_build_standard_graph)
    monkeypatch.setattr(graph.GraphRunner, "_ainvoke_loop", fake_ainvoke_loop)
    monkeypatch.setattr(graph.GraphRunner, "_initialize_memory_goal", lambda self, user_request: None)

    run_context = RunContext.create(
        workspace=tmp_path,
        model_name="test-model",
    )
    runner = graph.GraphRunner(
        task_runner_model=_DummyModel(),
        proposal_model=_DummyModel(),
        director_model=_DummyModel(),
        memory_patch_model=_DummyModel(),
        memory_store=_memory_store(tmp_path),
        run_context=run_context,
        skills_runtime=None,
    )

    asyncio.run(runner.arun("test request", lane="standard", proposal_review=False))

    assert captured["proposal"] == graph.render_proposal_skill_guide([])
    assert captured["director"] == graph.render_director_skill_guide([])


def test_graph_runner_uses_fallback_fast_skill_guide_when_skills_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_build_runtime_tool_surface(**kwargs):
        _ = kwargs
        return graph.RuntimeToolSurface(
            proposal_tools=[],
            director_tools=[],
            fast_director_tools=[],
            task_tools=[],
            task_runner_capability_guide_full="full guide",
            task_runner_capability_guide_short="short guide",
        )

    def fake_build_fast_graph(**kwargs):
        captured["fast"] = kwargs["fast_director_execution_context_guide"]
        return object()

    async def fake_ainvoke_loop(self, compiled, initial_state, config, workspace, lane="standard"):
        _ = (self, compiled, initial_state, config, workspace, lane)
        return {"tasks": [], "observations": [], "summary": "", "final_answer": "", "status": "done"}

    monkeypatch.setattr(graph, "build_runtime_tool_surface", fake_build_runtime_tool_surface)
    monkeypatch.setattr(graph, "build_fast_graph", fake_build_fast_graph)
    monkeypatch.setattr(graph.GraphRunner, "_ainvoke_loop", fake_ainvoke_loop)
    monkeypatch.setattr(graph.GraphRunner, "_initialize_memory_goal", lambda self, user_request: None)

    run_context = RunContext.create(
        workspace=tmp_path,
        model_name="test-model",
    )
    runner = graph.GraphRunner(
        task_runner_model=_DummyModel(),
        proposal_model=_DummyModel(),
        director_model=_DummyModel(),
        memory_patch_model=_DummyModel(),
        memory_store=_memory_store(tmp_path),
        run_context=run_context,
        skills_runtime=None,
    )

    asyncio.run(runner.arun("test request", lane="fast", proposal_review=False))

    assert captured["fast"] == graph.render_fast_director_skill_guide([])
