from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents import graph
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.tool_surface import RuntimeToolSurface


class _DummyModel:
    pass


class _FakeStateGraph:
    def __init__(self, _state_type):
        self.nodes = {}
        self.edges = []
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def compile(self, **kwargs):
        _ = kwargs
        return self


def _memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    return store


def test_build_standard_graph_respects_child_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(graph, "StateGraph", _FakeStateGraph)
    monkeypatch.setattr(graph, "_build_proposal_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_director_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_task_runner_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_memory_patcher_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_role_middleware", lambda **kwargs: [])

    compiled = graph.build_standard_graph(
        task_runner_model=_DummyModel(),
        proposal_model=_DummyModel(),
        director_model=_DummyModel(),
        memory_patch_model=_DummyModel(),
        memory_store=_memory_store(tmp_path),
        proposal_tools=[],
        director_tools=[],
        task_tools=[],
        memory_tools=[],
        proposal_execution_context_guide="guide",
        allow_human_intervention=False,
        allow_memory_patch=False,
    )

    run_task_node = compiled.nodes["run_task"]
    run_director_node = compiled.nodes["run_director"]
    assert run_task_node.keywords["intervention_goto"] == "run_director"
    assert run_director_node.keywords["allow_memory_patch"] is False


def test_build_fast_graph_respects_memory_patch_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(graph, "StateGraph", _FakeStateGraph)
    monkeypatch.setattr(graph, "_build_fast_director_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_task_runner_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_memory_patcher_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(graph, "_build_role_middleware", lambda **kwargs: [])

    compiled = graph.build_fast_graph(
        task_runner_model=_DummyModel(),
        director_model=_DummyModel(),
        memory_patch_model=_DummyModel(),
        memory_store=_memory_store(tmp_path),
        director_tools=[],
        task_tools=[],
        memory_tools=[],
        fast_director_execution_context_guide="guide",
        allow_memory_patch=False,
    )

    run_fast_director_node = compiled.nodes["run_fast_director"]
    assert run_fast_director_node.keywords["allow_memory_patch"] is False


def test_graph_runner_child_policy_skips_goal_init_history_and_literature_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {"init_goal": 0, "history": 0, "include_literature_tool": True}

    def fake_build_runtime_tool_surface(**kwargs):
        calls["include_literature_tool"] = kwargs["include_literature_tool"]
        return RuntimeToolSurface([], [], [], [], "full", "short")

    def fake_build_standard_graph(**kwargs):
        calls["allow_human_intervention"] = kwargs["allow_human_intervention"]
        calls["allow_memory_patch"] = kwargs["allow_memory_patch"]
        return object()

    async def fake_ainvoke_loop(self, compiled, initial_state, config, workspace, lane="standard"):
        _ = (self, compiled, initial_state, config, workspace, lane)
        return {"tasks": [], "observations": [], "summary": "ok", "status": "done"}

    class _HistoryReader:
        async def aload_context(self, **kwargs):
            _ = kwargs
            calls["history"] = int(calls["history"]) + 1
            raise AssertionError("history should not be called")

    monkeypatch.setattr(graph, "build_runtime_tool_surface", fake_build_runtime_tool_surface)
    monkeypatch.setattr(graph, "build_standard_graph", fake_build_standard_graph)
    monkeypatch.setattr(graph.GraphRunner, "_ainvoke_loop", fake_ainvoke_loop)
    monkeypatch.setattr(
        graph.GraphRunner,
        "_initialize_memory_goal",
        lambda self, user_request: calls.__setitem__("init_goal", int(calls["init_goal"]) + 1),
    )

    run_context = RunContext.create(workspace=tmp_path, model_name="test-model")
    runner = graph.GraphRunner(
        task_runner_model=_DummyModel(),
        proposal_model=_DummyModel(),
        director_model=_DummyModel(),
        memory_patch_model=_DummyModel(),
        memory_store=_memory_store(tmp_path),
        run_context=run_context,
        history_reader=_HistoryReader(),
        run_policy=graph.GraphRunPolicy(
            allow_memory_patch=False,
            allow_human_intervention=False,
            enable_literature_tool=False,
            enable_history_prefetch=False,
        ),
    )

    asyncio.run(runner.arun("test request", lane="standard", proposal_review=False))

    assert calls["init_goal"] == 0
    assert calls["history"] == 0
    assert calls["include_literature_tool"] is False
    assert calls["allow_human_intervention"] is False
    assert calls["allow_memory_patch"] is False


def test_graph_runner_unexpected_interrupt_fails_when_human_intervention_disabled(tmp_path: Path) -> None:
    run_context = RunContext.create(workspace=tmp_path, model_name="test-model")
    runner = graph.GraphRunner(
        task_runner_model=_DummyModel(),
        proposal_model=_DummyModel(),
        director_model=_DummyModel(),
        memory_patch_model=_DummyModel(),
        memory_store=_memory_store(tmp_path),
        run_context=run_context,
        run_policy=graph.GraphRunPolicy(allow_human_intervention=False),
    )

    async def _exercise() -> dict:
        runner._ainvoke_graph_once = lambda compiled, graph_input, config: asyncio.sleep(0, result={"__interrupt__": [{"value": {"type": "task_intervention"}}]})  # type: ignore[method-assign]
        return await runner._ainvoke_loop(
            compiled=object(),
            initial_state={"user_request": "x"},
            config={},
            workspace=tmp_path,
            lane="standard",
        )

    result = asyncio.run(_exercise())
    assert result["status"] == "failure"
    assert "human intervention is disabled" in result["summary"]
