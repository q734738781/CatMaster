from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents import graph
from catmaster.runtime.memory_store import MemoryStore


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
    bash = _DummyTool("bash_exec")

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
