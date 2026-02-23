from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator import Orchestrator
from catmaster.runtime.memory_store import MemoryStore


def test_initialize_memory_goal_is_idempotent_and_updates_primary(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    holder = SimpleNamespace(memory_store=store)
    goal_path = store.topics_dir / "GOAL.md"

    Orchestrator._initialize_memory_goal(holder, "first objective")
    first = goal_path.read_text(encoding="utf-8")
    assert "- Primary objective: first objective" in first
    assert first.count("## Change log") == 1

    Orchestrator._initialize_memory_goal(holder, "first objective")
    second = goal_path.read_text(encoding="utf-8")
    assert second == first

    Orchestrator._initialize_memory_goal(holder, "second objective")
    third = goal_path.read_text(encoding="utf-8")
    assert "- Primary objective: second objective" in third
    assert third.count("## Change log") == 1
    assert "first objective" in third
    assert "second objective" in third
