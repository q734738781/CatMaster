from __future__ import annotations

from types import SimpleNamespace

from catmaster.agents.graph import GraphRunner
from catmaster.runtime.memory_store import MemoryStore


def test_initialize_memory_goal_is_noop(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    holder = SimpleNamespace(memory_store=store)

    GraphRunner._initialize_memory_goal(holder, "first objective")

    goal_path = store.topics_dir / "GOAL.md"
    assert not goal_path.exists()
    index_text = store.read_index()
    assert "Goal / principles:" not in index_text
