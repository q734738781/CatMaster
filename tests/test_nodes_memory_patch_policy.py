from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage

from catmaster.agents.nodes import run_memory_patch
from catmaster.runtime.memory_store import MemoryStore


class _FakeAgent:
    def __init__(self, result: dict):
        self._result = result
        self.calls = 0

    async def ainvoke(self, payload):
        _ = payload
        self.calls += 1
        return self._result


def _memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    return store


def test_run_memory_patch_no_updates_short_circuit(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    agent = _FakeAgent({"messages": []})

    command = asyncio.run(
        run_memory_patch(
            {"pending_memory_updates": []},
            agent=agent,
            memory_store=store,
            run_id="run_01",
        )
    )

    assert command.goto == "summarize"
    assert agent.calls == 0
    result = command.update.get("memory_patch_result", {})
    assert result.get("status") == "done"
    assert result.get("applied_topics") == []


def test_run_memory_patch_records_event_and_clears_pending(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    agent = _FakeAgent(
        {
            "messages": [AIMessage(content="ok")],
            "structured_response": {
                "status": "done",
                "summary": "Memory updates applied.",
                "applied_topics": ["MEMORY/topics/FACTS.md"],
                "error": "",
                "needs_human": False,
            },
        }
    )

    command = asyncio.run(
        run_memory_patch(
            {
                "pending_memory_updates": [
                    {"topic": "MEMORY/topics/FACTS.md", "content": "Store final O-O distance."}
                ]
            },
            agent=agent,
            memory_store=store,
            run_id="run_01",
        )
    )

    assert command.goto == "summarize"
    assert agent.calls == 1
    assert command.update.get("pending_memory_updates") == []
    result = command.update.get("memory_patch_result", {})
    assert result.get("status") == "done"
    events = store.read_events_tail(limit=1)
    assert events
    assert events[-1].get("task_id") == "final_memory_update"
    assert events[-1].get("memory_patch_status") == "done"


def test_run_memory_patch_missing_structured_response_is_non_blocking(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    agent = _FakeAgent({"messages": [AIMessage(content="plain text")]})

    command = asyncio.run(
        run_memory_patch(
            {
                "pending_memory_updates": [
                    {"topic": "MEMORY/topics/FILES.md", "content": "Add final summary path."}
                ]
            },
            agent=agent,
            memory_store=store,
            run_id="run_01",
        )
    )

    assert command.goto == "summarize"
    result = command.update.get("memory_patch_result", {})
    assert result.get("status") == "blocked"
    assert command.update.get("contract_violation", {}).get("role") == "memory_patch"
