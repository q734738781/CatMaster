from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage

from catmaster.agents.nodes import run_director, run_proposal, run_task
from catmaster.runtime.memory_store import MemoryStore


class _FakeAgent:
    def __init__(self, result):
        self._result = result

    def invoke(self, payload):
        return self._result

    async def ainvoke(self, payload):
        _ = payload
        return self._result


def _memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    return store


def test_run_proposal_missing_structured_response_marks_failure(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    agent = _FakeAgent({"messages": [AIMessage(content="plain final text")]})

    out = asyncio.run(
        run_proposal(
            {"user_request": "draft plan"},
            agent=agent,
            memory_store=store,
            tools_description="bash_exec",
            run_dir=tmp_path,
        )
    )

    assert out.goto == "summarize"
    assert out.update.get("status") == "failure"
    assert out.update.get("contract_violation", {}).get("role") == "proposal"
    assert out.update.get("contract_violation", {}).get("reason") == "missing_structured_response"


def test_run_director_missing_structured_response_marks_failure(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    agent = _FakeAgent({"messages": [AIMessage(content="no structured response emitted")]})
    command = asyncio.run(
        run_director(
            {
                "user_request": "run",
                "proposal_md": "x",
                "work_packages": ["a"],
                "observations": [],
            },
            agent=agent,
            memory_store=store,
            tools_description="bash_exec",
        )
    )

    assert command.goto == "finalize_memory_patch"
    assert command.update.get("status") == "failure"
    assert command.update.get("contract_violation", {}).get("role") == "director"
    assert command.update.get("contract_violation", {}).get("reason") == "missing_structured_response"


def test_run_task_missing_structured_response_marks_failure(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    agent = _FakeAgent({"messages": [AIMessage(content="no structured response")]})

    out = asyncio.run(
        run_task(
            {"user_request": "run task", "current_task_packet": {"goal": "run task"}},
            agent=agent,
            memory_store=store,
        )
    )

    assert out.goto == "finalize_memory_patch"
    assert out.update.get("status") == "failure"
    assert out.update.get("contract_violation", {}).get("role") == "task_runner"
    assert out.update.get("contract_violation", {}).get("reason") == "missing_structured_response"
    assert out.update.get("task_result", {}).get("task_outcome") == "failure"
