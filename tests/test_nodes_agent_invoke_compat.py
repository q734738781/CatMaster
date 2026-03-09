from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage

from catmaster.agents.nodes import run_proposal
from catmaster.runtime.memory_store import MemoryStore


class _FallbackFakeAgent:
    def invoke(self, payload):
        if "remaining_steps" in payload:
            raise ValueError("ValidationError: remaining_steps extra_forbidden")
        return {
            "messages": [AIMessage(content="structured")],
            "structured_response": {
                "status": "success",
                "proposal_md": "# plan",
                "work_packages": ["wp1"],
                "error": "",
                "needs_human": False,
            },
        }


class _AsyncOnlyFallbackFakeAgent:
    def invoke(self, payload):
        _ = payload
        raise NotImplementedError("sync invoke is not supported")

    async def ainvoke(self, payload):
        if "remaining_steps" in payload:
            raise ValueError("ValidationError: remaining_steps extra_forbidden")
        return {
            "messages": [AIMessage(content="structured")],
            "structured_response": {
                "status": "success",
                "proposal_md": "# plan async",
                "work_packages": ["wp_async"],
                "error": "",
                "needs_human": False,
            },
        }


def _memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    return store


def test_run_proposal_retries_without_remaining_steps_when_rejected(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    out = asyncio.run(
        run_proposal(
            {"user_request": "draft plan"},
            agent=_FallbackFakeAgent(),
            memory_store=store,
            execution_context_guide="bash_exec",
            run_dir=tmp_path,
            max_steps=3,
        )
    )

    assert out.goto == "proposal_review"
    assert out.update.get("proposal_md") == "# plan"
    assert out.update.get("contract_violation") == {}


def test_run_proposal_supports_async_only_agent_with_remaining_steps_fallback(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    out = asyncio.run(
        run_proposal(
            {"user_request": "draft plan"},
            agent=_AsyncOnlyFallbackFakeAgent(),
            memory_store=store,
            execution_context_guide="bash_exec",
            run_dir=tmp_path,
            max_steps=3,
        )
    )

    assert out.goto == "proposal_review"
    assert out.update.get("proposal_md") == "# plan async"
    assert out.update.get("contract_violation") == {}
