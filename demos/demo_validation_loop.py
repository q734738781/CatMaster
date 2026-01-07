#!/usr/bin/env python3
"""
Deterministic demo for ToolExecutor validation loop.

It injects an invalid parameter into a write_file toolcall, then retries with a valid payload.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
import logging

from catmaster.agents.orchestrator import Orchestrator


class DummyLLM:
    model_name = "dummy"

    def invoke(self, _messages):  # pragma: no cover - demo stub
        class _Resp:
            content = "{}"
        return _Resp()


class DeterministicOrchestrator(Orchestrator):
    def __init__(self, decisions, **kwargs):
        super().__init__(**kwargs)
        self._decisions = decisions

    def _decide_next(self, memories, *, log_llm: bool = False, step: int = 0):  # noqa: D401
        return self._decisions[min(step, len(self._decisions) - 1)]

    def _llm_summary(self, memories, user_request: str) -> str:  # noqa: D401
        return self._summarize(memories, user_request)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = Path("workspace/demo_validation_loop").resolve()
    os.environ["CATMASTER_WORKSPACE"] = str(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    decisions = [
        {
            "action": "call",
            "method": "write_file",
            "params": {"path": "out.txt", "content": "hello", "bad_field": 1},
            "next_step": "write_file with correct params",
            "note": "inject invalid param",
            "reasoning": "simulate LLM mistake",
        },
        {
            "action": "call",
            "method": "write_file",
            "params": {"path": "out.txt", "content": "hello"},
            "next_step": "finish",
            "note": "retry with valid params",
            "reasoning": "fix validation error",
        },
        {
            "action": "finish_project",
            "method": None,
            "params": {},
            "next_step": "done",
            "note": "",
            "reasoning": "demo complete",
        },
    ]

    orch = DeterministicOrchestrator(
        decisions=decisions,
        llm=DummyLLM(),
        summary_llm=DummyLLM(),
        max_steps=len(decisions),
        log_llm_console=False,
        max_tool_attempts=3,
    )

    result = orch.run(
        "Demo: validate tool params and auto-retry",
        initial_plan={"todo": ["write a file with validation check"], "next_step": "write_file", "reasoning": ""},
    )

    print("\n=== Summary ===")
    print(result.get("summary", "No summary"))
    print("\nEvents log:")
    print(f"  {orch.run_context.run_dir / 'events.jsonl'}")


if __name__ == "__main__":
    main()
