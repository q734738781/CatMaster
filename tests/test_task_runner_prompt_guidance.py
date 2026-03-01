from __future__ import annotations
import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import TASK_RUNNER_SYSTEM_PROMPT


def test_task_runner_prompt_focuses_on_semantics_not_output_shape() -> None:
    content = TASK_RUNNER_SYSTEM_PROMPT
    assert "Use `status=\"done\"` when task is complete." in content
    assert "Use `status=\"blocked\"` only when still blocked" in content
    assert "Follow schema field descriptions for per-field content quality and placeholders" in content
    assert "Core output hygiene" not in content
    assert "TaskOutput" not in content
    assert "YAML" not in content
    assert "JSON" not in content
