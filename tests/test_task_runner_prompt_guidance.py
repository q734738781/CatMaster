from __future__ import annotations
import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import TASK_RUNNER_SYSTEM_PROMPT


def test_task_runner_prompt_focuses_on_semantics_not_output_shape() -> None:
    content = TASK_RUNNER_SYSTEM_PROMPT
    assert "Skills may be available for workflow guidance and parameter conventions." in content
    assert "Tool schemas are compact invocation interfaces, not complete SOP." in content
    assert "tool schemas remain authoritative for arguments and file outputs." in content
    assert "do not silently rely on tool defaults for method-critical toggles" in content
    assert "use it to determine method-critical defaults and evidence standards" in content
    assert "Before the first expensive, managed, or irreversible tool call" in content
    assert "preserve representative citations inline in the task handoff" in content
    assert "`run_literature_research` may be called only when the current task packet explicitly requires literature grounding" in content
    assert "Do not initiate literature research just because you want extra reassurance." in content
    assert "prefer the domain tool over re-implementing the same capability" in content
    assert "Use `status=\"done\"` when task is complete." in content
    assert "Use `status=\"blocked\"` only when still blocked" in content
    assert "Follow schema field descriptions for per-field content quality and placeholders" in content
    assert "Core output hygiene" not in content
    assert "TaskOutput" not in content
    assert "YAML" not in content
    assert "JSON" not in content
