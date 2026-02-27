from __future__ import annotations

from catmaster.agents.orchestrator_prompts import DIRECTOR_SYSTEM_PROMPT, DIRECTOR_CONTEXT_TEMPLATE


def test_director_prompt_includes_available_tools_and_constraints() -> None:
    system_content = DIRECTOR_SYSTEM_PROMPT
    human_content = DIRECTOR_CONTEXT_TEMPLATE
    assert "Available tools for task runner" in system_content
    assert "You may use helper tools for read/check inspection before deciding." in system_content
    assert "task_packet fields: goal, task_detail, expected_outputs, suggested_tools, reference_hint." in system_content
    assert "Never ask the worker to read metadata/internal run paths" in system_content
    assert "Never ask the worker to edit `MEMORY/**`" in system_content
    assert "Assume runtime environment is correctly configured per project README" in system_content
    assert "Do not revise or ask for confirmation for minor execution details" in system_content
    assert "Default priority: PerformNextTask > MinorReviseProposal > MajorReviseProposal." in system_content
    assert "StopAndSynthesize" in system_content
    assert "Do not treat proposal-format requirements" in system_content
    assert "Tool schemas are authoritative." in system_content
    assert "unresolved BLOCKING" in system_content
    assert "AlreadyDone" in human_content
    assert "Available tools for task runner:" in human_content
