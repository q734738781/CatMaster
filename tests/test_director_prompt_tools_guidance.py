from __future__ import annotations
import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import DIRECTOR_SYSTEM_PROMPT, DIRECTOR_CONTEXT_TEMPLATE


def test_director_prompt_includes_available_tools_and_constraints() -> None:
    system_content = DIRECTOR_SYSTEM_PROMPT
    human_content = DIRECTOR_CONTEXT_TEMPLATE
    assert "Available tools for task runner" in system_content
    assert "You may use helper tools for read/check inspection before deciding." in system_content
    assert "dispatch one concrete next worker action with minimal scope creep." in system_content
    assert "Never ask the worker to read metadata/internal run paths" in system_content
    assert "Never ask the worker to edit `MEMORY/**`" in system_content
    assert "Assume runtime environment is correctly configured per project README" in system_content
    assert "Do not revise or ask for confirmation for minor execution details" in system_content
    assert "Default priority: PerformNextTask > MinorReviseProposal > MajorReviseProposal." in system_content
    assert "StopAndSynthesize" in system_content
    assert "Do not treat proposal-format requirements" in system_content
    assert "Do not invent file paths, tool outputs, or numerical results" in system_content
    assert "unresolved BLOCKING" in system_content
    assert "DirectorOutput" not in system_content
    assert "expected_outputs default shape" not in system_content
    assert "`goal`:" not in system_content
    assert "`task_detail`:" not in system_content
    assert "`expected_outputs`:" not in system_content
    assert "`reference_hint`:" not in system_content
    assert "task_packet.suggested_tools" not in system_content
    assert "AlreadyDone" in human_content
    assert "Available tools for task runner:" in human_content
