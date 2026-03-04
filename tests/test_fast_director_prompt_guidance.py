from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    FAST_DIRECTOR_CONTEXT_TEMPLATE,
    FAST_DIRECTOR_SYSTEM_PROMPT,
)


def test_fast_director_prompt_has_no_proposal_lifecycle() -> None:
    system_content = FAST_DIRECTOR_SYSTEM_PROMPT
    human_content = FAST_DIRECTOR_CONTEXT_TEMPLATE

    assert "proposal-free dynamic execution controller" in system_content
    assert "You should not perform actual tasks. You should only dispatch tasks to the task runner agent." in system_content
    assert "Default to forwarding the minimal executable task spec." in system_content
    assert "PerformNextTask" in system_content
    assert "StopAndSynthesize" in system_content
    assert "MinorReviseProposal" not in system_content
    assert "MajorReviseProposal" not in system_content
    assert "Treat `.` as project files root" in system_content
    assert "Do not invent file paths, tool outputs, or numerical results" in system_content
    assert "`write_note`" in system_content
    assert "`apply_aider_edits`" not in system_content

    assert "Latest completed task outcome (authoritative-by-default evidence):" not in human_content
    assert "Recent task outcomes history (oldest -> newest, MarkdownKV records):" in human_content
    assert "Task status board (structured task history with outcomes):" not in human_content
    assert "Available tools for task runner:" in human_content
