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
    assert "reference absolute project-files-root path" in system_content
    assert "use relative paths in arguments by default" in system_content
    assert "`bash` command text is exempt" not in system_content
    assert "Do not invent file paths, tool outputs, or numerical results" in system_content
    assert "fill `update_memory`, otherwise return `[]`" in system_content
    assert "planning-only / explanation-only" in system_content
    assert 'FastDirectorOutput(state="StopAndSynthesize", ...)' in system_content
    assert "research-grounded" in system_content
    assert "short reference shortlist" in system_content
    assert "must not replace citations" in system_content
    assert "Intent routing rules (execution-priority):" in system_content
    assert "Read-only workspace QA" in system_content
    assert "General knowledge/comparison QA" in system_content
    assert "Literature-only grounding requests" in system_content
    assert "If uncertain whether execution is required, prefer `PerformNextTask`." in system_content
    assert "Skills may be available for domain SOP and parameter conventions." in system_content
    assert "Use literature grounding only when the user explicitly asks for papers/prior work/supporting evidence" in system_content
    assert "If `state=PerformNextTask`" in system_content
    assert "update_memory` must be `[]`" in system_content
    assert "`run_literature_research`" in system_content
    assert "`write_note`" not in system_content
    assert "`apply_aider_edits`" not in system_content

    assert "Latest completed task outcome (authoritative-by-default evidence):" not in human_content
    assert "Recent task outcomes history (oldest -> newest, MarkdownKV records):" in human_content
    assert "Task status board (structured task history with outcomes):" not in human_content
    assert "Relevant execution skills and task-runner capabilities:" in human_content
    assert "{execution_context_guide}" in human_content
    assert "Available tools for task runner:" not in human_content
