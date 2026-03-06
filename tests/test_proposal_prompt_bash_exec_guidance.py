from __future__ import annotations
import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_CONTEXT_TEMPLATE,
    PROPOSAL_SYSTEM_PROMPT,
    PROPOSAL_REVISION_CONTEXT_TEMPLATE,
)


def test_proposal_prompt_mentions_bash_exec_heredoc_and_no_persist() -> None:
    system_content = PROPOSAL_SYSTEM_PROMPT
    assert "Allowed helper tools in this stage" in system_content
    assert "Filesystem read/discovery tools" in system_content
    assert "`bash_exec`" in system_content
    assert "`apply_aider_edits`" not in system_content
    assert "`write_note`" not in system_content
    assert "`read_file`" not in system_content
    assert "`list_directory_with_sizes`" not in system_content
    assert "`get_file_info`" not in system_content
    assert "Do not modify files, update memory, or write notes in this stage" in system_content
    assert "Treat memory index as historical reference" in system_content
    assert "Assume runtime environment is correctly configured per project README." in system_content
    assert "Do NOT raise runtime/tooling environment prerequisites" in system_content
    assert "Plan primarily around skills, scientific stages, evidence contracts, and task packets" in system_content
    assert "Skills may be available for domain SOP and parameter conventions." in system_content
    assert "Do not invent nonexistent files, completed outputs, or numeric results." in system_content
    assert "Keep the body short" in system_content
    assert "Do not turn the proposal into a literature review" in system_content
    assert 'include an "Items needing human decision" section near the top.' in system_content
    assert "Include key parameters/defaults near the top with short rationale and confidence notes." in system_content
    assert "include key computational / geometric parameters" in system_content
    assert "Treat `.` as the project files root" in system_content
    assert "reference absolute project-files-root path" in system_content
    assert "use relative paths in arguments by default" in system_content
    assert "`bash_exec` command text is exempt" in system_content
    assert "ProposalOutput" not in system_content


def test_proposal_feedback_context_template_has_feedback_field() -> None:
    assert "INSTRUCTIONS REMINDER" in PROPOSAL_CONTEXT_TEMPLATE
    assert "AVAILABLE EXECUTION CAPABILITIES AND RELEVANT SKILLS" in PROPOSAL_CONTEXT_TEMPLATE
    assert "{execution_context_guide}" in PROPOSAL_CONTEXT_TEMPLATE
    assert "AVAILABLE TOOLS FOR TASK EXECUTION" not in PROPOSAL_CONTEXT_TEMPLATE
    assert "EXACT section order" not in PROPOSAL_CONTEXT_TEMPLATE
    assert "HUMAN FEEDBACK" in PROPOSAL_REVISION_CONTEXT_TEMPLATE
    assert "{feedback}" in PROPOSAL_REVISION_CONTEXT_TEMPLATE
    assert "AVAILABLE EXECUTION CAPABILITIES AND RELEVANT SKILLS" in PROPOSAL_REVISION_CONTEXT_TEMPLATE
    assert "{execution_context_guide}" in PROPOSAL_REVISION_CONTEXT_TEMPLATE
    assert "EXACT section order" not in PROPOSAL_REVISION_CONTEXT_TEMPLATE
