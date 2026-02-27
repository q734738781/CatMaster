from __future__ import annotations

from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    PROPOSAL_REVISION_CONTEXT_TEMPLATE,
)


def test_proposal_prompt_mentions_bash_exec_heredoc_and_no_persist() -> None:
    system_content = PROPOSAL_SYSTEM_PROMPT
    assert "Allowed helper tools in this stage" in system_content
    assert "- `bash_exec`" in system_content
    assert "python - <<'PY'" in system_content
    assert "avoid script persistence" in system_content
    assert "Assume runtime environment is correctly configured per project README." in system_content
    assert "Do NOT raise runtime/tooling environment prerequisites" in system_content
    assert 'The first section in proposal_md MUST be "Items needing human decision".' in system_content
    assert 'If no blocking decision is needed, still include the section and write "- (none)".' in system_content
    assert "Use a Markdown table with columns: | Parameter | Default | Confidence | Rationale |." in system_content
    assert "include key computational / geometric parameters" in system_content


def test_proposal_feedback_context_template_has_feedback_field() -> None:
    assert "HUMAN FEEDBACK" in PROPOSAL_REVISION_CONTEXT_TEMPLATE
    assert "{feedback}" in PROPOSAL_REVISION_CONTEXT_TEMPLATE
