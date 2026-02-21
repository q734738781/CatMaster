from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    build_proposal_feedback_prompt,
    build_proposal_prompt,
)


def test_proposal_prompt_mentions_bash_exec_heredoc_and_no_persist() -> None:
    prompt = build_proposal_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        memory_index_excerpt="",
        artifacts_index="[]",
        tools="bash_exec : run shell",
    )
    system_content = str(messages[0].content)
    assert "Allowed helper tools in this stage" in system_content
    assert "- `bash_exec`" in system_content
    assert "python - <<'PY'" in system_content
    assert "avoid script file persistence" in system_content
    assert "Assume runtime environment is correctly configured per project README." in system_content
    assert "Do NOT raise runtime/tooling environment prerequisites" in system_content
    assert 'The first section in proposal_md MUST be "Items needing human decision".' in system_content
    assert 'If no blocking decision is needed, still include the section and write "- (none)".' in system_content
    assert "Use a Markdown table with columns: | Parameter | Default | Confidence | Rationale |." in system_content
    assert "include key computational / geometric parameters" in system_content


def test_proposal_feedback_prompt_mentions_bash_exec_heredoc_and_no_persist() -> None:
    prompt = build_proposal_feedback_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        proposal_md="proposal",
        work_packages_json='["wp1"]',
        memory_index_excerpt="",
        artifacts_index="[]",
        tools="bash_exec : run shell",
        feedback="update",
    )
    system_content = str(messages[0].content)
    assert "Allowed helper tools in this stage" in system_content
    assert "- `bash_exec`" in system_content
    assert "python - <<'PY'" in system_content
    assert "avoid script file persistence" in system_content
    assert 'The first section MUST be "Items needing human decision"' in system_content
    assert 'render it as a Markdown table' in system_content
    assert "Render detail modifications as a Markdown table" in system_content
