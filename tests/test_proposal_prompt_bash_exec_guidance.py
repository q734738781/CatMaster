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
        whiteboard_full="",
        artifacts_index="[]",
        tools="bash_exec : run shell",
    )
    system_content = str(messages[0].content)
    assert "Allowed helper tools in this stage" in system_content
    assert "- `bash_exec`" in system_content
    assert "python - <<'PY'" in system_content
    assert "avoid script file persistence" in system_content


def test_proposal_feedback_prompt_mentions_bash_exec_heredoc_and_no_persist() -> None:
    prompt = build_proposal_feedback_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        proposal_md="proposal",
        work_packages_json='["wp1"]',
        whiteboard_full="",
        artifacts_index="[]",
        tools="bash_exec : run shell",
        feedback="update",
    )
    system_content = str(messages[0].content)
    assert "Allowed helper tools in this stage" in system_content
    assert "- `bash_exec`" in system_content
    assert "python - <<'PY'" in system_content
    assert "avoid script file persistence" in system_content
