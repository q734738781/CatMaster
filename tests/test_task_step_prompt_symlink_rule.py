from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import build_task_step_prompt


def test_task_step_prompt_includes_no_symlink_rule() -> None:
    prompt = build_task_step_prompt()
    messages = prompt.format_messages(
        goal="dummy",
        constraints="none",
        workspace_policy="none",
        whiteboard_excerpt="",
        artifact_slice="",
    )
    system_content = str(messages[0].content)
    assert "Symbolic link operations are forbidden in bash_exec" in system_content
