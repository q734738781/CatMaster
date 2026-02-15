from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import build_director_prompt


def test_director_prompt_includes_available_tools_and_constraints() -> None:
    prompt = build_director_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        proposal_md="demo proposal",
        work_packages_json='["wp1"]',
        whiteboard_full="",
        artifacts_index="[]",
        already_done_json="[]",
        tools="bash_exec : run shell",
    )
    system_content = str(messages[0].content)
    human_content = str(messages[1].content)
    assert "Available tools for task runner" in system_content
    assert 'suggested_tools must be selected from "Available tools for task runner"' in system_content
    assert "must be tool names (for tool calling), not shell commands" in system_content
    assert "Available tools for task runner:" in human_content
