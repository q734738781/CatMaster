from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    build_plan_feedback_prompt,
    build_plan_prompt,
    build_plan_repair_prompt,
)


def test_plan_prompt_requires_key_parameter_markdown_table() -> None:
    prompt = build_plan_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        tools="bash_exec : run shell",
        planner_tools="bash_exec : run shell",
    )
    system_content = str(messages[0].content)
    assert "present key parameters as a Markdown table" in system_content
    assert "| Parameter | Default / Choice | Rationale |" in system_content
    assert "key computational / geometric parameters" in system_content


def test_plan_repair_prompt_keeps_table_contract() -> None:
    prompt = build_plan_repair_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        tools="bash_exec : run shell",
        planner_tools="bash_exec : run shell",
        error="parse_error",
        raw="invalid",
    )
    system_content = str(messages[0].content)
    assert "keep key parameters in a Markdown table" in system_content
    assert "key computational / geometric parameters" in system_content


def test_plan_feedback_prompt_requires_table_for_detail_changes() -> None:
    prompt = build_plan_feedback_prompt()
    messages = prompt.format_messages(
        user_request="demo request",
        tools="bash_exec : run shell",
        planner_tools="bash_exec : run shell",
        plan_json='{"todo":["a"],"plan_description":"b"}',
        feedback="revise",
        feedback_history="[]",
    )
    system_content = str(messages[0].content)
    assert "Keep key parameters and detail changes in Markdown tables inside plan_description." in system_content
    assert "key computational / geometric parameters" in system_content
