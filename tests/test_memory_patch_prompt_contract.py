from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    build_memory_patch_prompt,
    build_memory_patch_repair_prompt,
    build_task_step_repair_prompt,
)


def test_memory_patch_prompt_requires_aider_edits_and_path_scope() -> None:
    prompt = build_memory_patch_prompt()
    messages = prompt.format_messages(
        run_id="run_01",
        task_id="task_01",
        task_goal="demo",
        outcome="success",
        structured_result_json="{}",
        memory_index_text="",
        topic_goal_text="# GOAL",
        topic_facts_text="# FACTS",
        topic_files_text="# FILES",
        topic_constraints_text="# CONSTRAINTS",
        topic_questions_text="# QUESTIONS",
        topic_runbook_text="# RUNBOOK",
    )
    system_content = str(messages[0].content)
    human_content = str(messages[1].content)
    assert "Output ONLY Aider SEARCH/REPLACE edit blocks." in system_content
    assert "Do NOT use markdown code fences." in system_content
    assert "Allowed paths to modify: MEMORY/**" in system_content
    assert "Never modify any other path." in system_content
    assert "Treat only text inside `<editable_file path=\"...\">...</editable_file>`" in system_content
    assert "Topic schema contract:" in system_content
    assert "Write-routing from task structured result:" in system_content
    assert "- PATH: <rel_path> | kind=<kind> | desc=<desc> | source=<task_id>" in system_content
    assert "<editable_file path=\"MEMORY/MEMORY.md\">" in human_content
    assert "Event path:" not in human_content
    assert "<editable_file path=\"MEMORY/topics/GOAL.md\">" in human_content
    assert "<editable_file path=\"MEMORY/topics/FACTS.md\">" in human_content
    assert "<editable_file path=\"MEMORY/topics/FILES.md\">" in human_content
    assert "<editable_file path=\"MEMORY/topics/CONSTRAINTS.md\">" in human_content
    assert "<editable_file path=\"MEMORY/topics/QUESTIONS.md\">" in human_content
    assert "<editable_file path=\"MEMORY/topics/RUNBOOK.md\">" in human_content


def test_memory_patch_repair_prompt_requires_aider_format_output() -> None:
    prompt = build_memory_patch_repair_prompt()
    messages = prompt.format_messages(
        previous_edit_text="MEMORY/MEMORY.md\n<<<<<<< SEARCH",
        apply_error="replace failed",
        apply_error_context_json='{"error_code":"replace_no_match"}',
        run_id="run_01",
        task_id="task_01",
        task_goal="demo",
        outcome="success",
        structured_result_json="{}",
        memory_index_text="",
        topic_goal_text="# GOAL",
        topic_facts_text="# FACTS",
        topic_files_text="# FILES",
        topic_constraints_text="# CONSTRAINTS",
        topic_questions_text="# QUESTIONS",
        topic_runbook_text="# RUNBOOK",
    )
    system_content = str(messages[0].content)
    human_content = str(messages[1].content)
    assert "Output ONLY corrected Aider SEARCH/REPLACE edit blocks" in system_content
    assert "allowed paths: MEMORY/** and notes/** only" in system_content
    assert "replace_no_match" in system_content
    assert "Keep the same topic schema contract and write-routing rules" in system_content
    assert "Apply error context (JSON, reference only)" in human_content


def test_task_step_repair_prompt_requires_single_tool_call() -> None:
    prompt = build_task_step_repair_prompt()
    messages = prompt.format_messages(
        goal="demo",
        task_detail="detail",
        expected_outputs="- out",
        suggested_tools="bash_exec",
        reference_hint="- MEMORY/topics/FACTS.md",
        workspace_policy="none",
        memory_index_excerpt="",
        error="parse_error",
        raw="invalid output",
    )
    system_content = str(messages[0].content)
    assert "This turn MUST be exactly one valid tool call." in system_content
    assert "Call exactly one tool in this turn." in system_content
    assert "Do not include any plain text outside the tool call." in system_content
