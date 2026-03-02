from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    build_memory_patch_prompt,
    build_memory_patch_repair_prompt,
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
    assert "Merge-first policy for `MEMORY/topics/FILES.md`:" in system_content
    assert "Do NOT append blindly. Canonicalize and merge before writing." in system_content
    assert "Exclude routine internal audit logs (`metadata/**`, `audit/**`, `.logs/**`)" in system_content
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
    assert "allowed paths: MEMORY/** only" in system_content
    assert "replace_no_match" in system_content
    assert "Keep the same topic schema contract and write-routing rules" in system_content
    assert "preserve merge-first behavior for `FILES.md`" in system_content
    assert "Apply error context (JSON, reference only)" in human_content


def test_task_runner_system_prompt_has_key_rules() -> None:
    from catmaster.agents.orchestrator_prompts import TASK_RUNNER_SYSTEM_PROMPT
    assert "Use tool calling from all available tools" in TASK_RUNNER_SYSTEM_PROMPT
    assert "status=\"done\"" in TASK_RUNNER_SYSTEM_PROMPT or "status=blocked" in TASK_RUNNER_SYSTEM_PROMPT
