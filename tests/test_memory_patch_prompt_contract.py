from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import (
    MEMORY_PATCHER_SYSTEM_PROMPT,
    MEMORY_PATCH_CONTEXT_TEMPLATE,
)


def test_memory_patcher_system_prompt_requires_read_then_patch() -> None:
    system_content = MEMORY_PATCHER_SYSTEM_PROMPT
    assert "Memory Patcher agent" in system_content
    assert "`apply_aider_edits`" in system_content
    assert "Only modify files under `MEMORY/**`." in system_content
    assert "If an edit apply fails, re-read the target file and retry" in system_content
    assert "FACTS keeps verified facts/results only" in system_content
    assert "FILES keeps artifact path/index entries only" in system_content


def test_memory_patch_context_template_contains_update_and_editable_sections() -> None:
    content = MEMORY_PATCH_CONTEXT_TEMPLATE.format(
        pending_memory_updates_json='[{"topic":"MEMORY/topics/FACTS.md","content":"x"}]',
        editable_file_snapshots='<editable_file path="MEMORY/topics/FACTS.md"># FACTS</editable_file>',
    )
    assert "Pending memory updates" in content
    assert "<editable_file path=\"MEMORY/topics/FACTS.md\">" in content
    assert "Aider SEARCH/REPLACE format reminder:" in content


def test_task_runner_system_prompt_has_key_rules() -> None:
    from catmaster.agents.orchestrator_prompts import TASK_RUNNER_SYSTEM_PROMPT
    assert "Use tool calling from all available tools" in TASK_RUNNER_SYSTEM_PROMPT
    assert "status=\"done\"" in TASK_RUNNER_SYSTEM_PROMPT or "status=blocked" in TASK_RUNNER_SYSTEM_PROMPT
