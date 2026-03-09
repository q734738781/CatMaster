from __future__ import annotations
import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator_prompts import TASK_RUNNER_SYSTEM_PROMPT, TASK_CONTEXT_TEMPLATE


def test_task_step_prompt_includes_no_symlink_rule() -> None:
    system_content = TASK_RUNNER_SYSTEM_PROMPT
    assert "Task detail defines the task invariants and done checks." in system_content
    assert "minimal non-destructive procedure that satisfies those invariants." in system_content
    assert "Do not rerun the same preparation tool with identical parameters" in system_content
    assert "Bundle independent filesystem operations in the same turn whenever possible" in system_content
    assert "do not issue concurrent write calls" in system_content
    assert "prefer a single focused bash_exec over many single-path filesystem tool calls" in system_content
    assert "Prefer read_multiple_files when you need several small text files at once." in system_content
    assert "Tool schemas are authoritative" in system_content
    assert "Do not initiate file edits on your own." in system_content
    assert "Only edit files when the current task packet explicitly requires" in system_content
    assert "For `MEMORY/**`, only update when scientific invariants" in system_content
    assert "Do NOT put function tool names into bash_exec commands" in system_content
    assert "ase, pymatgen, numpy, matplotlib, scipy, pandas, fitz, requests" in system_content
    assert "For remote/batch job failures, do one minimal triage" in system_content
    assert "Do not do open-ended exploration for remote failures (no SSH)." in system_content
    assert "minimal rerun/repair plan that reruns only the failed subset." in system_content
    assert "Debug triage should prioritize focused, minimal evidence extraction" in system_content
    assert "Prefer mature third-party libraries for parsing and post-analysis when available" in system_content
    assert "Internal metadata audit logs are not task inputs" in system_content
    assert "Follow schema field descriptions for per-field content quality and placeholders" in system_content
    assert "Keep handoff evidence-based and concise" in system_content

    human_content = TASK_CONTEXT_TEMPLATE
    assert "Task detail:" in human_content
    assert "Expected outputs:" in human_content
    assert "Suggested tools:" in human_content
    assert "Reference hint:" in human_content
