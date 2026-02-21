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
        memory_index_excerpt="",
        artifact_slice="",
    )
    system_content = str(messages[0].content)
    assert "Do NOT put function tool names into bash_exec commands" in system_content
    assert "ase, pymatgen, numpy, matplotlib, scipy, pandas, fitz, requests" in system_content
    assert "always write script files and execute from disk" in system_content
    assert "Use inline heredoc" in system_content
    assert "quick result analysis or file inspection" in system_content
    assert "actual workload execution" in system_content
    assert "python - <<'PY'" in system_content
    assert "Core output hygiene" in system_content
    assert "Do NOT paste raw tables, long snippets, logs, or scripts into task_finish.summary." in system_content
    assert "For remote/batch job failures, do one minimal triage" in system_content
    assert "Do not do open-ended exploration for remote failures (no SSH)." in system_content
    assert "minimal rerun/repair plan that reruns only the failed subset." in system_content
    assert "Debug triage is allowed to use grep/tail" in system_content
    assert "do not manually stitch results with repeated grep commands" in system_content
    assert "Result Handoff discipline" in system_content
    assert "reports/latest_run/** is for audit/debug" in system_content
    assert "primary script(s) written/executed (kind=script)" in system_content
    assert ".logs/bash_exec/..." in system_content
