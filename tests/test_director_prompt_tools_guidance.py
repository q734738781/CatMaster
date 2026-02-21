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
        memory_index_excerpt="",
        artifacts_index="[]",
        already_done_json="[]",
        tools="bash_exec : run shell",
    )
    system_content = str(messages[0].content)
    human_content = str(messages[1].content)
    assert "Available tools for task runner" in system_content
    assert 'task_packet.suggested_tools must be selected from "Available tools for task runner"' in system_content
    assert "reports/latest_run/** is an audit/debug snapshot" in system_content
    assert "Never ask the worker to read metadata/internal run paths" in system_content
    assert "Assume runtime environment is correctly configured per project README" in system_content
    assert "runtime/tooling environment prerequisites" in system_content
    assert "Do not revise or ask for confirmation for minor execution details" in system_content
    assert "Default priority: PerformNextTask > MinorReviseProposal > MajorReviseProposal." in system_content
    assert "If the worker reports remote job failures, default to MajorReviseProposal" in system_content
    assert "rerun only the failed subset" in system_content
    assert "do not restart successful jobs" in system_content
    assert "small/local edits that keep the same route" in system_content
    assert "clarifying wording, filling missing defaults" in system_content
    assert "route-level change is required" in system_content
    assert "methodological direction change" in system_content
    assert "unresolved BLOCKING human decisions" in system_content
    assert "Do not choose MajorReviseProposal when safe defaults or local edits can keep the current route valid." in system_content
    assert "sanitized summary; metadata/internal paths omitted" in human_content
    assert "Available tools for task runner:" in human_content
