from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator import Orchestrator


def test_resolve_task_goal_from_decision_uses_new_task_packet_fields() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    decision = {
        "state": "PerformNextTask",
        "task_packet": {
            "goal": "Run adsorption references",
            "task_detail": "Use D3 and keep slab constraints unchanged.",
            "expected_outputs": ["results/adsorption/reference_energies.json"],
            "suggested_tools": ["vasp_relax_prepare", "vasp_execute_batch"],
            "reference_hint": ["MEMORY/topics/FACTS.md", "rg keywords: D3 adsorption"],
        },
    }

    task_goal, packet = Orchestrator._resolve_task_goal_from_decision(orch, decision)
    assert "Goal: Run adsorption references" in task_goal
    assert "Task detail: Use D3 and keep slab constraints unchanged." in task_goal
    assert "Reference hint: MEMORY/topics/FACTS.md; rg keywords: D3 adsorption" in task_goal
    assert packet["goal"] == "Run adsorption references"
    assert packet["task_detail"] == "Use D3 and keep slab constraints unchanged."
    assert packet["reference_hint"] == ["MEMORY/topics/FACTS.md", "rg keywords: D3 adsorption"]


def test_resolve_task_goal_from_decision_requires_task_detail() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    decision = {
        "state": "PerformNextTask",
        "task_packet": {
            "goal": "Run adsorption references",
            "task_detail": "",
            "expected_outputs": [],
            "suggested_tools": ["bash"],
            "reference_hint": [],
        },
    }

    with pytest.raises(ValueError, match="task_packet.task_detail"):
        Orchestrator._resolve_task_goal_from_decision(orch, decision)


def test_resolve_task_goal_from_decision_requires_task_packet() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    decision = {
        "state": "PerformNextTask",
    }

    with pytest.raises(ValueError, match="missing task_packet"):
        Orchestrator._resolve_task_goal_from_decision(orch, decision)
