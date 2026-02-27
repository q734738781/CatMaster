from __future__ import annotations

import pytest
from pydantic import ValidationError

import json

pytest.importorskip("langchain_core")

from catmaster.agents.director_control_tools import as_langchain_control_tools
from catmaster.agents.response_schemas import DirectorOutput as DirectorDecideInput, TaskPacket


def test_perform_next_task_requires_task_packet() -> None:
    with pytest.raises(ValidationError, match="requires task_packet"):
        DirectorDecideInput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
        )


def test_perform_next_task_accepts_task_packet() -> None:
    payload = DirectorDecideInput(
        state="PerformNextTask",
        rationale="Dispatch next task.",
        task_packet={
            "goal": "Run adsorption references",
            "task_detail": "Enable D3 and keep slab setup fixed.",
            "expected_outputs": ["results/adsorption/reference_energies.json"],
            "suggested_tools": ["bash_exec"],
            "reference_hint": ["MEMORY/topics/FACTS.md", "rg keywords: adsorption D3"],
        },
    )
    assert payload.task_packet is not None
    assert payload.task_packet.task_detail == "Enable D3 and keep slab setup fixed."


def test_director_control_tool_serializes_task_packet_model_instance() -> None:
    tool = as_langchain_control_tools()[0]
    packet = TaskPacket(
        goal="Run task",
        task_detail="Use explicit settings.",
        expected_outputs=["reports/out.json"],
        suggested_tools=["bash_exec"],
        reference_hint=["MEMORY/topics/FACTS.md"],
    )
    out = tool.func(
        state="PerformNextTask",
        rationale="Proceed",
        task_packet=packet,
    )
    parsed = json.loads(out)
    assert parsed["tool_name"] == "director_decide"
    assert parsed["payload"]["task_packet"]["goal"] == "Run task"
