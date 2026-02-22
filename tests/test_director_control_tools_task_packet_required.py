from __future__ import annotations

import pytest
from pydantic import ValidationError

from catmaster.agents.director_control_tools import DirectorDecideInput


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
