from __future__ import annotations

import pytest
from pydantic import ValidationError

from catmaster.agents.response_schemas import DirectorOutput, FastDirectorOutput, TaskPacket


def test_perform_next_task_requires_task_packet() -> None:
    with pytest.raises(ValidationError, match="requires its matching payload"):
        DirectorOutput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
            perform_next_task=None,
            minor_revise_proposal=None,
            major_revise_proposal=None,
            stop_and_synthesize=None,
            update_memory=[],
        )


def test_perform_next_task_accepts_task_packet() -> None:
    payload = DirectorOutput(
        state="PerformNextTask",
        rationale="Dispatch next task.",
        perform_next_task={
            "task_packet": {
                "goal": "Run adsorption references",
                "task_detail": "Enable D3 and keep slab setup fixed.",
                "expected_outputs": ["results/adsorption/reference_energies.json"],
                "allowed_tools": ["execute"],
                "reference_hint": ["MEMORY/topics/FACTS.md", "rg keywords: adsorption D3"],
            },
        },
        minor_revise_proposal=None,
        major_revise_proposal=None,
        stop_and_synthesize=None,
        update_memory=[],
    )
    assert payload.perform_next_task is not None
    assert payload.perform_next_task.task_packet.task_detail == "Enable D3 and keep slab setup fixed."


def test_task_packet_model_instance_roundtrip() -> None:
    packet = TaskPacket(
        goal="Run task",
        task_detail="Use explicit settings.",
        expected_outputs=["reports/out.json"],
        allowed_tools=["execute"],
        reference_hint=["MEMORY/topics/FACTS.md"],
    )
    payload = DirectorOutput(
        state="PerformNextTask",
        rationale="Proceed",
        perform_next_task={"task_packet": packet},
        minor_revise_proposal=None,
        major_revise_proposal=None,
        stop_and_synthesize=None,
        update_memory=[],
    )
    assert payload.perform_next_task is not None
    assert payload.perform_next_task.task_packet.goal == "Run task"


def test_task_packet_reference_hint_requires_list() -> None:
    with pytest.raises(ValidationError):
        TaskPacket(
            goal="Run task",
            task_detail="Use explicit settings.",
            expected_outputs=["reports/out.json"],
            allowed_tools=["execute"],
            reference_hint="MEMORY/topics/FACTS.md",
        )


def test_director_output_requires_all_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        DirectorOutput(
            state="StopAndSynthesize",
            rationale="done",
            update_memory=[],
        )


def test_perform_next_task_rejects_redundant_deliverables_field() -> None:
    with pytest.raises(ValidationError):
        DirectorOutput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
            perform_next_task={
                "task_packet": {
                    "goal": "Run adsorption references",
                    "task_detail": "Enable D3 and keep slab setup fixed.",
                    "expected_outputs": ["results/adsorption/reference_energies.json"],
                    "allowed_tools": ["execute"],
                    "reference_hint": ["MEMORY/topics/FACTS.md"],
                },
                "deliverables": ["results/adsorption/reference_energies.json"],
            },
            minor_revise_proposal=None,
            major_revise_proposal=None,
            stop_and_synthesize=None,
            update_memory=[],
        )


def test_non_selected_payload_must_be_null() -> None:
    with pytest.raises(ValidationError, match="payload must be null when state=PerformNextTask"):
        DirectorOutput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
            perform_next_task={
                "task_packet": {
                    "goal": "Run adsorption references",
                    "task_detail": "Enable D3 and keep slab setup fixed.",
                    "expected_outputs": ["results/adsorption/reference_energies.json"],
                    "allowed_tools": ["execute"],
                    "reference_hint": ["MEMORY/topics/FACTS.md"],
                },
            },
            minor_revise_proposal=None,
            major_revise_proposal=None,
            stop_and_synthesize={"final_answer_md": "done"},
            update_memory=[],
        )


def test_fast_director_perform_next_task_requires_payload() -> None:
    with pytest.raises(ValidationError, match="requires its matching payload"):
        FastDirectorOutput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
            perform_next_task=None,
            stop_and_synthesize=None,
            update_memory=[],
        )


def test_fast_director_stop_and_synthesize_requires_payload() -> None:
    with pytest.raises(ValidationError, match="requires its matching payload"):
        FastDirectorOutput(
            state="StopAndSynthesize",
            rationale="Complete.",
            perform_next_task=None,
            stop_and_synthesize=None,
            update_memory=[],
        )


def test_fast_director_non_selected_payload_must_be_null() -> None:
    with pytest.raises(ValidationError, match="payload must be null when state=PerformNextTask"):
        FastDirectorOutput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
            perform_next_task={
                "task_packet": {
                    "goal": "Run adsorption references",
                    "task_detail": "Enable D3 and keep slab setup fixed.",
                    "expected_outputs": ["results/adsorption/reference_energies.json"],
                    "allowed_tools": ["execute"],
                    "reference_hint": ["MEMORY/topics/FACTS.md"],
                },
            },
            stop_and_synthesize={"final_answer_md": "done"},
            update_memory=[],
        )


def test_update_memory_requires_stop_state() -> None:
    with pytest.raises(ValidationError, match="update_memory must be \\[\\] unless state=StopAndSynthesize"):
        DirectorOutput(
            state="PerformNextTask",
            rationale="Dispatch next task.",
            perform_next_task={
                "task_packet": {
                    "goal": "Run adsorption references",
                    "task_detail": "Enable D3 and keep slab setup fixed.",
                    "expected_outputs": ["results/adsorption/reference_energies.json"],
                    "allowed_tools": ["execute"],
                    "reference_hint": ["MEMORY/topics/FACTS.md"],
                },
            },
            minor_revise_proposal=None,
            major_revise_proposal=None,
            stop_and_synthesize=None,
            update_memory=[{"topic": "MEMORY/topics/FACTS.md", "content": "Keep final D3 adsorption value."}],
        )


def test_stop_state_allows_update_memory() -> None:
    payload = FastDirectorOutput(
        state="StopAndSynthesize",
        rationale="Complete.",
        perform_next_task=None,
        stop_and_synthesize={"final_answer_md": "done"},
        update_memory=[{"topic": "MEMORY/topics/FILES.md", "content": "Record final summary artifact path."}],
    )
    assert len(payload.update_memory) == 1
