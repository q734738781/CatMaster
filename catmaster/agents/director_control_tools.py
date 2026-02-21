from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from catmaster.tools.registry import sanitize_json_schema


class TaskPacket(BaseModel):
    """Structured packet for a worker task."""

    goal: str = Field(..., description="Concrete task goal for worker execution.")
    success_criteria: str = Field(..., description="Completion criteria for this task.")
    expected_outputs: list[str] = Field(default_factory=list, description="Expected outputs and deliverables.")
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Optional tool-name hints for worker; advisory only.",
    )
    memory_hints: list[str] = Field(
        default_factory=list,
        description="Keyword hints for memory retrieval (rg query terms).",
    )
    path_hints: list[str] = Field(
        default_factory=list,
        description="Optional workspace-relative path hints relevant to the task.",
    )


class DirectorDecideInput(BaseModel):
    """Return the director's decision for the next action."""

    state: Literal[
        "PerformNextTask",
        "MinorReviseProposal",
        "MajorReviseProposal",
        "StopAndSynthesize",
    ] = Field(..., description="Decision state for the next action.")
    rationale: str = Field(..., description="Reasoning behind the decision.")

    # PerformNextTask fields
    task_packet: TaskPacket | None = Field(default=None, description="Structured worker task packet.")
    next_task_goal: str | None = Field(default=None, description="Concrete task goal to execute next.")
    suggested_tools: list[str] | None = Field(
        default=None,
        description="Optional suggested tool names for task runner; hints only, not mandatory.",
    )
    success_criteria: str | None = Field(default=None, description="Success criteria for the next task.")
    expected_outputs: str | None = Field(default=None, description="Expected outputs for the next task.")

    # Proposal revision fields
    updated_proposal_md: str | None = Field(default=None, description="Updated proposal markdown.")
    updated_work_packages: list[str] | None = Field(default=None, description="Updated ordered work packages.")
    change_log: str | None = Field(default=None, description="Summary of proposal changes.")

    # Major revise fields
    needs_human: bool | None = Field(default=None, description="Whether human approval is required.")
    questions_for_human: list[str] | None = Field(default=None, description="Questions for human decision.")

    # Stop fields
    stop_reason: str | None = Field(default=None, description="Reason for stopping.")
    deliverables: list[str] | None = Field(default=None, description="Expected deliverables when stopping.")


DIRECTOR_CONTROL_TOOL_NAMES = {"director_decide"}


def _schema_for(name: str, model: type[BaseModel], *, strict: bool) -> dict[str, Any]:
    description = (model.__doc__ or f"Input for {name}").strip()
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": sanitize_json_schema(model.model_json_schema()),
        "strict": strict,
    }


def get_director_control_tool_schemas(*, strict: bool = False) -> list[dict]:
    return [
        _schema_for("director_decide", DirectorDecideInput, strict=strict),
    ]


__all__ = [
    "DirectorDecideInput",
    "DIRECTOR_CONTROL_TOOL_NAMES",
    "get_director_control_tool_schemas",
]
