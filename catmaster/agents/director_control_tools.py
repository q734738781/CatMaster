from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from catmaster.tools.registry import sanitize_json_schema


class TaskPacket(BaseModel):
    """Structured packet for a worker task."""

    goal: str = Field(
        ...,
        description="Concrete worker goal in one short sentence.",
    )
    task_detail: str = Field(
        ...,
        description=(
            "Concise execution checklist: non-negotiable constraints, key parameters, and minimal done checks."
        ),
    )
    expected_outputs: list[str] = Field(
        default_factory=list,
        description="Only concrete deliverables needed for next-step decision.",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Optional concise tool-name hints for worker; advisory only.",
    )
    reference_hint: list[str] = Field(
        default_factory=list,
        description=(
            "Short high-value discovery hints (memory files, rg keywords, done-check points); avoid exhaustive lists."
        ),
    )


class DirectorDecideInput(BaseModel):
    """Return the director decision for next action using compact, high-signal fields."""

    state: Literal[
        "PerformNextTask",
        "MinorReviseProposal",
        "MajorReviseProposal",
        "StopAndSynthesize",
    ] = Field(..., description="Decision state for the next action.")
    rationale: str = Field(
        ...,
        description="Very brief decision rationale (usually 1-2 sentences).",
    )

    task_packet: TaskPacket | None = Field(default=None, description="Structured worker task packet.")

    updated_proposal_md: str | None = Field(default=None, description="Updated proposal markdown.")
    updated_work_packages: list[str] | None = Field(default=None, description="Updated ordered work packages.")
    change_log: str | None = Field(default=None, description="Summary of proposal changes.")

    needs_human: bool | None = Field(default=None, description="Whether human approval is required.")
    questions_for_human: list[str] | None = Field(default=None, description="Questions for human decision.")

    stop_reason: str | None = Field(default=None, description="Reason for stopping.")
    deliverables: list[str] | None = Field(default=None, description="Expected deliverables when stopping.")

    @model_validator(mode="after")
    def _validate_state_payload(self) -> "DirectorDecideInput":
        if self.state == "PerformNextTask" and self.task_packet is None:
            raise ValueError("PerformNextTask requires task_packet")
        return self


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


def _make_control_tool(name: str, model: type[BaseModel]) -> StructuredTool:
    description = (model.__doc__ or f"Input for {name}").strip()

    def _tool(**kwargs: Any) -> str:
        normalized = dict(kwargs)
        task_packet = normalized.get("task_packet")
        if isinstance(task_packet, BaseModel):
            normalized["task_packet"] = task_packet.model_dump(mode="json", exclude_none=True)
        payload = model.model_validate(normalized).model_dump(mode="json", exclude_none=True)
        return json.dumps(
            {
                "status": "control",
                "tool_name": name,
                "payload": payload,
            },
            ensure_ascii=False,
        )

    _tool.__name__ = name
    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
        args_schema=model,
        return_direct=True,
    )


def as_langchain_control_tools() -> list[StructuredTool]:
    return [
        _make_control_tool("director_decide", DirectorDecideInput),
    ]


__all__ = [
    "TaskPacket",
    "DirectorDecideInput",
    "DIRECTOR_CONTROL_TOOL_NAMES",
    "get_director_control_tool_schemas",
    "as_langchain_control_tools",
]
