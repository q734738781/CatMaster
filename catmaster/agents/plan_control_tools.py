from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from catmaster.tools.registry import sanitize_json_schema


class PlanFinishInput(BaseModel):
    """Return the finalized plan and task list."""

    todo: list[str] = Field(..., description="Ordered task list with milestone deliverables.")
    plan_description: str = Field(..., description="Brief rationale for the plan.")


class PlanFailInput(BaseModel):
    """Signal that planning failed and needs human intervention."""

    error: str = Field(..., description="Summary of the planning failure.")
    needs_human: bool = Field(default=True, description="Whether a human must intervene.")
    hint: str | None = Field(default=None, description="Optional hint for recovery.")


PLAN_CONTROL_TOOL_NAMES = {"plan_finish", "plan_fail"}


def _schema_for(name: str, model: type[BaseModel], *, strict: bool) -> dict[str, Any]:
    description = (model.__doc__ or f"Input for {name}").strip()
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": sanitize_json_schema(model.model_json_schema()),
        "strict": strict,
    }


def get_plan_control_tool_schemas(*, strict: bool = False) -> list[dict]:
    return [
        _schema_for("plan_finish", PlanFinishInput, strict=strict),
        _schema_for("plan_fail", PlanFailInput, strict=strict),
    ]


__all__ = [
    "PlanFinishInput",
    "PlanFailInput",
    "PLAN_CONTROL_TOOL_NAMES",
    "get_plan_control_tool_schemas",
]
