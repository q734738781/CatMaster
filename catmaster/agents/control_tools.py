from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from catmaster.tools.registry import sanitize_json_schema


class TaskFinishInput(BaseModel):
    """Signal that the current task is complete."""

    summary: str = Field(..., description="A detailed summary of what you have done in this task and the description of the task outcome.")
    artifacts: list[str] | None = Field(default=None, description="Relevant artifact paths for supporting your summary, if any.")


class TaskFailInput(BaseModel):
    """Signal that the current task failed and needs intervention."""

    error: str = Field(..., description="A detailed summary of the failure and the reason why you failed to complete the task.")
    needs_human: bool = Field(default=True, description="Whether a human must intervene.")
    hint: str | None = Field(default=None, description="Optional hint for recovery.")


CONTROL_TOOL_NAMES = {"task_finish", "task_fail"}


def _schema_for(name: str, model: type[BaseModel], *, strict: bool) -> dict[str, Any]:
    description = (model.__doc__ or f"Input for {name}").strip()
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": sanitize_json_schema(model.model_json_schema()),
        "strict": strict,
    }


def get_control_tool_schemas(*, strict: bool = False) -> list[dict]:
    return [
        _schema_for("task_finish", TaskFinishInput, strict=strict),
        _schema_for("task_fail", TaskFailInput, strict=strict),
    ]


__all__ = [
    "TaskFinishInput",
    "TaskFailInput",
    "CONTROL_TOOL_NAMES",
    "get_control_tool_schemas",
]
