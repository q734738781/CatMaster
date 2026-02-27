from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from catmaster.tools.registry import sanitize_json_schema


class TaskFileRecord(BaseModel):
    path: str = Field(..., description="Workspace-relative file path.")
    description: str | None = Field(default=None, description="Short one-line description.")
    kind: str | None = Field(default=None, description="File kind, e.g. script/output/report/log.")


class TaskDecisionRecord(BaseModel):
    decision: str = Field(..., description="Short decision statement.")
    rationale: str = Field(..., description="Brief rationale (one short sentence preferred).")


class TaskResultPayload(BaseModel):
    summary: str = Field(..., description="Concise task outcome summary with key file pointers.")
    facts: list[str] | None = Field(default=None, description="Only high-signal verified facts from this task.")
    files: list[TaskFileRecord] | None = Field(default=None, description="Only key reproducibility/user-facing files.")
    constraints: list[str] | None = Field(default=None, description="Only new/changed constraints.")
    open_questions: list[str] | None = Field(default=None, description="Only unresolved blockers affecting next steps.")
    decisions: list[TaskDecisionRecord] | None = Field(default=None, description="Only decisions materially affecting downstream work.")
    next_steps: list[str] | None = Field(default=None, description="Only immediate actionable next steps.")
    artifacts: list[str] | None = Field(default=None, description="Supporting artifact paths when needed.")


class TaskFinishInput(BaseModel):
    """Signal task completion with concise, non-duplicated structured handoff."""

    summary: str = Field(..., description="Concise completed-task summary with key file pointers.")
    facts: list[str] | None = Field(default=None, description="Only high-signal verified facts from this task.")
    files: list[TaskFileRecord] | None = Field(default=None, description="Only key files generated or used.")
    constraints: list[str] | None = Field(default=None, description="Only new/changed constraints.")
    open_questions: list[str] | None = Field(default=None, description="Only unresolved blockers affecting next tasks.")
    decisions: list[TaskDecisionRecord] | None = Field(default=None, description="Only key decisions with short rationale.")
    next_steps: list[str] | None = Field(default=None, description="Only immediate suggested next steps.")
    artifacts: list[str] | None = Field(default=None, description="Relevant supporting artifact paths when needed.")


class TaskFailInput(BaseModel):
    """Signal that the current task failed and needs intervention."""

    error: str = Field(..., description="A detailed summary of why the task failed.")
    needs_human: bool = Field(default=True, description="Whether a human must intervene.")
    hint: str | None = Field(default=None, description="Optional hint for recovery.")
    partial_result: TaskResultPayload | None = Field(
        default=None,
        description="Optional partial structured result already obtained before failure.",
    )


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


def _make_control_tool(name: str, model: type[BaseModel]) -> StructuredTool:
    description = (model.__doc__ or f"Input for {name}").strip()

    def _tool(**kwargs: Any) -> str:
        payload = model.model_validate(kwargs).model_dump(mode="json", exclude_none=True)
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
        _make_control_tool("task_finish", TaskFinishInput),
        _make_control_tool("task_fail", TaskFailInput),
    ]


__all__ = [
    "TaskFileRecord",
    "TaskDecisionRecord",
    "TaskResultPayload",
    "TaskFinishInput",
    "TaskFailInput",
    "CONTROL_TOOL_NAMES",
    "get_control_tool_schemas",
    "as_langchain_control_tools",
]
