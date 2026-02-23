from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from catmaster.tools.registry import sanitize_json_schema


class TaskFileRecord(BaseModel):
    path: str = Field(..., description="Workspace-relative file path.")
    description: str | None = Field(default=None, description="One-line description.")
    kind: str | None = Field(default=None, description="File kind, e.g. input/output/report.")


class TaskDecisionRecord(BaseModel):
    decision: str = Field(..., description="Decision statement.")
    rationale: str = Field(..., description="Brief rationale for the decision.")


class TaskResultPayload(BaseModel):
    summary: str = Field(..., description="Concise summary of task outcome.")
    facts: list[str] | None = Field(default=None, description="Verified facts from this task.")
    files: list[TaskFileRecord] | None = Field(default=None, description="Important files generated or used.")
    constraints: list[str] | None = Field(default=None, description="New or clarified constraints.")
    open_questions: list[str] | None = Field(default=None, description="Open questions remaining after this task.")
    decisions: list[TaskDecisionRecord] | None = Field(default=None, description="Key decisions with rationale.")
    next_steps: list[str] | None = Field(default=None, description="Suggested next steps.")
    artifacts: list[str] | None = Field(default=None, description="Supporting artifact paths.")


class TaskFinishInput(BaseModel):
    """Signal that the current task is complete."""

    summary: str = Field(..., description="A concise summary of completed task outcome.")
    facts: list[str] | None = Field(default=None, description="Verified facts from this task.")
    files: list[TaskFileRecord] | None = Field(default=None, description="Important files generated or used.")
    constraints: list[str] | None = Field(default=None, description="New or clarified constraints.")
    open_questions: list[str] | None = Field(default=None, description="Open questions remaining after this task.")
    decisions: list[TaskDecisionRecord] | None = Field(default=None, description="Key decisions with rationale.")
    next_steps: list[str] | None = Field(default=None, description="Suggested next steps.")
    artifacts: list[str] | None = Field(default=None, description="Relevant artifact paths supporting the summary.")


class TaskFailInput(BaseModel):
    """Signal that the current task failed and needs intervention."""

    error: str = Field(..., description="A detailed summary of the failure and the reason why you failed to complete the task.")
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


__all__ = [
    "TaskFinishInput",
    "TaskFailInput",
    "CONTROL_TOOL_NAMES",
    "get_control_tool_schemas",
]
