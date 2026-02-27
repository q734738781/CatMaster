"""Unified structured-output schemas for ReAct agents.

These schemas are used when agent termination mode is ``response_format``.
When termination mode is ``control_tools``, proposal/director/task use
explicit control tools instead.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared sub-models (carried over from old control_tools.py)
# ---------------------------------------------------------------------------

class TaskFileRecord(BaseModel):
    path: str = Field(..., description="Workspace-relative file path.")
    description: str | None = Field(default=None, description="Short one-line description.")
    kind: str | None = Field(default=None, description="File kind, e.g. script/output/report/log.")


class TaskDecisionRecord(BaseModel):
    decision: str = Field(..., description="Short decision statement.")
    rationale: str = Field(..., description="Brief rationale (one short sentence preferred).")


class TaskPacket(BaseModel):
    """Structured packet for a worker task."""

    goal: str = Field(..., description="Concrete worker goal in one short sentence.")
    task_detail: str = Field(
        ...,
        description="Concise execution checklist: non-negotiable constraints, key parameters, and minimal done checks.",
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
        description="Short high-value discovery hints (memory files, rg keywords, done-check points); avoid exhaustive lists.",
    )


# ---------------------------------------------------------------------------
# Proposal agent response
# ---------------------------------------------------------------------------

class ProposalOutput(BaseModel):
    """Structured output from the proposal agent."""

    status: Literal["success", "fail"] = Field(
        ..., description="Whether the proposal was successfully created.",
    )
    proposal_md: str = Field(
        default="",
        description="Full proposal in markdown format. Required when status=success.",
    )
    work_packages: list[str] = Field(
        default_factory=list,
        description="Ordered list of work packages / milestones. Required when status=success.",
    )
    error: str = Field(
        default="",
        description="Summary of the failure. Populated when status=fail.",
    )
    needs_human: bool = Field(
        default=False,
        description="Whether a human must intervene (status=fail only).",
    )


# ---------------------------------------------------------------------------
# Director agent response
# ---------------------------------------------------------------------------

class DirectorOutput(BaseModel):
    """Structured output from the director agent."""

    state: Literal[
        "PerformNextTask",
        "MinorReviseProposal",
        "MajorReviseProposal",
        "StopAndSynthesize",
    ] = Field(..., description="Decision state for the next action.")
    rationale: str = Field(..., description="Very brief decision rationale (usually 1-2 sentences).")

    task_packet: TaskPacket | None = Field(
        default=None, description="Structured worker task packet (PerformNextTask only).",
    )

    updated_proposal_md: str | None = Field(default=None, description="Updated proposal markdown.")
    updated_work_packages: list[str] | None = Field(default=None, description="Updated ordered work packages.")
    change_log: str | None = Field(default=None, description="Summary of proposal changes.")

    needs_human: bool | None = Field(default=None, description="Whether human approval is required.")
    questions_for_human: list[str] | None = Field(default=None, description="Questions for human decision.")

    stop_reason: str | None = Field(default=None, description="Reason for stopping.")
    deliverables: list[str] | None = Field(default=None, description="Expected deliverables when stopping.")

    @model_validator(mode="after")
    def _validate_state_payload(self) -> "DirectorOutput":
        if self.state == "PerformNextTask" and self.task_packet is None:
            raise ValueError("PerformNextTask requires task_packet")
        return self


# ---------------------------------------------------------------------------
# Task runner agent response
# ---------------------------------------------------------------------------

class TaskOutput(BaseModel):
    """Structured output from the task runner agent."""

    status: Literal["done", "blocked"] = Field(
        ..., description="Whether the task completed or is blocked.",
    )
    summary: str = Field(..., description="Concise task outcome/failure summary with key file pointers.")
    facts: list[str] = Field(default_factory=list, description="Only high-signal verified facts from this task.")
    files: list[TaskFileRecord] = Field(default_factory=list, description="Only key reproducibility/user-facing files.")
    constraints: list[str] = Field(default_factory=list, description="Only new/changed constraints.")
    open_questions: list[str] = Field(default_factory=list, description="Only unresolved blockers affecting next steps.")
    decisions: list[TaskDecisionRecord] = Field(default_factory=list, description="Only decisions materially affecting downstream work.")
    next_steps: list[str] = Field(default_factory=list, description="Only immediate actionable next steps.")
    artifacts: list[str] = Field(default_factory=list, description="Supporting artifact paths when needed.")
    error: str = Field(
        default="",
        description="Detailed failure reason. Populated when status=blocked.",
    )
    needs_human: bool = Field(
        default=False,
        description="Whether a human must intervene (status=blocked only).",
    )
    hint: str = Field(
        default="",
        description="Optional hint for recovery (status=blocked only).",
    )


__all__ = [
    "TaskFileRecord",
    "TaskDecisionRecord",
    "TaskPacket",
    "ProposalOutput",
    "DirectorOutput",
    "TaskOutput",
]
