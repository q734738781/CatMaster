"""Unified structured-output schemas for ReAct agents.

These schemas are the sole terminal handoff contract for proposal,
director, and task-runner agents.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class TaskFileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Workspace-relative artifact path (file or directory). "
            "Always use project-relative paths; do not use absolute paths."
        ),
    )
    description: str = Field(..., description="Short one-line description.")
    kind: str = Field(
        ...,
        description=(
            "Artifact category. Common values: script, output, report, log. "
            "Use 'dir' when the path points to a directory bundle."
        ),
    )


class TaskDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(..., description="Short decision statement.")
    rationale: str = Field(..., description="Brief rationale (one short sentence preferred).")


class TaskPacket(BaseModel):
    """Structured packet for the next worker action.

    Keep this packet concise and execution-oriented: state the concrete goal,
    critical invariants, minimal done checks, and short discovery hints.
    """

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., description="Concrete worker goal in one short sentence.")
    task_detail: str = Field(
        ...,
        description=(
            "Execution detail only: include key invariants, critical parameters/defaults, "
            "and clear done criteria; avoid copying long context/proposal text."
        ),
    )
    expected_outputs: list[str] = Field(
        ...,
        description=(
            "Flat list of concrete deliverables/evidence strings for next-step decision. "
            "Do not use nested objects or tree-shaped payloads. Use [] only when truly none."
        ),
    )
    suggested_tools: list[str] = Field(
        ...,
        description="Optional concise tool-name hints for worker; advisory only. Use [] when no hint is needed.",
    )
    reference_hint: list[str] = Field(
        ...,
        description=(
            "Short discovery/start hints only (list of short strings); "
            "do not write long narrative paragraphs. Use [] when no hint is needed."
        ),
    )


# ---------------------------------------------------------------------------
# Proposal agent response
# ---------------------------------------------------------------------------

class ProposalOutput(BaseModel):
    """Structured output from the proposal agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "fail"] = Field(
        ...,
        description="Terminal proposal status. Use 'success' when a usable proposal is produced, otherwise 'fail'.",
    )
    proposal_md: str = Field(
        ...,
        description=(
            "Full proposal markdown. For status=success, provide complete content. "
            "For status=fail, use empty string ''."
        ),
    )
    work_packages: list[str] = Field(
        ...,
        description=(
            "Ordered high-level work packages (milestones, not tool-by-tool scripts). "
            "For status=fail, use []."
        ),
    )
    error: str = Field(
        ...,
        description="Failure summary for status=fail. For status=success, use empty string ''.",
    )
    needs_human: bool = Field(
        ...,
        description="Whether human intervention is required for status=fail. For status=success, set false.",
    )

    @model_validator(mode="after")
    def _validate_status_payload(self) -> "ProposalOutput":
        if self.status == "success":
            if not self.proposal_md.strip():
                raise ValueError("status=success requires non-empty proposal_md")
            if self.error.strip():
                raise ValueError("status=success requires error=''")
            if self.needs_human:
                raise ValueError("status=success requires needs_human=false")
            return self

        # status == "fail"
        if self.proposal_md.strip():
            raise ValueError("status=fail requires proposal_md=''")
        if self.work_packages:
            raise ValueError("status=fail requires work_packages=[]")
        if not self.error.strip():
            raise ValueError("status=fail requires non-empty error")
        return self


# ---------------------------------------------------------------------------
# Director agent response
# ---------------------------------------------------------------------------

class PerformNextTaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_packet: TaskPacket = Field(...)


class ReviseProposalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updated_proposal_md: str | None = Field(
        ...,
        description="Updated proposal markdown body; use null when markdown body text does not need changes.",
    )
    updated_work_packages: list[str] = Field(
        ...,
        description="Updated ordered work packages. Use [] when no work-package updates are needed.",
    )
    change_log: str | None = Field(
        ...,
        description="Short change summary (1-3 concise points). Use null when not applicable.",
    )
    needs_human: bool = Field(..., description="Whether human approval or clarification is required.")
    questions_for_human: list[str] = Field(
        ...,
        description="Short question strings for human decisions. Use [] when no questions remain.",
    )


class StopAndSynthesizePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer_md: str = Field(
        ...,
        description=(
            "Concise final user-facing answer markdown. Include final requested values with units and "
            "project-relative evidence paths; avoid long narrative report sections."
        ),
    )


class MemoryUpdate(BaseModel):
    """Requested durable memory update item emitted by director."""

    model_config = ConfigDict(extra="forbid")

    topic: Literal[
        "MEMORY/MEMORY.md",
        "MEMORY/topics/GOAL.md",
        "MEMORY/topics/FACTS.md",
        "MEMORY/topics/FILES.md",
        "MEMORY/topics/CONSTRAINTS.md",
        "MEMORY/topics/QUESTIONS.md",
        "MEMORY/topics/RUNBOOK.md",
    ] = Field(
        ...,
        description=(
            "Target memory file to update. Topic semantics: "
            "GOAL=objective/scope/success criteria; "
            "FACTS=verified scientific facts/results only; "
            "FILES=artifact path index and file roles; "
            "CONSTRAINTS=hard limits/policies; "
            "QUESTIONS=unresolved questions; "
            "RUNBOOK=reusable procedural checklist; "
            "MEMORY.md=index-level concise summary/pointers."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "Brief durable information to record for this topic; keep concise and reusable. "
            "Match topic semantics strictly: "
            "for FACTS write only verifiable facts/results (include units/conditions when relevant), "
            "for FILES write only project-relative artifact paths and their roles. "
            "Do not mix file-index notes into FACTS, and do not put scientific conclusions into FILES."
        ),
    )


class DirectorOutput(BaseModel):
    """Structured output from the director agent."""

    model_config = ConfigDict(extra="forbid")

    state: Literal[
        "PerformNextTask",
        "MinorReviseProposal",
        "MajorReviseProposal",
        "StopAndSynthesize",
    ] = Field(..., description="Decision state for the next action.")
    rationale: str = Field(
        ...,
        description=(
            "Very brief decision rationale (usually 1-2 sentences): explain decision cause/tradeoff only; "
            "do not repeat long context, proposal text, or tool logs."
        ),
    )

    perform_next_task: PerformNextTaskPayload | None = Field(
        ...,
        description="Payload for PerformNextTask branch (must be non-null only when state=PerformNextTask).",
    )
    minor_revise_proposal: ReviseProposalPayload | None = Field(
        ...,
        description="Payload for MinorReviseProposal branch (must be non-null only when state=MinorReviseProposal).",
    )
    major_revise_proposal: ReviseProposalPayload | None = Field(
        ...,
        description="Payload for MajorReviseProposal branch (must be non-null only when state=MajorReviseProposal).",
    )
    stop_and_synthesize: StopAndSynthesizePayload | None = Field(
        ...,
        description="Payload for StopAndSynthesize branch (must be non-null only when state=StopAndSynthesize).",
    )
    update_memory: list[MemoryUpdate] = Field(
        ...,
        description=(
            "Memory updates to persist at run end. Use [] when no durable scientific invariant/result/constraint changed."
        ),
    )

    @model_validator(mode="after")
    def _validate_state_payload(self) -> "DirectorOutput":
        payloads = {
            "PerformNextTask": self.perform_next_task,
            "MinorReviseProposal": self.minor_revise_proposal,
            "MajorReviseProposal": self.major_revise_proposal,
            "StopAndSynthesize": self.stop_and_synthesize,
        }
        active_payload = payloads.get(self.state)
        if active_payload is None:
            raise ValueError(f"{self.state} requires its matching payload")
        for branch, payload in payloads.items():
            if branch != self.state and payload is not None:
                raise ValueError(f"{branch} payload must be null when state={self.state}")
        if self.state != "StopAndSynthesize" and self.update_memory:
            raise ValueError("update_memory must be [] unless state=StopAndSynthesize")
        return self


# ---------------------------------------------------------------------------
# Fast-director response
# ---------------------------------------------------------------------------

class FastDirectorOutput(BaseModel):
    """Structured output from the fast-lane director agent."""

    model_config = ConfigDict(extra="forbid")

    state: Literal[
        "PerformNextTask",
        "StopAndSynthesize",
    ] = Field(..., description="Decision state for the next action in fast lane.")
    rationale: str = Field(
        ...,
        description=(
            "Very brief decision rationale (usually 1-2 sentences): explain decision cause/tradeoff only; "
            "do not repeat long context or tool logs."
        ),
    )

    perform_next_task: PerformNextTaskPayload | None = Field(
        ...,
        description="Payload for PerformNextTask branch (must be non-null only when state=PerformNextTask).",
    )
    stop_and_synthesize: StopAndSynthesizePayload | None = Field(
        ...,
        description="Payload for StopAndSynthesize branch (must be non-null only when state=StopAndSynthesize).",
    )
    update_memory: list[MemoryUpdate] = Field(
        ...,
        description=(
            "Memory updates to persist at run end. Use [] when no durable scientific invariant/result/constraint changed."
        ),
    )

    @model_validator(mode="after")
    def _validate_state_payload(self) -> "FastDirectorOutput":
        payloads = {
            "PerformNextTask": self.perform_next_task,
            "StopAndSynthesize": self.stop_and_synthesize,
        }
        active_payload = payloads.get(self.state)
        if active_payload is None:
            raise ValueError(f"{self.state} requires its matching payload")
        for branch, payload in payloads.items():
            if branch != self.state and payload is not None:
                raise ValueError(f"{branch} payload must be null when state={self.state}")
        if self.state != "StopAndSynthesize" and self.update_memory:
            raise ValueError("update_memory must be [] unless state=StopAndSynthesize")
        return self


class MemoryPatchOutput(BaseModel):
    """Structured output from memory patcher agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "blocked"] = Field(..., description="Terminal memory patch status.")
    summary: str = Field(..., description="Concise patch result summary.")
    applied_topics: list[str] = Field(
        ...,
        description="Topics/files that were successfully updated. Use [] when none.",
    )
    error: str = Field(
        ...,
        description="Failure reason for status=blocked. For status=done, use empty string ''.",
    )
    needs_human: bool = Field(
        ...,
        description="Whether a human must intervene for status=blocked. For status=done, set false.",
    )

    @model_validator(mode="after")
    def _validate_status_payload(self) -> "MemoryPatchOutput":
        if self.status == "done":
            if self.error.strip():
                raise ValueError("status=done requires error=''")
            if self.needs_human:
                raise ValueError("status=done requires needs_human=false")
            return self
        if not self.error.strip():
            raise ValueError("status=blocked requires non-empty error")
        return self


# ---------------------------------------------------------------------------
# Task runner agent response
# ---------------------------------------------------------------------------

class TaskOutput(BaseModel):
    """Structured output from the task runner agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "blocked"] = Field(
        ...,
        description=(
            "Task terminal status. Use 'done' only when task acceptance criteria are satisfied; "
            "use 'blocked' when progress is blocked by unresolved issues."
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "Concise task outcome/failure summary with key file pointers. "
            "Do not paste long tables/logs/scripts; cite file paths instead."
        ),
    )
    facts: list[str] = Field(
        ...,
        description=(
            "Only high-value verified facts reusable downstream (include units/conditions/evidence path when relevant). "
            "Do not restate command traces. Use [] when none."
        ),
    )
    files: list[TaskFileRecord] = Field(
        ...,
        description=(
            "Only key reproducibility/user-facing artifact paths (files or directories). "
            "Use a directory path when it is a clearer bundle-level reference. "
            "Avoid listing low-value scratch files. Use [] when none."
        ),
    )
    constraints: list[str] = Field(..., description="Only new/changed constraints. Use [] when none.")
    open_questions: list[str] = Field(
        ...,
        description="Only unresolved blockers affecting next steps. Do not include speculative questions. Use [] when none.",
    )
    decisions: list[TaskDecisionRecord] = Field(
        ...,
        description="Only decisions materially affecting downstream work; avoid duplicating summary/facts content. Use [] when none.",
    )
    next_steps: list[str] = Field(..., description="Only immediate actionable next steps. Use [] when no next step is required.")
    artifacts: list[str] = Field(..., description="Supporting artifact paths when needed. Use [] when none.")
    error: str = Field(
        ...,
        description="Failure reason for status=blocked. For status=done, use empty string ''.",
    )
    needs_human: bool = Field(
        ...,
        description="Whether a human must intervene for status=blocked. For status=done, set false.",
    )
    hint: str = Field(
        ...,
        description="Optional concise recovery hint for status=blocked. For status=done, use empty string ''.",
    )

    @model_validator(mode="after")
    def _validate_status_payload(self) -> "TaskOutput":
        if self.status == "done":
            if self.error.strip():
                raise ValueError("status=done requires error=''")
            if self.needs_human:
                raise ValueError("status=done requires needs_human=false")
            if self.hint.strip():
                raise ValueError("status=done requires hint=''")
            return self

        # status == "blocked"
        if not self.error.strip():
            raise ValueError("status=blocked requires non-empty error")
        return self


__all__ = [
    "TaskFileRecord",
    "TaskDecisionRecord",
    "TaskPacket",
    "MemoryUpdate",
    "ProposalOutput",
    "PerformNextTaskPayload",
    "ReviseProposalPayload",
    "StopAndSynthesizePayload",
    "DirectorOutput",
    "FastDirectorOutput",
    "TaskOutput",
    "MemoryPatchOutput",
]
