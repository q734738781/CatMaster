from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "done", "error"]
CandidateAction = Literal["memory", "skill"]
CandidateStatus = Literal["proposed", "invalid", "approved", "rejected", "promoted", "conflict", "rolled_back"]
SKILL_GROUPS: tuple[str, ...] = (
    "materials_worker",
    "dynamics_worker",
    "ml_worker",
    "orca_xtb_worker",
    "research_specialist",
    "litreview_agent",
    "writing_specialist",
    "writing_quality",
    "execution",
)


@dataclass
class SelfEvolutionJob:
    job_id: str
    project_id: str
    run_id: str
    run_dir: str
    thread_id: str = ""
    trigger_kind: str = "post_run"
    status: JobStatus = "queued"
    attempt_count: int = 0
    candidate_id: str = ""
    model_config: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfEvolutionJob":
        fields = cls.__dataclass_fields__
        return cls(**{key: data.get(key) for key in fields if key in data})


@dataclass
class LearningCandidate:
    candidate_id: str
    project_id: str
    run_id: str
    thread_id: str
    action: CandidateAction
    status: CandidateStatus = "proposed"
    group: str = ""
    name: str = ""
    rationale: str = ""
    base_target_hash: str = ""
    bundle_hash: str = ""
    review: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def kind(self) -> str:
        return "memory_file" if self.action == "memory" else "skill_bundle"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind
        data["target"] = (
            {"path": "/memories/AGENTS.md"}
            if self.action == "memory"
            else {"group": self.group, "name": self.name}
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningCandidate":
        fields = cls.__dataclass_fields__
        return cls(**{key: data.get(key) for key in fields if key in data})


@dataclass
class ValidationReport:
    candidate_id: str
    valid: bool
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProposerResult(BaseModel):
    action: Literal["ignore", "memory", "skill"] = Field(
        description="Use ignore when the interaction does not justify durable workspace learning."
    )
    group: str = Field(
        default="",
        description="For skill only, the existing CatMaster skill group. Leave empty for ignore or memory.",
    )
    name: str = Field(
        default="",
        description="For skill only, the directory-matching skill name. Leave empty for ignore or memory.",
    )
    rationale: str = Field(default="", description="Concise evidence-grounded reason for this decision.")


class ReviewerResult(BaseModel):
    decision: Literal["approve", "reject"] = Field(
        description="Approve only the exact candidate skill bundle or complete memory file that was independently inspected."
    )
    rationale: str = Field(default="", description="Concise evidence-grounded review rationale.")


__all__ = [
    "CandidateAction",
    "CandidateStatus",
    "JobStatus",
    "LearningCandidate",
    "ProposerResult",
    "ReviewerResult",
    "SKILL_GROUPS",
    "SelfEvolutionJob",
    "ValidationReport",
]
