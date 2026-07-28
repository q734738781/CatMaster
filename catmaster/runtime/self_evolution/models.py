from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


JobStatus = Literal["queued", "running", "done", "error"]
CandidateAction = Literal["memory", "skill"]
CandidateStatus = Literal[
    "proposed",
    "invalid",
    "reviewed",
    "approved",
    "rejected",
    "promoted",
    "conflict",
    "rolled_back",
]
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


class ReviewChangePoint(BaseModel):
    title: str = Field(default="", description="Short name for one behaviorally meaningful change.")
    before: str = Field(default="", description="Plain-language description of the prior behavior or rule.")
    after: str = Field(default="", description="Plain-language description of the proposed future behavior.")
    evidence: str = Field(default="", description="Concrete trace evidence supporting this change.")
    evidence_source: str = Field(
        default="",
        description="Who supplied the evidence: user, repeated outcome, concrete failure, or agent inference.",
    )
    impact: str = Field(default="", description="Likely operational cost, benefit, or risk of this change.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_nulls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("title", "before", "after", "evidence", "evidence_source", "impact"):
            if data.get(key) is None:
                data[key] = ""
        return data


class ProportionalityAssessment(BaseModel):
    status: Literal["pass", "warning", "fail"] = Field(
        default="warning",
        description="Whether the proposed work is proportionate to the supported risk and scope.",
    )
    explanation: str = Field(default="", description="Short explanation of the proportionality judgment.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_nulls(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("status") is None:
            data["status"] = "warning"
        if data.get("explanation") is None:
            data["explanation"] = ""
        return data


class ReviewerResult(BaseModel):
    recommendation: Literal["approve", "reject", "needs_revision"] = Field(
        description=(
            "AI recommendation for the exact inspected candidate. This never authorizes skill promotion; "
            "a human remains the final promotion authority."
        )
    )
    summary: str = Field(default="", description="One plain-language sentence describing the candidate.")
    change_points: list[ReviewChangePoint] = Field(
        default_factory=list,
        description="One entry for every behaviorally meaningful change in the exact candidate.",
    )
    scope_assessment: str = Field(
        default="",
        description="Why the proposed activation scope is or is not supported by the trace.",
    )
    proportionality_assessment: ProportionalityAssessment = Field(
        default_factory=ProportionalityAssessment,
        description="Cost and burden assessment relative to the supported risk and task scope.",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Concrete overreach, ambiguity, conflict, or missing-evidence concerns; use an empty list if none.",
    )
    human_checks: list[str] = Field(
        default_factory=list,
        description="Small actionable checklist for the human promotion decision; use an empty list if none.",
    )
    rationale: str = Field(default="", description="Concise evidence-grounded review rationale.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("recommendation") and data.get("decision") in {"approve", "reject", "needs_revision"}:
            data["recommendation"] = data["decision"]
        for key in ("summary", "scope_assessment", "rationale"):
            if data.get(key) is None:
                data[key] = ""
        for key in ("change_points", "concerns", "human_checks"):
            if data.get(key) is None:
                data[key] = []
        if data.get("proportionality_assessment") is None:
            data["proportionality_assessment"] = {}
        return data

    @property
    def decision(self) -> Literal["approve", "reject", "needs_revision"]:
        """Compatibility view for callers that still use the former field name."""

        return self.recommendation


__all__ = [
    "CandidateAction",
    "CandidateStatus",
    "JobStatus",
    "LearningCandidate",
    "ProportionalityAssessment",
    "ProposerResult",
    "ReviewChangePoint",
    "ReviewerResult",
    "SKILL_GROUPS",
    "SelfEvolutionJob",
    "ValidationReport",
]
