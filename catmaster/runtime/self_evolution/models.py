from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from catmaster.runtime.skills.roots import ACTIVE_SKILL_GROUPS


JobStatus = Literal["queued", "running", "done", "error", "recovery_review"]
ReflectionKind = Literal[
    "no_change",
    "execution_lapse",
    "workspace_preference",
    "skill_revision",
    "skill_discovery",
]
SignalKind = Literal[
    "workspace_preference",
    "skill_revision",
    "skill_discovery",
]
ObservationStatus = Literal["open", "consolidated"]
CandidateRoute = Literal[
    "workspace_preference",
    "amend_existing_skill",
    "new_skill",
]
CandidateAction = Literal["memory", "skill"]
CandidateStatus = Literal[
    "pending",
    "review",
    "revision",
    "canary",
    "stable",
    "rejected",
    "inactive",
]
CANONICAL_CANDIDATE_STATUSES: frozenset[str] = frozenset(
    {"pending", "review", "revision", "canary", "stable", "rejected", "inactive"}
)
DeltaOperation = str
# Compatibility export for existing callers. Ownership lives in
# ``catmaster.runtime.skills.roots`` with the active runtime.
SKILL_GROUPS: tuple[str, ...] = ACTIVE_SKILL_GROUPS


@dataclass
class SelfEvolutionJob:
    job_id: str
    project_id: str
    run_id: str
    run_dir: str
    thread_id: str = ""
    episode_id: str = ""
    selected_item_ref: str = ""
    trigger_kind: str = "post_run"
    status: JobStatus = "queued"
    attempt_count: int = 0
    candidate_id: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)
    predecessor_job_id: str = ""
    model_config: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    owner: str = ""
    lease_until: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfEvolutionJob":
        fields = cls.__dataclass_fields__
        return cls(**{key: data.get(key) for key in fields if key in data})


@dataclass
class Observation:
    """One model-grounded durable-change signal.

    Workspace identity is deliberately omitted because the database path owns
    that scope. Do not add confidence, topic, model, token, or checksum fields.
    """

    observation_id: str
    run_id: str
    thread_id: str
    signal_kind: SignalKind
    target: str
    claim: str
    resolved_target: str = ""
    job_id: str = ""
    episode_id: str = ""
    item_ref: str = ""
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    outcome_ref: str = ""
    status: ObservationStatus = "open"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        fields = cls.__dataclass_fields__
        return cls(**{key: data.get(key) for key in fields if key in data})


@dataclass(frozen=True)
class CandidateRevision:
    candidate_id: str
    revision: int
    route: CandidateRoute
    target: dict[str, str]
    delta_operation: DeltaOperation
    evidence_ids: tuple[str, ...]
    applicability_boundary: tuple[str, ...]
    non_applicability: tuple[str, ...]
    expected_step_change: str
    created_at: str = ""

    @property
    def version(self) -> str:
        return f"r{self.revision:04d}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "evidence_ids",
            "applicability_boundary",
            "non_applicability",
        ):
            data[key] = list(data[key])
        data["version"] = self.version
        return data


@dataclass
class LearningCandidate:
    candidate_id: str
    project_id: str
    run_id: str
    thread_id: str
    action: CandidateAction
    episode_id: str = ""
    status: CandidateStatus = "pending"
    route: CandidateRoute = "amend_existing_skill"
    group: str = ""
    name: str = ""
    rationale: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    revision: int = 1
    base_target_hash: str = ""
    bundle_hash: str = ""
    review: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def kind(self) -> str:
        return "memory_file" if self.action == "memory" else "skill_bundle"

    @property
    def version(self) -> str:
        return f"r{max(1, int(self.revision or 1)):04d}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind
        data["version"] = self.version
        data["target"] = (
            {"path": "/memories/AGENTS.md"}
            if self.action == "memory"
            else {"group": self.group, "name": self.name}
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningCandidate":
        fields = cls.__dataclass_fields__
        values = {key: data.get(key) for key in fields if key in data}
        if not str(values.get("action") or "").strip():
            raise ValueError("candidate action is required")
        if not str(values.get("route") or "").strip():
            raise ValueError("candidate route is required")
        values["status"] = normalize_candidate_status(values.get("status"))
        return cls(**values)


def normalize_candidate_status(value: Any) -> CandidateStatus:
    """Validate the lifecycle state used by all active code paths."""

    status = str(value or "pending").strip()
    if status in CANONICAL_CANDIDATE_STATUSES:
        return status  # type: ignore[return-value]
    raise ValueError(f"unsupported candidate status: {status}")


@dataclass
class SkillRun:
    run_id: str
    skill_name: str
    skill_version: str
    presented: bool = False
    read: bool = False
    helper_used: bool = False
    outcome: str = "unknown"
    false_activation: bool = False
    partial: bool = False

    @property
    def used(self) -> bool:
        """A bounded activation signal, not a claim of causal task credit."""

        return bool(self.read or self.helper_used)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "used": self.used}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillRun":
        fields = cls.__dataclass_fields__
        return cls(**{key: data.get(key) for key in fields if key in data})


@dataclass
class ValidationReport:
    candidate_id: str
    valid: bool
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    loadable: bool = True
    repair_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextResult(BaseModel):
    """A completed textual response, without an inferred actionable decision."""

    text: str


class ReflectionResult(BaseModel):
    """Semantic judgment over one complete episode trajectory."""

    kind: ReflectionKind = Field(
        description=(
            "Choose no_change when no durable workspace behavior is justified; "
            "execution_lapse when existing guidance was sufficient but the agent did "
            "not follow it; workspace_preference for an explicit durable user "
            "convention; skill_revision for an existing skill; or skill_discovery "
            "for a genuinely independent reusable method."
        )
    )
    group: str = Field(
        default="",
        description="Exact existing or proposed CatMaster skill group. Leave empty for non-skill judgments.",
    )
    name: str = Field(
        default="",
        description=(
            "Exact existing/proposed skill directory name, or a concise stable topic "
            "name for workspace_preference. Leave empty for no_change or execution_lapse."
        ),
    )
    change: str = Field(
        default="",
        description="One concise, evidence-grounded behavior change. Leave empty when no durable change is justified.",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Exact source refs from the supplied trajectory that support the judgment.",
    )
    rationale: str = Field(
        default="",
        description="Concise explanation grounded in the trajectory and result.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_optional_controls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("group", "name", "change", "rationale"):
            if data.get(key) is None:
                data[key] = ""
        if data.get("evidence_refs") is None:
            data["evidence_refs"] = []
        return data


class ReflectionBatch(BaseModel):
    """Submit the final reflection findings after inspecting the authorized trace.

    Use this tool to submit findings for processing; explanations belong in text fields.
    """

    items: list[ReflectionResult] = Field(
        ...,
        min_length=1,
        description=(
            "All independently evidenced findings. Multiple findings may share "
            "a target; use no_change when no durable finding is supported."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_singular_result(cls, value: Any) -> Any:
        if isinstance(value, ReflectionResult):
            return {"items": [value]}
        if isinstance(value, dict) and "items" not in value and "kind" in value:
            return {"items": [value]}
        if isinstance(value, dict) and value.get("items") is None:
            return {**value, "items": []}
        return value

class ProposerResult(BaseModel):
    """Submit the final proposal decision after editing any candidate files.

    Use this tool to submit a proposal decision. Files contain the edits; rationale explains them.
    """

    action: Literal["defer", "ignore", "memory", "skill"] = Field(
        description=(
            "Use defer when a possible durable pattern needs more real episodes; use ignore "
            "when the current evidence is non-durable and should be consumed without a candidate."
        )
    )
    group: str = Field(
        default="",
        description="For skill only, the existing CatMaster skill group. Leave empty for defer, ignore, or memory.",
    )
    name: str = Field(
        default="",
        description="For skill only, one safe target directory name. Leave empty for defer, ignore, or memory.",
    )
    rationale: str = Field(default="", description="Concise evidence-grounded reason for this decision.")
    delta_operation: DeltaOperation = Field(
        default="replace",
        description=(
            "Optional short description of the edit mode, such as add, delete, "
            "replace, merge, or another accurate label."
        ),
    )
    applicability_boundary: list[str] = Field(
        default_factory=list,
        description="Concrete situations where the proposed behavior applies.",
    )
    non_applicability: list[str] = Field(
        default_factory=list,
        description="Concrete situations where the behavior must not activate.",
    )
    expected_step_change: str = Field(
        default="",
        description="Which decision or unnecessary step this revision is expected to change.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_optional_controls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("group", "name", "rationale", "expected_step_change"):
            if data.get(key) is None:
                data[key] = ""
        for key in ("applicability_boundary", "non_applicability"):
            if data.get(key) is None:
                data[key] = []
        if data.get("delta_operation") is None:
            data["delta_operation"] = "replace"
        return data


class ReviewChangePoint(BaseModel):
    title: str = Field(default="", description="Short name for one behaviorally meaningful change.")
    before: str = Field(default="", description="Plain-language description of the prior behavior or rule.")
    after: str = Field(default="", description="Plain-language description of the proposed future behavior.")
    evidence: str = Field(default="", description="Concrete evidence supporting this change.")
    evidence_source: str = Field(
        default="",
        description="Source type: user correction, verified outcome, counterexample, or unverified hypothesis.",
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
    """Submit the final review decision for the inspected candidate revision.

    Use this tool to submit a review decision; explanations belong in text fields.
    """

    recommendation: Literal["approve", "reject", "needs_revision"] = Field(
        description=(
            "Semantic decision for the exact inspected revision. Approve makes a valid revision "
            "eligible for workspace auto selection; needs_revision requests another bounded "
            "automatic revision; reject keeps this revision inactive."
        )
    )
    summary: str = Field(default="", description="One plain-language sentence describing the candidate.")
    change_points: list[ReviewChangePoint] = Field(
        default_factory=list,
        description="One entry for every behaviorally meaningful change in the exact candidate.",
    )
    evidence_sufficiency: str = Field(
        default="",
        description="Whether supporting and counterexample evidence is sufficient for the claimed scope.",
    )
    scope_assessment: str = Field(
        default="",
        description="Why the proposed activation scope is or is not supported by the evidence.",
    )
    proportionality_assessment: ProportionalityAssessment = Field(
        default_factory=ProportionalityAssessment,
        description="Cost and burden assessment relative to the supported risk and task scope.",
    )
    counterexamples: list[str] = Field(
        default_factory=list,
        description="Counterexamples that constrain the revision.",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Concrete overreach, ambiguity, conflict, or missing-evidence concerns.",
    )
    human_checks: list[str] = Field(
        default_factory=list,
        description=(
            "Only unresolved authorization, safety, or genuinely subjective user choices that "
            "must block activation. Leave empty for ordinary scope or quality checks."
        ),
    )
    rationale: str = Field(default="", description="Concise evidence-grounded review rationale.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_provider_nulls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in (
            "summary",
            "evidence_sufficiency",
            "scope_assessment",
            "rationale",
        ):
            if data.get(key) is None:
                data[key] = ""
        for key in ("change_points", "counterexamples", "concerns", "human_checks"):
            if data.get(key) is None:
                data[key] = []
        if data.get("proportionality_assessment") is None:
            data["proportionality_assessment"] = {}
        return data


__all__ = [
    "CandidateAction",
    "CandidateRevision",
    "CandidateRoute",
    "CandidateStatus",
    "CANONICAL_CANDIDATE_STATUSES",
    "DeltaOperation",
    "JobStatus",
    "LearningCandidate",
    "Observation",
    "ObservationStatus",
    "ReflectionKind",
    "ReflectionResult",
    "ReflectionBatch",
    "ProportionalityAssessment",
    "ProposerResult",
    "ReviewChangePoint",
    "ReviewerResult",
    "SKILL_GROUPS",
    "SelfEvolutionJob",
    "SignalKind",
    "SkillRun",
    "TextResult",
    "ValidationReport",
    "normalize_candidate_status",
]
