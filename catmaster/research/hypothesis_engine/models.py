from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HypothesisStatus(str, Enum):
    OPEN = "open"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    CONTESTED = "contested"


class EvidenceVerdict(str, Enum):
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    INCONCLUSIVE = "inconclusive"


class Band(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionLane(str, Enum):
    LITERATURE = "literature"
    EXPERIMENT = "experiment"
    WORKSPACE = "workspace"
    HUMAN = "human"


def _clean_unique_strings(values: list[str], field_name: str) -> list[str]:
    cleaned = [str(value or "").strip() for value in values]
    if any(not value for value in cleaned):
        raise ValueError(f"{field_name} values must not be empty")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field_name} values must be unique")
    return cleaned


class HypothesisDraft(BaseModel):
    """Scientific content returned by the hypothesis proposer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Short stable hypothesis id.")
    claim: str = Field(..., min_length=1, description="One falsifiable scientific claim.")
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this mechanism or explanation is scientifically plausible.",
    )
    predictions: list[str] = Field(
        ...,
        min_length=1,
        description="Observable predictions that distinguish this hypothesis from alternatives.",
    )
    derived_from: list[str] = Field(
        default_factory=list,
        description="Parent hypothesis ids for an evidence-driven revision; otherwise pass [].",
    )

    @field_validator("id", "claim", "rationale")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("scientific text must not be blank")
        return cleaned

    @field_validator("predictions")
    @classmethod
    def predictions_must_be_unique(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values, "prediction")

    @field_validator("derived_from")
    @classmethod
    def derivations_must_be_unique(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values, "derived_from") if values else []


class Hypothesis(HypothesisDraft):
    """Accepted campaign hypothesis with a status derived from evidence."""

    status: HypothesisStatus = HypothesisStatus.OPEN


class VerificationActionDraft(BaseModel):
    """Smallest useful scientific check proposed for one or more hypotheses."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Short stable verification id.")
    executor: ExecutionLane = Field(
        ...,
        description=(
            "literature delegates source work, experiment delegates computation, "
            "workspace stays with Research, and human asks the user."
        ),
    )
    question: str = Field(
        ...,
        min_length=1,
        description="The scientific distinction this verification must resolve.",
    )
    task: str = Field(
        ...,
        min_length=1,
        description="Bounded scientific work for the owning executor.",
    )
    target_hypotheses: list[str] = Field(
        ...,
        min_length=1,
        description="Every hypothesis that must be judged from this result.",
    )
    decision_rule: str = Field(
        ...,
        min_length=1,
        description="How possible outcomes support, oppose, or leave the targets inconclusive.",
    )
    prerequisite_action_ids: list[str] = Field(
        default_factory=list,
        description="Earlier verification ids that must complete first; otherwise pass [].",
    )
    information_value: Band = Field(
        Band.MEDIUM,
        description="Expected ability to distinguish live hypotheses: low, medium, or high.",
    )
    cost: Band = Field(
        Band.LOW,
        description="Coarse execution burden: low, medium, or high.",
    )

    @field_validator("id", "question", "task", "decision_rule")
    @classmethod
    def action_text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("verification text must not be blank")
        return cleaned

    @field_validator("target_hypotheses", "prerequisite_action_ids")
    @classmethod
    def action_lists_must_be_unique(
        cls,
        values: list[str],
        info,
    ) -> list[str]:
        return _clean_unique_strings(values, info.field_name) if values else []


class VerificationAction(VerificationActionDraft):
    """Accepted verification plus its current controller state."""

    status: ActionStatus = ActionStatus.PLANNED
    failure_reason: str = ""

    @model_validator(mode="after")
    def validate_failure_shape(self) -> VerificationAction:
        self.failure_reason = self.failure_reason.strip()
        if self.status is ActionStatus.FAILED and not self.failure_reason:
            raise ValueError("a failed verification must include a failure reason")
        if self.status is not ActionStatus.FAILED and self.failure_reason:
            raise ValueError("only a failed verification may include a failure reason")
        return self


class EvidenceEffect(BaseModel):
    """Independent scientific judgment for one hypothesis."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(..., min_length=1)
    verdict: EvidenceVerdict
    reason: str = Field(
        ...,
        min_length=1,
        description="Scientific reason grounded in the supplied result and decision rule.",
    )

    @field_validator("hypothesis_id", "reason")
    @classmethod
    def evidence_text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence text must not be blank")
        return cleaned


class EvidenceJudgment(BaseModel):
    """Structured output returned by the independent evidence judge."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(..., min_length=1)
    summary: str = Field(
        ...,
        min_length=1,
        description="Decision-relevant scientific result, without execution logs.",
    )
    source: str = Field(
        ...,
        min_length=1,
        description="DOI, URL, CatMaster artifact path, run id, or explicit user evidence.",
    )
    effects: list[EvidenceEffect] = Field(
        ...,
        min_length=1,
        description="Exactly one judgment for every target hypothesis.",
    )

    @field_validator("action_id", "summary", "source")
    @classmethod
    def judgment_text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence judgment text must not be blank")
        return cleaned

    @field_validator("effects")
    @classmethod
    def effects_must_be_unique(cls, effects: list[EvidenceEffect]) -> list[EvidenceEffect]:
        ids = [effect.hypothesis_id for effect in effects]
        if len(ids) != len(set(ids)):
            raise ValueError("an evidence judgment must address each hypothesis exactly once")
        return effects


class HypothesisPlan(BaseModel):
    """Structured output returned by the dedicated hypothesis proposer."""

    model_config = ConfigDict(extra="forbid")

    hypotheses: list[HypothesisDraft] = Field(default_factory=list, max_length=12)
    actions: list[VerificationActionDraft] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_plan(self) -> HypothesisPlan:
        if not self.hypotheses and not self.actions:
            raise ValueError("a hypothesis plan must contain scientific content")
        _unique_ids(list(self.hypotheses), "hypothesis")
        _unique_ids(list(self.actions), "verification")
        return self


class HypothesisEngineState(BaseModel):
    """Lean persistent state for scientific content and the active handoff."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[4] = 4
    revision: int = Field(0, ge=0)
    question: str = Field(..., min_length=1)
    hypotheses: list[Hypothesis] = Field(..., min_length=1)
    actions: list[VerificationAction] = Field(default_factory=list)
    evidence: list[EvidenceJudgment] = Field(default_factory=list)
    active_action_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_scientifically_complete_state(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version", 4)
        if version == 3:
            migrated = dict(value)
            migrated.pop("mode", None)
            migrated.pop("paused", None)
            migrated["schema_version"] = 4
            return migrated
        if version != 4:
            raise ValueError(
                "hypothesis campaign schemas before v3 are intentionally not migrated: "
                "regenerate the campaign through hypothesis_proposer so missing rationale, "
                "predictions, and decision rules are not invented"
            )
        return value

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("campaign question must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_campaign(self) -> HypothesisEngineState:
        hypothesis_ids, action_ids = _validate_scientific_network(
            self.hypotheses,
            self.actions,
        )
        actions_by_id = {action.id: action for action in self.actions}

        running = [
            action.id for action in self.actions if action.status is ActionStatus.RUNNING
        ]
        if self.active_action_id:
            if running != [self.active_action_id]:
                raise ValueError(
                    "active_action_id must identify the only running verification"
                )
        elif running:
            raise ValueError("a running verification requires active_action_id")

        evidence_actions: set[str] = set()
        for judgment in self.evidence:
            if judgment.action_id not in action_ids:
                raise ValueError(
                    f"evidence references unknown verification {judgment.action_id}"
                )
            if judgment.action_id in evidence_actions:
                raise ValueError(
                    f"verification {judgment.action_id} has more than one evidence judgment"
                )
            evidence_actions.add(judgment.action_id)
            action = actions_by_id[judgment.action_id]
            judged = {effect.hypothesis_id for effect in judgment.effects}
            expected = set(action.target_hypotheses)
            if judged != expected:
                raise ValueError(
                    f"evidence for {action.id} must judge exactly {sorted(expected)}; "
                    f"received {sorted(judged)}"
                )
            if action.status is not ActionStatus.COMPLETED:
                raise ValueError(
                    f"evidence may only belong to completed verification {action.id}"
                )
            unknown = judged - hypothesis_ids
            if unknown:
                raise ValueError(
                    f"evidence for {action.id} references unknown hypotheses: {sorted(unknown)}"
                )

        for action in self.actions:
            if action.status is ActionStatus.COMPLETED and action.id not in evidence_actions:
                raise ValueError(
                    f"completed verification {action.id} requires an evidence judgment"
                )
            if action.status is not ActionStatus.COMPLETED and action.id in evidence_actions:
                raise ValueError(
                    f"only completed verification {action.id} may have evidence"
                )

        verdicts: dict[str, set[EvidenceVerdict]] = {
            hypothesis_id: set() for hypothesis_id in hypothesis_ids
        }
        for judgment in self.evidence:
            for effect in judgment.effects:
                if effect.verdict is not EvidenceVerdict.INCONCLUSIVE:
                    verdicts[effect.hypothesis_id].add(effect.verdict)

        for hypothesis in self.hypotheses:
            seen = verdicts[hypothesis.id]
            if seen == {EvidenceVerdict.SUPPORTS}:
                hypothesis.status = HypothesisStatus.SUPPORTED
            elif seen == {EvidenceVerdict.OPPOSES}:
                hypothesis.status = HypothesisStatus.REJECTED
            elif len(seen) > 1:
                hypothesis.status = HypothesisStatus.CONTESTED
            else:
                hypothesis.status = HypothesisStatus.OPEN
        return self


def _unique_ids(items: list[Any], item_name: str) -> set[str]:
    ids = [str(item.id).strip() for item in items]
    if any(not item_id for item_id in ids):
        raise ValueError(f"{item_name} ids must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{item_name} ids must be unique")
    return set(ids)


def _validate_acyclic(dependencies: dict[str, list[str]], graph_name: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"{graph_name} contains a cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies.get(node, []):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node)


def _action_fingerprint(action: VerificationActionDraft) -> str:
    payload = {
        "executor": action.executor.value,
        "question": " ".join(action.question.lower().split()),
        "task": " ".join(action.task.lower().split()),
        "target_hypotheses": sorted(action.target_hypotheses),
        "decision_rule": " ".join(action.decision_rule.lower().split()),
        "prerequisite_action_ids": sorted(action.prerequisite_action_ids),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_scientific_network(
    hypotheses: list[HypothesisDraft] | list[Hypothesis],
    actions: list[VerificationActionDraft] | list[VerificationAction],
) -> tuple[set[str], set[str]]:
    hypothesis_ids = _unique_ids(list(hypotheses), "hypothesis")
    action_ids = _unique_ids(list(actions), "verification")

    hypothesis_dependencies: dict[str, list[str]] = {}
    for hypothesis in hypotheses:
        unknown = set(hypothesis.derived_from) - hypothesis_ids
        if unknown:
            raise ValueError(
                f"hypothesis {hypothesis.id} derives from unknown hypotheses: {sorted(unknown)}"
            )
        if hypothesis.id in hypothesis.derived_from:
            raise ValueError(f"hypothesis {hypothesis.id} cannot derive from itself")
        hypothesis_dependencies[hypothesis.id] = list(hypothesis.derived_from)
    _validate_acyclic(hypothesis_dependencies, "hypothesis derivation")

    action_dependencies: dict[str, list[str]] = {}
    fingerprints: dict[str, str] = {}
    for action in actions:
        unknown_targets = set(action.target_hypotheses) - hypothesis_ids
        if unknown_targets:
            raise ValueError(
                f"verification {action.id} targets unknown hypotheses: "
                f"{sorted(unknown_targets)}"
            )
        unknown_prerequisites = set(action.prerequisite_action_ids) - action_ids
        if unknown_prerequisites:
            raise ValueError(
                f"verification {action.id} has unknown prerequisites: "
                f"{sorted(unknown_prerequisites)}"
            )
        if action.id in action.prerequisite_action_ids:
            raise ValueError(f"verification {action.id} cannot depend on itself")
        action_dependencies[action.id] = list(action.prerequisite_action_ids)
        fingerprint = _action_fingerprint(action)
        duplicate = fingerprints.get(fingerprint)
        if duplicate:
            raise ValueError(
                f"verification {action.id} duplicates the scientific contract of {duplicate}"
            )
        fingerprints[fingerprint] = action.id
    _validate_acyclic(action_dependencies, "verification prerequisite")
    return hypothesis_ids, action_ids


__all__ = [
    "ActionStatus",
    "Band",
    "EvidenceEffect",
    "EvidenceJudgment",
    "EvidenceVerdict",
    "ExecutionLane",
    "Hypothesis",
    "HypothesisDraft",
    "HypothesisEngineState",
    "HypothesisPlan",
    "HypothesisStatus",
    "VerificationAction",
    "VerificationActionDraft",
]
