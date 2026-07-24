from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    ActionStatus,
    Band,
    EvidenceVerdict,
    HypothesisEngineState,
    HypothesisStatus,
    VerificationAction,
)


_BAND_SCORE = {
    Band.LOW: 1,
    Band.MEDIUM: 2,
    Band.HIGH: 3,
}


class ActionAssessment(BaseModel):
    """Small model-visible explanation of whether one verification can run."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    status: str
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    rationale: str = ""


def unresolved_hypothesis_ids(state: HypothesisEngineState) -> set[str]:
    return {
        hypothesis.id
        for hypothesis in state.hypotheses
        if hypothesis.status in {HypothesisStatus.OPEN, HypothesisStatus.CONTESTED}
    }


def eligibility_reasons(
    state: HypothesisEngineState,
    action: VerificationAction,
) -> list[str]:
    if state.active_action_id:
        return [f"active_verification:{state.active_action_id}"]
    if action.status is not ActionStatus.PLANNED:
        return [f"action_status:{action.status.value}"]

    reasons: list[str] = []
    actions_by_id = {item.id: item for item in state.actions}
    for prerequisite_id in action.prerequisite_action_ids:
        prerequisite = actions_by_id[prerequisite_id]
        if prerequisite.status is not ActionStatus.COMPLETED:
            reasons.append(f"prerequisite:{prerequisite_id}")

    unresolved = unresolved_hypothesis_ids(state)
    if not unresolved.intersection(action.target_hypotheses):
        reasons.append("no_unresolved_target")
    return reasons


def _status(action: VerificationAction, reasons: list[str]) -> str:
    if action.status is not ActionStatus.PLANNED:
        return action.status.value
    if any(reason.startswith("prerequisite:") for reason in reasons):
        return "locked"
    if "no_unresolved_target" in reasons:
        return "closed"
    if not reasons:
        return "eligible"
    return "blocked"


def _rationale(
    state: HypothesisEngineState,
    action: VerificationAction,
) -> str:
    unresolved = unresolved_hypothesis_ids(state)
    target_count = len(unresolved.intersection(action.target_hypotheses))
    noun = "hypothesis" if target_count == 1 else "hypotheses"
    return (
        f"{action.information_value.value} information value, "
        f"{action.cost.value} cost; tests {target_count} unresolved {noun}"
    )


def rank_actions(
    state: HypothesisEngineState,
) -> list[ActionAssessment]:
    assessments: list[ActionAssessment] = []
    actions_by_id = {action.id: action for action in state.actions}
    unresolved = unresolved_hypothesis_ids(state)
    for action in state.actions:
        reasons = eligibility_reasons(state, action)
        assessments.append(
            ActionAssessment(
                action_id=action.id,
                status=_status(action, reasons),
                eligible=not reasons,
                reasons=reasons,
                rationale=_rationale(state, action),
            )
        )

    def sort_key(assessment: ActionAssessment) -> tuple[int, int, int, int, str]:
        action = actions_by_id[assessment.action_id]
        unresolved_targets = len(unresolved.intersection(action.target_hypotheses))
        return (
            0 if assessment.eligible else 1,
            -_BAND_SCORE[action.information_value],
            _BAND_SCORE[action.cost],
            -unresolved_targets,
            action.id,
        )

    return sorted(assessments, key=sort_key)


def hypothesis_evidence_counts(
    state: HypothesisEngineState,
    hypothesis_id: str,
) -> dict[str, int]:
    counts = {
        EvidenceVerdict.SUPPORTS.value: 0,
        EvidenceVerdict.OPPOSES.value: 0,
        EvidenceVerdict.INCONCLUSIVE.value: 0,
    }
    for judgment in state.evidence:
        for effect in judgment.effects:
            if effect.hypothesis_id == hypothesis_id:
                counts[effect.verdict.value] += 1
    return counts


__all__ = [
    "ActionAssessment",
    "eligibility_reasons",
    "hypothesis_evidence_counts",
    "rank_actions",
    "unresolved_hypothesis_ids",
]
