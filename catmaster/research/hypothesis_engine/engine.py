from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .models import (
    ActionStatus,
    Band,
    EvidenceJudgment,
    ExecutionLane,
    Hypothesis,
    HypothesisDraft,
    HypothesisEngineState,
    HypothesisStatus,
    VerificationAction,
)
from .policy import (
    ActionAssessment,
    hypothesis_evidence_counts,
    rank_actions,
    unresolved_hypothesis_ids,
)


_DELEGATE_TARGET = {
    ExecutionLane.LITERATURE: "litreview_agent",
    ExecutionLane.EXPERIMENT: "experiment_specialist",
    ExecutionLane.WORKSPACE: "ResearchSpecialist",
    ExecutionLane.HUMAN: "user",
}


class ExecutionPacket(BaseModel):
    """Scientific handoff for one selected verification."""

    model_config = ConfigDict(extra="forbid")

    campaign_question: str
    action_id: str
    executor: ExecutionLane
    delegate_to: str
    question: str
    task: str
    hypotheses: list[HypothesisDraft]
    decision_rule: str
    information_value: Band
    cost: Band


class HypothesisEngine:
    """Serial controller that stores scientific content without doing science."""

    def __init__(self, state: HypothesisEngineState) -> None:
        self.state = HypothesisEngineState.model_validate(state.model_dump())

    def rank_actions(self) -> list[ActionAssessment]:
        return rank_actions(self.state)

    def select_next(self) -> VerificationAction | None:
        return self._select_next_from_state(self.state)

    def advance(self, action_id: str) -> ExecutionPacket:
        if self.state.active_action_id:
            raise ValueError(
                f"cannot advance while {self.state.active_action_id} is running"
            )
        if not unresolved_hypothesis_ids(self.state):
            raise ValueError("cannot advance because no unresolved hypothesis remains")

        requested = action_id.strip()
        if not requested:
            raise ValueError("advance requires an explicitly selected action id")
        action = self._action(requested)
        assessment = next(
            item for item in rank_actions(self.state) if item.action_id == action.id
        )
        if not assessment.eligible:
            raise ValueError(
                f"verification {action.id} is not eligible: "
                f"{', '.join(assessment.reasons)}"
            )

        updated = self.state.model_copy(deep=True)
        self._start_on_state(updated, action.id)
        self._commit(updated)
        packet = self.active_packet()
        assert packet is not None
        return packet

    def record_result(
        self,
        action_id: str,
        *,
        outcome: Literal["completed", "failed"],
        judgment: EvidenceJudgment | None = None,
        failure_reason: str = "",
    ) -> None:
        updated = self.state.model_copy(deep=True)
        action = self._active_action(updated, action_id)

        if outcome == "completed":
            if judgment is None:
                raise ValueError("completed verification requires an evidence judgment")
            if judgment.action_id != action.id:
                raise ValueError(
                    f"evidence action_id {judgment.action_id} does not match {action.id}"
                )
            action.status = ActionStatus.COMPLETED
            updated.evidence.append(judgment.model_copy(deep=True))
        elif outcome == "failed":
            reason = failure_reason.strip()
            if not reason:
                raise ValueError("failed verification requires a failure reason")
            if judgment is not None:
                raise ValueError("failed verification cannot include an evidence judgment")
            action.status = ActionStatus.FAILED
            action.failure_reason = reason
        else:
            raise ValueError(f"unknown verification outcome: {outcome}")

        updated.active_action_id = ""
        updated = HypothesisEngineState.model_validate(updated.model_dump())
        self._commit(updated)

    def extend(
        self,
        *,
        hypotheses: Sequence[Hypothesis] = (),
        actions: Sequence[VerificationAction] = (),
    ) -> HypothesisEngineState:
        if self.state.active_action_id:
            raise ValueError("cannot revise hypotheses while a verification is running")
        if not hypotheses and not actions:
            raise ValueError("campaign extension must add scientific content")

        updated = self.state.model_copy(deep=True)
        updated.hypotheses.extend(item.model_copy(deep=True) for item in hypotheses)
        updated.actions.extend(item.model_copy(deep=True) for item in actions)
        self._commit(updated)
        return self.state.model_copy(deep=True)

    def release_active(self, action_id: str) -> None:
        """Return an unlaunched reservation to planned state."""

        updated = self.state.model_copy(deep=True)
        action = self._active_action(updated, action_id)
        action.status = ActionStatus.PLANNED
        action.failure_reason = ""
        updated.active_action_id = ""
        self._commit(updated)

    def execution_packet(self, action_id: str) -> ExecutionPacket:
        action = self._action(action_id)
        if action.status is not ActionStatus.RUNNING:
            raise ValueError(f"verification {action.id} is not running")
        hypotheses_by_id = {
            hypothesis.id: hypothesis for hypothesis in self.state.hypotheses
        }
        contexts = [
            HypothesisDraft.model_validate(
                hypotheses_by_id[hypothesis_id].model_dump(exclude={"status"})
            )
            for hypothesis_id in action.target_hypotheses
        ]
        return ExecutionPacket(
            campaign_question=self.state.question,
            action_id=action.id,
            executor=action.executor,
            delegate_to=_DELEGATE_TARGET[action.executor],
            question=action.question,
            task=action.task,
            hypotheses=contexts,
            decision_rule=action.decision_rule,
            information_value=action.information_value,
            cost=action.cost,
        )

    def active_packet(self) -> ExecutionPacket | None:
        if not self.state.active_action_id:
            return None
        return self.execution_packet(self.state.active_action_id)

    def controller_snapshot(self) -> dict[str, Any]:
        ranking = self.rank_actions()
        packet = self.active_packet()
        unresolved = unresolved_hypothesis_ids(self.state)

        if packet is not None:
            phase = "running"
            status = "execution_required"
        elif not unresolved:
            phase = "stopped"
            status = "complete"
        else:
            phase = "ready"
            if any(item.eligible for item in ranking):
                status = "action_available"
            elif any(
                any(reason.startswith("prerequisite:") for reason in item.reasons)
                for item in ranking
            ):
                status = "prerequisite_blocked"
            else:
                status = "needs_hypothesis_revision"

        recommended = next((item for item in ranking if item.eligible), None)
        return {
            "status": status,
            "phase": phase,
            "revision": self.state.revision,
            "active_packet": packet.model_dump(mode="json") if packet else None,
            "recommended_action_id": recommended.action_id if recommended else "",
            "recommended_rationale": recommended.rationale if recommended else "",
        }

    def graph_projection(self) -> dict[str, Any]:
        assessments = {
            item.action_id: item for item in self.rank_actions()
        }
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for hypothesis in self.state.hypotheses:
            counts = hypothesis_evidence_counts(self.state, hypothesis.id)
            nodes.append(
                {
                    "id": f"hypothesis:{hypothesis.id}",
                    "kind": "hypothesis",
                    "label": hypothesis.claim,
                    "status": hypothesis.status.value,
                    "rationale": hypothesis.rationale,
                    "predictions": list(hypothesis.predictions),
                    "evidence_counts": counts,
                }
            )
            for parent_id in hypothesis.derived_from:
                edges.append(
                    {
                        "id": f"derives:{parent_id}:{hypothesis.id}",
                        "source": f"hypothesis:{parent_id}",
                        "target": f"hypothesis:{hypothesis.id}",
                        "kind": "derives",
                    }
                )

        for action in self.state.actions:
            assessment = assessments[action.id]
            nodes.append(
                {
                    "id": f"action:{action.id}",
                    "kind": "action",
                    "label": action.question,
                    "status": assessment.status,
                    "executor": action.executor.value,
                    "task": action.task,
                    "decision_rule": action.decision_rule,
                    "information_value": action.information_value.value,
                    "cost": action.cost.value,
                    "target_hypotheses": list(action.target_hypotheses),
                    "prerequisite_action_ids": list(action.prerequisite_action_ids),
                    "failure_reason": action.failure_reason,
                    "reasons": list(assessment.reasons),
                    "rationale": assessment.rationale,
                }
            )
            for hypothesis_id in action.target_hypotheses:
                edges.append(
                    {
                        "id": f"tested_by:{hypothesis_id}:{action.id}",
                        "source": f"hypothesis:{hypothesis_id}",
                        "target": f"action:{action.id}",
                        "kind": "tested_by",
                    }
                )
            for prerequisite_id in action.prerequisite_action_ids:
                edges.append(
                    {
                        "id": f"unlocks:{prerequisite_id}:{action.id}",
                        "source": f"action:{prerequisite_id}",
                        "target": f"action:{action.id}",
                        "kind": "unlocks",
                    }
                )

        for judgment in self.state.evidence:
            evidence_id = f"evidence:{judgment.action_id}"
            nodes.append(
                {
                    "id": evidence_id,
                    "kind": "evidence",
                    "label": judgment.summary,
                    "status": "judged",
                    "source": judgment.source,
                    "effects": [
                        effect.model_dump(mode="json") for effect in judgment.effects
                    ],
                }
            )
            edges.append(
                {
                    "id": f"produces:{judgment.action_id}",
                    "source": f"action:{judgment.action_id}",
                    "target": evidence_id,
                    "kind": "produces",
                }
            )
            for effect in judgment.effects:
                edges.append(
                    {
                        "id": (
                            f"{effect.verdict.value}:{judgment.action_id}:"
                            f"{effect.hypothesis_id}"
                        ),
                        "source": evidence_id,
                        "target": f"hypothesis:{effect.hypothesis_id}",
                        "kind": effect.verdict.value,
                    }
                )

        return {
            "schema_version": self.state.schema_version,
            "revision": self.state.revision,
            "question": self.state.question,
            "controller": self.controller_snapshot(),
            "nodes": nodes,
            "edges": edges,
        }

    def _select_next_from_state(
        self,
        state: HypothesisEngineState,
    ) -> VerificationAction | None:
        assessment = next(
            (
                item
                for item in rank_actions(state)
                if item.eligible
            ),
            None,
        )
        if assessment is None:
            return None
        return next(
            action for action in state.actions if action.id == assessment.action_id
        )

    def _start_on_state(
        self,
        state: HypothesisEngineState,
        action_id: str,
    ) -> None:
        action = next(action for action in state.actions if action.id == action_id)
        action.status = ActionStatus.RUNNING
        action.failure_reason = ""
        state.active_action_id = action.id

    @staticmethod
    def _active_action(
        state: HypothesisEngineState,
        action_id: str,
    ) -> VerificationAction:
        if not state.active_action_id:
            raise ValueError("campaign has no active verification")
        if action_id != state.active_action_id:
            raise ValueError(
                f"active verification is {state.active_action_id}, not {action_id}"
            )
        return next(action for action in state.actions if action.id == action_id)

    def _action(self, action_id: str) -> VerificationAction:
        action = next(
            (item for item in self.state.actions if item.id == action_id),
            None,
        )
        if action is None:
            raise ValueError(f"unknown verification: {action_id}")
        return action

    def _commit(self, updated: HypothesisEngineState) -> None:
        updated.revision = self.state.revision + 1
        self.state = HypothesisEngineState.model_validate(updated.model_dump())


__all__ = [
    "ExecutionPacket",
    "HypothesisEngine",
]
