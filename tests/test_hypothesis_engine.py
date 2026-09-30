from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from catmaster.research.hypothesis_engine import (
    ActionStatus,
    Band,
    EvidenceEffect,
    EvidenceJudgment,
    ExecutionLane,
    Hypothesis,
    HypothesisDraft,
    HypothesisEngine,
    HypothesisEngineState,
    HypothesisPlan,
    HypothesisStatus,
    VerificationAction,
    VerificationActionDraft,
)


def _hypothesis(hypothesis_id: str) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        claim=f"Mechanism {hypothesis_id} controls the observed promotion.",
        rationale=f"Mechanistic rationale for {hypothesis_id}.",
        predictions=[f"Distinct observable predicted by {hypothesis_id}."],
    )


def _action(
    action_id: str,
    *,
    targets: list[str] | None = None,
    executor: ExecutionLane = ExecutionLane.LITERATURE,
    information: Band = Band.HIGH,
    cost: Band = Band.LOW,
    prerequisites: list[str] | None = None,
    task_suffix: str = "",
) -> VerificationAction:
    return VerificationAction(
        id=action_id,
        executor=executor,
        question=f"Which mechanism survives {action_id}?",
        task=f"Run the bounded scientific check {action_id}.{task_suffix}",
        target_hypotheses=targets or ["h1", "h2"],
        decision_rule=(
            "Outcome one supports h1 and opposes h2; outcome two does the reverse; "
            "missing discrimination is inconclusive."
        ),
        prerequisite_action_ids=prerequisites or [],
        information_value=information,
        cost=cost,
    )


def _state(
    *,
    actions: list[VerificationAction] | None = None,
) -> HypothesisEngineState:
    return HypothesisEngineState(
        question="Which mechanism explains the promotion?",
        hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
        actions=[_action("a1")] if actions is None else actions,
    )


def _judgment(
    action_id: str,
    verdicts: dict[str, str] | None = None,
) -> EvidenceJudgment:
    chosen = verdicts or {"h1": "supports", "h2": "opposes"}
    return EvidenceJudgment(
        action_id=action_id,
        summary="The scientific result distinguishes the proposed mechanisms.",
        source="doi:10.1021/example",
        effects=[
            EvidenceEffect(
                hypothesis_id=hypothesis_id,
                verdict=verdict,
                reason=f"The result is {verdict} for {hypothesis_id}.",
            )
            for hypothesis_id, verdict in chosen.items()
        ],
    )


def test_execution_packet_transfers_scientific_content_not_audit_state() -> None:
    engine = HypothesisEngine(_state())

    packet = engine.advance("a1")

    assert packet is not None
    assert packet.delegate_to == "litreview_agent"
    assert [item.id for item in packet.hypotheses] == ["h1", "h2"]
    assert all(item.rationale and item.predictions for item in packet.hypotheses)
    assert "supports h1" in packet.decision_rule
    assert packet.information_value is Band.HIGH
    assert packet.cost is Band.LOW
    packet_fields = set(type(packet).model_fields)
    assert "run_id" not in packet_fields
    assert "attempt" not in packet_fields
    assert "resource_usage" not in packet_fields
    assert engine.state.active_action_id == "a1"
    assert engine.state.actions[0].status is ActionStatus.RUNNING


def test_independent_judgment_updates_all_target_hypotheses() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")

    engine.record_result(
        "a1",
        outcome="completed",
        judgment=_judgment("a1"),
    )

    assert {
        hypothesis.id: hypothesis.status for hypothesis in engine.state.hypotheses
    } == {
        "h1": HypothesisStatus.SUPPORTED,
        "h2": HypothesisStatus.REJECTED,
    }
    assert len(engine.state.evidence) == 1
    assert engine.controller_snapshot()["status"] == "complete"


def test_inconclusive_judgment_keeps_hypotheses_open() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")
    engine.record_result(
        "a1",
        outcome="completed",
        judgment=_judgment(
            "a1",
            {"h1": "inconclusive", "h2": "inconclusive"},
        ),
    )

    assert all(
        hypothesis.status is HypothesisStatus.OPEN
        for hypothesis in engine.state.hypotheses
    )
    assert engine.controller_snapshot()["status"] == "needs_hypothesis_revision"


def test_conflicting_independent_judgments_mark_hypothesis_contested() -> None:
    actions = [
        _action("a1"),
        _action("a2", task_suffix=" Use an independent method."),
    ]
    engine = HypothesisEngine(_state(actions=actions))
    engine.advance("a1")
    engine.record_result(
        "a1",
        outcome="completed",
        judgment=_judgment(
            "a1",
            {"h1": "supports", "h2": "inconclusive"},
        ),
    )
    engine.advance("a2")
    engine.record_result(
        "a2",
        outcome="completed",
        judgment=_judgment(
            "a2",
            {"h1": "opposes", "h2": "inconclusive"},
        ),
    )

    statuses = {item.id: item.status for item in engine.state.hypotheses}
    assert statuses["h1"] is HypothesisStatus.CONTESTED
    assert statuses["h2"] is HypothesisStatus.OPEN


def test_evidence_must_judge_every_target_exactly_once() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")

    with pytest.raises(ValueError, match="must judge exactly"):
        engine.record_result(
            "a1",
            outcome="completed",
            judgment=_judgment("a1", {"h1": "supports"}),
        )

    assert engine.state.active_action_id == "a1"
    assert not engine.state.evidence


def test_result_recording_cannot_create_hypotheses_or_actions() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")

    with pytest.raises(TypeError):
        engine.record_result(  # type: ignore[call-arg]
            "a1",
            outcome="completed",
            judgment=_judgment("a1"),
            new_hypotheses=[_hypothesis("h3")],
        )


def test_hypothesis_revision_is_a_separate_extension() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")
    engine.record_result(
        "a1",
        outcome="completed",
        judgment=_judgment(
            "a1",
            {"h1": "supports", "h2": "inconclusive"},
        ),
    )

    engine.extend(
        hypotheses=[
            Hypothesis(
                id="h3",
                claim="A boundary condition modifies the surviving mechanism.",
                rationale="The first result exposes a system-specific dependency.",
                predictions=["Changing the boundary condition changes the observable."],
                derived_from=["h1"],
            )
        ],
        actions=[
            _action(
                "a3",
                targets=["h2", "h3"],
                prerequisites=["a1"],
                task_suffix=" Test the evidence-driven revision.",
            )
        ],
    )

    assert engine.state.hypotheses[-1].derived_from == ["h1"]
    assert engine.state.actions[-1].id == "a3"
    assert engine.state.revision == 3


def test_ranking_prefers_information_then_cost() -> None:
    actions = [
        _action("low-info", information=Band.LOW, cost=Band.LOW),
        _action(
            "high-info-medium-cost",
            information=Band.HIGH,
            cost=Band.MEDIUM,
            task_suffix=" Distinct method.",
        ),
        _action(
            "high-info-high-cost",
            information=Band.HIGH,
            cost=Band.HIGH,
            task_suffix=" Expensive method.",
        ),
    ]
    engine = HypothesisEngine(_state(actions=actions))

    selected = engine.select_next()

    assert selected is not None
    assert selected.id == "high-info-medium-cost"


def test_high_cost_and_human_actions_have_no_controller_permission_gate() -> None:
    actions = [
        _action("expensive", cost=Band.HIGH),
        _action(
            "ask-user",
            executor=ExecutionLane.HUMAN,
            task_suffix=" Ask for the missing measurement.",
        ),
    ]
    engine = HypothesisEngine(_state(actions=actions))

    assessments = {item.action_id: item for item in engine.rank_actions()}

    assert assessments["expensive"].eligible
    assert assessments["ask-user"].eligible
    packet = engine.advance("expensive")
    assert packet.action_id == "expensive"


def test_advance_requires_action_selection() -> None:
    engine = HypothesisEngine(_state())

    with pytest.raises(ValueError, match="explicitly selected action id"):
        engine.advance("")


def test_prerequisite_unlocks_after_inconclusive_completion() -> None:
    actions = [
        _action("first"),
        _action(
            "second",
            prerequisites=["first"],
            task_suffix=" Independent follow-up.",
        ),
    ]
    engine = HypothesisEngine(_state(actions=actions))
    locked = next(item for item in engine.rank_actions() if item.action_id == "second")
    assert locked.status == "locked"

    engine.advance("first")
    engine.record_result(
        "first",
        outcome="completed",
        judgment=_judgment(
            "first",
            {"h1": "inconclusive", "h2": "inconclusive"},
        ),
    )

    unlocked = next(item for item in engine.rank_actions() if item.action_id == "second")
    assert unlocked.eligible


def test_failed_execution_is_not_evidence_and_is_not_retried() -> None:
    actions = [
        _action("first"),
        _action("alternative", task_suffix=" Independent route."),
    ]
    engine = HypothesisEngine(_state(actions=actions))
    selected = engine.select_next()
    assert selected is not None
    packet = engine.advance(selected.id)
    failed_action_id = packet.action_id

    engine.record_result(
        failed_action_id,
        outcome="failed",
        failure_reason="source unavailable",
    )

    assert not engine.state.evidence
    failed_action = next(
        action for action in engine.state.actions if action.id == failed_action_id
    )
    assert failed_action.status is ActionStatus.FAILED
    next_action = engine.select_next()
    assert next_action is not None
    assert next_action.id != failed_action_id


def test_release_active_returns_unlaunched_reservation_to_plan() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")

    engine.release_active("a1")

    assert engine.state.active_action_id == ""
    assert engine.state.actions[0].status is ActionStatus.PLANNED
    assert engine.controller_snapshot()["phase"] == "ready"


def test_graph_contains_only_hypotheses_checks_and_evidence() -> None:
    engine = HypothesisEngine(_state())
    engine.advance("a1")
    engine.record_result(
        "a1",
        outcome="completed",
        judgment=_judgment("a1"),
    )

    graph = engine.graph_projection()
    assert {node["kind"] for node in graph["nodes"]} == {
        "hypothesis",
        "action",
        "evidence",
    }
    assert {"tested_by", "produces", "supports", "opposes"}.issubset(
        {edge["kind"] for edge in graph["edges"]}
    )
    assert not any(node["kind"] == "run" for node in graph["nodes"])


def test_result_recording_does_not_reserve_another_action() -> None:
    actions = [_action("a1"), _action("a2", task_suffix=" Independent route.")]
    engine = HypothesisEngine(_state(actions=actions))
    engine.advance("a1")

    engine.record_result(
        "a1",
        outcome="completed",
        judgment=_judgment(
            "a1",
            {"h1": "inconclusive", "h2": "inconclusive"},
        ),
    )

    assert engine.state.active_action_id == ""
    assert engine.select_next() is not None


def test_hypothesis_and_action_dependency_cycles_are_rejected() -> None:
    with pytest.raises(ValidationError, match="hypothesis derivation contains a cycle"):
        HypothesisEngineState(
            question="cycle",
            hypotheses=[
                Hypothesis(
                    id="h1",
                    claim="H1.",
                    rationale="R1.",
                    predictions=["P1."],
                    derived_from=["h2"],
                ),
                Hypothesis(
                    id="h2",
                    claim="H2.",
                    rationale="R2.",
                    predictions=["P2."],
                    derived_from=["h1"],
                ),
            ],
        )

    with pytest.raises(ValidationError, match="verification prerequisite contains a cycle"):
        _state(
            actions=[
                _action("a1", prerequisites=["a2"]),
                _action("a2", prerequisites=["a1"], task_suffix=" Different."),
            ]
        )


def test_duplicate_scientific_verification_contract_is_rejected() -> None:
    duplicate = _action("a2")
    duplicate.question = _action("a1").question.replace("a1", "a2")
    duplicate.task = _action("a1").task.replace("a1", "a2")
    # Normalize both contracts to the same scientific text while retaining different ids.
    first = _action("a1")
    first.question = duplicate.question
    first.task = duplicate.task

    with pytest.raises(ValidationError, match="duplicates the scientific contract"):
        _state(actions=[first, duplicate])


def test_schema_v2_is_rejected_instead_of_inventing_missing_science() -> None:
    payload = _state().model_dump(mode="json")
    payload["schema_version"] = 2

    with pytest.raises(ValidationError, match="intentionally not migrated"):
        HypothesisEngineState.model_validate(payload)


def test_plan_schema_contains_only_scientific_proposal_content() -> None:
    plan = HypothesisPlan(
        hypotheses=[
            HypothesisDraft.model_validate(
                _hypothesis("h1").model_dump(exclude={"status"})
            )
        ],
        actions=[
            VerificationActionDraft.model_validate(
                _action("a1", targets=["h1"]).model_dump(
                    exclude={"status", "failure_reason"}
                )
            )
        ],
    )
    assert plan.hypotheses[0].predictions
    assert plan.actions[0].decision_rule
    assert set(HypothesisPlan.model_fields) == {"hypotheses", "actions"}


def test_state_has_no_run_budget_attempt_or_audit_ledger() -> None:
    fields = set(HypothesisEngineState.model_fields)
    assert fields == {
        "schema_version",
        "revision",
        "question",
        "hypotheses",
        "actions",
        "evidence",
        "active_action_id",
    }
    dumped = json.dumps(_state().model_dump(mode="json"))
    for removed in (
        "runs",
        "budget",
        "attempt",
        "resource_usage",
        "elapsed_wait_hours",
        "provenance_ref",
        "competing_pairs",
    ):
        assert removed not in dumped
