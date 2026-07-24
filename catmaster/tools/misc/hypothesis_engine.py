from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.research.hypothesis_engine import (
    EvidenceEffect,
    EvidenceJudgment,
    Hypothesis,
    HypothesisDraft,
    HypothesisEngine,
    HypothesisEngineState,
    VerificationAction,
    VerificationActionDraft,
)
from catmaster.research.hypothesis_engine.storage import (
    campaign_lock,
    engine_path,
    engine_relpath,
    load_engine,
    save_engine,
)
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_root


ResultOutcome = Literal["completed", "failed"]


CAMPAIGN_ID_DESCRIPTION = (
    "Hypothesis campaign id from the Research system context. In a Map-launched "
    "Research thread this is the source campaign id, not the execution thread id."
)


class InitializeHypothesisCampaignInput(BaseModel):
    """[research/control] Persist a scientific plan returned by hypothesis_proposer."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., min_length=1, description=CAMPAIGN_ID_DESCRIPTION)
    question: str = Field(..., min_length=1, description="Campaign scientific question.")
    hypotheses: list[HypothesisDraft] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Exact structured hypotheses returned by hypothesis_proposer.",
    )
    actions: list[VerificationActionDraft] = Field(
        ...,
        min_length=1,
        max_length=24,
        description="Exact discriminating checks returned by hypothesis_proposer.",
    )
class ExtendHypothesisCampaignInput(BaseModel):
    """[research/control] Apply a scientific revision returned by hypothesis_proposer."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., min_length=1, description=CAMPAIGN_ID_DESCRIPTION)
    expected_revision: int = Field(
        -1,
        ge=-1,
        description="Latest inspected revision, or -1 in one uninterrupted controller sequence.",
    )
    hypotheses: list[HypothesisDraft] = Field(
        default_factory=list,
        max_length=12,
        description="New evidence-driven hypotheses from hypothesis_proposer; otherwise pass [].",
    )
    actions: list[VerificationActionDraft] = Field(
        default_factory=list,
        max_length=24,
        description="New discriminating checks from hypothesis_proposer; otherwise pass [].",
    )
    @model_validator(mode="after")
    def require_content(self) -> ExtendHypothesisCampaignInput:
        if not self.hypotheses and not self.actions:
            raise ValueError("extension must include hypotheses or actions")
        return self


class InspectHypothesisCampaignInput(BaseModel):
    """[research/control] Read the scientific campaign and current next-step decision."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., min_length=1, description=CAMPAIGN_ID_DESCRIPTION)


class AdvanceHypothesisCampaignInput(BaseModel):
    """[research/control] Select one verification and return its scientific packet."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., min_length=1, description=CAMPAIGN_ID_DESCRIPTION)
    expected_revision: int = Field(
        -1,
        ge=-1,
        description="Latest inspected revision, or -1 in one uninterrupted controller sequence.",
    )
    action_id: str = Field(
        ...,
        min_length=1,
        description="Explicitly selected verification id.",
    )


class RecordHypothesisResultInput(BaseModel):
    """[research/control] Store execution failure or the evidence_judge decision."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., min_length=1, description=CAMPAIGN_ID_DESCRIPTION)
    expected_revision: int = Field(
        -1,
        ge=-1,
        description="Revision that owns the active verification.",
    )
    action_id: str = Field(..., min_length=1, description="Active verification id.")
    outcome: ResultOutcome
    evidence_summary: str = Field(
        "",
        description="Exact decision-relevant summary returned by evidence_judge.",
    )
    source: str = Field(
        "",
        description="Exact DOI, URL, artifact path, run id, or user evidence used by the judge.",
    )
    effects: list[EvidenceEffect] = Field(
        default_factory=list,
        description="Exact per-hypothesis effects returned by evidence_judge.",
    )
    failure_reason: str = Field(
        "",
        description="Concise execution failure; leave empty for completed work.",
    )
    @model_validator(mode="after")
    def validate_outcome(self) -> RecordHypothesisResultInput:
        if self.outcome == "completed":
            if not self.evidence_summary.strip() or not self.source.strip() or not self.effects:
                raise ValueError(
                    "completed verification requires evidence_summary, source, and effects"
                )
            if self.failure_reason.strip():
                raise ValueError("completed verification must leave failure_reason empty")
        else:
            if not self.failure_reason.strip():
                raise ValueError("failed verification requires failure_reason")
            if self.evidence_summary.strip() or self.source.strip() or self.effects:
                raise ValueError("failed verification must not include evidence judgment fields")
        return self


def _require_revision(engine: HypothesisEngine, expected_revision: int) -> None:
    if expected_revision >= 0 and engine.state.revision != expected_revision:
        raise ValueError(
            "stale campaign revision: "
            f"expected {expected_revision}, current {engine.state.revision}; inspect and retry"
        )


def _hypothesis(item: HypothesisDraft) -> Hypothesis:
    return Hypothesis.model_validate(item.model_dump())


def _action(item: VerificationActionDraft) -> VerificationAction:
    return VerificationAction.model_validate(item.model_dump())


def _artifact(
    tool_name: str,
    thread_id: str,
    engine: HypothesisEngine,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "data": {
            "thread_id": thread_id,
            "engine_path": engine_relpath(thread_id),
            "controller": engine.controller_snapshot(),
            "state": engine.state.model_dump(mode="json"),
            "ranking": [
                assessment.model_dump(mode="json")
                for assessment in engine.rank_actions()
            ],
            "graph": engine.graph_projection(),
        },
        "suppress_content_offload_ref": True,
    }


def _controller_content(
    engine: HypothesisEngine,
    thread_id: str,
    *,
    prefix: str,
) -> str:
    controller = engine.controller_snapshot()
    lines = [
        prefix,
        (
            f"Campaign {engine_relpath(thread_id)} is revision {engine.state.revision}, "
            f"phase {controller['phase']}, status {controller['status']}."
        ),
    ]
    packet = engine.active_packet()
    if packet is not None:
        lines.extend(
            [
                "",
                "EXECUTION PACKET",
                f"- action_id: {packet.action_id}",
                f"- delegate_to: {packet.delegate_to}",
                f"- question: {packet.question}",
                "- target hypotheses:",
            ]
        )
        for hypothesis in packet.hypotheses:
            lines.extend(
                [
                    f"  - {hypothesis.id}: {hypothesis.claim}",
                    f"    rationale: {hypothesis.rationale}",
                    "    predictions:",
                    *[f"      - {prediction}" for prediction in hypothesis.predictions],
                ]
            )
        lines.extend(
            [
                "- task:",
                packet.task,
                f"- decision rule: {packet.decision_rule}",
                (
                    f"- scientific tradeoff: information={packet.information_value.value}; "
                    f"cost={packet.cost.value}"
                ),
                (
                    "Execute exactly this packet through its named owner. If execution succeeds, "
                    "send the returned scientific result, source, target hypotheses, and decision "
                    "rule to evidence_judge. Record only that judge's structured decision before "
                    "starting another verification. Do not create a new hypothesis in the result call."
                ),
            ]
        )
    elif controller["recommended_action_id"]:
        lines.append(
            "Next allowed verification is "
            f"{controller['recommended_action_id']}: "
            f"{controller['recommended_rationale']}"
        )
    elif controller["status"] == "needs_hypothesis_revision":
        lines.append(
            "No useful verification remains for the unresolved hypotheses. Delegate the current "
            "question, hypotheses, evidence, and failed checks to hypothesis_proposer, then apply "
            "its structured revision with extend_hypothesis_campaign."
        )
    elif controller["status"] == "prerequisite_blocked":
        lines.append("The remaining verification is waiting for an earlier scientific check.")
    return "\n".join(lines)


def _raise_tool_error(tool_name: str, thread_id: str, exc: Exception) -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=f"{tool_name} failed: {exc}",
        artifact={
            "tool_name": tool_name,
            "data": {
                "thread_id": thread_id,
                "engine_path": engine_relpath(thread_id),
            },
        },
        error_code="hypothesis_campaign_error",
    ) from exc


def initialize_hypothesis_campaign(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[research/control] Persist hypothesis_proposer output."""

    tool_name = "initialize_hypothesis_campaign"
    thread_id = str(payload.get("thread_id") or "")
    try:
        params = InitializeHypothesisCampaignInput.model_validate(payload)
        root = workspace_root()
        with campaign_lock(root, params.thread_id):
            path = engine_path(root, params.thread_id)
            if path.exists():
                raise ValueError(
                    "campaign already exists; ask hypothesis_proposer for a revision "
                    "and extend it instead of replacing scientific state"
                )
            engine = HypothesisEngine(
                HypothesisEngineState(
                    question=params.question,
                    hypotheses=[_hypothesis(item) for item in params.hypotheses],
                    actions=[_action(item) for item in params.actions],
                )
            )
            save_engine(root, params.thread_id, engine)
        content = _controller_content(
            engine,
            params.thread_id,
            prefix=(
                f"Initialized {len(engine.state.hypotheses)} hypotheses and "
                f"{len(engine.state.actions)} scientific verifications from hypothesis_proposer."
            ),
        )
        return content, _artifact(tool_name, params.thread_id, engine)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _raise_tool_error(tool_name, thread_id, exc)


def extend_hypothesis_campaign(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[research/control] Apply a hypothesis_proposer revision without judging evidence."""

    tool_name = "extend_hypothesis_campaign"
    thread_id = str(payload.get("thread_id") or "")
    try:
        params = ExtendHypothesisCampaignInput.model_validate(payload)
        root = workspace_root()
        with campaign_lock(root, params.thread_id):
            engine = load_engine(root, params.thread_id)
            _require_revision(engine, params.expected_revision)
            engine.extend(
                hypotheses=[_hypothesis(item) for item in params.hypotheses],
                actions=[_action(item) for item in params.actions],
            )
            save_engine(root, params.thread_id, engine)
        content = _controller_content(
            engine,
            params.thread_id,
            prefix=(
                f"Applied hypothesis_proposer revision: "
                f"{len(params.hypotheses)} hypotheses and {len(params.actions)} verifications."
            ),
        )
        return content, _artifact(tool_name, params.thread_id, engine)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _raise_tool_error(tool_name, thread_id, exc)


def inspect_hypothesis_campaign(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[research/control] Read current scientific content and next-step status."""

    tool_name = "inspect_hypothesis_campaign"
    thread_id = str(payload.get("thread_id") or "")
    try:
        params = InspectHypothesisCampaignInput.model_validate(payload)
        engine = load_engine(workspace_root(), params.thread_id)
        content = _controller_content(
            engine,
            params.thread_id,
            prefix=(
                f"Inspected {len(engine.state.hypotheses)} hypotheses, "
                f"{len(engine.state.actions)} verifications, and "
                f"{len(engine.state.evidence)} evidence judgments."
            ),
        )
        return content, _artifact(tool_name, params.thread_id, engine)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _raise_tool_error(tool_name, thread_id, exc)


def advance_hypothesis_campaign(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[research/control] Select and return one scientific execution packet."""

    tool_name = "advance_hypothesis_campaign"
    thread_id = str(payload.get("thread_id") or "")
    try:
        params = AdvanceHypothesisCampaignInput.model_validate(payload)
        root = workspace_root()
        with campaign_lock(root, params.thread_id):
            engine = load_engine(root, params.thread_id)
            _require_revision(engine, params.expected_revision)
            engine.advance(params.action_id)
            save_engine(root, params.thread_id, engine)
        content = _controller_content(
            engine,
            params.thread_id,
            prefix="Advanced the campaign by one verification.",
        )
        return content, _artifact(tool_name, params.thread_id, engine)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _raise_tool_error(tool_name, thread_id, exc)


def record_hypothesis_result(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[research/control] Record execution failure or exact evidence_judge output."""

    tool_name = "record_hypothesis_result"
    thread_id = str(payload.get("thread_id") or "")
    try:
        params = RecordHypothesisResultInput.model_validate(payload)
        root = workspace_root()
        with campaign_lock(root, params.thread_id):
            engine = load_engine(root, params.thread_id)
            _require_revision(engine, params.expected_revision)
            judgment = (
                EvidenceJudgment(
                    action_id=params.action_id,
                    summary=params.evidence_summary,
                    source=params.source,
                    effects=params.effects,
                )
                if params.outcome == "completed"
                else None
            )
            engine.record_result(
                params.action_id,
                outcome=params.outcome,
                judgment=judgment,
                failure_reason=params.failure_reason,
            )
            save_engine(root, params.thread_id, engine)
        content = _controller_content(
            engine,
            params.thread_id,
            prefix=(
                "Recorded the independent evidence judgment."
                if params.outcome == "completed"
                else f"Recorded execution failure for {params.action_id}."
            ),
        )
        return content, _artifact(tool_name, params.thread_id, engine)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _raise_tool_error(tool_name, thread_id, exc)


__all__ = [
    "AdvanceHypothesisCampaignInput",
    "ExtendHypothesisCampaignInput",
    "InitializeHypothesisCampaignInput",
    "InspectHypothesisCampaignInput",
    "RecordHypothesisResultInput",
    "advance_hypothesis_campaign",
    "extend_hypothesis_campaign",
    "initialize_hypothesis_campaign",
    "inspect_hypothesis_campaign",
    "record_hypothesis_result",
]
