from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OrchestrationMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class NodeKind(str, Enum):
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    RESULT = "result"


class ExperimentState(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    HAS_RESULTS = "has_results"
    BLOCKED = "blocked"


class EdgeRelation(str, Enum):
    TESTS = "tests"
    PRODUCES = "produces"
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    INCONCLUSIVE = "inconclusive"
    SUGGESTS = "suggests"
    DEPENDS_ON = "depends_on"


class RefKind(str, Enum):
    THREAD = "thread"
    MESSAGE = "message"
    ARTIFACT = "artifact"
    RUN = "run"
    NOTE = "note"
    DOI = "doi"
    URL = "url"


class ExecutionLane(str, Enum):
    RESEARCH = "research"
    EXPERIMENT = "experiment"
    LITERATURE_REVIEW = "literature_review"


PriorityBand: TypeAlias = Literal["", "low", "medium", "high"]
ComputeCostBand: TypeAlias = Literal["", "none", "low", "medium", "high"]


def _clean_text(value: str) -> str:
    return str(value or "").strip()


def _clean_string_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = _clean_text(item)
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned


class HypothesisBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(..., min_length=1, max_length=2_000)
    rationale: str = Field("", max_length=4_000)
    predictions: list[str] = Field(default_factory=list, max_length=20)
    importance: PriorityBand = Field(
        "",
        description=(
            "Optional relative scientific importance within this Research Graph. "
            "Leave empty when it has not been assessed. This is not confidence "
            "that the hypothesis is true."
        ),
    )

    _clean_claim = field_validator("claim")(_clean_text)
    _clean_rationale = field_validator("rationale")(_clean_text)
    _clean_predictions = field_validator("predictions")(_clean_string_list)


class ExperimentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(..., min_length=1, max_length=2_000)
    plan_summary: str = Field(
        "",
        max_length=4_000,
        description=(
            "Optional while the proposal is a draft. Required before the "
            "experiment is marked ready to run."
        ),
    )
    decision_rule: str = Field(
        "",
        max_length=2_000,
        description=(
            "Optional while the proposal is a draft. Required before the "
            "experiment is marked ready to run."
        ),
    )
    blocking_reason: str = Field(
        "",
        max_length=2_000,
        description=(
            "Concrete scientific or practical reason this experiment cannot "
            "proceed. Leave empty unless the experiment is explicitly blocked."
        ),
    )
    execution_lane: ExecutionLane = ExecutionLane.EXPERIMENT
    expected_value: PriorityBand = Field(
        "",
        description=(
            "Optional expected decision value if this experiment produces a "
            "usable result. Leave empty when unknown. This is not a probability "
            "of success."
        ),
    )
    estimated_compute_cost: ComputeCostBand = Field(
        "",
        description=(
            "Optional coarse relative compute demand. Leave empty when unknown "
            "and do not invent a precise resource estimate."
        ),
    )

    _clean_objective = field_validator("objective")(_clean_text)
    _clean_plan = field_validator("plan_summary")(_clean_text)
    _clean_rule = field_validator("decision_rule")(_clean_text)
    _clean_blocking_reason = field_validator("blocking_reason")(_clean_text)


class ResultBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1, max_length=4_000)

    _clean_summary = field_validator("summary")(_clean_text)


NODE_BODY_MODELS: dict[NodeKind, type[BaseModel]] = {
    NodeKind.HYPOTHESIS: HypothesisBody,
    NodeKind.EXPERIMENT: ExperimentBody,
    NodeKind.RESULT: ResultBody,
}


def validate_node_body(kind: NodeKind | str, value: Any) -> dict[str, Any]:
    node_kind = NodeKind(kind)
    body = NODE_BODY_MODELS[node_kind].model_validate(value)
    return body.model_dump(mode="json")


class ResearchRefInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_kind: RefKind = Field(
        description="Kind of an existing durable source."
    )
    ref_id: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description=(
            "Exact existing identifier, DOI, or URL. Omit the reference when no "
            "durable identifier is available; never invent one."
        ),
    )

    _clean_id = field_validator("ref_id")(_clean_text)


class HypothesisSeed(HypothesisBody):
    title: str = Field("", max_length=300)
    refs: list[ResearchRefInput] = Field(default_factory=list, max_length=100)

    _clean_title = field_validator("title")(_clean_text)


class GraphCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4_000)
    title: str = Field("", max_length=300)
    completion_criterion: str = Field("", max_length=4_000)
    orchestration_mode: OrchestrationMode = OrchestrationMode.MANUAL
    initial_hypotheses: list[HypothesisSeed] = Field(default_factory=list, max_length=50)

    _clean_question = field_validator("question")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)
    _clean_completion = field_validator("completion_criterion")(_clean_text)

    @model_validator(mode="after")
    def _default_completion_criterion(self) -> "GraphCreateRequest":
        if not self.completion_criterion:
            self.completion_criterion = (
                "Reach a defensible answer to the research question using "
                "recorded Results and traceable sources."
            )
        return self


class ResearchHypothesisProposal(HypothesisSeed):
    """One temporary hypothesis branch produced by the planning subagent."""

    proposal_id: str = Field(..., min_length=1, max_length=160)

    _clean_proposal_id = field_validator("proposal_id")(_clean_text)


class ResearchExperimentProposal(ExperimentBody):
    """One temporary draft or runnable experiment produced by planning."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(..., min_length=1, max_length=160)
    title: str = Field("", max_length=300)
    tests_hypothesis_ids: list[str] = Field(default_factory=list, max_length=100)
    depends_on_experiment_ids: list[str] = Field(default_factory=list, max_length=100)
    refs: list[ResearchRefInput] = Field(default_factory=list, max_length=100)

    _clean_proposal_id = field_validator("proposal_id")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)
    _clean_tests = field_validator("tests_hypothesis_ids")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiment_ids")(
        _clean_string_list
    )


class ResearchGraphPlanningProposal(BaseModel):
    """Temporary graph mutation payload used only at the planning write boundary."""

    model_config = ConfigDict(extra="forbid")

    hypotheses: list[ResearchHypothesisProposal] = Field(
        default_factory=list,
        description=(
            "Scientifically distinct temporary hypotheses justified by the "
            "current evidence. Do not add variants merely to fill a quota."
        ),
    )
    experiments: list[ResearchExperimentProposal] = Field(
        default_factory=list,
        description=(
            "Scientifically distinct temporary checks justified by the current "
            "evidence. A branch may remain a draft with only an objective."
        ),
    )
    recommended_target_id: str = Field(
        "",
        max_length=160,
        description=(
            "Optional proposal_id or existing ready experiment ID recommended "
            "as the next route. Leave empty when evidence does not distinguish "
            "a useful next step."
        ),
    )
    recommendation_reason: str = Field(
        "",
        max_length=2_000,
        description="Short scientific reason for the recommendation, when present.",
    )

    _clean_recommended_target = field_validator("recommended_target_id")(_clean_text)
    _clean_recommendation_reason = field_validator("recommendation_reason")(_clean_text)

    @model_validator(mode="after")
    def _validate_proposal_links(self) -> "ResearchGraphPlanningProposal":
        proposal_ids = [
            item.proposal_id for item in [*self.hypotheses, *self.experiments]
        ]
        if any(not proposal_id for proposal_id in proposal_ids):
            raise ValueError("Planning proposal IDs must be non-empty.")
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("Planning proposal IDs must be unique.")
        if not self.hypotheses and not self.experiments and not self.recommended_target_id:
            raise ValueError(
                "A temporary plan must add a branch or recommend an existing "
                "ready experiment."
            )
        if self.recommended_target_id and not self.recommendation_reason:
            raise ValueError(
                "A recommended route requires a concise scientific reason."
            )
        return self


class GraphPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    title: str = Field("", max_length=300)
    question: str = Field("", max_length=4_000)
    completion_criterion: str = Field("", max_length=4_000)
    completed: bool = False
    orchestration_mode: OrchestrationMode = OrchestrationMode.MANUAL
    archived: bool = False

    _clean_title = field_validator("title")(_clean_text)
    _clean_question = field_validator("question")(_clean_text)
    _clean_completion = field_validator("completion_criterion")(_clean_text)


class HypothesisCreateRequest(HypothesisBody):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    title: str = Field("", max_length=300)
    suggested_by_result_ids: list[str] = Field(default_factory=list, max_length=100)
    refs: list[ResearchRefInput] = Field(default_factory=list, max_length=100)

    _clean_title = field_validator("title")(_clean_text)
    _clean_result_ids = field_validator("suggested_by_result_ids")(_clean_string_list)


class ExperimentCreateRequest(ExperimentBody):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    title: str = Field("", max_length=300)
    state: ExperimentState = ExperimentState.DRAFT
    tests_hypothesis_ids: list[str] = Field(default_factory=list, max_length=100)
    depends_on_experiment_ids: list[str] = Field(default_factory=list, max_length=100)
    refs: list[ResearchRefInput] = Field(default_factory=list, max_length=100)

    _clean_title = field_validator("title")(_clean_text)
    _clean_tests = field_validator("tests_hypothesis_ids")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiment_ids")(_clean_string_list)


class ResultJudgmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_node_id: str = Field(..., min_length=1, max_length=160)
    relation: Literal["supports", "opposes", "inconclusive"]

    _clean_hypothesis = field_validator("hypothesis_node_id")(_clean_text)


class ResultCreateRequest(ResultBody):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    title: str = Field("", max_length=300)
    experiment_node_id: str = Field(
        "",
        max_length=160,
        description=(
            "Producing Research Graph experiment ID. Leave empty for a sourced "
            "observation or result obtained outside this graph."
        ),
    )
    judgments: list[ResultJudgmentInput] = Field(default_factory=list, max_length=100)
    refs: list[ResearchRefInput] = Field(default_factory=list, max_length=100)

    _clean_title = field_validator("title")(_clean_text)
    _clean_experiment = field_validator("experiment_node_id")(_clean_text)

    @model_validator(mode="after")
    def _unique_judgments(self) -> "ResultCreateRequest":
        targets = [item.hypothesis_node_id for item in self.judgments]
        if len(targets) != len(set(targets)):
            raise ValueError(
                "A Result may judge each hypothesis at most once."
            )
        return self


class ResultJudgmentSetRequest(BaseModel):
    """Replace one Result-to-Hypothesis judgment, or leave it unjudged."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    relation: Literal["supports", "opposes", "inconclusive", "unjudged"]


class NodePatchRequest(BaseModel):
    """Human edit request. ``body`` is revalidated against the stored kind."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    expected_node_revision: int = Field(..., ge=1)
    title: str = Field(..., min_length=1, max_length=300)
    state: str = Field("", max_length=40)
    body: dict[str, Any] = Field(default_factory=dict)

    _clean_title = field_validator("title")(_clean_text)
    _clean_state = field_validator("state")(_clean_text)

    @model_validator(mode="before")
    @classmethod
    def _legacy_null_body(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("body") is None:
            return {**value, "body": {}}
        return value


class EdgeCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    source_node_id: str = Field(..., min_length=1, max_length=160)
    target_node_id: str = Field(..., min_length=1, max_length=160)
    relation: EdgeRelation

    _clean_source = field_validator("source_node_id")(_clean_text)
    _clean_target = field_validator("target_node_id")(_clean_text)


class RefCreateRequest(ResearchRefInput):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    node_id: str = Field(..., min_length=1, max_length=160)

    _clean_node = field_validator("node_id")(_clean_text)


class ExperimentLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    replicate: bool = False


class ExperimentBlockedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1, max_length=2_000)

    _clean_reason = field_validator("reason")(_clean_text)


class ThreadGraphBindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field("", max_length=160)
    focus_node_id: str = Field("", max_length=160)

    _clean_graph = field_validator("graph_id")(_clean_text)
    _clean_focus = field_validator("focus_node_id")(_clean_text)


class GraphContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_node_id: str = Field("", max_length=160)
    query: str = Field("", max_length=4_000)
    max_nodes: int = Field(24, ge=4, le=100)
    max_chars: int = Field(12_000, ge=2_000, le=40_000)

    _clean_focus = field_validator("focus_node_id")(_clean_text)
    _clean_query = field_validator("query")(_clean_text)


class GraphPlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    focus_node_id: str = Field("", max_length=160)

    _clean_focus = field_validator("focus_node_id")(_clean_text)


class PlanMaterializeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    proposal_id: str = Field(..., min_length=1, max_length=160)

    _clean_proposal = field_validator("proposal_id")(_clean_text)


__all__ = [
    "EdgeCreateRequest",
    "EdgeRelation",
    "ExecutionLane",
    "ExperimentBlockedRequest",
    "ExperimentBody",
    "ExperimentCreateRequest",
    "ExperimentLaunchRequest",
    "ExperimentState",
    "GraphContextRequest",
    "GraphCreateRequest",
    "GraphPlanningRequest",
    "GraphPatchRequest",
    "HypothesisBody",
    "HypothesisCreateRequest",
    "NodeKind",
    "NodePatchRequest",
    "OrchestrationMode",
    "PlanMaterializeRequest",
    "RefCreateRequest",
    "RefKind",
    "ResearchExperimentProposal",
    "ResearchGraphPlanningProposal",
    "ResearchHypothesisProposal",
    "ResearchRefInput",
    "ResultBody",
    "ResultCreateRequest",
    "ResultJudgmentInput",
    "ResultJudgmentSetRequest",
    "ThreadGraphBindingRequest",
    "validate_node_body",
]
