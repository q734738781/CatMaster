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


PriorityBand: TypeAlias = Literal["low", "medium", "high"]
ComputeCostBand: TypeAlias = Literal["none", "low", "medium", "high"]


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
        "medium",
        description=(
            "Relative scientific importance within this Research Graph. "
            "This is not confidence that the hypothesis is true."
        ),
    )

    _clean_claim = field_validator("claim")(_clean_text)
    _clean_rationale = field_validator("rationale")(_clean_text)
    _clean_predictions = field_validator("predictions")(_clean_string_list)


class ExperimentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(..., min_length=1, max_length=2_000)
    plan_summary: str = Field(..., min_length=1, max_length=4_000)
    decision_rule: str = Field(..., min_length=1, max_length=2_000)
    execution_lane: ExecutionLane = ExecutionLane.EXPERIMENT
    expected_value: PriorityBand = Field(
        "medium",
        description=(
            "Expected decision value if this experiment produces a usable result. "
            "This is not a probability of success."
        ),
    )
    estimated_compute_cost: ComputeCostBand = Field(
        "medium",
        description=(
            "Coarse relative compute demand: none, low, medium, or high. "
            "Do not invent a precise resource estimate when none is known."
        ),
    )

    _clean_objective = field_validator("objective")(_clean_text)
    _clean_plan = field_validator("plan_summary")(_clean_text)
    _clean_rule = field_validator("decision_rule")(_clean_text)


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

    ref_kind: RefKind
    ref_id: str = Field(..., min_length=1, max_length=2_000)

    _clean_id = field_validator("ref_id")(_clean_text)


class HypothesisSeed(HypothesisBody):
    title: str = Field("", max_length=300)

    _clean_title = field_validator("title")(_clean_text)


class GraphCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4_000)
    title: str = Field("", max_length=300)
    orchestration_mode: OrchestrationMode = OrchestrationMode.MANUAL
    initial_hypotheses: list[HypothesisSeed] = Field(default_factory=list, max_length=50)

    _clean_question = field_validator("question")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)


class ResearchExperimentProposal(ExperimentBody):
    """Graph-native branch proposal with qualitative value and compute bands."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field("", max_length=300)
    tests_hypothesis_ids: list[str] = Field(default_factory=list, max_length=100)
    depends_on_experiment_ids: list[str] = Field(default_factory=list, max_length=100)

    _clean_title = field_validator("title")(_clean_text)
    _clean_tests = field_validator("tests_hypothesis_ids")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiment_ids")(
        _clean_string_list
    )


class ResearchGraphPlanningProposal(BaseModel):
    """Bounded portfolio from the planning-only research subagent."""

    model_config = ConfigDict(extra="forbid")

    hypotheses: list[HypothesisSeed] = Field(default_factory=list, max_length=12)
    experiments: list[ResearchExperimentProposal] = Field(
        default_factory=list,
        max_length=24,
    )

    @model_validator(mode="after")
    def _has_scientific_step(self) -> "ResearchGraphPlanningProposal":
        if not self.hypotheses and not self.experiments:
            raise ValueError(
                "A planning proposal must contain a hypothesis or experiment."
            )
        return self


class ResearchEvidenceEffect(BaseModel):
    """Independent reasoning retained in the child thread; the graph stores its edge."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_node_id: str = Field(..., min_length=1, max_length=160)
    relation: Literal["supports", "opposes", "inconclusive"]
    reason: str = Field(..., min_length=1, max_length=2_000)

    _clean_hypothesis = field_validator("hypothesis_node_id")(_clean_text)
    _clean_reason = field_validator("reason")(_clean_text)


class ResearchGraphEvidenceJudgment(BaseModel):
    """Graph-native output from the evidence-only research subagent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1, max_length=4_000)
    effects: list[ResearchEvidenceEffect] = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    _clean_summary = field_validator("summary")(_clean_text)

    @field_validator("effects")
    @classmethod
    def _unique_hypotheses(
        cls,
        effects: list[ResearchEvidenceEffect],
    ) -> list[ResearchEvidenceEffect]:
        node_ids = [effect.hypothesis_node_id for effect in effects]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(
                "Evidence must judge each hypothesis node exactly once."
            )
        return effects


class GraphPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    title: str = Field("", max_length=300)
    question: str = Field("", max_length=4_000)
    orchestration_mode: OrchestrationMode = OrchestrationMode.MANUAL
    archived: bool = False

    _clean_title = field_validator("title")(_clean_text)
    _clean_question = field_validator("question")(_clean_text)


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
    experiment_node_id: str = Field(..., min_length=1, max_length=160)
    judgments: list[ResultJudgmentInput] = Field(default_factory=list, max_length=100)
    refs: list[ResearchRefInput] = Field(default_factory=list, max_length=100)

    _clean_title = field_validator("title")(_clean_text)
    _clean_experiment = field_validator("experiment_node_id")(_clean_text)


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
    "RefCreateRequest",
    "RefKind",
    "ResearchExperimentProposal",
    "ResearchGraphEvidenceJudgment",
    "ResearchGraphPlanningProposal",
    "ResearchRefInput",
    "ResultBody",
    "ResultCreateRequest",
    "ThreadGraphBindingRequest",
    "validate_node_body",
]
