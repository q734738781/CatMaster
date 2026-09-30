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
    REVISES = "revises"


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
    EXTERNAL = "external"


DEFAULT_COMPLETION_CRITERION = (
    "Reach a defensible answer to the research question using "
    "recorded Results and traceable sources."
)
PERSISTENT_RESEARCH_COMPLETION_CRITERION = (
    "Complete the deliverable or scientific stage explicitly requested in the "
    "research question using recorded Results and traceable sources. If the "
    "requested next step is laboratory or collaborator validation, persist "
    "implementation-ready external Experiment handoffs and stop automatic work "
    "before external execution."
)


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

    claim: str = Field(
        ...,
        min_length=1,
        description=(
            "Falsifiable physical, chemical, or materials claim, not a computational recipe."
        ),
    )
    rationale: str = Field("", description='Scientific reasoning motivating this hypothesis.')
    predictions: list[str] = Field(default_factory=list, description='Observable predictions that would distinguish this hypothesis.')
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

    objective: str = Field(
        ...,
        min_length=1,
        description=(
            "Scientific observable, comparison, or check; downstream workers own unspecified computation."
        ),
    )
    plan_summary: str = Field(
        "",
        description=(
            "Proposed scientific method: evidence/data, representation or measurement, "
            "comparison/baseline and validation needed to answer the objective. Preserve "
            "established scientific constraints; execution workers own unspecified numerical "
            "settings and implementation. Do not describe unperformed methods as tested."
        ),
    )
    decision_rule: str = Field(
        "",
        description=(
            "Optional while the proposal is a draft. Required before the "
            "experiment is marked ready to run. State which observations distinguish "
            "the alternatives and the scope of an inconclusive or negative finding."
        ),
    )
    blocking_reason: str = Field(
        "",
        description=(
            "Concrete scientific or practical reason this experiment cannot "
            "proceed. Leave empty unless the experiment is explicitly blocked."
        ),
    )
    execution_lane: ExecutionLane = Field(
        ExecutionLane.EXPERIMENT,
        description=(
            "Owner of execution. Use external for a complete laboratory or "
            "collaborator handoff that CatMaster must preserve but never launch; "
            "the returned observation is recorded later as a Result."
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

    # Old JSON records remain readable without rewriting checkpoint histories.
    methods: str = Field("missing due to old record", description=(
        "Method actually used: evidence/data, comparison or baseline, analysis and "
        "evaluation, and conditions that determine what was tested. For literature "
        "work describe search scope and how claims were compared. Scientific details only."))
    conclusion: str = Field("missing due to old record", description=(
        "What the result supports or leaves open, limited to the method's coverage; "
        "alternative explanations and the next discriminating hypothesis or experiment."))

    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "Observed or derived scientific outcome. Separate the observation "
            "from causal interpretation, and state modality, applicable conditions, "
            "or provenance in ordinary scientific language when they affect meaning. "
            "Do not assign a global evidence grade."
        ),
    )

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
        description=(
            "Exact existing identifier, DOI, or URL. Omit the reference when no "
            "durable identifier is available; never invent one. For message refs, "
            "a shared discussion's message_id is also a durable source."
        ),
    )

    _clean_id = field_validator("ref_id")(_clean_text)


class HypothesisSeed(HypothesisBody):
    title: str = Field("", description='Short display title; omit to derive from the scientific content.')
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Durable typed sources supporting this record; omit or pass [] when none are available.')

    _clean_title = field_validator("title")(_clean_text)


class GraphCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description='Scientific question defining this graph.')
    title: str = Field("", description='Short display title; omit to derive from the scientific content.')
    completion_criterion: str = Field("", description='Scientific condition for considering this graph answered; omit for the default criterion.')
    decision_preferences: str = Field(
        "",
        description=(
            "Optional stable comparison preferences explicitly stated by the user. "
            "Leave empty when the user has not supplied any."
        ),
    )
    orchestration_mode: OrchestrationMode = Field(OrchestrationMode.MANUAL, description='manual records work without autonomous graph orchestration; auto enables graph orchestration.')
    initial_hypotheses: list[HypothesisSeed] = Field(default_factory=list, description='Optional initial hypotheses; each becomes a new node, whose ID is returned.')

    _clean_question = field_validator("question")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)
    _clean_completion = field_validator("completion_criterion")(_clean_text)
    _clean_decision_preferences = field_validator("decision_preferences")(_clean_text)

    @model_validator(mode="after")
    def _default_completion_criterion(self) -> "GraphCreateRequest":
        if not self.completion_criterion:
            self.completion_criterion = DEFAULT_COMPLETION_CRITERION
        return self


class ResearchHypothesisProposal(HypothesisSeed):
    """One temporary hypothesis branch produced by the planning subagent."""

    proposal_id: str = Field(..., min_length=1, max_length=160)

    _clean_proposal_id = field_validator("proposal_id")(_clean_text)


class ResearchExperimentProposal(ExperimentBody):
    """One temporary draft or runnable experiment produced by planning."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(..., min_length=1, max_length=160)
    title: str = ""
    tests_hypothesis_ids: list[str] = Field(default_factory=list)
    depends_on_experiment_ids: list[str] = Field(default_factory=list)
    refs: list[ResearchRefInput] = Field(default_factory=list)

    _clean_proposal_id = field_validator("proposal_id")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)
    _clean_tests = field_validator("tests_hypothesis_ids")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiment_ids")(
        _clean_string_list
    )


class ResearchHypothesisDraft(BaseModel):
    """Science-first hypothesis input for the model-visible planning action."""

    claim: str = Field(
        ...,
        description=(
            "Falsifiable physical, chemical, or materials claim, not a computational recipe."
        ),
    )
    title: str = Field(
        "",
        description="Optional concise scientific label; leave empty to derive it from the claim.",
    )
    rationale: str = ""
    predictions: list[str] = Field(default_factory=list)
    importance: PriorityBand = Field(
        "",
        description=(
            "Optional relative scientific importance. Leave empty when it has "
            "not been assessed; this is not confidence or probability."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Optional existing DOI, URL, or workspace note path supporting this "
            "branch. Omit unavailable sources rather than inventing identifiers."
        ),
    )

    _clean_claim = field_validator("claim")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)
    _clean_rationale = field_validator("rationale")(_clean_text)
    _clean_predictions = field_validator("predictions")(_clean_string_list)
    _clean_sources = field_validator("sources")(_clean_string_list)


class ResearchExperimentDraft(BaseModel):
    """Science-first experiment input for the model-visible planning action."""

    objective: str = Field(
        ...,
        description="Scientific objective or discriminating check for this temporary route.",
    )
    title: str = Field(
        "",
        description="Optional concise scientific label; leave empty to derive it from the objective.",
    )
    plan_summary: str = Field(
        "",
        description=(
            "Optional planned scientific method, representation, measurement or "
            "comparison; execution workers choose implementation and unspecified numerical settings."
        ),
    )
    decision_rule: str = Field(
        "",
        description="Optional for a draft; include the outcome-specific rule when known.",
    )
    execution_lane: ExecutionLane = Field(
        ExecutionLane.EXPERIMENT,
        description=(
            "Owner of execution. Use external for an implementation-ready "
            "laboratory or collaborator protocol that must wait for a later "
            "human-supplied Result and must not enter automatic execution."
        ),
    )
    estimated_compute_cost: ComputeCostBand = ""
    tests_hypotheses: list[str] = Field(
        default_factory=list,
        description=(
            "Exact scientific titles or claims of the existing or newly proposed "
            "hypotheses this experiment tests."
        ),
    )
    depends_on_experiments: list[str] = Field(
        default_factory=list,
        description=(
            "Exact scientific titles or objectives of prerequisite experiments. "
            "Leave empty when there is no true execution prerequisite."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Optional existing DOI, URL, or workspace note path supporting this "
            "experiment. Omit unavailable sources rather than inventing identifiers."
        ),
    )

    _clean_objective = field_validator("objective")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)
    _clean_plan = field_validator("plan_summary")(_clean_text)
    _clean_rule = field_validator("decision_rule")(_clean_text)
    _clean_tests = field_validator("tests_hypotheses")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiments")(
        _clean_string_list
    )
    _clean_sources = field_validator("sources")(_clean_string_list)


class ResearchGraphPlanningDraft(BaseModel):
    """Science-first model input; the host adds internal planning identifiers."""

    hypotheses: list[ResearchHypothesisDraft] = Field(
        default_factory=list,
        description="Scientifically distinct temporary hypotheses supported by current evidence.",
    )
    experiments: list[ResearchExperimentDraft] = Field(
        default_factory=list,
        description=(
            "Temporary checks stated as observables, comparisons, and decision rules, not computational recipes."
        ),
    )
    recommended_route: str = Field(
        "",
        description=(
            "Optional exact scientific title, claim, or objective of the route to "
            "recommend. Leave empty when the evidence does not distinguish one."
        ),
    )
    recommendation_reason: str = Field(
        "",
        description="Short scientific reason for the recommendation, when present.",
    )

    _clean_recommended_route = field_validator("recommended_route")(_clean_text)
    _clean_recommendation_reason = field_validator("recommendation_reason")(_clean_text)

    @model_validator(mode="after")
    def _require_scientific_content(self) -> "ResearchGraphPlanningDraft":
        if not self.hypotheses and not self.experiments and not self.recommended_route:
            raise ValueError(
                "A temporary plan must add a scientific branch or recommend an "
                "existing ready experiment."
            )
        if self.recommended_route and not self.recommendation_reason:
            raise ValueError(
                "A recommended route requires a concise scientific reason."
            )
        return self


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
        description=(
            "Optional proposal_id or existing ready experiment ID recommended "
            "as the next route. Leave empty when evidence does not distinguish "
            "a useful next step."
        ),
    )
    recommendation_reason: str = Field(
        "",
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
    title: str = ""
    question: str = ""
    completion_criterion: str = ""
    decision_preferences: str = ""
    completed: bool = False
    orchestration_mode: OrchestrationMode = OrchestrationMode.MANUAL
    archived: bool = False

    _clean_title = field_validator("title")(_clean_text)
    _clean_question = field_validator("question")(_clean_text)
    _clean_completion = field_validator("completion_criterion")(_clean_text)
    _clean_decision_preferences = field_validator("decision_preferences")(_clean_text)


class HypothesisCreateRequest(HypothesisBody):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1, description='Latest inspected graph revision; stale writes fail without applying the mutation. Query current state before retrying.')
    title: str = Field("", description='Short display title; omit to derive from the scientific content.')
    suggested_by_result_ids: list[str] = Field(default_factory=list, description='Existing Result node IDs motivating this new hypothesis.')
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Durable typed sources supporting this record; omit or pass [] when none are available.')

    _clean_title = field_validator("title")(_clean_text)
    _clean_result_ids = field_validator("suggested_by_result_ids")(_clean_string_list)


class ExperimentCreateRequest(ExperimentBody):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1, description='Latest inspected graph revision; stale writes fail without applying the mutation. Query current state before retrying.')
    title: str = Field("", description='Short display title; omit to derive from the scientific content.')
    state: ExperimentState = Field(ExperimentState.DRAFT, description='Initial proposal state; draft needs further preparation, ready is eligible for execution. This call does not launch work.')
    tests_hypothesis_ids: list[str] = Field(default_factory=list, description='Existing hypothesis node IDs that this Experiment tests.')
    depends_on_experiment_ids: list[str] = Field(default_factory=list, description='Existing Experiment node IDs whose outputs are prerequisites for this Experiment.')
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Durable typed sources supporting this record; omit or pass [] when none are available.')

    _clean_title = field_validator("title")(_clean_text)
    _clean_tests = field_validator("tests_hypothesis_ids")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiment_ids")(_clean_string_list)


class ResultJudgmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_node_id: str = Field(..., min_length=1, max_length=160, description='Existing hypothesis node ID judged by this Result.')
    relation: Literal["supports", "opposes", "inconclusive"] = Field(..., description='Whether this Result supports, opposes, or is inconclusive for the specified hypothesis.')
    scope: str = Field("", description="Conditions and part of the claim this judgment actually addresses.")
    rationale: str = Field("", description="Scientific reason for this judgment; distinguish observation from interpretation.")

    _clean_hypothesis = field_validator("hypothesis_node_id")(_clean_text)


class ResultCreateRequest(ResultBody):
    model_config = ConfigDict(extra="forbid")

    methods: str = Field("", description=ResultBody.model_fields["methods"].description)
    conclusion: str = Field("", description=ResultBody.model_fields["conclusion"].description)

    expected_revision: int = Field(..., ge=1, description='Latest inspected graph revision; stale writes fail without applying the mutation. Query current state before retrying.')
    title: str = Field("", description='Short display title; omit to derive from the scientific content.')
    experiment_node_id: str = Field(
        "",
        max_length=160,
        description=(
            "Producing Research Graph experiment ID. Leave empty for a sourced "
            "observation or result obtained outside this graph."
        ),
    )
    judgments: list[ResultJudgmentInput] = Field(default_factory=list, description='Effects of this Result on existing hypotheses; each hypothesis can occur once.')
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Durable typed sources supporting this record; omit or pass [] when none are available.')

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

    expected_revision: int = Field(..., ge=1, description='Latest inspected graph revision; stale writes fail without applying the mutation. Query current state before retrying.')
    relation: Literal["supports", "opposes", "inconclusive", "unjudged"] = Field(..., description='Result-to-hypothesis evidence direction; unjudged removes this judgment without changing either node.')
    scope: str = Field("", description='Conditions and part of the hypothesis covered by the judgment.')
    rationale: str = Field("", description='Scientific reason for the evidence relationship.')


class ScientificRevisionRequest(BaseModel):
    """Link a new H or R to the older same-kind claim it revises."""

    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(..., ge=1, description='Latest inspected graph revision; stale writes fail without applying the mutation. Query current state before retrying.')
    new_node_id: str = Field(..., min_length=3, description='Existing new H or R node that revises the older node of the same kind.')
    old_node_id: str = Field(..., min_length=3, description='Existing older H or R node being revised; both nodes are retained.')
    action: Literal["replace", "qualify", "withdraw"] = Field(..., description='How the new claim revises the old one: replace, qualify its scope, or withdraw it.')
    scope: str = Field(..., min_length=1, description="Exactly which claim, conditions or use of the old record changes.")
    rationale: str = Field(..., min_length=1, description="Why the old interpretation changes; preserve still-valid observations.")


class ResearchDispositionRequest(BaseModel):
    """A consequential research stopping decision, not a task or queue record."""

    model_config = ConfigDict(extra="forbid")
    decision_id: str = Field("", description="Reuse the existing decision for the same unresolved issue and material evidence. Omit for a new issue.")
    disposition: Literal["stalled", "waiting", "boundary", "parked", "continue"] = Field(
        ..., description="For an unfulfilled requested stage: stalled requests independent reconsideration; "
        "parked is only available after that review. Waiting/boundary describe a real external dependency "
        "or authorization/resource boundary. An achieved deliverable uses graph completion, not a new stopping decision.")
    problem: str = Field(..., min_length=1, description="Unfulfilled part of the user's actual requested stage.")
    reason: str = Field(..., min_length=1, description='Concrete scientific or execution reason for this state change.')
    authorized_scope: str = Field(..., min_length=1, description="User-authorized work and stopping stage; literature/recommendations do not authorize calculations or lab work.")
    basis_node_ids: list[str] = Field(default_factory=list, description='Existing graph node IDs containing the evidence for this disposition.')
    resume_when: str = Field("", description="Open prerequisite or new evidence that would justify reopening.")
    validation_result_ids: list[str] = Field(default_factory=list, description="Actual Results from the one recommended validation, when finished.")
    exception: Literal["", "authorization", "unavailable", "moot", "equivalent_completed", "user_goal_met"] = Field("", description='Reason the reviewed validation cannot or need not run; empty when no exception applies. Does not grant authorization.')


class ResearchReviewRequest(BaseModel):
    """Independent reconsideration of one declared stalled research issue."""

    model_config = ConfigDict(extra="forbid")
    decision_id: str = Field(..., min_length=3, description='Existing stopping-decision ID in this bound graph being independently reviewed.')
    assessment: str = Field(..., min_length=1, description="What remains valid, what premise failed, and whether a missed action could change the situation.")
    remedy_experiment_id: str = Field("", description="Existing or newly recorded bounded Experiment for one useful authorized validation; empty when none exists.")
    authorization_basis: str = Field("", description="Which explicit user scope permits this particular validation. Recommendations alone never authorize a calculation.")
    resume_when: str = Field(..., min_length=1, description="Open premises and conditions for resuming if the validation adds no new information or no action is feasible.")


class NodePatchRequest(BaseModel):
    """Human edit request. ``body`` is revalidated against the stored kind."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    expected_node_revision: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
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


class ResearchGraphFocusRequest(BaseModel):
    """Select or clear the current thread's focus inside its bound graph."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        "",
        max_length=160,
        description="Node ID in the bound graph; leave empty to clear thread focus.",
    )

    _clean_node = field_validator("node_id")(_clean_text)


class BoundExperimentCreateRequest(BaseModel):
    """Create one explicit Experiment in the bound graph and focus this thread on it."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field("", description='Short display title; omit to derive from the scientific content.')
    objective: str = Field(..., min_length=1, description='Scientific question or result this new Experiment should resolve.')
    plan_summary: str = Field("", description='Brief method and inputs for this Experiment; omit when not yet determined.')
    decision_rule: str = Field("", description='How the outcome distinguishes the relevant hypotheses; omit when not yet determined.')
    tests_hypothesis_ids: list[str] = Field(default_factory=list, description='Existing hypothesis node IDs that this Experiment tests.')
    depends_on_experiment_ids: list[str] = Field(default_factory=list, description='Existing Experiment node IDs whose outputs are prerequisites for this Experiment.')
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Durable typed sources supporting this record; omit or pass [] when none are available.')

    _clean_title = field_validator("title")(_clean_text)
    _clean_objective = field_validator("objective")(_clean_text)
    _clean_plan = field_validator("plan_summary")(_clean_text)
    _clean_rule = field_validator("decision_rule")(_clean_text)
    _clean_tests = field_validator("tests_hypothesis_ids")(_clean_string_list)
    _clean_dependencies = field_validator("depends_on_experiment_ids")(
        _clean_string_list
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_null_controls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("title", "plan_summary", "decision_rule"):
            if data.get(key) is None:
                data[key] = ""
        for key in ("tests_hypothesis_ids", "depends_on_experiment_ids", "refs"):
            if data.get(key) is None:
                data[key] = []
        return data


class BoundResultUpdateRequest(BaseModel):
    """Correct the same scientific observation produced by the focused Experiment."""

    model_config = ConfigDict(extra="forbid")

    result_node_id: str = Field(..., min_length=3, max_length=160, description='Existing Result node ID produced by the currently focused Experiment.')
    summary: str = Field(..., min_length=1, description='Replacement scientific observation for the same Result, preserving its identity.')
    methods: str = Field("", description="Correct the actual method; omit to preserve the saved method.")
    conclusion: str = Field("", description="Correct the scoped interpretation and next question; omit to preserve it.")
    title: str = Field(
        "",
        description="Replacement title; leave empty to preserve the current title.",
    )
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Additional durable sources appended to the existing record; omit or pass [] to add none.')

    _clean_result = field_validator("result_node_id")(_clean_text)
    _clean_summary = field_validator("summary")(_clean_text)
    _clean_title = field_validator("title")(_clean_text)

    @model_validator(mode="before")
    @classmethod
    def _legacy_null_controls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("title") is None:
            data["title"] = ""
        if data.get("refs") is None:
            data["refs"] = []
        return data


class BoundExperimentResumeRequest(BaseModel):
    """Resume the focused blocked Experiment after its concrete blocker is removed."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(..., min_length=1, description='Why the focused Experiment blocker is resolved. This changes its graph state; it does not launch or resume computation.')
    refs: list[ResearchRefInput] = Field(default_factory=list, description='Additional durable sources appended to the existing record; omit or pass [] to add none.')

    _clean_reason = field_validator("reason")(_clean_text)

    @model_validator(mode="before")
    @classmethod
    def _legacy_null_refs(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("refs") is None:
            return {**value, "refs": []}
        return value


class BoundResultRetractRequest(BaseModel):
    """Retract one same-run category-error Result from the focused Experiment."""

    model_config = ConfigDict(extra="forbid")

    result_node_id: str = Field(..., min_length=3, max_length=160, description='Same-thread, same-run Result under the focused Experiment, mistakenly recorded as scientific evidence; not an arbitrary historical Result.')
    reason: str = Field(..., min_length=1, description='Concrete scientific or execution reason for this state change.')

    _clean_result = field_validator("result_node_id")(_clean_text)
    _clean_reason = field_validator("reason")(_clean_text)


class BoundGraphScopeUpdateRequest(BaseModel):
    """Revision-safely change explicitly user-directed scope fields on the bound graph."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field("", description='User-directed replacement title; omit or leave empty to preserve the saved value. Empty cannot clear this field.')
    question: str = Field("", description='User-directed replacement question; omit or leave empty to preserve the saved value. Empty cannot clear this field.')
    completion_criterion: str = Field("", description='User-directed replacement completion criterion; omit or leave empty to preserve the saved value. Empty cannot clear this field.')
    decision_preferences: str = Field(
        "",
        description=(
            "Stable comparison preferences explicitly stated by the user. "
            "Leave empty when the user has not supplied a new preference."
        ),
    )

    _clean_title = field_validator("title")(_clean_text)
    _clean_question = field_validator("question")(_clean_text)
    _clean_completion = field_validator("completion_criterion")(_clean_text)
    _clean_decision_preferences = field_validator("decision_preferences")(_clean_text)

    @model_validator(mode="after")
    def _require_one_change(self) -> "BoundGraphScopeUpdateRequest":
        if not (
            self.title
            or self.question
            or self.completion_criterion
            or self.decision_preferences
        ):
            raise ValueError("at least one graph scope field is required")
        return self


class ResultRetractRequest(BaseModel):
    """Revision-safe human Result deletion with a durable audit reason."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    expected_node_revision: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1)

    _clean_reason = field_validator("reason")(_clean_text)


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
    reason: str = Field(..., min_length=1)

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

    _clean_focus = field_validator("focus_node_id")(_clean_text)


class ResearchExperimentPairOutcomeDraft(BaseModel):
    """One clean, revision-bound relative judgment between A and B."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["a", "b", "indistinguishable", "neither"] = Field(..., description='Which of the A/B candidates supplied for this comparison is preferable, or whether they are indistinguishable or neither is suitable.')
    reason: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise scientific reason based on the canonical H/E/R state, "
            "original sources, and decision consequences of A and B."
        ),
    )
    decisive_source_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Only durable source handles that actually changed this comparison. "
            "Omit or pass [] when no additional source was decisive."
        ),
    )
    unresolved_tradeoff: str = Field(
        "",
        description=(
            "Scientific distinction still needed after this comparison; leave "
            "empty when the outcome resolves the pair."
        ),
    )

    _clean_reason = field_validator("reason")(_clean_text)
    _clean_source_refs = field_validator("decisive_source_refs")(_clean_string_list)
    _clean_tradeoff = field_validator("unresolved_tradeoff")(_clean_text)


class GraphPlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1)
    focus_node_id: str = Field("", max_length=160)

    _clean_focus = field_validator("focus_node_id")(_clean_text)


__all__ = [
    "BoundExperimentCreateRequest",
    "BoundExperimentResumeRequest",
    "BoundGraphScopeUpdateRequest",
    "BoundResultRetractRequest",
    "BoundResultUpdateRequest",
    "DEFAULT_COMPLETION_CRITERION",
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
    "PERSISTENT_RESEARCH_COMPLETION_CRITERION",
    "RefCreateRequest",
    "RefKind",
    "ResearchExperimentDraft",
    "ResearchExperimentPairOutcomeDraft",
    "ResearchExperimentProposal",
    "ResearchGraphPlanningDraft",
    "ResearchGraphPlanningProposal",
    "ResearchHypothesisDraft",
    "ResearchHypothesisProposal",
    "ResearchRefInput",
    "ResearchGraphFocusRequest",
    "ResultBody",
    "ResultCreateRequest",
    "ResultJudgmentInput",
    "ResultJudgmentSetRequest",
    "ScientificRevisionRequest",
    "ResearchDispositionRequest",
    "ResearchReviewRequest",
    "ResultRetractRequest",
    "ThreadGraphBindingRequest",
    "validate_node_body",
]
