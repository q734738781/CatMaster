from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.agents.response_schemas import MemoryUpdate
from catmaster.runtime.literature.models import PaperRecord


class ResearchArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(...)
    description: str = Field(...)
    kind: str = Field(...)


class ExperimentBriefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(...)
    hypothesis_ids: list[str] = Field(default_factory=list)
    lane: Literal["fast", "standard"] = Field(...)
    goal: str = Field(...)
    task_detail: str = Field(...)
    expected_outputs: list[str] = Field(default_factory=list)
    why_now: str = Field(...)
    stop_condition: str = Field(...)
    reference_hint: list[str] = Field(default_factory=list)


class ExperimentRunPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(...)
    brief: ExperimentBriefModel = Field(...)
    run_id: str = Field(...)
    run_dir: str = Field(...)
    lane: Literal["fast", "standard"] = Field(...)
    status: Literal["done", "failure", "needs_intervention", "interrupted_paused"] = Field(...)
    summary: str = Field(...)
    top_observations: list[str] = Field(default_factory=list)
    key_artifacts: list[ResearchArtifactRef] = Field(default_factory=list)
    final_report_path: str | None = Field(None)
    run_export_path: str | None = Field(None)
    child_memory_candidates: list[dict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class HypothesisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(...)
    text: str = Field(...)
    source: Literal["user_seed", "agent_refinement", "agent_local_expand"] = Field(...)
    status: Literal["seed", "active", "supported", "weakened", "discarded", "parked", "open"] = Field("seed")
    note: str = Field("")
    evidence_refs: list[str] = Field(default_factory=list)


class ResearchActionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(...)
    kind: Literal["literature", "experiment", "writer", "ask_human", "human_feedback", "conclusion"] = Field(...)
    status: str = Field(...)
    summary: str = Field(...)
    ref_path: str = Field(...)
    run_id: str | None = Field(None)


class ResearchBoard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(...)
    question: str = Field(...)
    exploration_policy: Literal["anchored", "local_expand", "open"] = Field(...)
    status: Literal["running", "needs_human", "done", "failed"] = Field("running")
    cycle_index: int = Field(0, ge=0)
    max_cycles: int = Field(..., ge=1)
    max_literature_queries: int = Field(..., ge=0)
    max_fast_runs: int = Field(..., ge=0)
    max_standard_runs: int = Field(..., ge=0)
    used_literature_queries: int = Field(0, ge=0)
    used_fast_runs: int = Field(0, ge=0)
    used_standard_runs: int = Field(0, ge=0)
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    action_refs: list[ResearchActionRef] = Field(default_factory=list)
    current_best_answer_md: str = Field("")
    supported_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    memory_promotion_candidates: list[dict] = Field(default_factory=list)
    latest_literature_ref: str | None = Field(None)
    latest_experiment_ref: str | None = Field(None)
    latest_writer_ref: str | None = Field(None)
    latest_human_questions: list[str] = Field(default_factory=list)
    human_feedback_summary: str = Field("")
    history_context_summary: str = Field("")


class DossierExperimentRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(...)
    lane: str = Field(...)
    goal: str = Field(...)
    summary: str = Field(...)
    status: str = Field(...)
    key_artifact_paths: list[str] = Field(default_factory=list)


class ResearchDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(...)
    question: str = Field(...)
    exploration_policy: str = Field(...)
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    literature_summary: str = Field("")
    key_papers: list[PaperRecord] = Field(default_factory=list)
    experiment_rows: list[DossierExperimentRow] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    key_artifacts: list[ResearchArtifactRef] = Field(default_factory=list)
    citation_shortlist: list[dict] = Field(default_factory=list)
    final_answer_md: str = Field(...)
    confidence: str = Field(...)
    memory_promotion_candidates: list[dict] = Field(default_factory=list)


class ConclusionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer_md: str = Field(...)
    supported_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(...)
    memory_promotion_candidates: list[MemoryUpdate] = Field(default_factory=list)


class ResearchPlannerContextPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(...)
    campaign_summary_md: str = Field("")
    hypothesis_snapshot_md: str = Field("")
    recent_actions_md: str = Field("")
    budget_snapshot_md: str = Field("")
    durable_memory_summary_md: str = Field("")
    history_summary_md: str = Field("")
    human_feedback_md: str = Field("")
    context_review_md: str = Field("")
    workspace_summary_md: str = Field("")
    latest_literature_summary_md: str = Field("")
    latest_experiment_summary_md: str = Field("")
    current_best_answer_md: str = Field("")
    open_questions_md: str = Field("")


class ResearchContextReviewPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(...)
    history_focus_md: str = Field("")
    durable_memory_focus_md: str = Field("")
    workspace_focus_md: str = Field("")
    citations: list[dict] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


__all__ = [
    "ConclusionRecord",
    "DossierExperimentRow",
    "ExperimentBriefModel",
    "ExperimentRunPack",
    "HypothesisRecord",
    "ResearchActionRef",
    "ResearchArtifactRef",
    "ResearchBoard",
    "ResearchContextReviewPack",
    "ResearchDossier",
    "ResearchPlannerContextPack",
]
