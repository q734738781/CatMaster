from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.agents.response_schemas import MemoryUpdate
from catmaster.runtime.research.models import ExperimentBriefModel


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(...)
    seed_hypotheses: list[str] = Field(default_factory=list)
    exploration_policy: Literal["anchored", "local_expand", "open"] = Field("anchored")
    max_cycles: int = Field(6, ge=1)
    max_literature_queries: int = Field(4, ge=0)
    max_fast_runs: int = Field(3, ge=0)
    max_standard_runs: int = Field(2, ge=0)
    writing_mode: Literal["none", "internal_report", "paper_outline", "section_draft", "full_draft"] = Field("none")
    target_section: str | None = Field(None)
    allow_deep_report: bool = Field(False)
    campaign_title: str | None = Field(None)


class HypothesisUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(...)
    status: Literal["active", "supported", "weakened", "discarded", "parked", "open"] = Field(...)
    note: str = Field("")


class NewHypothesisProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(...)
    parent_hypothesis_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(...)


class RunLiteraturePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(...)
    depth: Literal["none", "quick", "standard", "focused", "deep_report"] = Field(...)
    topic: str | None = Field(None)
    seed_papers: list[str] = Field(default_factory=list)
    why_now: str = Field(...)


class ExperimentBrief(ExperimentBriefModel):
    model_config = ConfigDict(extra="forbid")


class AskHumanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(default_factory=list)
    blocking_reason: str = Field(...)
    context: str = Field(...)


class RunWriterPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    why_now: str = Field(...)
    use_existing_evidence_only: bool = Field(True)


class ConcludePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer_md: str = Field(...)
    supported_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(...)
    memory_promotion_candidates: list[MemoryUpdate] = Field(default_factory=list)


class ResearchLeadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["RunLiterature", "RunExperiment", "RunWriter", "AskHuman", "Conclude"] = Field(...)
    rationale: str = Field(...)
    current_best_answer_md: str | None = Field(None)
    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    new_hypotheses: list[NewHypothesisProposal] = Field(default_factory=list)
    run_literature: RunLiteraturePayload | None = Field(None)
    run_experiment: ExperimentBrief | None = Field(None)
    run_writer: RunWriterPayload | None = Field(None)
    ask_human: AskHumanPayload | None = Field(None)
    conclude: ConcludePayload | None = Field(None)

    @model_validator(mode="after")
    def _validate_branch_payload(self) -> "ResearchLeadOutput":
        active = {
            "RunLiterature": self.run_literature,
            "RunExperiment": self.run_experiment,
            "RunWriter": self.run_writer,
            "AskHuman": self.ask_human,
            "Conclude": self.conclude,
        }
        for name, payload in active.items():
            if self.state == name and payload is None:
                raise ValueError(f"state={self.state} requires payload {name}")
            if self.state != name and payload is not None:
                raise ValueError(f"state={self.state} forbids payload {name}")
        if self.state == "AskHuman":
            if not self.ask_human or not [q for q in self.ask_human.questions if str(q).strip()]:
                raise ValueError("AskHuman requires non-empty questions")
        if self.state == "Conclude":
            if not self.conclude or not str(self.conclude.final_answer_md or "").strip():
                raise ValueError("Conclude requires non-empty final_answer_md")
        if self.state == "RunExperiment":
            brief = self.run_experiment
            if brief is None:
                raise ValueError("RunExperiment requires experiment brief")
            if not str(brief.goal or "").strip():
                raise ValueError("RunExperiment requires goal")
            if not str(brief.task_detail or "").strip():
                raise ValueError("RunExperiment requires task_detail")
            if not list(brief.expected_outputs or []):
                raise ValueError("RunExperiment requires expected_outputs")
        return self
__all__ = [
    "AskHumanPayload",
    "ConcludePayload",
    "ExperimentBrief",
    "HypothesisUpdate",
    "NewHypothesisProposal",
    "ResearchLeadOutput",
    "ResearchRequest",
    "RunWriterPayload",
    "RunLiteraturePayload",
]
