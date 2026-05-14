from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ElementPosterior(BaseModel):
    element: str
    p_keep: float = Field(ge=0.0, le=1.0)
    p_top10: float = Field(ge=0.0, le=1.0)
    score_mean: float
    score_std: float = Field(ge=0.0)
    uncertainty: float = Field(ge=0.0)
    support_count: int = Field(ge=0)


class Stage1Result(BaseModel):
    campaign_id: str
    top10_pool: list[str]
    shadow_pool: list[str]
    summary: str
    artifact_paths: dict[str, str]


class Stage2Candidate(BaseModel):
    formula: str
    stability_score: float
    diffusion_score: float
    activation_barrier_ev: float
    volume_deformation: float
    exploration_mode: Literal["exploit", "explore", "surprise"]


class Stage2Result(BaseModel):
    campaign_id: str
    pareto_formulas: list[str]
    dft_queue: list[str]
    experiment_queue: list[str]
    summary: str
    artifact_paths: dict[str, str]
