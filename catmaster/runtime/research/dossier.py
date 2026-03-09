from __future__ import annotations

from catmaster.runtime.literature.models import LiteratureContextPack, PaperRecord

from .models import (
    ConclusionRecord,
    DossierExperimentRow,
    ExperimentRunPack,
    ResearchArtifactRef,
    ResearchBoard,
    ResearchDossier,
)


def build_research_dossier(
    *,
    board: ResearchBoard,
    conclusion: ConclusionRecord,
    literature_packs: list[LiteratureContextPack],
    experiment_packs: list[ExperimentRunPack],
) -> ResearchDossier:
    literature_summary = "\n\n".join(
        pack.summary.strip() for pack in literature_packs if str(pack.summary or "").strip()
    ).strip()
    key_papers: list[PaperRecord] = []
    seen_papers: set[str] = set()
    for pack in literature_packs:
        for paper in pack.key_papers:
            key = paper.paper_id or paper.doi or paper.title.strip().lower()
            if not key or key in seen_papers:
                continue
            seen_papers.add(key)
            key_papers.append(paper)

    experiment_rows = [
        DossierExperimentRow(
            experiment_id=pack.experiment_id,
            lane=pack.lane,
            goal=pack.brief.goal,
            summary=pack.summary,
            status=pack.status,
            key_artifact_paths=[artifact.path for artifact in pack.key_artifacts],
        )
        for pack in experiment_packs
    ]
    key_artifacts: list[ResearchArtifactRef] = []
    seen_artifacts: set[str] = set()
    for pack in experiment_packs:
        for artifact in pack.key_artifacts:
            key = artifact.path.strip()
            if not key or key in seen_artifacts:
                continue
            seen_artifacts.add(key)
            key_artifacts.append(artifact)

    citation_shortlist: list[dict] = []
    seen_citations: set[str] = set()
    for pack in literature_packs:
        for citation in pack.citations:
            if not isinstance(citation, dict):
                continue
            key = str(citation.get("url") or citation.get("doi") or citation.get("title") or "").strip()
            if not key or key in seen_citations:
                continue
            seen_citations.add(key)
            citation_shortlist.append(dict(citation))

    return ResearchDossier(
        campaign_id=board.campaign_id,
        question=board.question,
        exploration_policy=board.exploration_policy,
        hypotheses=list(board.hypotheses),
        literature_summary=literature_summary,
        key_papers=key_papers,
        experiment_rows=experiment_rows,
        supported_claims=list(conclusion.supported_claims),
        open_questions=list(conclusion.open_questions),
        recommended_next_steps=list(conclusion.recommended_next_steps),
        key_artifacts=key_artifacts,
        citation_shortlist=citation_shortlist,
        final_answer_md=conclusion.final_answer_md,
        confidence=conclusion.confidence,
        memory_promotion_candidates=[item.model_dump() for item in conclusion.memory_promotion_candidates],
    )


__all__ = ["build_research_dossier"]
