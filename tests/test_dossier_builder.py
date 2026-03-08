from __future__ import annotations

from catmaster.agents.response_schemas import MemoryUpdate
from catmaster.runtime.literature.models import LiteratureContextPack, PaperRecord
from catmaster.runtime.research import (
    ConclusionRecord,
    ExperimentBriefModel,
    ExperimentRunPack,
    HypothesisRecord,
    ResearchArtifactRef,
    ResearchBoard,
)
from catmaster.runtime.research.dossier import build_research_dossier


def test_build_research_dossier_aggregates_packs() -> None:
    board = ResearchBoard(
        campaign_id="camp",
        question="Q",
        exploration_policy="anchored",
        max_cycles=3,
        max_literature_queries=2,
        max_fast_runs=1,
        max_standard_runs=1,
        hypotheses=[HypothesisRecord(hypothesis_id="H1", text="seed", source="user_seed", status="supported")],
    )
    literature_pack = LiteratureContextPack(
        query="q",
        depth="quick",
        topic="topic",
        summary="Representative papers support bridge-like adsorption.",
        key_papers=[PaperRecord(title="Paper A", year=2024, source="semantic_scholar")],
        citations=[{"title": "Paper A", "url": "https://example.com/a"}],
        confidence="medium",
    )
    experiment_pack = ExperimentRunPack(
        experiment_id="exp_001",
        brief=ExperimentBriefModel(
            title="exp",
            hypothesis_ids=["H1"],
            lane="fast",
            goal="g",
            task_detail="d",
            expected_outputs=["o"],
            why_now="now",
            stop_condition="stop",
        ),
        run_id="run1",
        run_dir="runs/run1",
        lane="fast",
        status="done",
        summary="Bridge remained stable.",
        key_artifacts=[ResearchArtifactRef(path="results/a.vasp", description="best", kind="output")],
    )
    conclusion = ConclusionRecord(
        final_answer_md="Bridge-like adsorption is best supported so far.",
        supported_claims=["Bridge remained stable in the child run."],
        open_questions=["Need higher-fidelity validation."],
        recommended_next_steps=["Run a standard validation."],
        confidence="medium",
        memory_promotion_candidates=[MemoryUpdate(topic="MEMORY/topics/FACTS.md", content="Bridge was stable.")],
    )

    dossier = build_research_dossier(
        board=board,
        conclusion=conclusion,
        literature_packs=[literature_pack],
        experiment_packs=[experiment_pack],
    )

    assert dossier.campaign_id == "camp"
    assert dossier.final_answer_md.startswith("Bridge-like adsorption")
    assert dossier.literature_summary.startswith("Representative papers")
    assert dossier.key_papers[0].title == "Paper A"
    assert dossier.experiment_rows[0].experiment_id == "exp_001"
    assert dossier.key_artifacts[0].path == "results/a.vasp"
    assert dossier.memory_promotion_candidates[0]["topic"] == "MEMORY/topics/FACTS.md"
