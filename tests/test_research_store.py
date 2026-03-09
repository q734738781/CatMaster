from __future__ import annotations

from pathlib import Path

from catmaster.agents.research_schemas import ConcludePayload, ResearchRequest
from catmaster.runtime.literature.models import LiteratureContextPack
from catmaster.runtime.research import (
    ExperimentBriefModel,
    ExperimentRunPack,
    HypothesisRecord,
    ResearchArtifactRef,
    ResearchBoard,
    ResearchDossier,
    ResearchStore,
)


def test_research_store_round_trip(tmp_path: Path) -> None:
    store = ResearchStore(workspace=tmp_path, campaign_id="camp_001")
    store.ensure_exists()

    request = ResearchRequest(question="What controls CO adsorption on Fe(110)?", seed_hypotheses=["H1"])
    request_path = store.write_request(request)
    assert request_path == "research_campaigns/camp_001/request.json"

    board = ResearchBoard(
        campaign_id="camp_001",
        question=request.question,
        exploration_policy="anchored",
        max_cycles=3,
        max_literature_queries=2,
        max_fast_runs=1,
        max_standard_runs=1,
        hypotheses=[
            HypothesisRecord(
                hypothesis_id="H1",
                text="Bridge site stabilizes CO more strongly than ontop.",
                source="user_seed",
            )
        ],
    )
    board_path = store.save_board(board)
    assert board_path == "research_campaigns/camp_001/board.json"
    loaded = store.load_board()
    assert loaded.question == request.question
    assert (store.metadata_root / "board.md").exists()

    lit_pack = LiteratureContextPack(
        query="CO adsorption Fe(110)",
        depth="quick",
        topic="CO adsorption",
        summary="Literature suggests Fe(110) bridge and hollow sites are common starting points.",
        confidence="medium",
    )
    lit_path = store.persist_literature_pack(lit_pack, action_id="lit_001")
    assert lit_path == "research_campaigns/camp_001/literature/lit-001.json"

    exp_pack = ExperimentRunPack(
        experiment_id="exp_001",
        brief=ExperimentBriefModel(
            title="Fast CO placement check",
            hypothesis_ids=["H1"],
            lane="fast",
            goal="Generate stable CO placements.",
            task_detail="Screen bridge and ontop sites with bounded relaxations.",
            expected_outputs=["ranked structures"],
            why_now="Need a bounded first-pass experiment.",
            stop_condition="Return when at least one stable geometry is found.",
        ),
        run_id="run_child",
        run_dir="runs/run_child",
        lane="fast",
        status="done",
        summary="Bridge placement remained most stable after relaxation.",
        key_artifacts=[ResearchArtifactRef(path="results/bridge.vasp", description="Best structure", kind="output")],
    )
    exp_path = store.persist_experiment_pack(exp_pack, action_id="exp_001")
    assert exp_path == "research_campaigns/camp_001/experiments/exp-001.json"

    conclusion_path = store.persist_conclusion(
        ConcludePayload(
            final_answer_md="Bridge-like coordination appears most plausible so far.",
            supported_claims=["Bridge geometry is stable in the bounded child run."],
            open_questions=["Needs higher-level validation."],
            recommended_next_steps=["Run a standard validation."],
            confidence="medium",
        )
    )
    assert conclusion_path == "research_campaigns/camp_001/conclusion/conclusion.json"

    dossier = ResearchDossier(
        campaign_id="camp_001",
        question=request.question,
        exploration_policy="anchored",
        hypotheses=board.hypotheses,
        literature_summary=lit_pack.summary,
        experiment_rows=[],
        final_answer_md="Bridge-like coordination appears most plausible so far.",
        confidence="medium",
    )
    dossier_json, dossier_file = store.persist_dossier(dossier)
    assert dossier_json == "research_campaigns/camp_001/dossier/dossier.json"
    assert dossier_file == "research/camp_001/dossier/RESEARCH_DOSSIER.md"
    assert (tmp_path / "files" / dossier_file).exists()

    action_log_path = store.append_action_log({"action_id": "lit_001", "kind": "literature"})
    assert action_log_path == "research_campaigns/camp_001/actions.jsonl"

    assert store.load_request()["question"] == request.question
    assert len(store.load_action_log()) == 1
    assert len(store.load_literature_packs()) == 1
    assert len(store.load_experiment_packs()) == 1
    assert store.load_conclusion() is not None
    assert store.load_dossier() is not None
