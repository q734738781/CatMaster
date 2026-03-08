from __future__ import annotations

from pathlib import Path

from catmaster.runtime.literature.models import LiteratureContextPack
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import (
    ExperimentBriefModel,
    ExperimentRunPack,
    HypothesisRecord,
    ResearchActionRef,
    ResearchBoard,
    ResearchContextBuilder,
    ResearchStore,
)
from catmaster.runtime.research.models import ResearchArtifactRef


def test_research_context_builder_uses_persisted_campaign_and_memory_state(tmp_path: Path) -> None:
    memory_store = MemoryStore.create_default(workspace=tmp_path)
    memory_store.ensure_exists()
    (memory_store.topics_dir / "CONSTRAINTS.md").write_text(
        "# CONSTRAINTS\n\n## TL;DR\n- Hard constraints: keep the slab fixed and compare only bounded geometries.\n\n## Hard Constraints\n- (empty)\n",
        encoding="utf-8",
    )
    (memory_store.topics_dir / "FILES.md").write_text(
        "# FILES\n\n## TL;DR\n- Entry points: results/bridge.vasp\n\n## Index\n- results/bridge.vasp | best structure\n",
        encoding="utf-8",
    )
    memory_store.refresh_index_from_topics()

    store = ResearchStore(workspace=tmp_path, campaign_id="camp")
    board = ResearchBoard(
        campaign_id="camp",
        question="What controls CO adsorption on Fe(110)?",
        exploration_policy="anchored",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=1,
        cycle_index=2,
        used_literature_queries=1,
        used_fast_runs=1,
        hypotheses=[HypothesisRecord(hypothesis_id="H1", text="Bridge is favored", source="user_seed")],
        action_refs=[
            ResearchActionRef(action_id="lit_001", kind="literature", status="done", summary="lit summary", ref_path="research_campaigns/camp/literature/lit-001.json"),
            ResearchActionRef(action_id="exp_001", kind="experiment", status="done", summary="exp summary", ref_path="research_campaigns/camp/experiments/exp-001.json", run_id="child"),
        ],
        current_best_answer_md="Bridge is favored so far.",
        open_questions=["Need validation on other coverages."],
        latest_human_questions=["Should we trust the bridge preference at finite coverage?"],
        human_feedback_summary="User wants finite-coverage validation before concluding.",
        history_context_summary="Prior standard runs on Fe surfaces exist.",
    )
    store.save_board(board)
    store.persist_literature_pack(
        LiteratureContextPack(query="CO Fe(110)", depth="quick", topic="CO", summary="Literature supports bridge-like motifs.", confidence="medium"),
        action_id="lit_001",
    )
    store.persist_experiment_pack(
        ExperimentRunPack(
            experiment_id="exp_001",
            brief=ExperimentBriefModel(
                title="fast check",
                hypothesis_ids=["H1"],
                lane="fast",
                goal="g",
                task_detail="d",
                expected_outputs=["o"],
                why_now="now",
                stop_condition="stop",
            ),
            run_id="child",
            run_dir="runs/child",
            lane="fast",
            status="done",
            summary="Bridge remained stable.",
            key_artifacts=[ResearchArtifactRef(path="results/bridge.vasp", description="best", kind="output")],
        ),
        action_id="exp_001",
    )

    builder = ResearchContextBuilder(store=store, memory_store=memory_store)
    pack = builder.build_planner_context(board=board, history_summary=board.history_context_summary)
    rendered = builder.render(pack)

    assert "cycle: 2/4" in pack.campaign_summary_md
    assert "CONSTRAINTS.md TL;DR" in pack.durable_memory_summary_md
    assert "keep the slab fixed and compare only bounded geometries" in pack.durable_memory_summary_md
    assert "results/bridge.vasp" in pack.workspace_summary_md
    assert "lit_001" in pack.recent_actions_md
    assert "Bridge remained stable" in pack.latest_experiment_summary_md
    assert "finite-coverage validation" in pack.human_feedback_md
    assert "Campaign goal:\nConduct research about the following request and converge on the best supported answer." in rendered
    assert "Request: What controls CO adsorption on Fe(110)?" in rendered
    assert "What controls CO adsorption on Fe(110)?" in rendered
