from __future__ import annotations

import asyncio
import json
from pathlib import Path

from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import (
    ExperimentBriefModel,
    ExperimentRunPack,
    HypothesisRecord,
    ResearchActionRef,
    ResearchBoard,
    ResearchContextReviewer,
    ResearchStore,
)
from catmaster.runtime.literature.models import LiteratureContextPack
from catmaster.runtime.run_ledger.history_reader import HistoryReader
from catmaster.runtime.run_ledger.models import RunLedgerEntry, RunSearchHit
from catmaster.runtime.run_ledger.store import RunLedgerStore
from catmaster.tools.base import system_root, workspace_scope


class _StubSearcher:
    def __init__(self, hits):
        self._hits = list(hits)

    async def asearch(self, **kwargs):
        _ = kwargs
        return list(self._hits)


def test_research_context_reviewer_selects_history_memory_and_workspace(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        memory_store = MemoryStore.create_default(workspace=tmp_path)
        memory_store.ensure_exists()
        (memory_store.topics_dir / "CONSTRAINTS.md").write_text(
            "# CONSTRAINTS\n\n## TL;DR\n- Hard constraints: keep the slab and functional fixed while comparing motifs.\n",
            encoding="utf-8",
        )
        (memory_store.topics_dir / "QUESTIONS.md").write_text(
            "# QUESTIONS\n\n## TL;DR\n- Active blockers: finite-coverage ordering remains unclear.\n",
            encoding="utf-8",
        )

        research_store = ResearchStore(workspace=tmp_path, campaign_id="camp")
        research_store.persist_literature_pack(
            LiteratureContextPack(
                query="Fe(110) adsorption coverage ordering",
                depth="standard",
                topic="coverage",
                summary="Bridge dominates low coverage while ontop becomes competitive as coverage increases.",
                confidence="medium",
            ),
            action_id="lit_001",
        )
        research_store.persist_experiment_pack(
            ExperimentRunPack(
                experiment_id="exp_001",
                brief=ExperimentBriefModel(
                    title="coverage check",
                    hypothesis_ids=["H1"],
                    lane="fast",
                    goal="Check if bridge remains preferred across coverage points.",
                    task_detail="Run a bounded coverage sweep.",
                    expected_outputs=["coverage ordering summary"],
                    why_now="Need to resolve current uncertainty.",
                    stop_condition="Ordering is clear enough for next planning step.",
                ),
                run_id="child_001",
                run_dir="runs/child_001",
                lane="fast",
                status="done",
                summary="Bridge stayed stable in the fast run.",
            ),
            action_id="exp_001",
        )
        board = ResearchBoard(
            campaign_id="camp",
            question="Which adsorption motif is most robust on Fe(110)?",
            exploration_policy="anchored",
            max_cycles=4,
            max_literature_queries=2,
            max_fast_runs=2,
            max_standard_runs=1,
            used_literature_queries=1,
            used_fast_runs=1,
            hypotheses=[HypothesisRecord(hypothesis_id="H1", text="Bridge is most robust", source="user_seed")],
            open_questions=["How does finite coverage change ordering?"],
            latest_human_questions=["Prioritize finite-coverage validation first."],
            action_refs=[
                ResearchActionRef(
                    action_id="exp_001",
                    kind="experiment",
                    status="done",
                    summary="Bridge stayed stable in the fast run.",
                    ref_path="research_campaigns/camp/experiments/exp-001.json",
                    run_id="child_001",
                )
            ],
            latest_experiment_ref="research_campaigns/camp/experiments/exp-001.json",
        )

        ledger_store = RunLedgerStore.create_default(workspace=tmp_path)
        sys_root = system_root(workspace=tmp_path)
        run_dir = sys_root / "runs" / "run_001"
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        final_rel = "runs/run_001/reports/FINAL_REPORT.md"
        export_rel = "runs/run_001/reports/RUN_EXPORT.json"
        (sys_root / final_rel).write_text(
            "# Final Report\n\n## Coverage\nBridge remains favorable at low coverage; ontop becomes competitive later.\n",
            encoding="utf-8",
        )
        (sys_root / export_rel).write_text(json.dumps({"answer_summary": "Bridge dominates low coverage."}), encoding="utf-8")
        entry = RunLedgerEntry(
            project_id="project_ws_demo",
            run_id="run_001",
            lane="standard",
            status="done",
            request="Fe(110) adsorption ordering",
            answer_summary="Bridge dominates low coverage.",
            search_blob_text="bridge ontop coverage Fe110",
            final_report_relpath=final_rel,
            run_export_relpath=export_rel,
            ts_start="2026-03-05T10:00:00Z",
            ts_end="2026-03-05T10:01:00Z",
            model_name="test-model",
            provider="openrouter",
        )
        ledger_store.upsert_entry(entry)
        hit = RunSearchHit(
            run_id="run_001",
            project_id="project_ws_demo",
            lane="standard",
            status="done",
            score=1.0,
            source="hybrid",
            request=entry.request,
            answer_summary=entry.answer_summary,
            final_report_relpath=entry.final_report_relpath,
            run_export_relpath=entry.run_export_relpath,
        )
        history_reader = HistoryReader(
            searcher=_StubSearcher([hit]),  # type: ignore[arg-type]
            run_ledger_store=ledger_store,
            system_root=sys_root,
            rerank_model=None,
        )

        reviewer = ResearchContextReviewer(
            history_reader=history_reader,
            store=research_store,
            memory_store=memory_store,
            project_id="project_ws_demo",
        )
        pack = asyncio.run(reviewer.areview(board=board))
        rendered = reviewer.render(pack)

        assert "Reviewed historical runs" in rendered
        assert "Reviewed durable memory" in rendered
        assert "Reviewed workspace/artifacts" in rendered
        assert "MEMORY/topics/CONSTRAINTS.md" in rendered
        assert "research_campaigns/camp/experiments/exp-001.json" in rendered
        assert "Hypothesis focus: H1[seed] Bridge is most robust" in pack.query
        assert "Latest literature: query=Fe(110) adsorption coverage ordering" in pack.query
        assert "Latest experiment: id=exp_001 ; lane=fast ; goal=Check if bridge remains preferred across coverage points." in pack.query
        assert "Current frontier:" in pack.query
        assert pack.citations
