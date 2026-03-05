from __future__ import annotations

import asyncio
import json

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


def test_history_reader_builds_context_pack_with_citations(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        store = RunLedgerStore.create_default(workspace=tmp_path)
        sys_root = system_root(workspace=tmp_path)
        run_dir = sys_root / "runs" / "run_001"
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)

        final_rel = "runs/run_001/reports/FINAL_REPORT.md"
        export_rel = "runs/run_001/reports/RUN_EXPORT.json"
        (sys_root / final_rel).write_text(
            "# Final Report\n\n## Result\nontop site is stable.\n",
            encoding="utf-8",
        )
        (sys_root / export_rel).write_text(
            json.dumps({"answer_summary": "ontop is best"}, ensure_ascii=False),
            encoding="utf-8",
        )

        entry = RunLedgerEntry(
            project_id="project_ws_demo",
            run_id="run_001",
            lane="standard",
            status="done",
            request="CO adsorption on Fe(111)",
            answer_summary="ontop is best",
            search_blob_text="CO adsorption ontop Fe(111)",
            final_report_relpath=final_rel,
            run_export_relpath=export_rel,
            ts_start="2026-03-05T10:00:00Z",
            ts_end="2026-03-05T10:01:00Z",
            model_name="test-model",
            provider="openrouter",
        )
        store.upsert_entry(entry)

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
        reader = HistoryReader(
            searcher=_StubSearcher([hit]),  # type: ignore[arg-type]
            run_ledger_store=store,
            system_root=sys_root,
            rerank_model=None,
        )
        pack = asyncio.run(
            reader.aload_context(
                query="best CO adsorption site on Fe(111)",
                project_id="project_ws_demo",
            )
        )
        assert pack.selected_runs == ["run_001"]
        assert "Relevant historical runs" in pack.context_text
        assert final_rel not in pack.context_text
        assert str(sys_root) not in pack.context_text
        assert pack.citations
        assert pack.citations[0]["run_id"] == "run_001"
        assert pack.citations[0]["path"] == final_rel
