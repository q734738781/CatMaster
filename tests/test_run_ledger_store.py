from __future__ import annotations

from catmaster.runtime.run_ledger.models import RunLedgerEntry
from catmaster.runtime.run_ledger.store import RunLedgerStore
from catmaster.tools.base import workspace_scope


def _sample_entry(*, run_id: str = "run_001") -> RunLedgerEntry:
    return RunLedgerEntry(
        project_id="project_ws_demo",
        run_id=run_id,
        lane="standard",
        status="done",
        request="Find best adsorption site for CO on Fe(111)",
        answer_summary="ontop site is best under current settings.",
        search_blob_text="request: CO on Fe111; tools: build_slab, place_adsorbate, vasp_relax_prepare",
        final_report_relpath=f"runs/{run_id}/reports/FINAL_REPORT.md",
        run_export_relpath=f"runs/{run_id}/reports/RUN_EXPORT.json",
        ts_start="2026-03-05T10:00:00Z",
        ts_end="2026-03-05T10:10:00Z",
        model_name="test-model",
        provider="openrouter",
    )


def test_run_ledger_store_upsert_and_sparse_search(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        store = RunLedgerStore.create_default(workspace=tmp_path)
        entry = _sample_entry(run_id="run_aaa")
        store.upsert_entry(entry)

        loaded = store.get_entry("run_aaa")
        assert loaded is not None
        assert loaded.project_id == "project_ws_demo"
        assert loaded.answer_summary.startswith("ontop")

        hits = store.search_sparse(
            project_id="project_ws_demo",
            query="Fe(111) adsorption CO",
            limit=5,
            lane="standard",
            status="done",
        )
        assert hits
        assert hits[0].run_id == "run_aaa"
        assert hits[0].source == "sparse"
