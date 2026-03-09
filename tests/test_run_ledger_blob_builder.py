from __future__ import annotations

import json
from pathlib import Path

from catmaster.runtime.run_ledger.blob_builder import build_run_search_blob


def test_build_run_search_blob_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_001"
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": "run_001"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "task_state.json").write_text(
        json.dumps(
            {
                "user_request": "Evaluate CO adsorption on Fe(111)",
                "summary": "Bridge and ontop were tested; ontop is lower in energy.",
                "tasks": [
                    {"goal": "Build slab"},
                    {"goal": "Enumerate adsorption structures"},
                    {"goal": "Compare energies"},
                ],
                "observations": [{"output_poscar_rel": "ads/ontop_0.vasp"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "tool_trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tool_name": "build_slab"}),
                json.dumps({"payload": {"tool_name": "place_adsorbate"}}),
                json.dumps({"tool_name": "build_slab"}),
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "reports" / "FINAL_REPORT.md").write_text(
        "# Final Report\n\nCO adsorption on Fe(111) completed.\n",
        encoding="utf-8",
    )

    blob = build_run_search_blob(run_dir)
    assert blob.run_id == "run_001"
    assert "Evaluate CO adsorption on Fe(111)" in blob.search_blob_text
    assert "build_slab" in blob.tool_names
    assert "place_adsorbate" in blob.tool_names
    assert len(blob.search_blob_text) <= 5500
