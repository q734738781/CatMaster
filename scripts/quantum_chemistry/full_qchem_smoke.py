#!/usr/bin/env python3
"""Exercise the current native xTB, CREST, and ORCA prepare/execute surfaces."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable

from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.analysis import analyze_orca_results, analyze_xtb_results
from catmaster.tools.base import ensure_project_space_layout, resolve_workspace_path, workspace_scope
from catmaster.tools.execution import remote_submission
from catmaster.tools.geometry_inputs import crest_prepare, orca_nebts_prepare, orca_prepare, xtb_prepare


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(
    records: list[dict[str, Any]],
    name: str,
    function: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    print(f"[smoke] START {name}")
    try:
        content, artifact = function(payload)
    except Exception as exc:
        records.append(
            {
                "step": name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[smoke] FAIL  {name}: {type(exc).__name__}: {exc}")
        return None
    records.append({"step": name, "status": "passed", "content": content, "artifact": artifact})
    print(f"[smoke] PASS  {name}")
    return artifact


def _submit(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    with toolcall_context(f"qchem_smoke_{Path(str(payload['work_dir'])).name}", audience="orca_xtb_worker"):
        return remote_submission(payload)


def _prepare_inputs() -> None:
    _write(
        resolve_workspace_path("structures/h2o.xyz"),
        "3\nwater\nO 0 0 0\nH 0.758602 0 0.504284\nH -0.758602 0 0.504284\n",
    )
    _write(
        resolve_workspace_path("structures/nh3.xyz"),
        "4\nammonia\nN 0 0 0.1\nH 0.9377 0 -0.2813\nH -0.46885 0.81207 -0.2813\nH -0.46885 -0.81207 -0.2813\n",
    )
    _write(
        resolve_workspace_path("structures/ethanol.xyz"),
        "9\nethanol\nC 0 0 0\nC 1.51 0 0\nO 2.09 1.25 0\nH -0.54 0.93 0\nH -0.54 -0.465 0.89\nH -0.54 -0.465 -0.89\nH 1.96 -0.54 0.89\nH 1.96 -0.54 -0.89\nH 3.05 1.17 0\n",
    )
    _write(
        resolve_workspace_path("structures/ethanol_alt.xyz"),
        "9\nethanol-alt\nC 0 0 0\nC 1.51 0 0\nO 2.09 1.25 0\nH -0.54 0.93 0\nH -0.54 -0.465 0.89\nH -0.54 -0.465 -0.89\nH 1.96 -0.54 0.89\nH 1.96 -0.54 -0.89\nH 2.85 1.65 0.76\n",
    )


def run_smoke(workspace: Path, *, check_interval: int) -> dict[str, Any]:
    ensure_project_space_layout(workspace, create=True)
    records: list[dict[str, Any]] = []
    with workspace_scope(workspace):
        _prepare_inputs()

        crest_artifact = _run(
            records,
            "crest_prepare_ethanol",
            crest_prepare,
            {
                "output_root": "prepared/crest_ethanol",
                "argv": ["input.xyz", "--gfn2", "--ewin", "6.0"],
                "asset_mappings": [
                    {"source_path": "structures/ethanol.xyz", "stage_path": "input.xyz"}
                ],
            },
        )
        if crest_artifact:
            crest_submit = _run(
                records,
                "crest_execute_ethanol",
                _submit,
                {
                    "work_dir": crest_artifact["data"]["stage_path"],
                    "task_name": "crest_execute",
                    "submission_config": {"check_interval": check_interval},
                },
            )
            if crest_submit and crest_submit["data"]["attempt_paths"]:
                _run(
                    records,
                    "analyze_crest_ethanol",
                    analyze_xtb_results,
                    {"result_root": crest_submit["data"]["attempt_paths"][0]},
                )

        for label, argv in (
            ("h2o_sp", ["coord.xyz", "--gfn", "2"]),
            ("h2o_opt", ["coord.xyz", "--gfn", "2", "--opt", "normal"]),
            ("nh3_hess", ["coord.xyz", "--gfn", "2", "--hess"]),
        ):
            source = "structures/nh3.xyz" if label.startswith("nh3") else "structures/h2o.xyz"
            prepared = _run(
                records,
                f"xtb_prepare_{label}",
                xtb_prepare,
                {
                    "output_root": f"prepared/xtb_{label}",
                    "argv": argv,
                    "asset_mappings": [{"source_path": source, "stage_path": "coord.xyz"}],
                },
            )
            if not prepared:
                continue
            submitted = _run(
                records,
                f"xtb_execute_{label}",
                _submit,
                {
                    "work_dir": prepared["data"]["stage_path"],
                    "task_name": "xtb_execute",
                    "submission_config": {"check_interval": check_interval},
                },
            )
            if submitted and submitted["data"]["attempt_paths"]:
                _run(
                    records,
                    f"analyze_xtb_{label}",
                    analyze_xtb_results,
                    {"result_root": submitted["data"]["attempt_paths"][0]},
                )

        orca_payloads = {
            "h2o_sp": {
                "input_path": "structures/h2o.xyz",
                "simple_keywords": ["B3LYP", "def2-SVP"],
            },
            "nh3_optfreq": {
                "input_path": "structures/nh3.xyz",
                "simple_keywords": ["B3LYP", "def2-SVP", "Opt", "Freq"],
            },
            "h2o_tddft": {
                "input_path": "structures/h2o.xyz",
                "simple_keywords": ["B3LYP", "def2-SVP"],
                "input_blocks": ["%tddft\n  NRoots 3\nend"],
            },
            "nh3_nmr": {
                "input_path": "structures/nh3.xyz",
                "simple_keywords": ["B3LYP", "def2-SVP", "NMR"],
            },
            "ethanol_scan": {
                "input_path": "structures/ethanol.xyz",
                "simple_keywords": ["r2SCAN-3c", "Opt"],
                "input_blocks": ["%geom\n  Scan\n    D 0 1 2 8 = -180, 180, 8\n  end\nend"],
            },
            "nh3_optts": {
                "input_path": "structures/nh3.xyz",
                "simple_keywords": ["B3LYP", "def2-SVP", "OptTS"],
                "input_blocks": ["%geom\n  Calc_Hess true\n  Recalc_Hess 4\nend"],
            },
            "nh3_irc": {
                "input_path": "structures/nh3.xyz",
                "simple_keywords": ["B3LYP", "def2-SVP", "IRC"],
                "input_blocks": ["%irc\n  Direction both\nend"],
            },
        }
        for label, payload in orca_payloads.items():
            prepared = _run(
                records,
                f"orca_prepare_{label}",
                orca_prepare,
                {
                    **payload,
                    "output_root": f"prepared/orca_{label}",
                    "charge": 0,
                    "multiplicity": 1,
                },
            )
            if label != "h2o_sp" or not prepared:
                continue
            submitted = _run(
                records,
                "orca_execute_h2o_sp",
                _submit,
                {
                    "work_dir": prepared["data"]["records"][0]["stage_path"],
                    "task_name": "orca_execute",
                    "submission_config": {"check_interval": check_interval},
                },
            )
            if submitted:
                _run(
                    records,
                    "analyze_orca_h2o_sp",
                    analyze_orca_results,
                    {"result_root": prepared["data"]["records"][0]["stage_path"]},
                )

        _run(
            records,
            "orca_nebts_prepare_ethanol",
            orca_nebts_prepare,
            {
                "reactant_path": "structures/ethanol.xyz",
                "product_path": "structures/ethanol_alt.xyz",
                "output_root": "prepared/orca_nebts_ethanol",
                "simple_keywords": ["B3LYP", "def2-SVP", "NEB-TS"],
                "neb_block": "%neb\n  Product \"product.xyz\"\n  NImages 4\nend",
                "charge": 0,
                "multiplicity": 1,
            },
        )

    passed = sum(record["status"] == "passed" for record in records)
    failed = sum(record["status"] == "failed" for record in records)
    summary = {"workspace": str(workspace), "passed": passed, "failed": failed, "records": records}
    summary_path = workspace / "files" / "reports" / "qchem_full_smoke_summary.json"
    _write(summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[smoke] summary_json={summary_path}")
    print(f"[smoke] passed={passed} failed={failed}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native quantum-chemistry smoke matrix.")
    parser.add_argument("--workspace", default="tmp_qchem_full_smoke")
    parser.add_argument("--check-interval", type=int, default=10)
    args = parser.parse_args()
    summary = run_smoke(Path(args.workspace).expanduser().resolve(), check_interval=args.check_interval)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
