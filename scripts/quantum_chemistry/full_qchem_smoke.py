from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

from catmaster.tools.analysis import analyze_orca_results, analyze_xtb_results
from catmaster.tools.base import ensure_project_space_layout, resolve_workspace_path, workspace_scope
from catmaster.tools.execution import crest_conformer_search, orca_execute_batch, xtb_run_batch
from catmaster.tools.geometry_inputs import (
    create_molecule_from_smiles,
    enumerate_molecular_conformers,
    extract_optimized_molecules,
    filter_conformer_ensemble,
    orca_irc_prepare,
    orca_nebts_prepare,
    orca_optts_prepare,
    orca_prepare,
    orca_scan_prepare,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_run_dir(src_rel: str, dest_rel: str) -> None:
    src = resolve_workspace_path(src_rel, must_exist=True)
    dest = resolve_workspace_path(dest_rel)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)


def _record_pass(records: list[dict[str, Any]], name: str, content: str, artifact: dict[str, Any]) -> None:
    records.append(
        {
            "step": name,
            "status": "passed",
            "content": content,
            "artifact": artifact,
        }
    )


def _record_fail(records: list[dict[str, Any]], name: str, exc: Exception) -> None:
    records.append(
        {
            "step": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


def _step(records: list[dict[str, Any]], name: str, fn, payload: dict[str, Any]) -> dict[str, Any] | None:
    print(f"[smoke] START {name}")
    try:
        content, artifact = fn(payload)
        _record_pass(records, name, content, artifact)
        print(f"[smoke] PASS  {name}")
        return artifact
    except Exception as exc:
        _record_fail(records, name, exc)
        print(f"[smoke] FAIL  {name}: {type(exc).__name__}: {exc}")
        return None


def _prepare_inputs() -> None:
    molecules = {
        "structures/h2o.xyz": "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.758602 0.000000 0.504284\nH -0.758602 0.000000 0.504284\n",
        "structures/nh3.xyz": "4\nammonia\nN 0.000000 0.000000 0.100000\nH 0.937700 0.000000 -0.281300\nH -0.468850 0.812070 -0.281300\nH -0.468850 -0.812070 -0.281300\n",
        "structures/ethanol.xyz": "9\nethanol\nC 0.000000 0.000000 0.000000\nC 1.510000 0.000000 0.000000\nO 2.090000 1.250000 0.000000\nH -0.540000 0.930000 0.000000\nH -0.540000 -0.465000 0.890000\nH -0.540000 -0.465000 -0.890000\nH 1.960000 -0.540000 0.890000\nH 1.960000 -0.540000 -0.890000\nH 3.050000 1.170000 0.000000\n",
        "structures/ethanol_alt.xyz": "9\nethanol-alt\nC 0.000000 0.000000 0.000000\nC 1.510000 0.000000 0.000000\nO 2.090000 1.250000 0.000000\nH -0.540000 0.930000 0.000000\nH -0.540000 -0.465000 0.890000\nH -0.540000 -0.465000 -0.890000\nH 1.960000 -0.540000 0.890000\nH 1.960000 -0.540000 -0.890000\nH 2.850000 1.650000 0.760000\n",
    }
    for rel, text in molecules.items():
        _write_text(resolve_workspace_path(rel), text)


def run_smoke(workspace: Path, *, check_interval: int) -> dict[str, Any]:
    ensure_project_space_layout(workspace, create=True)
    records: list[dict[str, Any]] = []
    with workspace_scope(workspace):
        _prepare_inputs()

        _step(
            records,
            "create_molecule_from_smiles_ethanol",
            create_molecule_from_smiles,
            {
                "smiles": "CCO",
                "output_path": "generated/ethanol_from_smiles",
                "fmt": "both",
            },
        )
        _step(
            records,
            "enumerate_molecular_conformers_ethanol",
            enumerate_molecular_conformers,
            {
                "smiles": "CCO",
                "output_dir": "generated/ethanol_conformers",
                "max_conformers": 12,
                "rms_prune_threshold": 0.2,
            },
        )
        _step(
            records,
            "filter_conformer_ensemble_ethanol",
            filter_conformer_ensemble,
            {
                "input_dir": "generated/ethanol_conformers",
                "output_dir": "generated/ethanol_conformers_filtered",
                "energy_window_kcal_mol": 5.0,
                "rmsd_threshold_angstrom": 0.35,
            },
        )

        _step(
            records,
            "crest_conformer_search_ethanol_standard",
            crest_conformer_search,
            {
                "input_path": "structures/ethanol.xyz",
                "output_root": "runs/crest_ethanol_standard",
                "mode": "standard",
                "method": "gfn2",
                "ewin": 6.0,
                "rthr": 0.125,
                "ethr": 0.05,
                "bthr": 0.01,
                "check_interval": check_interval,
            },
        )
        _step(
            records,
            "crest_conformer_search_ethanol_constrained",
            crest_conformer_search,
            {
                "input_path": "structures/ethanol.xyz",
                "output_root": "runs/crest_ethanol_constrained",
                "mode": "constrained",
                "method": "gfn2",
                "ewin": 6.0,
                "rthr": 0.125,
                "ethr": 0.05,
                "bthr": 0.01,
                "frozen_atom_indices": [2, 8],
                "distance_constraints": [{"atom1": 2, "atom2": 8, "value_angstrom": 0.98}],
                "check_interval": check_interval,
            },
        )
        _step(
            records,
            "crest_conformer_search_ethanol_nci",
            crest_conformer_search,
            {
                "input_path": "structures/ethanol.xyz",
                "output_root": "runs/crest_ethanol_nci",
                "mode": "nci",
                "method": "gfn2",
                "ewin": 6.0,
                "rthr": 0.125,
                "ethr": 0.05,
                "bthr": 0.01,
                "check_interval": check_interval,
            },
        )
        _step(records, "analyze_xtb_results_crest_standard", analyze_xtb_results, {"result_root": "runs/crest_ethanol_standard"})

        _step(
            records,
            "xtb_run_batch_h2o_sp",
            xtb_run_batch,
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "runs/xtb_h2o_sp",
                "mode": "sp",
                "gfn": "gfn2",
                "check_interval": check_interval,
            },
        )
        _step(records, "analyze_xtb_results_h2o_sp", analyze_xtb_results, {"result_root": "runs/xtb_h2o_sp"})

        _step(
            records,
            "xtb_run_batch_h2o_opt",
            xtb_run_batch,
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "runs/xtb_h2o_opt",
                "mode": "opt",
                "gfn": "gfn2",
                "check_interval": check_interval,
            },
        )
        _step(records, "analyze_xtb_results_h2o_opt", analyze_xtb_results, {"result_root": "runs/xtb_h2o_opt"})

        _step(
            records,
            "xtb_run_batch_nh3_hess",
            xtb_run_batch,
            {
                "input_path": "structures/nh3.xyz",
                "output_root": "runs/xtb_nh3_hess",
                "mode": "hess",
                "gfn": "gfn2",
                "check_interval": check_interval,
            },
        )
        _step(records, "analyze_xtb_results_nh3_hess", analyze_xtb_results, {"result_root": "runs/xtb_nh3_hess"})

        _step(
            records,
            "xtb_run_batch_h2o_md",
            xtb_run_batch,
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "runs/xtb_h2o_md",
                "mode": "md",
                "gfn": "gfn2",
                "temperature": 300.0,
                "md_time_ps": 0.1,
                "timestep_fs": 0.5,
                "md_dump_fs": 5.0,
                "check_interval": check_interval,
            },
        )
        _step(records, "analyze_xtb_results_h2o_md", analyze_xtb_results, {"result_root": "runs/xtb_h2o_md"})

        _step(
            records,
            "extract_optimized_molecules_xtb",
            extract_optimized_molecules,
            {
                "input_dir": "runs",
                "output_dir": "runs/extracted_xtb",
                "source": "xtb",
                "include_failed": False,
            },
        )

        sp_art = _step(
            records,
            "orca_prepare_h2o_sp",
            orca_prepare,
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "prepared/orca_h2o_sp",
                "task": "sp",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
            },
        )
        optfreq_art = _step(
            records,
            "orca_prepare_nh3_optfreq",
            orca_prepare,
            {
                "input_path": "structures/nh3.xyz",
                "output_root": "prepared/orca_nh3_optfreq",
                "task": "optfreq",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
            },
        )
        td_art = _step(
            records,
            "orca_prepare_h2o_td",
            orca_prepare,
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "prepared/orca_h2o_td",
                "task": "td",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
                "safe_patch": {"nroots": 3},
            },
        )
        nmr_art = _step(
            records,
            "orca_prepare_nh3_nmr",
            orca_prepare,
            {
                "input_path": "structures/nh3.xyz",
                "output_root": "prepared/orca_nh3_nmr",
                "task": "nmr",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
            },
        )

        combined_root = resolve_workspace_path("prepared/orca_full_batch")
        if combined_root.exists():
            shutil.rmtree(combined_root)
        combined_root.mkdir(parents=True, exist_ok=True)
        for label, art in (
            ("sp_h2o", sp_art),
            ("optfreq_nh3", optfreq_art),
            ("td_h2o", td_art),
            ("nmr_nh3", nmr_art),
        ):
            if not art:
                continue
            run_dir_rel = art["data"]["records"][0]["run_dir_rel"]
            _copy_run_dir(run_dir_rel, f"prepared/orca_full_batch/{label}")

        _step(
            records,
            "orca_execute_batch_full",
            orca_execute_batch,
            {
                "input_dir": "prepared/orca_full_batch",
                "output_root": "runs/orca_full_batch",
                "check_interval": check_interval,
            },
        )
        _step(records, "analyze_orca_results_full", analyze_orca_results, {"result_root": "runs/orca_full_batch"})
        _step(
            records,
            "extract_optimized_molecules_orca",
            extract_optimized_molecules,
            {
                "input_dir": "runs/orca_full_batch",
                "output_dir": "runs/extracted_orca",
                "source": "orca",
                "include_failed": True,
            },
        )

        _step(
            records,
            "orca_scan_prepare_ethanol",
            orca_scan_prepare,
            {
                "input_path": "structures/ethanol.xyz",
                "output_root": "prepared/orca_scan_ethanol",
                "task": "opt",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
                "scan_type": "dihedral",
                "atom_indices": [0, 1, 2, 8],
                "start_value": -180.0,
                "end_value": 180.0,
                "steps": 8,
            },
        )
        _step(
            records,
            "orca_optts_prepare_nh3",
            orca_optts_prepare,
            {
                "input_path": "structures/nh3.xyz",
                "output_root": "prepared/orca_optts_nh3",
                "task": "opt",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
                "calc_hess": True,
                "recalc_hess": 4,
            },
        )
        _step(
            records,
            "orca_nebts_prepare_ethanol",
            orca_nebts_prepare,
            {
                "reactant_path": "structures/ethanol.xyz",
                "product_path": "structures/ethanol_alt.xyz",
                "output_root": "prepared/orca_nebts_ethanol",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
                "nimages": 4,
                "variant": "default",
            },
        )
        _step(
            records,
            "orca_irc_prepare_nh3",
            orca_irc_prepare,
            {
                "input_path": "structures/nh3.xyz",
                "output_root": "prepared/orca_irc_nh3",
                "task": "freq",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "nprocs": 4,
                "maxcore_mb": 512,
            },
        )

    passed = sum(1 for rec in records if rec["status"] == "passed")
    failed = sum(1 for rec in records if rec["status"] == "failed")
    summary = {
        "workspace": str(workspace),
        "passed": passed,
        "failed": failed,
        "records": records,
    }
    summary_path = workspace / "files" / "reports" / "qchem_full_smoke_summary.json"
    _dump_json(summary_path, summary)
    print(f"[smoke] summary_json={summary_path}")
    print(f"[smoke] passed={passed} failed={failed}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full smoke matrix for the molecular quantum-chemistry lane.")
    parser.add_argument(
        "--workspace",
        default="/home/chenhh/python_projects/CatMaster/tmp_qchem_full_smoke",
        help="Project-space root for smoke outputs.",
    )
    parser.add_argument("--check-interval", type=int, default=10, help="DPDispatcher polling interval.")
    args = parser.parse_args()
    summary = run_smoke(Path(args.workspace).expanduser().resolve(), check_interval=int(args.check_interval))
    return 0 if int(summary["failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
