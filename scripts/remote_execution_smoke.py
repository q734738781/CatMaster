#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROJECT_SPACE = Path(os.environ.get("CATMASTER_REMOTE_SMOKE_PROJECT", "/tmp/catmaster_remote_execution_smoke"))


@dataclass
class SmokeContext:
    project_space: Path
    run_id: str
    files_root: Path
    metadata_root: Path
    run_dir: Path

    @property
    def case_root_rel(self) -> str:
        return f"remote_execution_smoke/{self.run_id}"

    @property
    def report_path(self) -> Path:
        return self.files_root / self.case_root_rel / "remote_execution_smoke_report.json"


@dataclass
class CaseResult:
    name: str
    status: str
    started_at: str
    finished_at: str
    elapsed_s: float
    stage_rel: str = ""
    stage_dir: str = ""
    remote_context_id: str = ""
    submission_hash: str = ""
    receipt_rel: str = ""
    checks: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    runner: Callable[[SmokeContext, argparse.Namespace], dict[str, Any]]


_REGISTRY = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_run_id(raw: str | None = None) -> str:
    if raw:
        text = raw
    else:
        text = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return token or "run"


def _init_context(project_space: Path, run_id: str) -> SmokeContext:
    from catmaster.tools.base import ensure_project_space_layout

    layout = ensure_project_space_layout(project_space, create=True)
    metadata_root = layout["metadata_root"]
    run_dir = metadata_root / "runs" / f"remote_execution_smoke_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return SmokeContext(
        project_space=layout["project_space_root"],
        run_id=run_id,
        files_root=layout["files_root"],
        metadata_root=metadata_root,
        run_dir=run_dir,
    )


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        from catmaster.tools.registry import ToolRegistry

        _REGISTRY = ToolRegistry()
    return _REGISTRY


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"JSON file is not an object: {path}")
    return data


def _materialize_artifact(ctx: SmokeContext, artifact: dict[str, Any]) -> dict[str, Any]:
    refs = artifact.get("offload_refs") or []
    if not refs:
        return artifact
    ref = refs[0]
    if isinstance(ref, dict):
        ref = ref.get("offload_ref") or ref.get("path") or ""
    ref_path = ctx.files_root / str(ref)
    loaded = _read_json(ref_path)
    return loaded


def _invoke_tool(
    ctx: SmokeContext,
    tool_name: str,
    payload: dict[str, Any],
    *,
    audience: str = "",
) -> tuple[Any, dict[str, Any]]:
    tools = _registry().as_langchain_tools(
        allowlist=[tool_name],
        workspace=str(ctx.project_space),
        run_dir=str(ctx.run_dir),
        audience=audience,
    )
    if len(tools) != 1:
        raise RuntimeError(f"tool not available: {tool_name}")
    content, artifact = tools[0].func(**payload)
    if not isinstance(artifact, dict):
        raise RuntimeError(f"{tool_name} returned non-dict artifact: {type(artifact).__name__}")
    return content, _materialize_artifact(ctx, artifact)


def _case_stage(ctx: SmokeContext, name: str) -> tuple[str, Path]:
    stage_rel = f"{ctx.case_root_rel}/{name}"
    stage_dir = ctx.files_root / stage_rel
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_rel, stage_dir


def _write_o2_xyz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "2",
                "O2 remote smoke",
                "O 0.000000 0.000000 0.000000",
                "O 0.000000 0.000000 1.210000",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_water_xyz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "3",
                "H2O remote smoke",
                "O 0.000000 0.000000 0.000000",
                "H 0.758602 0.000000 0.504284",
                "H -0.758602 0.000000 0.504284",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_o2_poscar(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "O2 remote smoke",
                "1.0",
                "15.000000 0.000000 0.000000",
                "0.000000 15.000000 0.000000",
                "0.000000 0.000000 15.000000",
                "O",
                "2",
                "Cartesian",
                "7.500000 7.500000 7.500000",
                "7.500000 7.500000 8.710000",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _status_ok(stage_dir: Path) -> dict[str, Any]:
    status = _read_json(stage_dir / "status.json")
    rc = status.get("returncode")
    if rc != 0:
        raise AssertionError(
            "remote command failed "
            f"returncode={rc} stdout_tail={status.get('stdout_tail')!r} "
            f"stderr_tail={status.get('stderr_tail')!r}"
        )
    return status


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{label} is not numeric: {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AssertionError(f"{label} is not finite: {value!r}")
    return parsed


def _extract_last_float(pattern: str, path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    raw = matches[-1]
    if isinstance(raw, tuple):
        raw = next((item for item in raw if item), "")
    try:
        return float(raw)
    except Exception:
        return None


def _submit_remote(
    ctx: SmokeContext,
    *,
    work_dir: str,
    task_name: str,
    audience: str,
    check_interval: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _content, artifact = _invoke_tool(
        ctx,
        "remote_submission",
        {
            "work_dir": work_dir,
            "task_name": task_name,
            "params": params or {},
            "config": {"check_interval": int(check_interval)},
        },
        audience=audience,
    )
    data = artifact.get("data") or {}
    if not isinstance(data, dict):
        raise AssertionError("remote_submission artifact.data is not a dict")
    if not data.get("task_state_counts"):
        raise AssertionError("DPDispatcher returned no task_state_counts")
    return data


def _remote_context(data: dict[str, Any]) -> dict[str, str]:
    return {
        "remote_context_id": str(data.get("remote_context_id") or ""),
        "submission_hash": str(data.get("submission_hash") or ""),
        "receipt_rel": str(data.get("receipt_rel") or ""),
    }


def run_mace_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "mace_o2_sp")
    input_dir = stage_dir / "input"
    _write_o2_poscar(input_dir / "O2.vasp")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="mace_sp_dir",
        audience="materials_worker",
        check_interval=args.mace_check_interval,
        params={
            "model": args.mace_model,
            "head": args.mace_head,
            "default_dtype": args.mace_dtype,
            "device": args.mace_device,
        },
    )
    _status_ok(stage_dir)
    batch = _read_json(stage_dir / "output" / "batch_summary.json")
    if batch.get("errors"):
        raise AssertionError(f"MACE returned errors: {batch.get('errors')!r}")
    results = batch.get("results") or []
    if not results:
        raise AssertionError("MACE batch_summary.json has no results")
    energy = _finite_number(results[0].get("summary", {}).get("energy_eV"), "MACE energy_eV")
    if not (stage_dir / "output" / "O2" / "summary.json").is_file():
        raise AssertionError("MACE summary.json was not downloaded")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["status.json returncode=0", "batch_summary has one finite energy", "output/O2/summary.json exists"],
        "details": {"energy_eV": energy, "task_data": data},
        **_remote_context(data),
    }


def _assert_uma_batch_energy(stage_dir: Path, output_name: str) -> tuple[dict[str, Any], float]:
    batch = _read_json(stage_dir / "output" / "batch_summary.json")
    if batch.get("errors"):
        raise AssertionError(f"UMA returned errors: {batch.get('errors')!r}")
    results = batch.get("results") or []
    if not results:
        raise AssertionError("UMA batch_summary.json has no results")
    energy = _finite_number(results[0].get("summary", {}).get("energy_eV"), "UMA energy_eV")
    if not (stage_dir / "output" / output_name / "summary.json").is_file():
        raise AssertionError(f"UMA summary.json was not downloaded for {output_name}")
    return batch, energy


def _assert_uma_batch_relax(stage_dir: Path, output_name: str, structure_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = _read_json(stage_dir / "output" / "batch_summary.json")
    if batch.get("errors"):
        raise AssertionError(f"UMA relaxation returned errors: {batch.get('errors')!r}")
    results = batch.get("results") or []
    if not results:
        raise AssertionError("UMA relaxation batch_summary.json has no results")
    summary = results[0].get("summary") or {}
    _finite_number(summary.get("final_energy_eV"), "UMA final_energy_eV")
    _finite_number(summary.get("max_force_abs_eVA"), "UMA max_force_abs_eVA")
    if not isinstance(summary.get("converged"), bool):
        raise AssertionError(f"UMA relaxation summary lacks boolean convergence flag: {summary!r}")
    output_dir = stage_dir / "output" / output_name
    if not (output_dir / "summary.json").is_file():
        raise AssertionError(f"UMA relaxation summary.json was not downloaded for {output_name}")
    if not (output_dir / structure_name).is_file():
        raise AssertionError(f"UMA relaxation output structure was not downloaded: {output_dir / structure_name}")
    return batch, summary


def run_uma_mol_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "uma_h2o_sp")
    input_dir = stage_dir / "input"
    _write_water_xyz(input_dir / "H2O.xyz")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="uma_sp_dir",
        audience="orca_xtb_worker",
        check_interval=args.uma_check_interval,
        params={
            "model": args.uma_model,
            "uma_task": "omol",
            "charge": 0,
            "spin": args.uma_mol_spin,
            "device": args.uma_device,
        },
    )
    _status_ok(stage_dir)
    batch, energy = _assert_uma_batch_energy(stage_dir, "H2O")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["status.json returncode=0", "UMA OMOL batch_summary has one finite energy", "output/H2O/summary.json exists"],
        "details": {"energy_eV": energy, "summary": batch, "task_data": data},
        **_remote_context(data),
    }


def run_uma_mol_relax(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "uma_h2o_relax")
    input_dir = stage_dir / "input"
    _write_water_xyz(input_dir / "H2O.xyz")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="uma_relax_dir",
        audience="orca_xtb_worker",
        check_interval=args.uma_check_interval,
        params={
            "model": args.uma_model,
            "uma_task": "omol",
            "charge": 0,
            "spin": args.uma_mol_spin,
            "device": args.uma_device,
            "fmax": args.uma_relax_fmax,
            "steps": args.uma_relax_steps,
            "relax_cell": "false",
        },
    )
    _status_ok(stage_dir)
    batch, summary = _assert_uma_batch_relax(stage_dir, "H2O", "opt.xyz")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": [
            "status.json returncode=0",
            "UMA OMOL relaxation has finite final energy and max force",
            "output/H2O/summary.json and opt.xyz exist",
        ],
        "details": {"summary": summary, "batch_summary": batch, "task_data": data},
        **_remote_context(data),
    }


def run_uma_mat_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "uma_o2_box_sp")
    input_dir = stage_dir / "input"
    _write_o2_poscar(input_dir / "O2.vasp")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="uma_sp_dir",
        audience="materials_worker",
        check_interval=args.uma_check_interval,
        params={
            "model": args.uma_model,
            "uma_task": args.uma_task,
            "charge": 0,
            "spin": 0,
            "device": args.uma_device,
        },
    )
    _status_ok(stage_dir)
    batch, energy = _assert_uma_batch_energy(stage_dir, "O2")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["status.json returncode=0", "UMA periodic batch_summary has one finite energy", "output/O2/summary.json exists"],
        "details": {"energy_eV": energy, "summary": batch, "task_data": data},
        **_remote_context(data),
    }


def run_uma_mat_relax(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "uma_o2_box_relax")
    input_dir = stage_dir / "input"
    _write_o2_poscar(input_dir / "O2.vasp")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="uma_relax_dir",
        audience="materials_worker",
        check_interval=args.uma_check_interval,
        params={
            "model": args.uma_model,
            "uma_task": args.uma_task,
            "charge": 0,
            "spin": 0,
            "device": args.uma_device,
            "fmax": args.uma_relax_fmax,
            "steps": args.uma_relax_steps,
            "relax_cell": "false",
        },
    )
    _status_ok(stage_dir)
    batch, summary = _assert_uma_batch_relax(stage_dir, "O2", "opt.vasp")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": [
            "status.json returncode=0",
            "UMA periodic relaxation has finite final energy and max force",
            "output/O2/summary.json and opt.vasp exist",
        ],
        "details": {"summary": summary, "batch_summary": batch, "task_data": data},
        **_remote_context(data),
    }


def run_vasp_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    input_rel, input_dir = _case_stage(ctx, "vasp_o2_input")
    _write_o2_poscar(input_dir / "O2.vasp")
    prepared_rel = f"{ctx.case_root_rel}/vasp_o2_sp"
    _content, prep_artifact = _invoke_tool(
        ctx,
        "vasp_prepare",
        {
            "input_path": f"{input_rel}/O2.vasp",
            "output_root": prepared_rel,
            "preset": "static",
            "regime": "gas",
            "k_product": 1,
            "user_incar_patch": {
                "NSW": 0,
                "ISPIN": 2,
                "MAGMOM": {"O": 1.0},
                "NELM": int(args.vasp_nelm),
            },
            "patch_policy": "force",
        },
        audience="materials_worker",
    )
    stage_dir = ctx.files_root / prepared_rel
    for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR"):
        if not (stage_dir / name).is_file():
            raise AssertionError(f"vasp_prepare did not write {name}")
    data = _submit_remote(
        ctx,
        work_dir=prepared_rel,
        task_name="vasp_execute",
        audience="materials_worker",
        check_interval=args.vasp_check_interval,
    )
    _status_ok(stage_dir)
    output_names = ("OUTCAR", "OSZICAR", "vasprun.xml")
    if not any((stage_dir / name).is_file() for name in output_names):
        raise AssertionError(f"VASP did not return any expected output: {output_names}")
    return {
        "stage_rel": prepared_rel,
        "stage_dir": str(stage_dir),
        "checks": ["vasp_prepare wrote INCAR/KPOINTS/POSCAR/POTCAR", "status.json returncode=0", "VASP output file exists"],
        "details": {"prepare_data": prep_artifact.get("data") or {}, "task_data": data},
        **_remote_context(data),
    }


def run_xtb_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "xtb_o2_sp")
    _write_o2_xyz(stage_dir / "input.xyz")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="xtb_run",
        audience="orca_xtb_worker",
        check_interval=args.xtb_check_interval,
        params={"input_name": "input.xyz", "mode": "sp", "gfn": args.xtb_gfn, "charge": 0, "uhf": 2},
    )
    _status_ok(stage_dir)
    summary = _read_json(stage_dir / "xtb_summary.json")
    if summary.get("completed") is not True:
        raise AssertionError(f"xTB did not complete: {summary!r}")
    energy_hartree = _extract_last_float(r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", stage_dir / "xtb_stdout.out")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["status.json returncode=0", "xtb_summary.completed=true"],
        "details": {"energy_hartree": energy_hartree, "summary": summary, "task_data": data},
        **_remote_context(data),
    }


def run_orca_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    input_rel, input_dir = _case_stage(ctx, "orca_o2_input")
    _write_o2_xyz(input_dir / "O2.xyz")
    output_rel = f"{ctx.case_root_rel}/orca_o2_sp"
    _content, prep_artifact = _invoke_tool(
        ctx,
        "orca_prepare",
        {
            "input_path": f"{input_rel}/O2.xyz",
            "output_root": output_rel,
            "task": "sp",
            "method": args.orca_method,
            "basis": args.orca_basis,
            "charge": 0,
            "multiplicity": 3,
            "maxcore_mb": int(args.orca_maxcore_mb),
            "tightness": "loose",
            "safe_patch": {"scf_maxiter": int(args.orca_scf_maxiter)},
        },
        audience="orca_xtb_worker",
    )
    records = (prep_artifact.get("data") or {}).get("records") or []
    if not records:
        raise AssertionError("orca_prepare returned no prepared records")
    stage_rel = str(records[0]["run_dir_rel"])
    stage_dir = ctx.files_root / stage_rel
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="orca_execute",
        audience="orca_xtb_worker",
        check_interval=args.orca_check_interval,
    )
    _status_ok(stage_dir)
    summary = _read_json(stage_dir / "orca_summary.json")
    if summary.get("completed") is not True:
        raise AssertionError(f"ORCA did not complete normally: {summary!r}")
    energy_hartree = _extract_last_float(
        r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        stage_dir / "job.out",
    )
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["orca_prepare wrote job.inp", "status.json returncode=0", "orca_summary.completed=true"],
        "details": {"energy_hartree": energy_hartree, "summary": summary, "prepare_data": prep_artifact.get("data") or {}, "task_data": data},
        **_remote_context(data),
    }


def run_cp2k_sp(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    input_rel, input_dir = _case_stage(ctx, "cp2k_o2_input")
    _write_o2_xyz(input_dir / "O2.xyz")
    output_rel = f"{ctx.case_root_rel}/cp2k_o2_sp"
    _content, prep_artifact = _invoke_tool(
        ctx,
        "cp2k_prepare",
        {
            "input_path": f"{input_rel}/O2.xyz",
            "output_root": output_rel,
            "recipe": "sp",
            "settings": {
                "periodic": "none",
                "cell_abc": [12, 12, 12],
                "xc": args.cp2k_xc,
                "max_scf": int(args.cp2k_max_scf),
            },
        },
        audience="materials_worker",
    )
    records = (prep_artifact.get("data") or {}).get("records") or []
    if not records:
        raise AssertionError("cp2k_prepare returned no prepared records")
    stage_rel = str(records[0]["stage_dir_rel"])
    stage_dir = ctx.files_root / stage_rel
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="cp2k_execute",
        audience="materials_worker",
        check_interval=args.cp2k_check_interval,
    )
    _status_ok(stage_dir)
    summary = _read_json(stage_dir / "cp2k_summary.json")
    if summary.get("completed") is not True:
        raise AssertionError(f"CP2K did not complete: {summary!r}")
    energy_hartree = _extract_last_float(
        r"ENERGY\|\s+Total\s+FORCE_EVAL.*?:\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        stage_dir / "job.out",
    )
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["cp2k_prepare wrote job.inp", "status.json returncode=0", "cp2k_summary.completed=true"],
        "details": {"energy_hartree": energy_hartree, "summary": summary, "prepare_data": prep_artifact.get("data") or {}, "task_data": data},
        **_remote_context(data),
    }


def run_lammps_min(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    input_rel, input_dir = _case_stage(ctx, "lammps_o2_input")
    _write_o2_xyz(input_dir / "O2.xyz")
    _ff_content, ff_artifact = _invoke_tool(
        ctx,
        "lammps_forcefield_validate",
        {
            "forcefield_card": {
                "units": "metal",
                "atom_style": "atomic",
                "pair_style": "lj/cut 8.5",
                "pair_coeff": ["* * 0.0103 3.0"],
            }
        },
        audience="dynamics_worker",
    )
    output_rel = f"{ctx.case_root_rel}/lammps_o2_min"
    _content, prep_artifact = _invoke_tool(
        ctx,
        "lammps_prepare",
        {
            "input_path": f"{input_rel}/O2.xyz",
            "output_root": output_rel,
            "recipe": "minimize",
            "forcefield_card_path": (ff_artifact.get("data") or {})["output_path_rel"],
            "settings": {"cell_abc": [20, 20, 20], "thermo": 1},
        },
        audience="dynamics_worker",
    )
    records = (prep_artifact.get("data") or {}).get("records") or []
    if not records:
        raise AssertionError("lammps_prepare returned no prepared records")
    stage_rel = str(records[0]["stage_dir_rel"])
    stage_dir = ctx.files_root / stage_rel
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="lammps_execute",
        audience="dynamics_worker",
        check_interval=args.lammps_check_interval,
    )
    _status_ok(stage_dir)
    summary = _read_json(stage_dir / "lammps_summary.json")
    if summary.get("completed") is not True:
        raise AssertionError(f"LAMMPS did not complete: {summary!r}")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["lammps_prepare wrote in.lammps", "status.json returncode=0", "lammps_summary.completed=true"],
        "details": {"summary": summary, "prepare_data": prep_artifact.get("data") or {}, "task_data": data},
        **_remote_context(data),
    }


def run_crest_quick(ctx: SmokeContext, args: argparse.Namespace) -> dict[str, Any]:
    stage_rel, stage_dir = _case_stage(ctx, "crest_h2o_quick")
    _write_water_xyz(stage_dir / "input.xyz")
    data = _submit_remote(
        ctx,
        work_dir=stage_rel,
        task_name="crest_run",
        audience="orca_xtb_worker",
        check_interval=args.crest_check_interval,
        params={
            "input_name": "input.xyz",
            "mode": "standard",
            "method": args.crest_method,
            "ewin": args.crest_ewin,
            "rthr": args.crest_rthr,
            "ethr": args.crest_ethr,
            "bthr": args.crest_bthr,
        },
    )
    _status_ok(stage_dir)
    summary = _read_json(stage_dir / "crest_summary.json")
    if summary.get("completed") is not True:
        raise AssertionError(f"CREST did not complete: {summary!r}")
    return {
        "stage_rel": stage_rel,
        "stage_dir": str(stage_dir),
        "checks": ["status.json returncode=0", "crest_summary.completed=true"],
        "details": {"summary": summary, "task_data": data},
        **_remote_context(data),
    }


CASES: dict[str, CaseSpec] = {
    "mace_sp": CaseSpec("mace_sp", "MACE GPU O2 single-point energy through mace_sp_dir.", run_mace_sp),
    "uma_mol_sp": CaseSpec("uma_mol_sp", "FairChem UMA OMOL H2O single-point energy through uma_sp_dir.", run_uma_mol_sp),
    "uma_mol_relax": CaseSpec("uma_mol_relax", "FairChem UMA OMOL H2O short relaxation through uma_relax_dir.", run_uma_mol_relax),
    "uma_mat_sp": CaseSpec("uma_mat_sp", "FairChem UMA periodic O2-box single-point energy through uma_sp_dir.", run_uma_mat_sp),
    "uma_mat_relax": CaseSpec("uma_mat_relax", "FairChem UMA periodic O2-box short relaxation through uma_relax_dir.", run_uma_mat_relax),
    "vasp_sp": CaseSpec("vasp_sp", "VASP CPU O2 static single-point through vasp_prepare + vasp_execute.", run_vasp_sp),
    "xtb_sp": CaseSpec("xtb_sp", "xTB CPU O2 single-point through xtb_run.", run_xtb_sp),
    "orca_sp": CaseSpec("orca_sp", "ORCA CPU O2 triplet single-point through orca_prepare + orca_execute.", run_orca_sp),
    "cp2k_sp": CaseSpec("cp2k_sp", "CP2K CPU O2 single-point through cp2k_prepare + cp2k_execute.", run_cp2k_sp),
    "lammps_min": CaseSpec("lammps_min", "LAMMPS CPU/GPU-detect LJ minimization through lammps_prepare + lammps_execute.", run_lammps_min),
    "crest_quick": CaseSpec("crest_quick", "CREST CPU quick H2O conformer-search executable check through crest_run.", run_crest_quick),
}

SUITES: dict[str, list[str]] = {
    "core": ["mace_sp", "xtb_sp", "orca_sp"],
    "materials": ["mace_sp", "vasp_sp"],
    "uma": ["uma_mol_sp", "uma_mol_relax", "uma_mat_sp", "uma_mat_relax"],
    "qchem": ["xtb_sp", "orca_sp"],
    "dynamics": ["cp2k_sp", "lammps_min"],
    "all": ["mace_sp", "vasp_sp", "xtb_sp", "orca_sp", "cp2k_sp", "lammps_min", "crest_quick"],
}


def _parse_case_list(values: list[str] | None, suite: str) -> list[str]:
    selected: list[str] = []
    if values:
        for value in values:
            selected.extend(item.strip() for item in str(value).split(",") if item.strip())
    else:
        selected.extend(SUITES[suite])
    unknown = [name for name in selected if name not in CASES]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    deduped: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _run_case(ctx: SmokeContext, spec: CaseSpec, args: argparse.Namespace) -> CaseResult:
    started = time.time()
    started_at = _now_iso()
    try:
        payload = spec.runner(ctx, args)
        status = "passed"
        error = ""
    except Exception as exc:
        payload = {}
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finished = time.time()
    result = CaseResult(
        name=spec.name,
        status=status,
        started_at=started_at,
        finished_at=_now_iso(),
        elapsed_s=round(finished - started, 3),
        stage_rel=str(payload.get("stage_rel") or ""),
        stage_dir=str(payload.get("stage_dir") or ""),
        remote_context_id=str(payload.get("remote_context_id") or ""),
        submission_hash=str(payload.get("submission_hash") or ""),
        receipt_rel=str(payload.get("receipt_rel") or ""),
        checks=list(payload.get("checks") or []),
        details=dict(payload.get("details") or {}),
        error=error,
    )
    return result


def _write_report(ctx: SmokeContext, args: argparse.Namespace, selected: list[str], results: list[CaseResult]) -> None:
    payload = {
        "run_id": ctx.run_id,
        "created_at": _now_iso(),
        "project_space": str(ctx.project_space),
        "files_root": str(ctx.files_root),
        "run_dir": str(ctx.run_dir),
        "suite": args.suite,
        "selected_cases": selected,
        "results": [asdict(item) for item in results],
        "summary": {
            "passed": sum(1 for item in results if item.status == "passed"),
            "failed": sum(1 for item in results if item.status == "failed"),
        },
    }
    ctx.report_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_case_catalog() -> None:
    print("Suites:")
    for name, cases in SUITES.items():
        print(f"  {name}: {', '.join(cases)}")
    print("\nCases:")
    for name, spec in CASES.items():
        print(f"  {name}: {spec.description}")


def build_parser() -> argparse.ArgumentParser:
    def _env_int(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw in (None, ""):
            return None
        return int(str(raw).strip())

    parser = argparse.ArgumentParser(
        description=(
            "Submit real CatMaster DPDispatcher remote smoke jobs. "
            "These are actual calculations, not dry-runs."
        )
    )
    parser.add_argument("--list", action="store_true", help="List suites/cases and exit without submitting jobs.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="core", help="Case suite to run when --case is not provided.")
    parser.add_argument("--case", action="append", help="Case name or comma-separated case names. Overrides --suite.")
    parser.add_argument("--project-space", default=str(DEFAULT_PROJECT_SPACE), help="Project-space root used for staged files and reports.")
    parser.add_argument("--run-id", default="", help="Stable run id. Default is a timestamp.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed case.")
    parser.add_argument("--check-interval", type=int, default=int(os.environ.get("CATMASTER_REMOTE_CHECK_INTERVAL", "30")))

    parser.add_argument("--mace-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_MACE_CHECK_INTERVAL"))
    parser.add_argument("--mace-model", default=os.environ.get("CATMASTER_REMOTE_MACE_MODEL", "mh-1"))
    parser.add_argument("--mace-head", default=os.environ.get("CATMASTER_REMOTE_MACE_HEAD", "omat_pbe"))
    parser.add_argument("--mace-dtype", choices=("float32", "float64"), default=os.environ.get("CATMASTER_REMOTE_MACE_DTYPE", "float32"))
    parser.add_argument("--mace-device", default=os.environ.get("CATMASTER_REMOTE_MACE_DEVICE", "auto"))

    parser.add_argument("--uma-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_UMA_CHECK_INTERVAL"))
    parser.add_argument("--uma-model", default=os.environ.get("CATMASTER_REMOTE_UMA_MODEL", "uma-s-1p2"))
    parser.add_argument("--uma-task", choices=("auto", "omat", "oc20", "oc22", "oc25", "odac", "omc"), default=os.environ.get("CATMASTER_REMOTE_UMA_TASK", "omat"))
    parser.add_argument("--uma-device", default=os.environ.get("CATMASTER_REMOTE_UMA_DEVICE", "auto"))
    parser.add_argument("--uma-mol-spin", type=int, default=int(os.environ.get("CATMASTER_REMOTE_UMA_MOL_SPIN", "1")))
    parser.add_argument("--uma-relax-fmax", type=float, default=float(os.environ.get("CATMASTER_REMOTE_UMA_RELAX_FMAX", "0.05")))
    parser.add_argument("--uma-relax-steps", type=int, default=int(os.environ.get("CATMASTER_REMOTE_UMA_RELAX_STEPS", "5")))

    parser.add_argument("--vasp-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_VASP_CHECK_INTERVAL"))
    parser.add_argument("--vasp-nelm", type=int, default=int(os.environ.get("CATMASTER_REMOTE_VASP_NELM", "40")))

    parser.add_argument("--xtb-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_XTB_CHECK_INTERVAL"))
    parser.add_argument("--xtb-gfn", choices=("gfn2", "gfn1", "gfnff"), default=os.environ.get("CATMASTER_REMOTE_XTB_GFN", "gfn2"))

    parser.add_argument("--orca-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_ORCA_CHECK_INTERVAL"))
    parser.add_argument("--orca-method", default=os.environ.get("CATMASTER_REMOTE_ORCA_METHOD", "HF"))
    parser.add_argument("--orca-basis", default=os.environ.get("CATMASTER_REMOTE_ORCA_BASIS", "STO-3G"))
    parser.add_argument("--orca-maxcore-mb", type=int, default=int(os.environ.get("CATMASTER_REMOTE_ORCA_MAXCORE_MB", "1000")))
    parser.add_argument("--orca-scf-maxiter", type=int, default=int(os.environ.get("CATMASTER_REMOTE_ORCA_SCF_MAXITER", "80")))

    parser.add_argument("--cp2k-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_CP2K_CHECK_INTERVAL"))
    parser.add_argument("--cp2k-xc", default=os.environ.get("CATMASTER_REMOTE_CP2K_XC", "PBE"))
    parser.add_argument("--cp2k-max-scf", type=int, default=int(os.environ.get("CATMASTER_REMOTE_CP2K_MAX_SCF", "50")))

    parser.add_argument("--lammps-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_LAMMPS_CHECK_INTERVAL"))

    parser.add_argument("--crest-check-interval", type=int, default=_env_int("CATMASTER_REMOTE_CREST_CHECK_INTERVAL"))
    parser.add_argument("--crest-method", choices=("gfn2", "gfn1", "gfnff"), default=os.environ.get("CATMASTER_REMOTE_CREST_METHOD", "gfn2"))
    parser.add_argument("--crest-ewin", type=float, default=float(os.environ.get("CATMASTER_REMOTE_CREST_EWIN", "2.0")))
    parser.add_argument("--crest-rthr", type=float, default=float(os.environ.get("CATMASTER_REMOTE_CREST_RTHR", "0.25")))
    parser.add_argument("--crest-ethr", type=float, default=float(os.environ.get("CATMASTER_REMOTE_CREST_ETHR", "0.1")))
    parser.add_argument("--crest-bthr", type=float, default=float(os.environ.get("CATMASTER_REMOTE_CREST_BTHR", "0.05")))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "mace_check_interval",
        "uma_check_interval",
        "vasp_check_interval",
        "xtb_check_interval",
        "orca_check_interval",
        "cp2k_check_interval",
        "lammps_check_interval",
        "crest_check_interval",
    ):
        if getattr(args, name) is None:
            setattr(args, name, args.check_interval)
    if args.list:
        _print_case_catalog()
        return 0

    selected = _parse_case_list(args.case, args.suite)
    ctx = _init_context(Path(args.project_space).expanduser().resolve(), _safe_run_id(args.run_id))
    print(f"Project space: {ctx.project_space}")
    print(f"Run id: {ctx.run_id}")
    print(f"Selected cases: {', '.join(selected)}")
    print("These cases submit real DPDispatcher jobs.\n")

    results: list[CaseResult] = []
    for name in selected:
        spec = CASES[name]
        print(f"[remote-smoke] START {name}: {spec.description}", flush=True)
        result = _run_case(ctx, spec, args)
        results.append(result)
        if result.status == "passed":
            print(f"[remote-smoke] PASS  {name} elapsed={result.elapsed_s}s stage={result.stage_rel}", flush=True)
        else:
            print(f"[remote-smoke] FAIL  {name} elapsed={result.elapsed_s}s error={result.error}", flush=True)
        if result.status == "failed" and args.stop_on_failure:
            break

    _write_report(ctx, args, selected, results)
    passed = sum(1 for item in results if item.status == "passed")
    failed = sum(1 for item in results if item.status == "failed")
    print(f"\nReport: {ctx.report_path}")
    print(f"Summary: passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
