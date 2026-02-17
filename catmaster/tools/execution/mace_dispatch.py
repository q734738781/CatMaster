from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os
import shlex

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    DispatchRequest,
    dispatch_task,
    dispatch_submission,
    TaskSpec,
    BatchDispatchRequest,
    make_work_base,
)
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.execution.task_payloads import render_task_fields
import shutil
from pydantic import BaseModel, Field


class MaceRelaxInput(BaseModel):
    """Deprecated: single MACE relaxation. Use mace_relax_batch with input/output roots."""

    structure_file: str = Field(
        ...,
        description="Input structure file with lattice information (Supports POSCAR/CIF, xyz files are NOT supported).",
    )
    output_root: Optional[str] = Field(
        None,
        description="Output directory for relaxation results. Defaults to the input file's directory.",
    )
    fmax: float = Field(0.02, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: str = Field(
        "mh-1",
        description=(
            "MACE model name. Recommended options: "
            "'mh-1' (slower, higher accuracy) or "
            "'medium-mpa-0' (faster, lower accuracy)."
        ),
        examples=["mh-1", "medium-mpa-0"],
    )
    head: Optional[str] = Field(
        "omat_pbe",
        description="Model head for multi-head models (e.g. 'omat_pbe'). Use empty string to disable.",
    )
    dispersion: bool = Field(
        False,
        description="Enable dispersion correction in mace_mp. Default: false.",
    )
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


class MaceRelaxBatchInput(BaseModel):
    """Submit multiple MACE relaxations in one DPDispatcher submission. Preferred tool for batch MACE relaxations."""

    input_dir: str = Field(
        ...,
        description="Root directory containing structure files with lattice (POSCAR/CIF).",
    )
    output_root: str = Field(
        ...,
        description=(
            "Root directory to store batch outputs. Results mirror the input directory structure; each input file "
            "expands to a folder without suffix (e.g. input_dir/a/c.vasp -> output_root/a/c/opt.vasp). "
            "Must be outside input_dir."
        ),
    )
    fmax: float = Field(0.02, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: str = Field(
        "mh-1",
        description=(
            "MACE model name. Recommended options: "
            "'mh-1' (slower, higher accuracy) or "
            "'medium-mpa-0' (faster, lower accuracy)."
        ),
        examples=["mh-1", "medium-mpa-0"],
    )
    head: Optional[str] = Field(
        "omat_pbe",
        description="Model head for multi-head models (e.g. 'omat_pbe'). Use empty string to disable.",
    )
    dispersion: bool = Field(
        False,
        description="Enable dispersion correction in mace_mp. Default: false.",
    )
    relax_lattice: bool = Field(
        False,
        description="Whether to relax lattice/cell together with atomic positions via ASE FrechetCellFilter.",
    )
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


class MaceSPBatchInput(BaseModel):
    """Run MACE single-point jobs for structures under one directory."""

    input_dir: str = Field(
        ...,
        description="Input directory containing structures (POSCAR/CONTCAR/CIF/VASP).",
    )
    output_root: str = Field(
        ...,
        description="Output root for batch results. Must be outside input_dir.",
    )
    model: str = Field(
        "mh-1",
        description="MACE model name (e.g., mh-1, medium-mpa-0).",
        examples=["mh-1", "medium-mpa-0"],
    )
    head: Optional[str] = Field(
        "omat_pbe",
        description="Model head for multi-head models. Use empty string to disable.",
    )
    dispersion: bool = Field(
        False,
        description="Enable dispersion correction.",
    )
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


def _resolve_machine_for_resources(resources_key: str) -> str:
    reg = MachineRegister()
    res_cfg = reg.get_resources(resources_key)
    resolved = res_cfg.get("machine")
    if not resolved:
        raise KeyError(f"Resources '{resources_key}' missing machine binding")
    return str(resolved)


def _resolve_mace_head(head_value: Any) -> Optional[str]:
    raw = "omat_pbe" if head_value is None else head_value
    head = str(raw).strip()
    return head or None


def mace_relax(payload: Dict[str, Any]) -> Dict[str, Any]:
    _ = MaceRelaxInput(**payload)
    return create_tool_output(
        tool_name="mace_relax",
        success=False,
        error="Single-file MACE relaxation is deprecated. Use mace_relax_batch with input_root/output_root.",
    )

def _is_structure_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if name in {"POSCAR", "CONTCAR"}:
        return True
    return path.suffix.lower() in {".vasp", ".poscar", ".cif"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _collect_structure_files(root: Path, *, exclude_root: Path | None = None) -> list[Path]:
    files: list[Path] = []
    skip_prefixes = ("mace_batch_", "mace_sp_batch_", "vasp_batch_")
    internal_dirs = {"metadata", ".catmaster"}
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if exclude_root is not None and _is_within(path, exclude_root):
            dirnames[:] = []
            continue
        if any(part.startswith(skip_prefixes) for part in path.parts):
            dirnames[:] = []
            continue
        if any(part in internal_dirs for part in path.parts):
            dirnames[:] = []
            continue
        if "summary.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in internal_dirs and not d.startswith(skip_prefixes)
        ]
        for fname in filenames:
            p = path / fname
            if _is_structure_file(p):
                files.append(p)
    return sorted(files, key=lambda p: str(p))


def mace_relax_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MaceRelaxBatchInput(**payload)
    reg = TaskRegistry()
    cfg = reg.get("mace_relax_dir")
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError("mace_relax_dir missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)
    model = params.model
    if not model:
        raise ValueError("model is required.")
    head = _resolve_mace_head(params.head)
    head_arg = shlex.quote(head or "")
    dispersion = bool(params.dispersion)
    relax_lattice = bool(params.relax_lattice)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        return create_tool_output(
            "mace_relax_batch",
            success=False,
            error=f"input_dir is not a directory: {input_root}",
        )
    if params.output_root is None:
        return create_tool_output(
            "mace_relax_batch",
            success=False,
            error="output_root is required for directory batch relaxations.",
        )
    output_root = resolve_workspace_path(params.output_root)
    if output_root.exists() and not output_root.is_dir():
        return create_tool_output(
            "mace_relax_batch",
            success=False,
            error=f"output_root is not a directory: {output_root}",
        )
    if _is_within(output_root, input_root):
        return create_tool_output(
            "mace_relax_batch",
            success=False,
            error="output_root must not be inside input_dir to avoid mixing inputs with outputs.",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    structures = _collect_structure_files(input_root, exclude_root=None)
    if not structures:
        return create_tool_output(
            "mace_relax_batch",
            success=False,
            error="No structure files found in input_dir (expected POSCAR/CONTCAR/CIF/VASP files).",
        )

    work_base = make_work_base("mace_batch")
    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_input = stage_root / "input"
    stage_output = stage_root / "output"
    shutil.copytree(input_root, stage_input)
    stage_output.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve().parents[2] / "remote" / "gpu" / "mace_relax.py"
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing MACE remote script: {script_src}")
    script_dst = stage_root / "task_script" / "mace_relax.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_src, script_dst)

    cfg = reg.get("mace_relax_dir")

    ctx = {
        "input_path": "input",
        "output_root": "output",
        "fmax": params.fmax,
        "maxsteps": params.maxsteps,
        "steps": params.maxsteps,
        "model": model,
        "head": head_arg,
        "dispersion": "true" if dispersion else "false",
        "relax_lattice": "true" if relax_lattice else "false",
    }
    rendered = render_task_fields(cfg, ctx, stage_root)
    task = TaskSpec(
        command=rendered["command"],
        task_work_path=".",
        forward_files=rendered["forward_files"],
        backward_files=rendered["backward_files"],
    )

    batch_req = BatchDispatchRequest(
        machine=machine,
        resources=resources_key,
        work_base=work_base,
        local_root=str(output_root),
        tasks=[task],
        forward_common_files=[],
        backward_common_files=[],
        clean_remote=False,
        check_interval=params.check_interval,
    )

    result = dispatch_submission(batch_req)

    if stage_output.exists():
        shutil.copytree(stage_output, output_root, dirs_exist_ok=True)

    if stage_root.exists():
        shutil.rmtree(stage_root)

    return create_tool_output(
        tool_name="mace_relax_batch",
        success=True,
        data={
            "task_states": result.task_states,
            "submission_dir": workspace_relpath(Path(result.submission_dir)) if result.submission_dir else "",
            "work_base": result.work_base,
            "input_root_rel": workspace_relpath(input_root),
            "output_root_rel": workspace_relpath(output_root),
            "batch_summary_rel": workspace_relpath(output_root / "batch_summary.json") if (output_root / "batch_summary.json").exists() else None,
            "structures_found": len(structures),
            "model": model,
            "head": head,
            "dispersion": dispersion,
            "relax_lattice": relax_lattice,
        },
        execution_time=result.duration_s,
    )


def mace_sp_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MaceSPBatchInput(**payload)
    reg = TaskRegistry()
    cfg = reg.get("mace_sp_dir")
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError("mace_sp_dir missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)
    model = params.model
    if not model:
        raise ValueError("model is required.")
    head = _resolve_mace_head(params.head)
    head_arg = shlex.quote(head or "")
    dispersion = bool(params.dispersion)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        return create_tool_output(
            "mace_sp_batch",
            success=False,
            error=f"input_dir is not a directory: {input_root}",
        )
    output_root = resolve_workspace_path(params.output_root)
    if output_root.exists() and not output_root.is_dir():
        return create_tool_output(
            "mace_sp_batch",
            success=False,
            error=f"output_root is not a directory: {output_root}",
        )
    if _is_within(output_root, input_root):
        return create_tool_output(
            "mace_sp_batch",
            success=False,
            error="output_root must not be inside input_dir to avoid mixing inputs with outputs.",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    structures = _collect_structure_files(input_root, exclude_root=None)
    if not structures:
        return create_tool_output(
            "mace_sp_batch",
            success=False,
            error="No structure files found in input_dir (expected POSCAR/CONTCAR/CIF/VASP files).",
        )

    work_base = make_work_base("mace_sp_batch")
    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_input = stage_root / "input"
    stage_output = stage_root / "output"
    shutil.copytree(input_root, stage_input)
    stage_output.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve().parents[2] / "remote" / "gpu" / "mace_sp.py"
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing MACE SP remote script: {script_src}")
    script_dst = stage_root / "task_script" / "mace_sp.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_src, script_dst)

    ctx = {
        "input_path": "input",
        "output_root": "output",
        "model": model,
        "head": head_arg,
        "dispersion": "true" if dispersion else "false",
    }
    rendered = render_task_fields(cfg, ctx, stage_root)
    task = TaskSpec(
        command=rendered["command"],
        task_work_path=".",
        forward_files=rendered["forward_files"],
        backward_files=rendered["backward_files"],
    )

    batch_req = BatchDispatchRequest(
        machine=machine,
        resources=resources_key,
        work_base=work_base,
        local_root=str(output_root),
        tasks=[task],
        forward_common_files=[],
        backward_common_files=[],
        clean_remote=False,
        check_interval=params.check_interval,
    )

    result = dispatch_submission(batch_req)

    if stage_output.exists():
        shutil.copytree(stage_output, output_root, dirs_exist_ok=True)

    if stage_root.exists():
        shutil.rmtree(stage_root)

    return create_tool_output(
        tool_name="mace_sp_batch",
        success=True,
        data={
            "task_states": result.task_states,
            "submission_dir": workspace_relpath(Path(result.submission_dir)) if result.submission_dir else "",
            "work_base": result.work_base,
            "input_root_rel": workspace_relpath(input_root),
            "output_root_rel": workspace_relpath(output_root),
            "batch_summary_rel": workspace_relpath(output_root / "batch_summary.json") if (output_root / "batch_summary.json").exists() else None,
            "structures_found": len(structures),
            "model": model,
            "head": head,
            "dispersion": dispersion,
        },
        execution_time=result.duration_s,
    )


def _read_summary(path: Path) -> Dict[str, Any]:
    try:
        import json
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _build_mace_relax_request(
    params: Any,
    *,
    machine: str,
    resources: str,
    work_dir: Path,
    dest_structure: Path,
    registry: TaskRegistry | None = None,
):
    reg = registry or TaskRegistry()
    cfg = reg.get("mace_relax")

    work_base = work_dir.name if work_dir.name not in ("", ".") else make_work_base("mace")
    local_root = work_dir.parent

    ctx = {
        "structure_file": dest_structure.name,
        "structure": dest_structure.name,
        "fmax": params.fmax,
        "maxsteps": params.maxsteps,
        "steps": params.maxsteps,
        "model": params.model,
        "head": shlex.quote(_resolve_mace_head(getattr(params, "head", None)) or ""),
        "dispersion": "true" if bool(getattr(params, "dispersion", False)) else "false",
        "relax_lattice": "true" if bool(getattr(params, "relax_lattice", False)) else "false",
    }

    rendered = render_task_fields(cfg, ctx, work_dir)

    return DispatchRequest(
        machine=machine,
        resources=resources,
        command=rendered["command"],
        work_base=work_base,
        task_work_path=rendered["task_work_path"],
        forward_files=rendered["forward_files"],
        backward_files=rendered["backward_files"],
        forward_common_files=rendered["forward_common_files"],
        backward_common_files=rendered["backward_common_files"],
        local_root=str(local_root),
        check_interval=params.check_interval,
    )


__all__ = [
    "MaceRelaxInput",
    "MaceRelaxBatchInput",
    "MaceSPBatchInput",
    "mace_relax_batch",
    "mace_sp_batch",
]
