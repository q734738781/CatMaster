from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    DispatchRequest,
    dispatch_task,
    dispatch_submission,
    TaskSpec,
    BatchDispatchRequest,
    make_work_base,
)
from catmaster.tools.execution.resource_router import ResourceRouter, Route
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.execution.task_payloads import render_task_fields
import shutil
from pydantic import BaseModel, Field


class MaceRelaxInput(BaseModel):
    """Submit a single MACE relaxation."""

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
    model: Optional[str] = Field(None, description="MACE model name; defaults from router config (medium-mpa-0).")
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
            "Root directory to store batch outputs. Results are flattened under output_root using '__' to encode the "
            "relative input path (without file suffix). For example, input 'a/b/CO.vasp' -> output "
            "'<output_root>/a__b__CO/'. Must be outside input_dir."
        ),
    )
    fmax: float = Field(0.02, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: Optional[str] = Field(None, description="MACE model name; defaults from router config (medium-mpa-0).")
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


def mace_relax(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MaceRelaxInput(**payload)
    router = ResourceRouter()
    route = router.route("mace_relax")

    model = params.model or route.defaults.get("model")
    if not model:
        raise ValueError("model is required; set in payload or router defaults")

    structure_path = resolve_workspace_path(params.structure_file, must_exist=True)
    output_root = resolve_workspace_path(params.output_root) if params.output_root else structure_path.parent
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"output_root is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root
    dest_structure = work_dir / structure_path.name
    if dest_structure.resolve() != structure_path.resolve():
        dest_structure.write_bytes(structure_path.read_bytes())

    params.model = model
    dispatch_req = _build_mace_relax_request(params, route=route, work_dir=work_dir, dest_structure=dest_structure)
    result = dispatch_task(dispatch_req)

    summary = _read_summary(work_dir / "summary.json")

    return create_tool_output(
        tool_name="mace_relax",
        success=True,
        data={
            "task_states": result.task_states,
            "work_base": result.work_base,
            "relaxed_structure_rel": workspace_relpath(work_dir / summary.get("output_structure") if summary.get("output_structure") else work_dir / "CONTCAR"),
            "trajectory_rel": workspace_relpath(work_dir / "opt.traj"),
            "optimization_log_rel": workspace_relpath(work_dir / "opt.log"),
            "summary_json_rel": workspace_relpath(work_dir / "summary.json"),
            "converged": summary.get("converged"),
            "final_energy_eV": summary.get("final_energy_eV"),
            "max_force": summary.get("max_force"),
            "nsteps": summary.get("nsteps"),
        },
        execution_time=result.duration_s,
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
    skip_prefixes = ("mace_batch_", "vasp_batch_")
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if exclude_root is not None and _is_within(path, exclude_root):
            dirnames[:] = []
            continue
        if any(part.startswith(skip_prefixes) for part in path.parts):
            dirnames[:] = []
            continue
        if ".catmaster" in path.parts:
            dirnames[:] = []
            continue
        if "summary.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d != ".catmaster" and not d.startswith(skip_prefixes)
        ]
        for fname in filenames:
            p = path / fname
            if _is_structure_file(p):
                files.append(p)
    return sorted(files, key=lambda p: str(p))


def mace_relax_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MaceRelaxBatchInput(**payload)
    router = ResourceRouter()
    route = router.route("mace_relax")

    model = params.model or route.defaults.get("model")
    if not model:
        raise ValueError("model is required; set in payload or router defaults")

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

    reg = TaskRegistry()
    cfg = reg.get("mace_relax_dir")

    ctx = {
        "input_path": "input",
        "output_root": "output",
        "fmax": params.fmax,
        "maxsteps": params.maxsteps,
        "steps": params.maxsteps,
        "model": model,
    }
    rendered = render_task_fields(cfg, ctx, stage_root)
    task = TaskSpec(
        command=rendered["command"],
        task_work_path=".",
        forward_files=rendered["forward_files"],
        backward_files=rendered["backward_files"],
    )

    batch_req = BatchDispatchRequest(
        machine=route.machine,
        resources=route.resources,
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
            "submission_dir": result.submission_dir,
            "work_base": result.work_base,
            "input_root_rel": workspace_relpath(input_root),
            "output_root_rel": workspace_relpath(output_root),
            "batch_summary_rel": workspace_relpath(output_root / "batch_summary.json") if (output_root / "batch_summary.json").exists() else None,
            "structures_found": len(structures),
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

def _build_mace_relax_request(params: Any, *, route: Route, work_dir: Path, dest_structure: Path, registry: TaskRegistry | None = None):
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
    }

    rendered = render_task_fields(cfg, ctx, work_dir)

    return DispatchRequest(
        machine=route.machine,
        resources=route.resources,
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
    "mace_relax",
    "mace_relax_batch",
]
