from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
    fmax: float = Field(0.03, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: Optional[str] = Field(None, description="MACE model name; defaults from router config (medium-mpa-0).")
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


class MaceRelaxBatchInput(BaseModel):
    """Submit multiple MACE relaxations in one DPDispatcher submission. Preferred tool for batch MACE relaxations."""

    structure_files: list[str] = Field(..., description="List of structure files with lattice (POSCAR/CIF).")
    fmax: float = Field(0.03, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
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
    parent = structure_path.parent
    base = structure_path.stem
    candidate = parent / base
    idx = 1
    while candidate.exists():
        candidate = parent / f"{base}_run{idx}"
        idx += 1
    work_dir = candidate
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_structure = work_dir / structure_path.name
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

def mace_relax_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MaceRelaxBatchInput(**payload)
    router = ResourceRouter()
    route = router.route("mace_relax")

    model = params.model or route.defaults.get("model")
    if not model:
        raise ValueError("model is required; set in payload or router defaults")

    structures = [resolve_workspace_path(p, must_exist=True) for p in params.structure_files]
    work_base = make_work_base("mace_batch")
    base_dir = structures[0].parent

    reg = TaskRegistry()
    cfg = reg.get("mace_relax")

    tasks: list[TaskSpec] = []
    task_meta = []
    for struct in structures:
        task_dir = base_dir / work_base / struct.stem
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        dest_structure = task_dir / struct.name
        dest_structure.write_bytes(struct.read_bytes())

        ctx = {
            "structure_file": dest_structure.name,
            "structure": dest_structure.name,
            "fmax": params.fmax,
            "maxsteps": params.maxsteps,
            "steps": params.maxsteps,
            "model": model,
        }
        rendered = render_task_fields(cfg, ctx, task_dir)

        tasks.append(
            TaskSpec(
                command=rendered["command"],
                task_work_path=task_dir.relative_to(base_dir / work_base).as_posix(),
                forward_files=rendered["forward_files"],
                backward_files=rendered["backward_files"],
            )
        )
        task_meta.append(
            {
                "input_struct_rel": workspace_relpath(struct),
                "stage_dir": task_dir,
                "backward_files": rendered["backward_files"],
            }
        )

    batch_req = BatchDispatchRequest(
        machine=route.machine,
        resources=route.resources,
        work_base=work_base,
        local_root=str(base_dir),
        tasks=tasks,
        forward_common_files=[],
        backward_common_files=[],
        clean_remote=False,
        check_interval=params.check_interval,
    )

    result = dispatch_submission(batch_req)

    outputs = []
    for meta in task_meta:
        stage = meta["stage_dir"]
        back_files = []
        for pattern in meta["backward_files"]:
            for p in stage.glob(pattern):
                back_files.append(workspace_relpath(p))
        outputs.append(
            {
                "input_structure_rel": meta["input_struct_rel"],
                "output_files": back_files,
            }
        )

    return create_tool_output(
        tool_name="mace_relax_batch",
        success=True,
        data={
            "task_states": result.task_states,
            "submission_dir": result.submission_dir,
            "work_base": result.work_base,
            "outputs": outputs,
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
