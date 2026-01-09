from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Any, Dict

from catmaster.tools.base import create_tool_output, resolve_workspace_path
from catmaster.tools.base import workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    DispatchRequest,
    dispatch_task,
    dispatch_submission,
    TaskSpec,
    BatchDispatchRequest,
    make_work_base,
)
from catmaster.tools.execution.resource_router import ResourceRouter, Route
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.execution.task_payloads import render_task_fields
import shutil
import shutil
from pydantic import BaseModel, Field


class VaspExecuteInput(BaseModel):
    """Submit a single VASP run."""

    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    check_interval: int = Field(30, description="Polling interval seconds")


class VaspExecuteBatchInput(BaseModel):
    """Submit multiple VASP runs in one DPDispatcher submission. Preferred tool for batch VASP runs."""

    input_dirs: list[str] = Field(..., description="List of VASP input directories.")
    check_interval: int = Field(30, description="Polling interval seconds")


_INCAR_KEY_RE = re.compile(r"(?i)^\s*(NCORE|NPAR)\b")


def _maybe_autoset_ncore(input_dir: Path, *, resources_key: str) -> Dict[str, Any]:
    """
    If INCAR does not specify NCORE/NPAR, append `NCORE` chosen by:
    1) target ~ sqrt(cpu_per_node)
    2) NCORE must divide cpu_per_node

    Returns a small dict for logging/debugging.
    """
    reg = MachineRegister()
    res_cfg = reg.get_resources(resources_key)
    cpu_per_node = res_cfg.get("cpu_per_node")

    try:
        cpu_per_node_i = int(cpu_per_node)
    except Exception:
        return {"patched": False, "reason": "cpu_per_node_missing_or_invalid", "cpu_per_node": cpu_per_node}

    target = math.sqrt(cpu_per_node_i)
    factors = [f for f in range(1, cpu_per_node_i + 1) if cpu_per_node_i % f == 0]

    # choose the smallest factor >= target; if none, fallback to largest factor
    higher_or_equal = [f for f in factors if f >= target]
    if higher_or_equal:
        ncore = min(higher_or_equal)
    else:
        ncore = max(factors)

    incar = input_dir / "INCAR"
    if not incar.exists():
        return {"patched": False, "reason": "incar_missing", "ncore": ncore}

    lines = incar.read_text(errors="ignore").splitlines()

    # If user already set NCORE/NPAR, do nothing (respect explicit intent)
    for ln in lines:
        # strip inline comments commonly used in INCAR
        head = ln.split("!")[0].split("#")[0].strip()
        if not head:
            continue
        if _INCAR_KEY_RE.match(head):
            return {"patched": False, "reason": "already_set", "ncore": ncore}

    lines.append(f"NCORE = {ncore}")
    incar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "patched": True,
        "reason": "autoset",
        "ncore": ncore,
        "cpu_per_node": cpu_per_node_i,
        "target": target,
    }


def vasp_execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = VaspExecuteInput(**payload)
    router = ResourceRouter()
    route = router.route("vasp_execute")

    # --- NCORE hack lives here (resource-aware, LLM-agnostic) ---
    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    ncore_info = _maybe_autoset_ncore(input_dir, resources_key=route.resources)

    dispatch_req = _build_vasp_execute_request(params, route=route)
    result = dispatch_task(dispatch_req)

    return create_tool_output(
        tool_name="vasp_execute",
        success=True,
        data={
            "task_states": result.task_states,
            "download_path": result.output_dir,
            "submission_dir": result.submission_dir,
            "work_base": result.work_base,
            #"ncore_autoset": ncore_info,
        },
        execution_time=result.duration_s,
    )

def vasp_execute_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = VaspExecuteBatchInput(**payload)
    router = ResourceRouter()
    route = router.route("vasp_execute")

    input_dirs = [resolve_workspace_path(p, must_exist=True) for p in params.input_dirs]
    work_base = make_work_base("vasp_batch")
    parents = {d.parent for d in input_dirs}
    local_root = parents.pop() if len(parents) == 1 else input_dirs[0].parent

    reg = TaskRegistry()
    cfg = reg.get("vasp_execute")

    tasks: list[TaskSpec] = []
    task_meta = []
    for inp in input_dirs:
        stage_dir = local_root / work_base / inp.stem
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        shutil.copytree(inp, stage_dir)

        _maybe_autoset_ncore(stage_dir, resources_key=route.resources)

        rendered = render_task_fields(cfg, {}, stage_dir)
        tasks.append(
            TaskSpec(
                command=rendered["command"],
                task_work_path=stage_dir.relative_to(local_root / work_base).as_posix(),
                forward_files=rendered["forward_files"],
                backward_files=rendered["backward_files"],
            )
        )
        task_meta.append(
            {
                "input_dir": inp,
                "stage_dir": stage_dir,
                "backward_files": rendered["backward_files"],
            }
        )

    batch_req = BatchDispatchRequest(
        machine=route.machine,
        resources=route.resources,
        work_base=work_base,
        local_root=str(local_root),
        tasks=tasks,
        forward_common_files=[],
        backward_common_files=[],
        clean_remote=False,
        check_interval=params.check_interval,
    )

    result = dispatch_submission(batch_req)

    outputs = []
    for meta in task_meta:
        inp = meta["input_dir"]
        stage = meta["stage_dir"]
        for bf in meta["backward_files"]:
            for p in stage.glob(bf):
                dest = inp / p.name
                dest.write_bytes(p.read_bytes())
        outputs.append(
            {
                "input_dir_rel": workspace_relpath(inp),
                "output_files": meta["backward_files"],
            }
        )

    return create_tool_output(
        tool_name="vasp_execute_batch",
        success=True,
        data={
            "task_states": result.task_states,
            "submission_dir": result.submission_dir,
            "work_base": result.work_base,
            "outputs": outputs,
        },
        execution_time=result.duration_s,
    )

def _build_vasp_execute_request(params: Any, *, route: Route, registry: TaskRegistry | None = None) -> DispatchRequest:
    reg = registry or TaskRegistry()
    cfg = reg.get("vasp_execute")

    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    work_base = (input_dir.name if input_dir.name not in ("", ".") else make_work_base("vasp"))
    local_root = input_dir.parent

    rendered = render_task_fields(cfg, {}, input_dir)

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
    "VaspExecuteInput",
    "VaspExecuteBatchInput",
    "vasp_execute",
    "vasp_execute_batch",
]
