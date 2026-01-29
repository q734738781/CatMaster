from __future__ import annotations

import re
import math
import os
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
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.execution.task_payloads import render_task_fields
import shutil
from pydantic import BaseModel, Field


class VaspExecuteInput(BaseModel):
    """Submit a single VASP run."""

    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    resources: str = Field("vasp_cpu", description="DPDispatcher resources preset key")
    machine: str | None = Field(None, description="Override machine key; default derived from resources preset")
    check_interval: int = Field(30, description="Polling interval seconds")


class VaspExecuteBatchInput(BaseModel):
    """Submit multiple VASP runs in one DPDispatcher submission. Preferred tool for batch VASP runs."""

    input_dir: str = Field(
        ...,
        description="Root directory containing VASP input subdirectories (each with INCAR/KPOINTS/POSCAR/POTCAR). e.g. input_dir/task01/INCAR...,input_dir/task02/INCAR... ",
    )
    output_dir: str = Field(
        ...,
        description=(
            "Root directory to store batch outputs. Results are written to output_dir using the same subdirectory "
            "layout as input_dir. Must not be inside input_dir. Staging directories under output_dir are cleaned "
            "after completion."
        ),
    )
    resources: str = Field("vasp_cpu", description="DPDispatcher resources preset key")
    machine: str | None = Field(None, description="Override machine key; default derived from resources preset")
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


def _resolve_machine_for_resources(resources_key: str, *, machine: str | None = None) -> str:
    if machine:
        return machine
    reg = MachineRegister()
    res_cfg = reg.get_resources(resources_key)
    resolved = res_cfg.get("machine")
    if not resolved:
        raise KeyError(f"Resources '{resources_key}' missing machine binding")
    return str(resolved)


def _is_vasp_input_dir(path: Path) -> bool:
    required = ("POTCAR", "INCAR")
    return path.is_dir() and all((path / name).is_file() for name in required)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _collect_vasp_input_dirs(root: Path, *, exclude_root: Path | None = None) -> list[Path]:
    input_dirs: list[Path] = []
    skip_prefixes = ("vasp_batch_", "mace_batch_")
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if exclude_root is not None and _is_within(path, exclude_root):
            dirnames[:] = []
            continue
        rel = path.resolve().relative_to(root.resolve())
        if any(part.startswith(skip_prefixes) for part in rel.parts):
            dirnames[:] = []
            continue
        if ".catmaster" in path.parts:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d != ".catmaster" and not d.startswith(skip_prefixes)
        ]
        if _is_vasp_input_dir(path):
            if path != root:
                input_dirs.append(path)
            dirnames[:] = []
    return sorted(input_dirs, key=lambda p: str(p))


def vasp_execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = VaspExecuteInput(**payload)
    resources_key = params.resources
    machine = _resolve_machine_for_resources(resources_key, machine=params.machine)

    # --- NCORE hack lives here (resource-aware, LLM-agnostic) ---
    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    ncore_info = _maybe_autoset_ncore(input_dir, resources_key=resources_key)

    dispatch_req = _build_vasp_execute_request(params, machine=machine, resources=resources_key)
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
    resources_key = params.resources
    machine = _resolve_machine_for_resources(resources_key, machine=params.machine)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        return create_tool_output(
            "vasp_execute_batch",
            success=False,
            error=f"input_dir is not a directory: {input_root}",
        )
    output_root = resolve_workspace_path(params.output_dir)
    if output_root.exists() and not output_root.is_dir():
        return create_tool_output(
            "vasp_execute_batch",
            success=False,
            error=f"output_dir is not a directory: {output_root}",
        )
    if _is_within(output_root, input_root):
        return create_tool_output(
            "vasp_execute_batch",
            success=False,
            error="output_dir must not be inside input_dir to avoid mixing inputs with outputs.",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    exclude_root = None
    input_dirs = _collect_vasp_input_dirs(input_root, exclude_root=exclude_root)
    if not input_dirs:
        return create_tool_output(
            "vasp_execute_batch",
            success=False,
            error="No VASP input subdirectories found (expected subdirs containing POTCAR and INCAR).",
        )
    work_base = make_work_base("vasp_batch")
    local_root = output_root

    reg = TaskRegistry()
    cfg = reg.get("vasp_execute")

    tasks: list[TaskSpec] = []
    task_meta = []
    for inp in input_dirs:
        rel_path = inp.relative_to(input_root)
        stage_dir = local_root / work_base / rel_path
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(inp, stage_dir)

        _maybe_autoset_ncore(stage_dir, resources_key=resources_key)

        rendered = render_task_fields(cfg, {}, stage_dir)
        tasks.append(
            TaskSpec(
                command=rendered["command"],
                task_work_path=rel_path.as_posix(),
                forward_files=rendered["forward_files"],
                backward_files=rendered["backward_files"],
            )
        )
        task_meta.append(
            {
                "input_dir": inp,
                "stage_dir": stage_dir,
                "output_dir": output_root / rel_path,
                "backward_files": rendered["backward_files"],
            }
        )

    batch_req = BatchDispatchRequest(
        machine=machine,
        resources=resources_key,
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
        output_dir = meta["output_dir"]
        if output_dir.exists():
            shutil.copytree(inp, output_dir, dirs_exist_ok=True)
        else:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(inp, output_dir)
        for bf in meta["backward_files"]:
            for p in stage.glob(bf):
                dest = output_dir / p.name
                if p.is_dir():
                    shutil.copytree(p, dest, dirs_exist_ok=True)
                else:
                    dest.write_bytes(p.read_bytes())
        outputs.append(
            {
                "input_dir_rel": workspace_relpath(inp),
                "output_dir_rel": workspace_relpath(output_dir),
                "output_files": meta["backward_files"],
            }
        )

    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)

    return create_tool_output(
        tool_name="vasp_execute_batch",
        success=True,
        data={
            "task_states": result.task_states,
            "submission_dir": result.submission_dir,
            "work_base": result.work_base,
            "input_root_rel": workspace_relpath(input_root),
            "output_root_rel": workspace_relpath(output_root),
            "outputs": outputs,
        },
        execution_time=result.duration_s,
    )

def _build_vasp_execute_request(
    params: Any,
    *,
    machine: str,
    resources: str,
    registry: TaskRegistry | None = None,
) -> DispatchRequest:
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
    "VaspExecuteInput",
    "VaspExecuteBatchInput",
    "vasp_execute",
    "vasp_execute_batch",
]
