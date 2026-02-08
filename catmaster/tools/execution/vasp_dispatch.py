from __future__ import annotations

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
    """Deprecated: single VASP run. Use vasp_execute_batch with input/output roots."""

    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    check_interval: int = Field(30, description="Polling interval seconds")


class VaspExecuteBatchInput(BaseModel):
    """Submit multiple VASP runs in one DPDispatcher submission. Preferred tool for batch VASP runs."""

    input_dir: str = Field(
        ...,
        description=(
            "Root directory containing VASP input subdirectories. Any subfolder containing both INCAR and POTCAR "
            "is treated as a VASP calc folder. Subfolders are discovered recursively."
        ),
    )
    output_dir: str = Field(
        ...,
        description=(
            "Root directory to store batch outputs. Results mirror the input subfolder layout under output_dir. "
            "Must not be inside input_dir. Staging directories under output_dir are cleaned after completion."
        ),
    )
    check_interval: int = Field(30, description="Polling interval seconds")


def _resolve_machine_for_resources(resources_key: str) -> str:
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
    _ = VaspExecuteInput(**payload)
    return create_tool_output(
        tool_name="vasp_execute",
        success=False,
        error="Single-folder VASP relaxation is deprecated. Use vasp_execute_batch with input_root/output_root.",
    )


def vasp_execute_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = VaspExecuteBatchInput(**payload)
    reg = TaskRegistry()
    cfg = reg.get("vasp_execute")
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError("vasp_execute missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        return create_tool_output(
            "vasp_execute_batch",
            success=False,
            error=f"input_dir is not a directory: {input_root}",
        )
    if _is_vasp_input_dir(input_root):
        return create_tool_output(
            "vasp_execute_batch",
            success=False,
            error="input_dir is a single VASP calc folder. Provide a root with subfolders for batch execution.",
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

    tasks: list[TaskSpec] = []
    task_meta = []
    for inp in input_dirs:
        rel_path = inp.relative_to(input_root)
        stage_dir = local_root / work_base / rel_path
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(inp, stage_dir)
        script_src = Path(__file__).resolve().parents[2] / "remote" / "cpu" / "vasp_boot.py"
        if not script_src.is_file():
            raise FileNotFoundError(f"Missing VASP boot script: {script_src}")
        script_dst = stage_dir / "task_script" / "vasp_boot.py"
        script_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_src, script_dst)

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
            "submission_dir": workspace_relpath(Path(result.submission_dir)) if result.submission_dir else "",
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
    "vasp_execute_batch",
]
