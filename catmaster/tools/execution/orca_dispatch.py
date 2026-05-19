from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    STATUS_FILE_NAME,
    BatchDispatchRequest,
    TaskSpec,
    dispatch_submission,
    make_work_base,
    remote_context_from_exception,
    remote_context_from_result,
)
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_payloads import render_task_fields
from catmaster.tools.execution.task_registry import TaskRegistry

_BATCH_STATE_FILENAME = "_BATCH_STATE.json"


def _success(
    tool_name: str,
    *,
    content: str,
    data: dict[str, Any],
    warnings: list[str] | None = None,
    execution_time: float | None = None,
) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    if execution_time is not None:
        artifact["execution_time"] = execution_time
    return content, artifact


def _fail(
    tool_name: str,
    *,
    message: str,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    error_code: str = "",
) -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
        artifact={"tool_name": tool_name, "data": data or {}, "warnings": warnings or []},
        error_code=error_code,
    )


def _resolve_machine_for_resources(resources_key: str) -> str:
    reg = MachineRegister()
    res_cfg = reg.get_resources(resources_key)
    resolved = res_cfg.get("machine")
    if not resolved:
        raise KeyError(f"Resources '{resources_key}' missing machine binding")
    return str(resolved)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _discover_orca_job_dirs(root: Path) -> list[Path]:
    if root.is_file():
        raise ValueError("orca_execute_batch expects a directory, not a single file.")
    if (root / "job.inp").is_file():
        return [root]
    job_dirs: list[Path] = []
    skip_prefixes = ("orca_batch_", "xtb_batch_", "crest_batch_", "vasp_batch_", "mace_batch_")
    internal_dirs = {"metadata", ".catmaster"}
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part.startswith(skip_prefixes) for part in path.parts):
            dirnames[:] = []
            continue
        if any(part in internal_dirs for part in path.parts):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in internal_dirs and not d.startswith(skip_prefixes)]
        if "job.inp" in filenames:
            job_dirs.append(path)
            dirnames[:] = []
    return sorted(job_dirs, key=lambda p: str(p))


def _write_batch_state(
    output_root: Path,
    *,
    work_base: str,
    state: str,
    details: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "work_base": work_base,
        "state": state,
        "timestamp": float(time.time()),
    }
    if details:
        payload["details"] = details
    state_path = output_root / _BATCH_STATE_FILENAME
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return state_path


def _copy_stage_tree(stage_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(
        stage_dir,
        output_dir,
        ignore=shutil.ignore_patterns("task_script", "__pycache__"),
        dirs_exist_ok=False,
    )


class OrcaExecuteBatchInput(BaseModel):
    """[orca/execute] Submit one prepared ORCA job directory or a prepared batch root via DPDispatcher."""

    input_dir: str = Field(..., description="Prepared ORCA job directory or a directory containing multiple prepared ORCA jobs.")
    output_root: str = Field(..., description="Output root where collected ORCA result folders will be written.")
    task_name: str = Field("orca_execute", description="DPDispatcher task template name.")
    check_interval: int = Field(30, ge=1, description="Polling interval seconds.")


def orca_execute_batch(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = OrcaExecuteBatchInput(**payload)
    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_dir.is_dir():
        _fail("orca_execute_batch", message=f"input_dir is not a directory: {input_dir}", error_code="invalid_input_dir")
    output_root = resolve_workspace_path(params.output_root)
    if _is_within(output_root, input_dir):
        _fail("orca_execute_batch", message="output_root must not be inside input_dir.", error_code="output_inside_input")
    job_dirs = _discover_orca_job_dirs(input_dir)
    if not job_dirs:
        _fail("orca_execute_batch", message="No ORCA job directories found (expected job.inp).", error_code="no_jobs")

    registry = TaskRegistry()
    cfg = registry.get(params.task_name)
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError(f"{params.task_name} missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)

    output_root.mkdir(parents=True, exist_ok=True)
    work_base = make_work_base("orca_batch")
    local_root = output_root
    script_src = Path(__file__).resolve().parents[2] / "remote" / "cpu" / "orca_boot.py"
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing ORCA boot script: {script_src}")

    tasks: list[TaskSpec] = []
    task_meta: list[dict[str, Any]] = []
    single_mode = (input_dir / "job.inp").is_file()
    for job_dir in job_dirs:
        rel_name = Path(job_dir.name) if single_mode else job_dir.relative_to(input_dir)
        stage_dir = local_root / work_base / rel_name
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        shutil.copytree(job_dir, stage_dir)
        script_dst = stage_dir / "task_script" / "orca_boot.py"
        script_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_src, script_dst)
        rendered = render_task_fields(cfg, {}, stage_dir)
        backward_files = list(rendered["backward_files"])
        if "*" not in backward_files and STATUS_FILE_NAME not in backward_files:
            backward_files.append(STATUS_FILE_NAME)
        tasks.append(
            TaskSpec(
                command=rendered["command"],
                task_work_path=rel_name.as_posix(),
                forward_files=rendered["forward_files"],
                backward_files=backward_files,
            )
        )
        task_meta.append(
            {
                "stage_dir": stage_dir,
                "output_dir": output_root / rel_name,
                "input_dir_rel": workspace_relpath(job_dir),
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
        tool_name="orca_execute_batch",
    )

    dispatch_error: Exception | None = None
    result = None
    outputs: list[dict[str, Any]] = []
    warnings: list[str] = []
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": len(tasks)})
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": len(tasks)})
        result = dispatch_submission(batch_req)
    except Exception as exc:
        dispatch_error = exc
    finally:
        for meta in task_meta:
            try:
                _copy_stage_tree(meta["stage_dir"], meta["output_dir"])
                outputs.append(
                    {
                        "input_dir_rel": meta["input_dir_rel"],
                        "output_dir_rel": workspace_relpath(meta["output_dir"]),
                    }
                )
            except Exception as exc:
                warnings.append(f"collect failed for {meta['input_dir_rel']}: {type(exc).__name__}: {exc}")
        _write_batch_state(
            output_root,
            work_base=work_base,
            state="collected_partial" if dispatch_error else "collected_complete",
            details={
                "tasks": len(tasks),
                "outputs_collected": len(outputs),
                "errors": [str(dispatch_error)] if dispatch_error else [],
            },
        )
    stage_root = output_root / work_base
    if stage_root.exists():
        try:
            shutil.rmtree(stage_root)
        except Exception as exc:
            warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")

    if dispatch_error is not None:
        _fail(
            "orca_execute_batch",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            data={
                "input_dir_rel": workspace_relpath(input_dir),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                "outputs": outputs,
                **remote_context_from_exception(dispatch_error),
            },
            warnings=warnings,
            error_code="dispatch_failed",
        )

    data = {
        "task_name": params.task_name,
        "resources_key": resources_key,
        "input_dir_rel": workspace_relpath(input_dir),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "outputs": outputs,
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        "task_states": result.task_states if result else [],
        "work_base": result.work_base if result else work_base,
        **remote_context_from_result(result),
    }
    content = (
        "orca_execute_batch completed.\n"
        f"task_name={params.task_name} jobs={len(job_dirs)} outputs_collected={len(outputs)}\n"
        f"output_root_rel={data['output_root_rel']} batch_state_rel={data['batch_state_rel']}"
    )
    return _success(
        "orca_execute_batch",
        content=content,
        data=data,
        warnings=warnings,
        execution_time=result.duration_s if result else None,
    )


__all__ = ["OrcaExecuteBatchInput", "orca_execute_batch"]
