from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path
from catmaster.tools.base import workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    STATUS_FILE_NAME,
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

BATCH_STATE_FILENAME = "_BATCH_STATE.json"


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
    details: list[str] = [str(message).strip()]
    if isinstance(data, dict):
        for key in ("input_root_rel", "output_root_rel", "batch_state_rel", "work_base"):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            details.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(details),
        artifact={"tool_name": tool_name, "data": data or {}, "warnings": warnings or []},
        error_code=error_code,
    )


class VaspExecuteInput(BaseModel):
    """Deprecated: single VASP run. Use vasp_execute_batch with input/output roots."""

    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    check_interval: int = Field(30, description="Polling interval seconds")


class VaspExecuteBatchInput(BaseModel):
    """[vasp/execute] Submit VASP runs in one DPDispatcher submission (single-folder or recursive batch modes)."""

    input_dir: str = Field(
        ...,
        description=(
            "Input directory for VASP execution. Two modes are supported: "
            "(1) single-folder mode: input_dir itself is a VASP calc folder containing INCAR and POTCAR; "
            "(2) batch mode: recursively discover subfolders containing INCAR and POTCAR. "
            "Nested mode is forbidden when input_dir itself is a calc folder and descendants also contain calc folders."
        ),
    )
    output_dir: str = Field(
        ...,
        description=(
            "Root directory to store outputs. In single-folder mode, results go to output_dir/<basename(input_dir)>. "
            "In batch mode, results mirror discovered subfolder layout under output_dir. "
            "Must not be inside input_dir. Staging directories under output_dir are cleaned after completion."
        ),
    )
    check_interval: int = Field(30, description="Polling interval seconds")
    task_name: str = Field(
        "vasp_execute",
        description=(
            "DPDispatcher task template name. Use `vasp_execute` for general VASP runs or a dedicated preset such as "
            "`vasp_execute_neb` when a workflow should use separate submission resources/config."
        ),
    )


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


def _paths_overlap(a: Path, b: Path) -> bool:
    return _is_within(a, b) or _is_within(b, a)


def _collect_vasp_input_dirs(root: Path, *, exclude_root: Path | None = None) -> list[Path]:
    input_dirs: list[Path] = []
    skip_prefixes = ("vasp_batch_", "mace_batch_")
    internal_dirs = {"metadata", ".catmaster"}
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if exclude_root is not None and _is_within(path, exclude_root):
            dirnames[:] = []
            continue
        rel = path.resolve().relative_to(root.resolve())
        if any(part.startswith(skip_prefixes) for part in rel.parts):
            dirnames[:] = []
            continue
        if any(part in internal_dirs for part in path.parts):
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in internal_dirs and not d.startswith(skip_prefixes)
        ]
        if _is_vasp_input_dir(path):
            if path != root:
                input_dirs.append(path)
            dirnames[:] = []
    return sorted(input_dirs, key=lambda p: str(p))


def _has_nested_vasp_input_dirs(root: Path) -> bool:
    for dirpath, _, _ in os.walk(root):
        path = Path(dirpath)
        if path == root:
            continue
        if _is_vasp_input_dir(path):
            return True
    return False


def _write_batch_state(
    output_root: Path,
    *,
    work_base: str,
    state: str,
    details: Dict[str, Any] | None = None,
) -> Path:
    payload: Dict[str, Any] = {
        "work_base": work_base,
        "state": state,
        "timestamp": float(time.time()),
    }
    if details:
        payload["details"] = details
    state_path = output_root / BATCH_STATE_FILENAME
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return state_path


def _collect_vasp_outputs(task_meta: List[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[str]]:
    outputs: list[Dict[str, Any]] = []
    warnings: list[str] = []
    for meta in task_meta:
        inp = meta["input_dir"]
        stage = meta["stage_dir"]
        output_dir = meta["output_dir"]
        backward_files = list(meta.get("backward_files") or [])
        try:
            if output_dir.exists():
                shutil.copytree(inp, output_dir, dirs_exist_ok=True)
            else:
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(inp, output_dir)
            for bf in backward_files:
                for p in stage.glob(bf):
                    dest = output_dir / p.name
                    if p.is_dir():
                        shutil.copytree(p, dest, dirs_exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dest)
            outputs.append(
                {
                    "input_dir_rel": workspace_relpath(inp),
                    "output_dir_rel": workspace_relpath(output_dir),
                    "output_files": backward_files,
                }
            )
        except Exception as exc:
            warnings.append(f"collect failed for {workspace_relpath(inp)}: {type(exc).__name__}: {exc}")
    return outputs, warnings


def vasp_execute(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _ = VaspExecuteInput(**payload)
    _fail(
        "vasp_execute",
        message="Single-folder VASP relaxation is deprecated. Use vasp_execute_batch with input_root/output_root.",
        error_code="deprecated_single_mode",
    )


def vasp_execute_batch(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[vasp/execute] Submit one VASP calc folder or a discovered batch of calc folders via DPDispatcher."""
    params = VaspExecuteBatchInput(**payload)
    reg = TaskRegistry()
    task_name = str(params.task_name or "vasp_execute").strip() or "vasp_execute"
    cfg = reg.get(task_name)
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError(f"{task_name} missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        _fail(
            "vasp_execute_batch",
            message=f"input_dir is not a directory: {input_root}",
            error_code="invalid_input_dir",
        )
    single_folder_mode = _is_vasp_input_dir(input_root)
    output_root = resolve_workspace_path(params.output_dir)
    if output_root.exists() and not output_root.is_dir():
        _fail(
            "vasp_execute_batch",
            message=f"output_dir is not a directory: {output_root}",
            error_code="invalid_output_dir",
        )
    if _is_within(output_root, input_root):
        _fail(
            "vasp_execute_batch",
            message="output_dir must not be inside input_dir to avoid mixing inputs with outputs.",
            error_code="output_inside_input",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    exclude_root = None
    nested_calc_dirs = _collect_vasp_input_dirs(input_root, exclude_root=exclude_root)
    if single_folder_mode:
        if _has_nested_vasp_input_dirs(input_root):
            _fail(
                "vasp_execute_batch",
                message=(
                    "Nested calc folders are not allowed: input_dir is already a VASP calc folder "
                    "and descendant calc folders were also found. Split inputs before submission."
                ),
                error_code="nested_calc_forbidden",
            )
        input_dirs = [input_root]
    else:
        input_dirs = nested_calc_dirs
    if not input_dirs:
        _fail(
            "vasp_execute_batch",
            message="No VASP calc folders found (expected directories containing both POTCAR and INCAR).",
            error_code="no_calc_dirs",
        )
    nested_examples: list[str] = []
    for inp in input_dirs:
        if not _has_nested_vasp_input_dirs(inp):
            continue
        nested_examples.append(workspace_relpath(inp))
        if len(nested_examples) >= 3:
            break
    if nested_examples:
        examples = "; ".join(nested_examples)
        _fail(
            "vasp_execute_batch",
            message=(
                "Nested calc folders are not allowed inside submitted calc directories. "
                f"Examples: {examples}. Split or relocate descendant calc folders before submission."
            ),
            data={
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "nested_calc_dirs": nested_examples,
            },
            error_code="nested_calc_forbidden",
        )

    overlap_examples: list[str] = []
    overlap_pairs: list[dict[str, str]] = []
    for inp in input_dirs:
        rel_path = Path(inp.name) if single_folder_mode else inp.relative_to(input_root)
        target_output_dir = output_root / rel_path
        if not _paths_overlap(target_output_dir, inp):
            continue
        in_rel = workspace_relpath(inp)
        out_rel = workspace_relpath(target_output_dir)
        overlap_pairs.append({"input_dir_rel": in_rel, "output_dir_rel": out_rel})
        if len(overlap_examples) < 3:
            overlap_examples.append(f"{in_rel} -> {out_rel}")
    if overlap_pairs:
        examples = "; ".join(overlap_examples)
        _fail(
            "vasp_execute_batch",
            message=(
                "output_dir mapping overlaps input calc directories (in-place collect conflict). "
                f"Examples: {examples}. Choose a separate output_dir."
            ),
            data={
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "overlaps": overlap_pairs,
            },
            error_code="output_overlaps_input",
        )

    work_base = make_work_base("vasp_batch")
    local_root = output_root

    tasks: list[TaskSpec] = []
    task_meta = []
    for inp in input_dirs:
        rel_path = Path(inp.name) if single_folder_mode else inp.relative_to(input_root)
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

        rendered = render_task_fields(cfg, {"task_name": task_name}, stage_dir)
        backward_files = list(rendered["backward_files"])
        if "*" not in backward_files and STATUS_FILE_NAME not in backward_files:
            backward_files.append(STATUS_FILE_NAME)
        tasks.append(
            TaskSpec(
                command=rendered["command"],
                task_work_path=rel_path.as_posix(),
                forward_files=rendered["forward_files"],
                backward_files=backward_files,
            )
        )
        task_meta.append(
            {
                "input_dir": inp,
                "stage_dir": stage_dir,
                "output_dir": output_root / rel_path,
                "backward_files": backward_files,
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

    dispatch_error: Exception | None = None
    result = None
    outputs: list[Dict[str, Any]] = []
    collect_warnings: list[str] = []
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": len(tasks)})
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": len(tasks)})
        result = dispatch_submission(batch_req)
    except Exception as exc:
        dispatch_error = exc
    finally:
        outputs, collect_warnings = _collect_vasp_outputs(task_meta)
        final_state = "collected_partial" if dispatch_error else "collected_complete"
        _write_batch_state(
            output_root,
            work_base=work_base,
            state=final_state,
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
            collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")

    if dispatch_error is not None:
        _fail(
            "vasp_execute_batch",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            warnings=collect_warnings,
            data={
                "work_base": work_base,
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "outputs": outputs,
                "batch_state_rel": workspace_relpath(state_path),
            },
            error_code="dispatch_failed",
        )

    data = {
        "task_states": result.task_states if result else [],
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        "work_base": result.work_base if result else work_base,
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "outputs": outputs,
        "batch_state_rel": workspace_relpath(state_path),
        "task_name": task_name,
        "resources_key": resources_key,
    }
    first_output = outputs[0]["output_dir_rel"] if outputs else ""
    lines = [
        "vasp_execute_batch completed.",
        f"task_name={task_name} calc_dirs={len(input_dirs)} outputs_collected={len(outputs)} task_states={len(data['task_states'])}",
        f"output_root_rel={data['output_root_rel']} batch_state_rel={data['batch_state_rel']}",
    ]
    if data["submission_dir"]:
        lines.append(f"submission_dir={data['submission_dir']}")
    if first_output:
        lines.append(f"first_output_dir={first_output}")
    content = "\n".join(lines)
    return _success(
        "vasp_execute_batch",
        content=content,
        warnings=collect_warnings,
        data=data,
        execution_time=result.duration_s if result else None,
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
