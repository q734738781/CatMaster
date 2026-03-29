from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    BatchDispatchRequest,
    TaskSpec,
    dispatch_submission,
    make_work_base,
)
from catmaster.tools.execution.mace_dispatch import (
    _collect_mace_outputs,
    _is_likely_local_model_path,
    _resolve_machine_for_resources,
    _resolve_mace_head,
    _select_model_file_from_dir,
    _stage_mace_model,
    _write_batch_state,
)
from catmaster.tools.execution.task_payloads import render_task_fields
from catmaster.tools.execution.task_registry import TaskRegistry

_IMAGE_FILE_RE = re.compile(r"^(?P<idx>\d{2})(?P<ext>\.(vasp|poscar|cif))$", re.IGNORECASE)


class MaceNebBatchInput(BaseModel):
    """[mace/execute] Submit MACE NEB jobs through DPDispatcher from explicit image-tree task directories."""

    input_root: str = Field(
        ...,
        description=(
            "Task root for MACE NEB image trees. Two modes are supported: "
            "(1) single-task mode: input_root itself contains numbered image files such as 00.vasp, 01.vasp, ...; "
            "(2) batch mode: input_root contains task subdirectories, each with numbered image files. "
            "Nested task directories deeper than one level are forbidden."
        ),
    )
    output_root: str = Field(
        ...,
        description=(
            "Output root for NEB results. In single-task mode, results go to output_root/<basename(input_root)>. "
            "In batch mode, results mirror task directory names under output_root."
        ),
    )
    fmax: float = Field(0.05, gt=0, description="NEB optimizer force threshold in eV/Angstrom.")
    steps: int = Field(300, ge=1, description="Maximum FIRE optimization steps.")
    climb: bool = Field(
        False,
        description="Enable climbing-image NEB. Defaults to plain NEB for coarse convergence; set true for CI-NEB refinement.",
    )
    model: str = Field(
        "mh-1",
        description=(
            "MACE model identifier or workspace-local trained-model path. "
            "Local paths may point to one model file or a directory containing a unique preferred model artifact."
        ),
        examples=["mh-1", "medium-mpa-0", "models/best.model"],
    )
    head: Optional[str] = Field("omat_pbe", description="Model head for multi-head models; use empty string to disable.")
    dispersion: bool = Field(False, description="Enable dispersion correction when supported by the underlying calculator.")
    overwrite: bool = Field(False, description="If true, overwrite existing per-task output directories.")


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
        for key in ("input_root_rel", "output_root_rel", "batch_summary_rel", "batch_state_rel", "work_base"):
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


def _parse_image_file(path: Path) -> int | None:
    match = _IMAGE_FILE_RE.match(path.name)
    if not match:
        return None
    return int(match.group("idx"))


def _list_task_image_files(task_dir: Path) -> list[Path]:
    subdirs = sorted(path for path in task_dir.iterdir() if path.is_dir())
    if subdirs:
        raise ValueError(
            f"Nested directories are forbidden inside MACE NEB task directories: {workspace_relpath(task_dir)}"
        )
    indexed: list[tuple[int, Path]] = []
    for path in sorted(task_dir.iterdir()):
        if not path.is_file():
            continue
        idx = _parse_image_file(path)
        if idx is None:
            continue
        indexed.append((idx, path))
    if len(indexed) < 2:
        raise ValueError(
            f"MACE NEB task directory must contain numbered image files like 00.vasp, 01.vasp, ...: {workspace_relpath(task_dir)}"
        )
    expected = list(range(len(indexed)))
    found = [idx for idx, _ in indexed]
    if found != expected:
        raise ValueError(
            f"MACE NEB image numbering must be contiguous from 00. Found {found} in {workspace_relpath(task_dir)}"
        )
    return [path for _, path in indexed]


def _is_task_dir(path: Path) -> bool:
    try:
        _list_task_image_files(path)
    except Exception:
        return False
    return True


def _discover_task_dirs(input_root: Path) -> tuple[list[Path], bool]:
    if _is_task_dir(input_root):
        return [input_root], True
    child_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not child_dirs:
        raise ValueError(
            "input_root is neither a MACE NEB task directory nor a directory of task subdirectories."
        )
    stray_files = sorted(path.name for path in input_root.iterdir() if path.is_file())
    if stray_files:
        raise ValueError(
            f"Batch MACE NEB input_root must contain only task directories. Unexpected files: {', '.join(stray_files[:5])}"
        )
    task_dirs: list[Path] = []
    for child in child_dirs:
        if not _is_task_dir(child):
            raise ValueError(
                f"Each child of batch input_root must be a task directory with flat numbered image files: {workspace_relpath(child)}"
            )
        task_dirs.append(child)
    return task_dirs, False


def _resolve_local_model(model_value: str) -> tuple[str, str, str]:
    requested = str(model_value or "").strip()
    if not requested:
        raise ValueError("model is required")

    local_path: Path | None = None
    try:
        local_path = resolve_workspace_path(requested, must_exist=True)
    except (FileNotFoundError, ValueError):
        local_path = None

    if local_path is None:
        if _is_likely_local_model_path(requested):
            raise FileNotFoundError(f"Local MACE model path does not exist inside project files root: {requested}")
        return requested, "pretrained", requested

    if local_path.is_file():
        return str(local_path.resolve()), "local_file", workspace_relpath(local_path)
    if local_path.is_dir():
        selected = _select_model_file_from_dir(local_path)
        return str(selected.resolve()), "local_dir", workspace_relpath(local_path)
    raise ValueError(f"Unsupported model path type: {local_path}")


def mace_neb_batch(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[mace/execute] Submit a MACE NEB batch through DPDispatcher."""
    params = MaceNebBatchInput(**payload)
    reg = TaskRegistry()
    cfg = reg.get("mace_neb_dir")
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError("mace_neb_dir missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)
    head = _resolve_mace_head(params.head)

    input_root = resolve_workspace_path(params.input_root, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    if not input_root.is_dir():
        _fail("mace_neb_batch", message=f"input_root is not a directory: {input_root}", error_code="invalid_input_root")
    if output_root.exists() and not output_root.is_dir():
        _fail("mace_neb_batch", message=f"output_root is not a directory: {output_root}", error_code="invalid_output_root")
    try:
        output_root.resolve().relative_to(input_root.resolve())
        _fail("mace_neb_batch", message="output_root must not be inside input_root.", error_code="output_inside_input")
    except ValueError:
        pass
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        task_dirs, single_task_mode = _discover_task_dirs(input_root)
    except Exception as exc:
        _fail(
            "mace_neb_batch",
            message=f"Invalid MACE NEB input layout: {exc}",
            data={"input_root_rel": workspace_relpath(input_root), "output_root_rel": workspace_relpath(output_root)},
            error_code="invalid_task_layout",
        )

    existing_output_dirs = [output_root / task_dir.name for task_dir in task_dirs]
    conflicts = [path for path in existing_output_dirs if path.exists()]
    if conflicts and not params.overwrite:
        _fail(
            "mace_neb_batch",
            message=f"output task directory already exists: {workspace_relpath(conflicts[0])}. Set overwrite=true to replace.",
            data={
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
            },
            error_code="output_exists",
        )
    for path in conflicts:
        shutil.rmtree(path)

    work_base = make_work_base("mace_neb_batch")
    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_input = stage_root / "input"
    stage_output = stage_root / "output"
    stage_output.mkdir(parents=True, exist_ok=True)
    if single_task_mode:
        shutil.copytree(input_root, stage_input / input_root.name)
    else:
        shutil.copytree(input_root, stage_input)
    script_src = Path(__file__).resolve().parents[2] / "remote" / "gpu" / "mace_neb.py"
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing MACE NEB remote script: {script_src}")
    script_dst = stage_root / "task_script" / "mace_neb.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_src, script_dst)

    try:
        model_spec = _stage_mace_model(params.model, stage_root=stage_root)
    except Exception as exc:
        _fail(
            "mace_neb_batch",
            message=f"Invalid model specification {params.model!r}: {exc}",
            data={"input_root_rel": workspace_relpath(input_root), "output_root_rel": workspace_relpath(output_root)},
            error_code="invalid_model",
        )

    ctx = {
        "input_path": "input",
        "output_root": "output",
        "fmax": params.fmax,
        "steps": params.steps,
        "climb": "true" if params.climb else "false",
        "model": shlex.quote(model_spec.command_arg),
        "head": shlex.quote(head or ""),
        "dispersion": "true" if params.dispersion else "false",
    }
    rendered = render_task_fields(cfg, ctx, stage_root)
    if model_spec.asset_dir_rel and model_spec.asset_dir_rel not in rendered["forward_files"]:
        rendered["forward_files"].append(model_spec.asset_dir_rel)

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
        check_interval=30,
    )

    dispatch_error: Exception | None = None
    result = None
    collect_info: Dict[str, Any] = {}
    collect_warnings: list[str] = []
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": len(task_dirs)})
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": len(task_dirs)})
        result = dispatch_submission(batch_req)
    except Exception as exc:
        dispatch_error = exc
    finally:
        collect_info, collect_warnings = _collect_mace_outputs(stage_root, stage_output, output_root)
        final_state = "collected_partial" if dispatch_error else "collected_complete"
        _write_batch_state(
            output_root,
            work_base=work_base,
            state=final_state,
            details={"tasks": len(task_dirs), "errors": [str(dispatch_error)] if dispatch_error else []},
        )
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except Exception as exc:
                collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")

    if dispatch_error is not None:
        _fail(
            "mace_neb_batch",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            warnings=collect_warnings,
            data={
                "work_base": work_base,
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                **collect_info,
            },
            error_code="dispatch_failed",
        )

    data = {
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "batch_summary_rel": collect_info.get("batch_summary_rel"),
        "task_count": len(task_dirs),
        "single_task_mode": single_task_mode,
        "model": model_spec.requested,
        "model_command_arg": model_spec.command_arg,
        "model_source_kind": model_spec.source_kind,
        "model_source_rel": model_spec.source_rel,
        "model_asset_rel": model_spec.asset_model_rel,
        "head": head,
        "climb": params.climb,
        "fmax": params.fmax,
        "steps": params.steps,
        **collect_info,
    }
    lines = [
        "mace_neb_batch completed.",
        f"task_count={len(task_dirs)} task_states={len(result.task_states) if result else 0}",
        f"output_root_rel={data['output_root_rel']}",
        f"batch_state_rel={data['batch_state_rel']}",
    ]
    if data.get("batch_summary_rel"):
        lines.append(f"batch_summary_rel={data['batch_summary_rel']}")
    return "\n".join(lines), {
        "tool_name": "mace_neb_batch",
        "data": data,
        "warnings": collect_warnings,
        "execution_time": result.duration_s if result else None,
    }


__all__ = ["MaceNebBatchInput", "mace_neb_batch", "_resolve_local_model"]
