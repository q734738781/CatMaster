from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import os
import shlex

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    STATUS_FILE_NAME,
    STDOUT_FILE_NAME,
    STDERR_FILE_NAME,
    DispatchRequest,
    dispatch_task,
    dispatch_submission,
    remote_context_from_exception,
    remote_context_from_result,
    TaskSpec,
    BatchDispatchRequest,
    make_work_base,
)
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskConfig, TaskRegistry
from catmaster.tools.execution.task_payloads import render_task_fields
import shutil
from pydantic import BaseModel, Field, field_validator

BATCH_STATE_FILENAME = "_BATCH_STATE.json"
_LOCAL_MODEL_FILE_SUFFIXES = {".model", ".pt", ".pth", ".ckpt"}
_PREFERRED_MODEL_TOKENS = ("best", "final")


def _legacy_mace_task(kind: Literal["relax", "sp", "md"]) -> TaskConfig:
    """Private compatibility card for deprecated Python wrappers.

    These wrappers are not registered agent tools. Keeping their command
    rendering local avoids resurrecting removed provider-named task cards.
    """

    if kind == "relax":
        command = (
            "python task_script/mace_relax.py --input {input} --output_root {output_root} "
            "--fmax {fmax} --steps {steps} --model {model} --head {head} "
            "--dispersion {dispersion} --relax_lattice {relax_lattice} "
            "--default_dtype {default_dtype} --enable_cueq {enable_cueq} --device auto"
        )
        defaults = {
            "input": "input",
            "output_root": "output",
            "fmax": 0.02,
            "steps": 500,
            "model": "mh-1",
            "head": "omat_pbe",
            "dispersion": False,
            "relax_lattice": False,
            "default_dtype": "float64",
            "enable_cueq": False,
        }
        script = "task_script/mace_relax.py"
    elif kind == "sp":
        command = (
            "python task_script/mace_sp.py --input {input} --output_root {output_root} "
            "--model {model} --head {head} --dispersion {dispersion} "
            "--default_dtype {default_dtype} --device auto"
        )
        defaults = {
            "input": "input",
            "output_root": "output",
            "model": "mh-1",
            "head": "omat_pbe",
            "dispersion": False,
            "default_dtype": "float64",
        }
        script = "task_script/mace_sp.py"
    else:
        command = (
            "python task_script/mace_md.py --input {input} --output_root {output_root} "
            "--params {params} --device {device}"
        )
        defaults = {"input": "input", "output_root": "output", "params": "params/md_params.json", "device": "auto"}
        script = "task_script/mace_md.py"
    return TaskConfig(
        command=command,
        resources="mace_gpu",
        defaults=defaults,
        forward_files=["input", "params", script] if kind == "md" else ["input", script],
        backward_files=["output"],
    )


@dataclass(frozen=True)
class _ResolvedMaceModel:
    requested: str
    command_arg: str
    source_kind: Literal["pretrained", "local_file", "local_dir"]
    source_path: Path | None = None
    source_rel: str | None = None
    asset_dir_rel: str | None = None
    asset_model_rel: str | None = None


class MaceRelaxInput(BaseModel):
    """Deprecated: single MACE relaxation. Use mace_relax_batch with input/output roots."""

    structure_file: str = Field(
        ...,
        description="Input periodic structure file with lattice information (supports POSCAR/CONTCAR/.vasp/.poscar/.cif; xyz files are NOT supported).",
    )
    output_root: Optional[str] = Field(
        None,
        description="Output directory for relaxation results. Defaults to the input file's directory.",
    )
    fmax: float = Field(0.02, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
    steps: int = Field(500, ge=1, description="Maximum optimization steps passed to --steps.")
    model: str = Field(
        "mh-1",
        description=(
            "MACE model identifier or workspace-local trained-model path. "
            "Recommended pretrained options: "
            "'mh-1' (slower, higher accuracy) or "
            "'medium-mpa-0' (faster, lower accuracy). "
            "Local paths may point to one model file or to a directory containing a unique preferred model artifact."
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
    """[mace/execute] Submit multiple MACE relaxations in one DPDispatcher submission."""

    input_dir: str = Field(
        ...,
        description="Root directory containing periodic structure files with lattice (POSCAR/CONTCAR/.vasp/.poscar/.cif).",
    )
    output_root: str = Field(
        ...,
        description="Output root for mirrored batch results. Must be outside input_dir.",
    )
    fmax: float = Field(0.02, gt=0, description="Force threshold for relaxation in eV/Angstrom.")
    steps: int = Field(500, ge=1, description="Maximum optimization steps passed to --steps.")
    model: str = Field(
        "mh-1",
        description="MACE model identifier or workspace-local trained-model path.",
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
    default_dtype: Literal["float32", "float64"] = Field(
        "float64",
        description="MACE calculator precision.",
    )
    enable_cueq: bool = Field(
        False,
        description="Enable cuEquivariance acceleration. Requires a CUDA device on the managed MACE resource.",
    )
    relax_lattice: bool = Field(
        False,
        description="Whether to relax lattice/cell together with atomic positions via ASE FrechetCellFilter.",
    )
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


class MaceSPBatchInput(BaseModel):
    """[mace/execute] Run MACE single-point jobs for structures under one directory."""

    input_dir: str = Field(
        ...,
        description="Input directory containing periodic structures (POSCAR/CONTCAR/.vasp/.poscar/.cif).",
    )
    output_root: str = Field(
        ...,
        description="Output root for batch results. Must be outside input_dir.",
    )
    model: str = Field(
        "mh-1",
        description=(
            "MACE model identifier or workspace-local trained-model path "
            "(e.g., mh-1, medium-mpa-0, models/best.model)."
        ),
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
    default_dtype: Literal["float32", "float64"] = Field(
        "float64",
        description="MACE calculator precision.",
    )
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


class MaceMDBatchInput(BaseModel):
    """[mace/execute] Run ASE-backed MACE MD for structures under one directory."""

    input_dir: str = Field(
        ...,
        description="Input directory containing periodic structures or molecules (POSCAR/CONTCAR/.vasp/.poscar/.cif/.xyz).",
    )
    output_root: str = Field(
        ...,
        description="Output root for MD trajectories, logs, final structures, and per-structure summaries. Must be outside input_dir.",
    )
    model: str = Field(
        "mh-1",
        description="MACE model identifier or workspace-local trained-model path.",
        examples=["mh-1", "medium-mpa-0", "models/best.model"],
    )
    head: Optional[str] = Field(
        "omat_pbe",
        description="Model head for multi-head models. Use empty string to disable.",
    )
    dispersion: bool = Field(False, description="Enable dispersion correction.")
    default_dtype: Literal["float32", "float64"] = Field(
        "float32",
        description="Floating-point precision passed to the MACE calculator. MD defaults to float32 for throughput.",
    )
    md_config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form MD config object. Keep it minimal; for example "
            "{'dynamics': {'ensemble': 'nve'}} or "
            "{'dynamics': {'ensemble': 'nvt'}, 'thermostat': {'type': 'langevin'}}. "
            "Optional CUDA acceleration belongs under calculator, for example "
            "{'calculator': {'enable_cueq': true, 'compile_mode': 'reduce-overhead'}}. "
            "Use the MACE MD skill for full ASE parameter templates."
        ),
    )
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")

    @field_validator("md_config")
    @classmethod
    def _validate_md_config_is_object(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("md_config must be an object.")
        return value


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


def _is_likely_local_model_path(raw_model: str) -> bool:
    text = str(raw_model or "").strip()
    if not text:
        return False
    path = Path(text)
    if path.is_absolute() or text.startswith(".") or path.suffix.lower() in _LOCAL_MODEL_FILE_SUFFIXES:
        return True
    if "/" in text or "\\" in text:
        try:
            candidate = resolve_workspace_path(text, must_exist=False)
        except Exception:
            return False
        return candidate.parent.exists()
    return False


def _select_model_file_from_dir(model_dir: Path) -> Path:
    all_files: list[Path] = []
    model_like: list[Path] = []
    preferred: list[Path] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        all_files.append(path)
        if path.suffix.lower() not in _LOCAL_MODEL_FILE_SUFFIXES:
            continue
        model_like.append(path)
        lower_name = path.name.lower()
        if any(token in lower_name for token in _PREFERRED_MODEL_TOKENS):
            preferred.append(path)

    if len(preferred) == 1:
        return preferred[0]
    if len(model_like) == 1:
        return model_like[0]
    if not model_like and len(all_files) == 1:
        return all_files[0]

    candidates = preferred or model_like or all_files
    preview = ", ".join(str(path.relative_to(model_dir)) for path in candidates[:5])
    raise ValueError(
        "model directory must contain exactly one usable model artifact, "
        "or exactly one preferred artifact containing 'best'/'final'. "
        f"Candidates: {preview or '(none)'}"
    )


def _stage_mace_model(model_value: Any, *, stage_root: Path) -> _ResolvedMaceModel:
    requested = str(model_value or "").strip()
    if not requested:
        raise ValueError("model is required.")

    local_path: Path | None = None
    try:
        local_path = resolve_workspace_path(requested, must_exist=True)
    except (FileNotFoundError, ValueError):
        local_path = None

    if local_path is None:
        if _is_likely_local_model_path(requested):
            raise FileNotFoundError(
                f"Local MACE model path does not exist inside project files root: {requested}"
            )
        return _ResolvedMaceModel(
            requested=requested,
            command_arg=requested,
            source_kind="pretrained",
        )

    assets_root = stage_root / "assets" / "models"
    assets_root.mkdir(parents=True, exist_ok=True)

    if local_path.is_file():
        dest = assets_root / local_path.name
        shutil.copy2(local_path, dest)
        return _ResolvedMaceModel(
            requested=requested,
            command_arg=dest.relative_to(stage_root).as_posix(),
            source_kind="local_file",
            source_path=local_path,
            source_rel=workspace_relpath(local_path),
            asset_dir_rel=(stage_root / "assets").relative_to(stage_root).as_posix(),
            asset_model_rel=dest.relative_to(stage_root).as_posix(),
        )

    if not local_path.is_dir():
        raise ValueError(f"Unsupported local model path type: {local_path}")

    selected = _select_model_file_from_dir(local_path)
    dest_root = assets_root / (local_path.name or "model_dir")
    shutil.copytree(local_path, dest_root)
    dest_model = dest_root / selected.relative_to(local_path)
    return _ResolvedMaceModel(
        requested=requested,
        command_arg=dest_model.relative_to(stage_root).as_posix(),
        source_kind="local_dir",
        source_path=local_path,
        source_rel=workspace_relpath(local_path),
        asset_dir_rel=(stage_root / "assets").relative_to(stage_root).as_posix(),
        asset_model_rel=dest_model.relative_to(stage_root).as_posix(),
    )


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
        for key in (
            "input_root_rel",
            "output_root_rel",
            "batch_state_rel",
            "status_file_rel",
            "stdout_file_rel",
            "stderr_file_rel",
            "batch_summary_rel",
            "work_base",
        ):
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


def mace_relax(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _ = MaceRelaxInput(**payload)
    _fail(
        "mace_relax",
        message="Single-file MACE relaxation is deprecated. Use mace_relax_batch with input_root/output_root.",
        error_code="deprecated_single_mode",
    )

def _is_structure_file(path: Path, *, allow_xyz: bool = False) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if name in {"POSCAR", "CONTCAR"}:
        return True
    suffixes = {".vasp", ".poscar", ".cif"}
    if allow_xyz:
        suffixes.add(".xyz")
    return path.suffix.lower() in suffixes


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _collect_structure_files(
    root: Path,
    *,
    exclude_root: Path | None = None,
    allow_xyz: bool = False,
) -> list[Path]:
    files: list[Path] = []
    skip_prefixes = ("mace_batch_", "mace_sp_batch_", "mace_md_batch_", "vasp_batch_")
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
            if _is_structure_file(p, allow_xyz=allow_xyz):
                files.append(p)
    return sorted(files, key=lambda p: str(p))


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


def _collect_mace_outputs(stage_root: Path, stage_output: Path, output_root: Path) -> tuple[Dict[str, Any], list[str]]:
    warnings: list[str] = []
    out: Dict[str, Any] = {}

    try:
        if stage_output.exists():
            shutil.copytree(stage_output, output_root, dirs_exist_ok=True)
    except Exception as exc:
        warnings.append(f"collect output failed: {type(exc).__name__}: {exc}")

    for src_name, field_name in (
        (STATUS_FILE_NAME, "status_file_rel"),
        (STDOUT_FILE_NAME, "stdout_file_rel"),
        (STDERR_FILE_NAME, "stderr_file_rel"),
    ):
        src = stage_root / src_name
        dst = output_root / src_name
        try:
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        except Exception as exc:
            warnings.append(f"collect {src_name} failed: {type(exc).__name__}: {exc}")
        out[field_name] = workspace_relpath(dst) if dst.is_file() else None

    out["batch_summary_rel"] = (
        workspace_relpath(output_root / "batch_summary.json")
        if (output_root / "batch_summary.json").exists()
        else None
    )
    return out, warnings


def mace_relax_batch(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[mace/execute] Submit a directory batch of MACE relaxations through DPDispatcher."""
    params = MaceRelaxBatchInput(**payload)
    cfg = _legacy_mace_task("relax")
    resources_key = "mace_gpu"
    machine = _resolve_machine_for_resources(resources_key)
    head = _resolve_mace_head(params.head)
    head_arg = shlex.quote(head or "")
    dispersion = bool(params.dispersion)
    relax_lattice = bool(params.relax_lattice)
    enable_cueq = bool(params.enable_cueq)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        _fail(
            "mace_relax_batch",
            message=f"input_dir is not a directory: {input_root}",
            error_code="invalid_input_dir",
        )
    if params.output_root is None:
        _fail(
            "mace_relax_batch",
            message="output_root is required for directory batch relaxations.",
            error_code="missing_output_root",
        )
    output_root = resolve_workspace_path(params.output_root)
    if output_root.exists() and not output_root.is_dir():
        _fail(
            "mace_relax_batch",
            message=f"output_root is not a directory: {output_root}",
            error_code="invalid_output_root",
        )
    if _is_within(output_root, input_root):
        _fail(
            "mace_relax_batch",
            message="output_root must not be inside input_dir to avoid mixing inputs with outputs.",
            error_code="output_inside_input",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    structures = _collect_structure_files(input_root, exclude_root=None)
    if not structures:
        _fail(
            "mace_relax_batch",
            message="No structure files found in input_dir (expected POSCAR/CONTCAR/.vasp/.poscar/.cif files).",
            error_code="no_structures",
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

    try:
        model_spec = _stage_mace_model(params.model, stage_root=stage_root)
    except Exception as exc:
        _fail(
            "mace_relax_batch",
            message=f"Invalid model specification {params.model!r}: {exc}",
            data={
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
            },
            error_code="invalid_model",
        )
    model_arg = shlex.quote(model_spec.command_arg)

    ctx = {
        "input": "input",
        "output_root": "output",
        "fmax": params.fmax,
        "steps": params.steps,
        "model": model_arg,
        "head": head_arg,
        "dispersion": "true" if dispersion else "false",
        "default_dtype": params.default_dtype,
        "enable_cueq": "true" if enable_cueq else "false",
        "relax_lattice": "true" if relax_lattice else "false",
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
        check_interval=params.check_interval,
        tool_name="mace_relax_batch",
    )

    dispatch_error: Exception | None = None
    result = None
    collect_info: Dict[str, Any] = {}
    collect_warnings: list[str] = []
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": 1})
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": 1})
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
            details={
                "tasks": 1,
                "errors": [str(dispatch_error)] if dispatch_error else [],
            },
        )
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except Exception as exc:
                collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")

    if dispatch_error is not None:
        _fail(
            "mace_relax_batch",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            warnings=collect_warnings,
            data={
                "work_base": work_base,
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                **remote_context_from_exception(dispatch_error),
                **collect_info,
            },
            error_code="dispatch_failed",
        )

    states = result.task_states if result else []
    data = {
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        "work_base": result.work_base if result else work_base,
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "structures_found": len(structures),
        "model": model_spec.requested,
        "model_command_arg": model_spec.command_arg,
        "model_source_kind": model_spec.source_kind,
        "model_source_rel": model_spec.source_rel,
        "model_asset_rel": model_spec.asset_model_rel,
        "head": head,
        "dispersion": dispersion,
        "default_dtype": params.default_dtype,
        "enable_cueq": enable_cueq,
        "relax_lattice": relax_lattice,
        **compact_list_for_artifact(
            states,
            count_key="task_states_count",
            inline_key="task_states",
            preview_key="task_states_preview",
            truncated_key="task_states_truncated",
            max_inline=20,
        ),
        **remote_context_from_result(result),
        **collect_info,
    }
    lines = [
        "mace_relax_batch completed.",
        f"structures_found={len(structures)} task_states={len(states)}",
        f"output_root_rel={data['output_root_rel']} batch_state_rel={data['batch_state_rel']}",
    ]
    status_file_rel = str(data.get("status_file_rel") or "")
    stdout_file_rel = str(data.get("stdout_file_rel") or "")
    stderr_file_rel = str(data.get("stderr_file_rel") or "")
    if status_file_rel:
        lines.append(f"status_file_rel={status_file_rel}")
    if stdout_file_rel:
        lines.append(f"stdout_file_rel={stdout_file_rel}")
    if stderr_file_rel:
        lines.append(f"stderr_file_rel={stderr_file_rel}")
    batch_summary_rel = str(data.get("batch_summary_rel") or "")
    if batch_summary_rel:
        lines.append(f"batch_summary_rel={batch_summary_rel}")
    content = "\n".join(lines)
    return _success(
        "mace_relax_batch",
        content=content,
        warnings=collect_warnings,
        data=data,
        execution_time=result.duration_s if result else None,
    )


def mace_sp_batch(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[mace/execute] Submit a directory batch of MACE single-point jobs through DPDispatcher."""
    params = MaceSPBatchInput(**payload)
    cfg = _legacy_mace_task("sp")
    resources_key = "mace_gpu"
    machine = _resolve_machine_for_resources(resources_key)
    head = _resolve_mace_head(params.head)
    head_arg = shlex.quote(head or "")
    dispersion = bool(params.dispersion)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        _fail(
            "mace_sp_batch",
            message=f"input_dir is not a directory: {input_root}",
            error_code="invalid_input_dir",
        )
    output_root = resolve_workspace_path(params.output_root)
    if output_root.exists() and not output_root.is_dir():
        _fail(
            "mace_sp_batch",
            message=f"output_root is not a directory: {output_root}",
            error_code="invalid_output_root",
        )
    if _is_within(output_root, input_root):
        _fail(
            "mace_sp_batch",
            message="output_root must not be inside input_dir to avoid mixing inputs with outputs.",
            error_code="output_inside_input",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    structures = _collect_structure_files(input_root, exclude_root=None, allow_xyz=True)
    if not structures:
        _fail(
            "mace_sp_batch",
            message="No structure files found in input_dir (expected POSCAR/CONTCAR/.vasp/.poscar/.cif/.xyz files).",
            error_code="no_structures",
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

    try:
        model_spec = _stage_mace_model(params.model, stage_root=stage_root)
    except Exception as exc:
        _fail(
            "mace_sp_batch",
            message=f"Invalid model specification {params.model!r}: {exc}",
            data={
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
            },
            error_code="invalid_model",
        )
    model_arg = shlex.quote(model_spec.command_arg)

    ctx = {
        "input": "input",
        "output_root": "output",
        "model": model_arg,
        "head": head_arg,
        "dispersion": "true" if dispersion else "false",
        "default_dtype": params.default_dtype,
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
        check_interval=params.check_interval,
        tool_name="mace_sp_batch",
    )

    dispatch_error: Exception | None = None
    result = None
    collect_info: Dict[str, Any] = {}
    collect_warnings: list[str] = []
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": 1})
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": 1})
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
            details={
                "tasks": 1,
                "errors": [str(dispatch_error)] if dispatch_error else [],
            },
        )
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except Exception as exc:
                collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")

    if dispatch_error is not None:
        _fail(
            "mace_sp_batch",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            warnings=collect_warnings,
            data={
                "work_base": work_base,
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                **remote_context_from_exception(dispatch_error),
                **collect_info,
            },
            error_code="dispatch_failed",
        )

    states = result.task_states if result else []
    data = {
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        "work_base": result.work_base if result else work_base,
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "structures_found": len(structures),
        "model": model_spec.requested,
        "model_command_arg": model_spec.command_arg,
        "model_source_kind": model_spec.source_kind,
        "model_source_rel": model_spec.source_rel,
        "model_asset_rel": model_spec.asset_model_rel,
        "head": head,
        "dispersion": dispersion,
        "default_dtype": params.default_dtype,
        **compact_list_for_artifact(
            states,
            count_key="task_states_count",
            inline_key="task_states",
            preview_key="task_states_preview",
            truncated_key="task_states_truncated",
            max_inline=20,
        ),
        **remote_context_from_result(result),
        **collect_info,
    }
    lines = [
        "mace_sp_batch completed.",
        f"structures_found={len(structures)} task_states={len(states)}",
        f"output_root_rel={data['output_root_rel']} batch_state_rel={data['batch_state_rel']}",
    ]
    status_file_rel = str(data.get("status_file_rel") or "")
    stdout_file_rel = str(data.get("stdout_file_rel") or "")
    stderr_file_rel = str(data.get("stderr_file_rel") or "")
    if status_file_rel:
        lines.append(f"status_file_rel={status_file_rel}")
    if stdout_file_rel:
        lines.append(f"stdout_file_rel={stdout_file_rel}")
    if stderr_file_rel:
        lines.append(f"stderr_file_rel={stderr_file_rel}")
    batch_summary_rel = str(data.get("batch_summary_rel") or "")
    if batch_summary_rel:
        lines.append(f"batch_summary_rel={batch_summary_rel}")
    content = "\n".join(lines)
    return _success(
        "mace_sp_batch",
        content=content,
        warnings=collect_warnings,
        data=data,
        execution_time=result.duration_s if result else None,
    )


def mace_md_batch(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[mace/execute] Submit a directory batch of MACE MD jobs through DPDispatcher."""
    params = MaceMDBatchInput(**payload)
    cfg = _legacy_mace_task("md")
    resources_key = "mace_gpu"
    machine = _resolve_machine_for_resources(resources_key)
    head = _resolve_mace_head(params.head)
    dispersion = bool(params.dispersion)

    input_root = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_root.is_dir():
        _fail(
            "mace_md_batch",
            message=f"input_dir is not a directory: {input_root}",
            error_code="invalid_input_dir",
        )
    output_root = resolve_workspace_path(params.output_root)
    if output_root.exists() and not output_root.is_dir():
        _fail(
            "mace_md_batch",
            message=f"output_root is not a directory: {output_root}",
            error_code="invalid_output_root",
        )
    if _is_within(output_root, input_root):
        _fail(
            "mace_md_batch",
            message="output_root must not be inside input_dir to avoid mixing inputs with outputs.",
            error_code="output_inside_input",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    structures = _collect_structure_files(input_root, exclude_root=None, allow_xyz=True)
    if not structures:
        _fail(
            "mace_md_batch",
            message="No structure files found in input_dir (expected POSCAR/CONTCAR/.vasp/.poscar/.cif/.xyz files).",
            error_code="no_structures",
        )

    work_base = make_work_base("mace_md_batch")
    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_input = stage_root / "input"
    stage_output = stage_root / "output"
    shutil.copytree(input_root, stage_input)
    stage_output.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve().parents[2] / "remote" / "gpu" / "mace_md.py"
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing MACE MD remote script: {script_src}")
    script_dst = stage_root / "task_script" / "mace_md.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_src, script_dst)

    try:
        model_spec = _stage_mace_model(params.model, stage_root=stage_root)
    except Exception as exc:
        _fail(
            "mace_md_batch",
            message=f"Invalid model specification {params.model!r}: {exc}",
            data={
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
            },
            error_code="invalid_model",
        )

    params_dir = stage_root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / "md_params.json"
    params_payload = {
        "schema_version": 2,
        "model": model_spec.command_arg,
        "head": head,
        "dispersion": dispersion,
        "default_dtype": params.default_dtype,
        "md_config": params.md_config,
    }
    params_path.write_text(json.dumps(params_payload, indent=2) + "\n", encoding="utf-8")

    ctx = {
        "input": "input",
        "output_root": "output",
        "params": "params/md_params.json",
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
        check_interval=params.check_interval,
        tool_name="mace_md_batch",
    )

    dispatch_error: Exception | None = None
    result = None
    collect_info: Dict[str, Any] = {}
    collect_warnings: list[str] = []
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": 1})
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": 1})
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
            details={
                "tasks": 1,
                "errors": [str(dispatch_error)] if dispatch_error else [],
            },
        )
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except Exception as exc:
                collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")

    if dispatch_error is not None:
        _fail(
            "mace_md_batch",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            warnings=collect_warnings,
            data={
                "work_base": work_base,
                "input_root_rel": workspace_relpath(input_root),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                **remote_context_from_exception(dispatch_error),
                **collect_info,
            },
            error_code="dispatch_failed",
        )

    states = result.task_states if result else []
    data = {
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        "work_base": result.work_base if result else work_base,
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "structures_found": len(structures),
        "model": model_spec.requested,
        "model_command_arg": model_spec.command_arg,
        "model_source_kind": model_spec.source_kind,
        "model_source_rel": model_spec.source_rel,
        "model_asset_rel": model_spec.asset_model_rel,
        "head": head,
        "dispersion": dispersion,
        "default_dtype": params.default_dtype,
        "md_config": params.md_config,
        "params_file_rel": workspace_relpath(params_path),
        **compact_list_for_artifact(
            states,
            count_key="task_states_count",
            inline_key="task_states",
            preview_key="task_states_preview",
            truncated_key="task_states_truncated",
            max_inline=20,
        ),
        **remote_context_from_result(result),
        **collect_info,
    }
    lines = [
        "mace_md_batch completed.",
        f"structures_found={len(structures)} task_states={len(states)}",
        f"output_root_rel={data['output_root_rel']} batch_state_rel={data['batch_state_rel']}",
    ]
    status_file_rel = str(data.get("status_file_rel") or "")
    stdout_file_rel = str(data.get("stdout_file_rel") or "")
    stderr_file_rel = str(data.get("stderr_file_rel") or "")
    if status_file_rel:
        lines.append(f"status_file_rel={status_file_rel}")
    if stdout_file_rel:
        lines.append(f"stdout_file_rel={stdout_file_rel}")
    if stderr_file_rel:
        lines.append(f"stderr_file_rel={stderr_file_rel}")
    batch_summary_rel = str(data.get("batch_summary_rel") or "")
    if batch_summary_rel:
        lines.append(f"batch_summary_rel={batch_summary_rel}")
    content = "\n".join(lines)
    return _success(
        "mace_md_batch",
        content=content,
        warnings=collect_warnings,
        data=data,
        execution_time=result.duration_s if result else None,
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
    model_spec = _stage_mace_model(getattr(params, "model", None), stage_root=work_dir)

    work_base = work_dir.name if work_dir.name not in ("", ".") else make_work_base("mace")
    local_root = work_dir.parent

    ctx = {
        "structure_file": dest_structure.name,
        "structure": dest_structure.name,
        "fmax": params.fmax,
        "steps": params.steps,
        "model": shlex.quote(model_spec.command_arg),
        "head": shlex.quote(_resolve_mace_head(getattr(params, "head", None)) or ""),
        "dispersion": "true" if bool(getattr(params, "dispersion", False)) else "false",
        "relax_lattice": "true" if bool(getattr(params, "relax_lattice", False)) else "false",
    }

    rendered = render_task_fields(cfg, ctx, work_dir)
    if model_spec.asset_dir_rel and model_spec.asset_dir_rel not in rendered["forward_files"]:
        rendered["forward_files"].append(model_spec.asset_dir_rel)

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
    "MaceMDBatchInput",
    "mace_relax_batch",
    "mace_sp_batch",
    "mace_md_batch",
]
