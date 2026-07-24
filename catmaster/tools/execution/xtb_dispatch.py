from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath
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
_MOLECULE_EXTS = {".xyz", ".mol", ".sdf", ".mol2", ".pdb", ".vasp", ".cif"}


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


def _discover_molecule_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    skip_prefixes = ("xtb_batch_", "crest_batch_", "orca_batch_", "vasp_batch_", "mace_batch_")
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
        for filename in filenames:
            candidate = path / filename
            if candidate.suffix.lower() in _MOLECULE_EXTS or filename in {"POSCAR", "CONTCAR"}:
                files.append(candidate)
    return sorted(files, key=lambda p: str(p))


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


class CrestDistanceConstraint(BaseModel):
    atom1: int = Field(..., ge=0)
    atom2: int = Field(..., ge=0)
    value_angstrom: float = Field(..., gt=0.0)


class CrestAngleConstraint(BaseModel):
    atom1: int = Field(..., ge=0)
    atom2: int = Field(..., ge=0)
    atom3: int = Field(..., ge=0)
    value_degree: float = Field(..., gt=0.0)


class CrestDihedralConstraint(BaseModel):
    atom1: int = Field(..., ge=0)
    atom2: int = Field(..., ge=0)
    atom3: int = Field(..., ge=0)
    atom4: int = Field(..., ge=0)
    value_degree: float = Field(..., description="Target dihedral angle in degrees.")


class CrestConformerSearchInput(BaseModel):
    """[xtb/execute] Run CREST conformer search for one molecule or a batch of molecules via DPDispatcher."""

    input_path: str = Field(..., description="Single structure file or directory containing molecules.")
    output_root: str = Field(..., description="Output root where collected CREST result folders will be written.")
    mode: str = Field("standard", pattern="^(standard|nci|constrained)$", description="CREST workflow mode.")
    method: str = Field("gfn2", pattern="^(gfn2|gfn1|gfnff)$", description="Semiempirical method family.")
    ewin: float = Field(6.0, ge=0.0, description="Energy window in kcal/mol.")
    rthr: float = Field(0.125, ge=0.0, description="CREST RMSD threshold.")
    ethr: float = Field(0.05, ge=0.0, description="CREST energy threshold.")
    bthr: float = Field(0.01, ge=0.0, description="CREST rotational-constant threshold.")
    charge: int = Field(0, description="Molecular charge.")
    uhf: int = Field(0, ge=0, description="Number of unpaired electrons.")
    solvent: str | None = Field(None, description="Optional ALPB solvent.")
    frozen_atom_indices: list[int] = Field(default_factory=list, description="0-based atom indices to freeze in constrained mode.")
    distance_constraints: list[CrestDistanceConstraint] = Field(default_factory=list, description="Optional distance constraints.")
    angle_constraints: list[CrestAngleConstraint] = Field(default_factory=list, description="Optional angle constraints.")
    dihedral_constraints: list[CrestDihedralConstraint] = Field(default_factory=list, description="Optional dihedral constraints.")
    task_name: str = Field("crest_run", description="DPDispatcher task template name.")
    check_interval: int = Field(30, ge=1, description="Polling interval seconds.")


def _submit_molecule_batch(
    *,
    tool_name: str,
    input_path: Path,
    output_root: Path,
    task_name: str,
    work_prefix: str,
    script_name: str,
    context_builder,
    check_interval: int,
) -> tuple[str, dict[str, Any]]:
    registry = TaskRegistry()
    cfg = registry.get(task_name)
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError(f"{task_name} missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)
    structures = _discover_molecule_files(input_path)
    if not structures:
        _fail(tool_name, message=f"No supported molecular inputs found under {input_path}", error_code="no_structures")
    if input_path.is_dir() and _is_within(output_root, input_path):
        _fail(tool_name, message="output_root must not be inside input_path.", error_code="output_inside_input")
    output_root.mkdir(parents=True, exist_ok=True)

    work_base = make_work_base(work_prefix)
    local_root = output_root
    tasks: list[TaskSpec] = []
    task_meta: list[dict[str, Any]] = []
    script_src = Path(__file__).resolve().parents[2] / "remote" / "cpu" / script_name
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing task script: {script_src}")

    for structure in structures:
        rel_name = Path(structure.stem) if input_path.is_file() else structure.relative_to(input_path).with_suffix("")
        stage_dir = local_root / work_base / rel_name
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        input_name = "input" + (structure.suffix if structure.suffix else ".xyz")
        shutil.copy2(structure, stage_dir / input_name)
        script_dst = stage_dir / "task_script" / script_name
        script_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_src, script_dst)
        ctx = context_builder(stage_dir=stage_dir, input_name=input_name)
        rendered = render_task_fields(cfg, ctx, stage_dir)
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
                "input_rel": workspace_relpath(structure),
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
        check_interval=check_interval,
        tool_name=tool_name,
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
                        "input_rel": meta["input_rel"],
                        "output_dir_rel": workspace_relpath(meta["output_dir"]),
                    }
                )
            except Exception as exc:
                warnings.append(f"collect failed for {meta['input_rel']}: {type(exc).__name__}: {exc}")
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
    states = result.task_states if result else []
    execution_summary_path = output_root / f"{tool_name}_summary.json"
    execution_summary_path.write_text(
        json.dumps(
            {
                "tool_name": tool_name,
                "task_name": task_name,
                "work_base": work_base,
                "input_path_rel": workspace_relpath(input_path),
                "output_root_rel": workspace_relpath(output_root),
                "task_states": states,
                "outputs": outputs,
                "warnings": warnings,
                **remote_context_from_result(result),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    execution_summary_rel = workspace_relpath(execution_summary_path)

    if dispatch_error is not None:
        _fail(
            tool_name,
            message=f"DPDispatcher submission failed: {dispatch_error}",
            data={
                "input_path_rel": workspace_relpath(input_path),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                "execution_summary_rel": execution_summary_rel,
                **compact_list_for_artifact(
                    outputs,
                    count_key="outputs_count",
                    inline_key="outputs",
                    preview_key="outputs_preview",
                    truncated_key="outputs_truncated",
                    full_rel_key="outputs_full_rel",
                    full_rel=execution_summary_rel,
                    max_inline=5,
                ),
                **compact_list_for_artifact(
                    states,
                    count_key="task_states_count",
                    inline_key="task_states",
                    preview_key="task_states_preview",
                    truncated_key="task_states_truncated",
                    full_rel_key="task_states_full_rel",
                    full_rel=execution_summary_rel,
                    max_inline=20,
                ),
                **remote_context_from_exception(dispatch_error),
            },
            warnings=warnings,
            error_code="dispatch_failed",
        )

    data = {
        "task_name": task_name,
        "resources_key": resources_key,
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "execution_summary_rel": execution_summary_rel,
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        "work_base": result.work_base if result else work_base,
        **compact_list_for_artifact(
            outputs,
            count_key="outputs_count",
            inline_key="outputs",
            preview_key="outputs_preview",
            truncated_key="outputs_truncated",
            full_rel_key="outputs_full_rel",
            full_rel=execution_summary_rel,
            max_inline=5,
        ),
        **compact_list_for_artifact(
            states,
            count_key="task_states_count",
            inline_key="task_states",
            preview_key="task_states_preview",
            truncated_key="task_states_truncated",
            full_rel_key="task_states_full_rel",
            full_rel=execution_summary_rel,
            max_inline=20,
        ),
        **remote_context_from_result(result),
    }
    content = (
        f"{tool_name} completed.\n"
        f"task_name={task_name} inputs={len(structures)} outputs_collected={len(outputs)}\n"
        f"output_root_rel={data['output_root_rel']} batch_state_rel={data['batch_state_rel']}"
    )
    return _success(
        tool_name,
        content=content,
        data=data,
        warnings=warnings,
        execution_time=result.duration_s if result else None,
    )


def _serialize_constraints(params: CrestConformerSearchInput, stage_dir: Path) -> str:
    if params.mode != "constrained" and not any(
        (params.frozen_atom_indices, params.distance_constraints, params.angle_constraints, params.dihedral_constraints)
    ):
        return ""
    lines = ["$constrain"]
    if params.frozen_atom_indices:
        atoms = ", ".join(str(int(idx) + 1) for idx in params.frozen_atom_indices)
        lines.append(f"  atoms: {atoms}")
    for item in params.distance_constraints:
        lines.append(
            f"  distance: {int(item.atom1) + 1}, {int(item.atom2) + 1}, {float(item.value_angstrom):.6f}"
        )
    for item in params.angle_constraints:
        lines.append(
            f"  angle: {int(item.atom1) + 1}, {int(item.atom2) + 1}, {int(item.atom3) + 1}, {float(item.value_degree):.6f}"
        )
    for item in params.dihedral_constraints:
        lines.append(
            "  dihedral: "
            f"{int(item.atom1) + 1}, {int(item.atom2) + 1}, {int(item.atom3) + 1}, {int(item.atom4) + 1}, {float(item.value_degree):.6f}"
        )
    lines.append("$end")
    constraint_path = stage_dir / "constraints.inp"
    constraint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return constraint_path.name


def crest_conformer_search(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = CrestConformerSearchInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    return _submit_molecule_batch(
        tool_name="crest_conformer_search",
        input_path=input_path,
        output_root=output_root,
        task_name=params.task_name,
        work_prefix="crest_batch",
        script_name="crest_boot.py",
        context_builder=lambda *, stage_dir, input_name: {
            "input": input_name,
            "mode": params.mode,
            "method": params.method,
            "ewin": params.ewin,
            "rthr": params.rthr,
            "ethr": params.ethr,
            "bthr": params.bthr,
            "charge": params.charge,
            "uhf": params.uhf,
            "solvent": params.solvent or "__none__",
            "constraint_file": _serialize_constraints(params, stage_dir) or "__none__",
        },
        check_interval=params.check_interval,
    )


__all__ = [
    "CrestConformerSearchInput",
    "crest_conformer_search",
]
