from __future__ import annotations

"""Utility builders that translate tool inputs into DPDispatcher payloads."""

import shutil
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from catmaster.tools.execution.dpdispatcher_runner import DispatchRequest, make_work_base
from catmaster.tools.execution.resource_router import Route
from catmaster.tools.execution.task_registry import (
    TaskConfig,
    TaskRegistry,
    format_list,
    format_template,
)
from catmaster.tools.base import resolve_workspace_path


def _dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _expand_forward_files(patterns: Sequence[str], base_dir: Path, ctx: Mapping[str, Any]) -> List[str]:
    include_all = False
    expanded: List[str] = []
    for item in patterns or []:
        if item == "*":
            include_all = True
            continue
        expanded.append(format_template(item, ctx))
    if include_all or not expanded:
        expanded.extend([p.name for p in base_dir.iterdir() if p.is_file()])
    return _dedup(expanded)


def _render_task_fields(cfg: TaskConfig, ctx: Mapping[str, Any], base_dir: Path) -> dict:
    """Materialize command and file lists from a TaskConfig."""

    command = format_template(cfg.command, ctx)
    forward_files = _expand_forward_files(cfg.forward_files, base_dir, ctx)
    backward_files = _dedup(format_list(cfg.backward_files, ctx))
    forward_common_files = _dedup(format_list(cfg.forward_common_files, ctx))
    backward_common_files = _dedup(format_list(cfg.backward_common_files, ctx))
    task_work_path = cfg.task_work_path or "."

    return {
        "command": command,
        "task_work_path": task_work_path,
        "forward_files": forward_files,
        "backward_files": backward_files,
        "forward_common_files": forward_common_files,
        "backward_common_files": backward_common_files,
    }


def build_mace_relax_request(params: Any, *, route: Route, registry: TaskRegistry | None = None) -> DispatchRequest:
    """Create a DispatchRequest for `mace_relax` using task configs and router info."""

    reg = registry or TaskRegistry()
    cfg = reg.get("mace_relax")

    work_dir = resolve_workspace_path(params.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    src_structure = resolve_workspace_path(params.structure_file, must_exist=True)
    if not src_structure.exists():
        raise FileNotFoundError(f"structure_file not found: {src_structure}")

    dest_structure = work_dir / src_structure.name
    if dest_structure.resolve() != src_structure.resolve():
        shutil.copy(src_structure, dest_structure)

    work_base = params.work_base or (work_dir.name if work_dir.name not in ("", ".") else make_work_base("mace"))
    local_root = work_dir.parent

    ctx = {
        "structure_file": dest_structure.name,
        "structure": dest_structure.name,
        "fmax": params.fmax,
        "maxsteps": params.maxsteps,
        "steps": params.maxsteps,
        "model": params.model,
    }

    rendered = _render_task_fields(cfg, ctx, work_dir)

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


def build_vasp_execute_request(params: Any, *, route: Route, registry: TaskRegistry | None = None) -> DispatchRequest:
    """Create a DispatchRequest for `vasp_execute` using task configs and router info."""

    reg = registry or TaskRegistry()
    cfg = reg.get("vasp_execute")

    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    work_base = (input_dir.name if input_dir.name not in ("", ".") else make_work_base("vasp"))
    local_root = input_dir.parent

    rendered = _render_task_fields(cfg, {}, input_dir)

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
