from __future__ import annotations

"""Utility builders that translate tool inputs into DPDispatcher payloads."""

import shutil
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from catmaster.tools.execution.dpdispatcher_runner import DispatchRequest, make_work_base
from catmaster.tools.execution.task_registry import (
    TaskConfig,
    TaskRegistry,
    format_list,
    format_template,
)


def dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def expand_forward_files(patterns: Sequence[str], base_dir: Path, ctx: Mapping[str, Any]) -> List[str]:
    include_all = False
    expanded: List[str] = []
    for item in patterns or []:
        if item == "*":
            include_all = True
            continue
        expanded.append(format_template(item, ctx))
    if include_all or not expanded:
        expanded.extend([p.name for p in base_dir.iterdir() if p.is_file()])
    return dedup(expanded)


def render_task_fields(cfg: TaskConfig, ctx: Mapping[str, Any], base_dir: Path) -> dict:
    """Materialize command and file lists from a TaskConfig."""

    command = format_template(cfg.command, ctx)
    forward_files = expand_forward_files(cfg.forward_files, base_dir, ctx)
    backward_files = dedup(format_list(cfg.backward_files, ctx))
    forward_common_files = dedup(format_list(cfg.forward_common_files, ctx))
    backward_common_files = dedup(format_list(cfg.backward_common_files, ctx))
    task_work_path = cfg.task_work_path or "."

    return {
        "command": command,
        "task_work_path": task_work_path,
        "forward_files": forward_files,
        "backward_files": backward_files,
        "forward_common_files": forward_common_files,
        "backward_common_files": backward_common_files,
    }
