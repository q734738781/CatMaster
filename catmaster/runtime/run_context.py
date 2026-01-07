#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunContext manages per-run metadata and standardized run directory layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import subprocess
import uuid

from catmaster.tools.base import workspace_root


def _find_git_root(start: Path) -> Optional[Path]:
    for path in [start] + list(start.parents):
        if (path / ".git").exists():
            return path
    return None


def _get_git_commit(start: Path) -> str:
    git_root = _find_git_root(start)
    if not git_root:
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=git_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _default_project_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d")
    return f"project_{stamp}_{uuid.uuid4().hex[:8]}"


def _default_run_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


@dataclass
class RunContext:
    project_id: str
    run_id: str
    workspace: Path
    run_dir: Path
    model_name: str
    start_time: str
    git_commit: str

    @classmethod
    def create(
        cls,
        *,
        workspace: Optional[Path] = None,
        run_dir: Optional[Path] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_name: str = "unknown",
    ) -> "RunContext":
        ws = (workspace or (run_dir if run_dir else workspace_root())).resolve()
        project_id = project_id or _default_project_id()
        run_id = run_id or _default_run_id()
        resolved_run_dir = (ws / "metadata").resolve()
        if not str(resolved_run_dir).startswith(str(ws)):
            raise ValueError(f"metadata dir must be under workspace root: {resolved_run_dir}")
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        start_time = datetime.utcnow().isoformat() + "Z"
        git_commit = _get_git_commit(ws)
        ctx = cls(
            project_id=project_id,
            run_id=run_id,
            workspace=ws,
            run_dir=resolved_run_dir,
            model_name=model_name,
            start_time=start_time,
            git_commit=git_commit,
        )
        ctx.write_meta()
        return ctx

    @classmethod
    def load(cls, workspace: Path) -> "RunContext":
        ws = Path(workspace).expanduser().resolve()
        resolved_run_dir = (ws / "metadata").resolve()
        meta_path = resolved_run_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if not str(resolved_run_dir).startswith(str(ws)):
            raise ValueError(f"metadata dir must be under workspace root: {resolved_run_dir}")
        project_id = meta.get("project_id") or _default_project_id()
        run_id = meta.get("run_id") or _default_run_id()
        model_name = meta.get("model_name") or "unknown"
        start_time = meta.get("start_time") or datetime.utcnow().isoformat() + "Z"
        git_commit = meta.get("git_commit") or _get_git_commit(ws)
        return cls(
            project_id=project_id,
            run_id=run_id,
            workspace=ws,
            run_dir=resolved_run_dir,
            model_name=model_name,
            start_time=start_time,
            git_commit=git_commit,
        )

    def meta(self) -> dict:
        return {
            "project_id": self.project_id,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "model_name": self.model_name,
            "start_time": self.start_time,
        }

    def write_meta(self) -> None:
        meta_path = self.run_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(self.meta(), fh, ensure_ascii=False, indent=2)
