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
import uuid

from catmaster.tools.base import ensure_system_root, system_root, workspace_root


def _default_project_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d")
    return f"project_{stamp}_{uuid.uuid4().hex[:8]}"


def _default_run_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class RunContext:
    project_id: str
    run_id: str
    workspace: Path
    run_dir: Path
    model_name: str
    start_time: str

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
        ws = (workspace or workspace_root()).resolve()
        ensure_system_root()
        project_id = project_id or _default_project_id()
        run_id = run_id or _default_run_id()
        resolved_run_dir = Path(run_dir).expanduser().resolve() if run_dir else (system_root() / "runs" / run_id).resolve()
        sys_root = system_root().resolve()
        if not str(resolved_run_dir).startswith(str(sys_root)):
            raise ValueError(f"run_dir must be under system root: {resolved_run_dir}")
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        start_time = datetime.utcnow().isoformat() + "Z"
        ctx = cls(
            project_id=project_id,
            run_id=run_id,
            workspace=ws,
            run_dir=resolved_run_dir,
            model_name=model_name,
            start_time=start_time,
        )
        ctx.write_meta()
        return ctx

    @classmethod
    def load(cls, run_dir: Path) -> "RunContext":
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        sys_root = system_root().resolve()
        if not str(resolved_run_dir).startswith(str(sys_root)):
            raise ValueError(f"run_dir must be under system root: {resolved_run_dir}")
        meta_path = resolved_run_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"run meta not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        workspace_value = meta.get("workspace")
        if not workspace_value:
            raise ValueError("run meta missing workspace")
        ws = Path(workspace_value).expanduser().resolve()
        return cls(
            project_id=meta.get("project_id") or _default_project_id(),
            run_id=meta.get("run_id") or _default_run_id(),
            workspace=ws,
            run_dir=resolved_run_dir,
            model_name=meta.get("model_name") or "unknown",
            start_time=meta.get("start_time") or datetime.utcnow().isoformat() + "Z",
        )

    def meta(self) -> dict:
        return {
            "project_id": self.project_id,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "model_name": self.model_name,
            "start_time": self.start_time,
        }

    def write_meta(self) -> None:
        meta_path = self.run_dir / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(self.meta(), ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["RunContext"]
