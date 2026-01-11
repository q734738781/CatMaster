#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified trace store for events, toolcalls, and patches.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass(frozen=True)
class TraceStore:
    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def event_path(self) -> Path:
        return self.run_dir / "event_trace.jsonl"

    @property
    def tool_path(self) -> Path:
        return self.run_dir / "tool_trace.jsonl"

    @property
    def patch_path(self) -> Path:
        return self.run_dir / "patch_trace.jsonl"

    def append_event(self, record: Dict[str, Any]) -> None:
        self._append(self.event_path, record)

    def append_toolcall(self, record: Dict[str, Any]) -> None:
        self._append(self.tool_path, record)

    def append_patch(self, record: Dict[str, Any]) -> None:
        self._append(self.patch_path, record)

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("ts", _now_iso())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["TraceStore"]
