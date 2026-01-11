#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artifact log stored as a CSV file for task-level outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv
import os

from catmaster.tools.base import workspace_root


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _canonical_path(path: str) -> str:
    root = workspace_root()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    try:
        return str(p.relative_to(root))
    except Exception:
        raise ValueError(f"Artifact path escapes workspace: {p}")


@dataclass(frozen=True)
class ArtifactLog:
    path: Path

    header: tuple[str, ...] = ("path", "description", "type", "updated_time")

    def ensure_exists(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.header)

    def load(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows: List[Dict[str, str]] = []
            for row in reader:
                path = (row.get("path") or "").strip()
                if not path:
                    continue
                rows.append({
                    "path": path,
                    "description": (row.get("description") or "").strip(),
                    "type": (row.get("type") or "").strip(),
                    "updated_time": (row.get("updated_time") or "").strip(),
                })
            return rows

    def update(self, entries: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        existing = {entry["path"]: entry for entry in self.load() if entry.get("path")}
        for entry in entries:
            normalized = self._normalize_entry(entry)
            if normalized is None:
                continue
            existing[normalized["path"]] = normalized
        merged = [existing[key] for key in sorted(existing.keys())]
        self._write(merged)
        return merged

    def _write(self, rows: List[Dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.header))
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "path": row.get("path", ""),
                    "description": row.get("description", ""),
                    "type": row.get("type", ""),
                    "updated_time": row.get("updated_time", ""),
                })

    @classmethod
    def infer_type(cls, path_text: str) -> str:
        path = Path(path_text)
        if not path.is_absolute():
            path = workspace_root() / path
        if path.exists():
            return "dir" if path.is_dir() else "file"
        if path_text.endswith("/") or path_text.endswith(os.sep):
            return "dir"
        return "file"

    def _normalize_entry(self, entry: Dict[str, str]) -> Optional[Dict[str, str]]:
        path = (entry.get("path") or "").strip()
        if not path:
            return None
        try:
            normalized_path = _canonical_path(path)
        except Exception:
            return None
        description = (entry.get("description") or "").strip()
        type_value = (entry.get("type") or "").strip().lower()
        if type_value not in {"file", "dir"}:
            type_value = "file"
        return {
            "path": normalized_path,
            "description": description,
            "type": type_value,
            "updated_time": _now_iso(),
        }


__all__ = ["ArtifactLog"]
