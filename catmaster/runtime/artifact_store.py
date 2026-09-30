#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArtifactStore persists toolcall inputs/outputs under the run directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


class ArtifactStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.toolcalls_dir = self.run_dir / "toolcalls"
        self.toolcalls_dir.mkdir(parents=True, exist_ok=True)

    def toolcall_refs(self, toolcall_id: str) -> Dict[str, str]:
        toolcall_dir = self.toolcalls_dir / toolcall_id
        input_path = toolcall_dir / "input.json"
        output_path = toolcall_dir / "output.json"
        return {
            "toolcall_dir": str(toolcall_dir.relative_to(self.run_dir)),
            "input_ref": str(input_path.relative_to(self.run_dir)),
            "output_ref": str(output_path.relative_to(self.run_dir)),
        }

    def write_input(self, toolcall_id: str, payload: Dict[str, Any]) -> Path:
        toolcall_dir = self.toolcalls_dir / toolcall_id
        toolcall_dir.mkdir(parents=True, exist_ok=True)
        path = toolcall_dir / "input.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2)
        return path

    def write_output(self, toolcall_id: str, payload: Dict[str, Any]) -> Path:
        toolcall_dir = self.toolcalls_dir / toolcall_id
        toolcall_dir.mkdir(parents=True, exist_ok=True)
        path = toolcall_dir / "output.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2)
        return path
