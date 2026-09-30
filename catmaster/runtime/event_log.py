#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append-only JSONL event log for run auditing.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


class EventLog:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        record: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
        }
        if payload is not None:
            record["payload"] = payload
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
