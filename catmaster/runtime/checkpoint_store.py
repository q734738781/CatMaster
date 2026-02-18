from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CheckpointStore:
    """Persistent checkpoint stream for fine-grained resume."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.root = self.run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def trace_path(self) -> Path:
        return self.root / "checkpoint_trace.jsonl"

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    def append(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            "payload": payload or {},
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_latest(self, snapshot: Dict[str, Any]) -> None:
        data = {
            "ts": time.time(),
            "snapshot": snapshot or {},
        }
        tmp = self.latest_path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.latest_path)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        if not self.latest_path.exists():
            return None
        try:
            data = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        snapshot = data.get("snapshot")
        return snapshot if isinstance(snapshot, dict) else None


__all__ = ["CheckpointStore"]
