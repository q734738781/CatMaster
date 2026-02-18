from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class InterruptRequest:
    requested: bool = False
    request_ts: float = 0.0
    source: str = ""
    note: str = ""
    acked: bool = False
    ack_ts: float = 0.0
    phase: str = ""
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": bool(self.requested),
            "request_ts": float(self.request_ts or 0.0),
            "source": self.source or "",
            "note": self.note or "",
            "acked": bool(self.acked),
            "ack_ts": float(self.ack_ts or 0.0),
            "phase": self.phase or "",
            "details": dict(self.details or {}),
        }


class RunControl:
    """In-memory run control plane for interrupt/resume coordination."""

    def __init__(self, *, run_id: str = "") -> None:
        self.run_id = run_id or ""
        self._lock = threading.Lock()
        self._interrupt = InterruptRequest()

    def request_interrupt(self, *, source: str = "ui", note: str = "") -> Dict[str, Any]:
        with self._lock:
            if not self._interrupt.requested:
                self._interrupt.requested = True
                self._interrupt.request_ts = time.time()
                self._interrupt.source = source or "ui"
                self._interrupt.note = note or ""
                self._interrupt.acked = False
                self._interrupt.ack_ts = 0.0
                self._interrupt.phase = ""
                self._interrupt.details = {}
            return self._interrupt.to_dict()

    def is_interrupt_requested(self) -> bool:
        with self._lock:
            return bool(self._interrupt.requested)

    def ack_interrupt(self, *, phase: str = "", details: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            if self._interrupt.requested:
                self._interrupt.acked = True
                self._interrupt.ack_ts = time.time()
                self._interrupt.phase = phase or self._interrupt.phase
                self._interrupt.details = dict(details or {})
            return self._interrupt.to_dict()

    def clear_interrupt(self) -> Dict[str, Any]:
        with self._lock:
            self._interrupt = InterruptRequest()
            return self._interrupt.to_dict()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._interrupt.to_dict()


__all__ = ["RunControl", "InterruptRequest"]
