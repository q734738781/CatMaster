from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from catmaster.ui import Reporter
from catmaster.ui.events import UIEvent


class PromptBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: Optional[Dict[str, Any]] = None
        self._responses: Dict[str, str] = {}

    def request_prompt(self, kind: str, payload: Dict[str, Any]) -> str:
        prompt_id = uuid.uuid4().hex
        with self._cond:
            self._pending = {
                "prompt_id": prompt_id,
                "kind": kind,
                "payload": payload,
                "created_at": time.time(),
            }
            while prompt_id not in self._responses:
                self._cond.wait()
            text = self._responses.pop(prompt_id, "")
            if self._pending and self._pending.get("prompt_id") == prompt_id:
                self._pending = None
            return text

    def submit(self, prompt_id: str, text: str) -> bool:
        with self._cond:
            if not self._pending or self._pending.get("prompt_id") != prompt_id:
                return False
            self._responses[prompt_id] = text
            self._cond.notify_all()
            return True

    def get_pending(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._pending:
                return None
            return dict(self._pending)


class WebReporter(Reporter):
    def __init__(self, *, broker: PromptBroker, max_events: int = 2000) -> None:
        self._broker = broker
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._pending_write: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._run_dir: Optional[Path] = None
        self._event_log_path: Optional[Path] = None
        self._final_summary: Optional[str] = None

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def is_live(self) -> bool:
        return True

    def set_run_dir(self, run_dir: Path) -> None:
        run_dir = Path(run_dir).expanduser().resolve()
        with self._lock:
            self._run_dir = run_dir
            self._event_log_path = run_dir / "ui_events.jsonl"
            pending = list(self._pending_write)
            self._pending_write.clear()
        if pending and self._event_log_path:
            for event in pending:
                self._append_event_file(event)

    def get_run_dir(self) -> Optional[Path]:
        with self._lock:
            return self._run_dir

    def emit(self, event: UIEvent) -> None:
        try:
            payload = asdict(event)
        except Exception:
            payload = {
                "ts": getattr(event, "ts", time.time()),
                "level": getattr(event, "level", "info"),
                "category": getattr(event, "category", "event"),
                "name": getattr(event, "name", "EVENT"),
                "payload": getattr(event, "payload", {}) or {},
                "run_id": getattr(event, "run_id", None),
                "task_id": getattr(event, "task_id", None),
                "step_id": getattr(event, "step_id", None),
            }
        with self._lock:
            self._seq += 1
            payload["seq"] = self._seq
            self._events.append(payload)
            if self._event_log_path:
                self._append_event_file(payload)
            else:
                self._pending_write.append(payload)

    def get_events_since(self, last_seq: int) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            if not self._events:
                return [], last_seq
            events = [event for event in self._events if event.get("seq", 0) > last_seq]
            new_seq = events[-1]["seq"] if events else last_seq
        return events, new_seq

    def prompt_proposal_feedback(self, *, todo: List[str], proposal_description: str) -> str:
        return self._broker.request_prompt("proposal_review", {
            "todo": list(todo),
            "proposal_description": proposal_description or "",
        })

    def prompt_hitl_feedback(self, *, report_text: str, report_path: str) -> str:
        return self._broker.request_prompt("hitl", {
            "report_text": report_text or "",
            "report_path": report_path or "",
        })

    def prompt_interrupt_feedback(self, *, guidance: str, run_id: str, phase: str) -> str:
        return self._broker.request_prompt("interrupt_feedback", {
            "guidance": guidance or "",
            "run_id": run_id or "",
            "phase": phase or "",
        })

    def show_final_summary(self, summary: str) -> None:
        with self._lock:
            self._final_summary = summary

    def get_final_summary(self) -> Optional[str]:
        with self._lock:
            return self._final_summary

    def _append_event_file(self, payload: Dict[str, Any]) -> None:
        path = self._event_log_path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return
