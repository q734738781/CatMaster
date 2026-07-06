from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, ObservabilityStore
from catmaster.tools.base import system_root

from .thread_models import ThreadEventEnvelope
from .thread_store import _safe_id


def format_sse(event: ThreadEventEnvelope) -> str:
    data = event.model_dump(mode="json")
    return (
        f"id: {event.seq}\n"
        f"event: {event.event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


class ThreadEventBroker:
    """Replayable per-workspace thread event broker backed by ObservabilityStore."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        run_dir_resolver: Callable[[str, dict[str, Any]], Path | None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = system_root(self.workspace) / "threads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_root = system_root(self.workspace) / "runs"
        self._run_dir_resolver = run_dir_resolver
        self._condition = threading.Condition()
        self._last_seq: dict[str, int] = {}
        self._events_by_thread: dict[str, list[ThreadEventEnvelope]] = {}

    def events_path(self, thread_id: str) -> Path:
        """Legacy JSONL path used only for reading old thread stream files."""
        return self.root / _safe_id(thread_id, label="thread_id") / "events.jsonl"

    def emit(
        self,
        thread_id: str,
        event: str,
        *,
        message_id: str = "",
        status: str = "",
        data: dict[str, Any] | None = None,
    ) -> ThreadEventEnvelope:
        tid = _safe_id(thread_id, label="thread_id")
        with self._condition:
            seq = self._next_seq_unlocked(tid)
            envelope = ThreadEventEnvelope(
                seq=seq,
                event=str(event or "error"),
                thread_id=tid,
                message_id=str(message_id or ""),
                status=str(status or ""),
                data=dict(data or {}),
            )
            self._events_by_thread.setdefault(tid, []).append(envelope)
            self._last_seq[tid] = seq
            self._persist_observation(tid, envelope)
            self._condition.notify_all()
            return envelope

    def replay(self, thread_id: str, *, last_seq: int = 0, limit: int = 1000) -> list[ThreadEventEnvelope]:
        tid = _safe_id(thread_id, label="thread_id")
        capped_limit = min(5000, max(1, int(limit or 1000)))
        by_seq: dict[int, ThreadEventEnvelope] = {}
        for event in self._read_observed_events(tid, last_seq=last_seq, limit=capped_limit):
            by_seq[int(event.seq or 0)] = event
        for event in self._read_legacy_events(tid, last_seq=last_seq, limit=capped_limit):
            by_seq.setdefault(int(event.seq or 0), event)
        with self._condition:
            for event in self._events_by_thread.get(tid, []):
                if int(event.seq or 0) > int(last_seq or 0):
                    by_seq[int(event.seq or 0)] = event
        return [by_seq[seq] for seq in sorted(by_seq)[:capped_limit]]

    def latest_seq(self, thread_id: str) -> int:
        tid = _safe_id(thread_id, label="thread_id")
        with self._condition:
            current = int(self._last_seq.get(tid) or 0)
        return max(current, self._read_last_seq(tid), self._read_observed_last_seq(tid))

    def wait_for_events(
        self,
        thread_id: str,
        *,
        last_seq: int = 0,
        timeout_s: float = 10.0,
        limit: int = 200,
    ) -> tuple[list[ThreadEventEnvelope], int]:
        tid = _safe_id(thread_id, label="thread_id")
        deadline = time.time() + max(0.1, float(timeout_s or 10.0))
        while True:
            events = self.replay(tid, last_seq=last_seq, limit=limit)
            if events:
                return events, int(events[-1].seq)
            remaining = deadline - time.time()
            if remaining <= 0:
                return [], int(last_seq or 0)
            with self._condition:
                self._condition.wait(timeout=remaining)

    def _next_seq_unlocked(self, thread_id: str) -> int:
        current = int(self._last_seq.get(thread_id) or 0)
        if current <= 0:
            current = max(self._read_last_seq(thread_id), self._read_observed_last_seq(thread_id))
        return current + 1

    def _persist_observation(self, thread_id: str, envelope: ThreadEventEnvelope) -> None:
        run_dir = self._run_dir_for_event(thread_id, envelope.data)
        if run_dir is None:
            return
        try:
            ObservabilityStore(run_dir).record_thread_event(envelope)
        except Exception:
            return

    def _run_dir_for_event(self, thread_id: str, data: dict[str, Any]) -> Path | None:
        if self._run_dir_resolver is not None:
            try:
                resolved = self._run_dir_resolver(thread_id, data)
            except Exception:
                resolved = None
            if resolved is not None:
                return Path(resolved)
        run_id = _event_run_id(data)
        if not run_id:
            return None
        return self.runs_root / _safe_id(run_id, label="run_id")

    def _candidate_run_dirs(self, thread_id: str) -> list[Path]:
        tid = _safe_id(thread_id, label="thread_id")
        candidates: list[Path] = []
        if self._run_dir_resolver is not None:
            try:
                resolved = self._run_dir_resolver(tid, {})
            except Exception:
                resolved = None
            if resolved is not None:
                candidates.append(Path(resolved))
        try:
            for db_path in self.runs_root.glob(f"*/{OBSERVABILITY_DB_NAME}"):
                candidates.append(db_path.parent)
        except Exception:
            pass
        seen: set[str] = set()
        unique: list[Path] = []
        for path in candidates:
            key = str(Path(path).expanduser().resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(Path(path))
        return unique

    def _read_observed_events(self, thread_id: str, *, last_seq: int, limit: int) -> list[ThreadEventEnvelope]:
        events: list[ThreadEventEnvelope] = []
        for run_dir in self._candidate_run_dirs(thread_id):
            try:
                rows = ObservabilityStore(run_dir).read_thread_events_page(thread_id, last_seq=last_seq, limit=limit)
            except Exception:
                continue
            for row in rows:
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                try:
                    events.append(ThreadEventEnvelope.model_validate(payload))
                except Exception:
                    continue
        events.sort(key=lambda item: int(item.seq or 0))
        return events[:limit]

    def _read_observed_last_seq(self, thread_id: str) -> int:
        last = 0
        for run_dir in self._candidate_run_dirs(thread_id):
            try:
                last = max(last, ObservabilityStore(run_dir).latest_thread_event_seq(thread_id))
            except Exception:
                continue
        return last

    def _read_last_seq(self, thread_id: str) -> int:
        path = self.events_path(thread_id)
        if not path.exists():
            return 0
        last = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        last = max(last, int(payload.get("seq") or 0))
                    except Exception:
                        continue
        except Exception:
            return last
        return last

    def _read_legacy_events(self, thread_id: str, *, last_seq: int, limit: int) -> list[ThreadEventEnvelope]:
        path = self.events_path(thread_id)
        if not path.exists():
            return []
        events: list[ThreadEventEnvelope] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                        event = ThreadEventEnvelope.model_validate(payload)
                    except Exception:
                        continue
                    if int(event.seq or 0) > int(last_seq or 0):
                        events.append(event)
                        if len(events) >= limit:
                            break
        except Exception:
            return events
        return events


def _event_run_id(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("run_id", "active_run_id"):
        text = str(data.get(key) or "").strip()
        if text:
            return text
    receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
    text = str(receipt.get("run_id") or "").strip()
    if text:
        return text
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
    return str(meta.get("run_id") or "").strip()


__all__ = ["ThreadEventBroker", "format_sse"]
