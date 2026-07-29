from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.storage import connect_workspace_db, ensure_workspace_ui_events
from catmaster.tools.base import system_root

from .thread_models import ThreadEventEnvelope
from .thread_store import _safe_id


def format_sse(event: Any) -> str:
    data = event.model_dump(mode="json", exclude_none=True)
    return (
        f"id: {event.seq}\n"
        f"event: {event.event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


class ThreadEventBroker:
    """Replayable cross-process thread stream backed by the workspace outbox."""

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
        with connect_workspace_db(self.workspace) as connection:
            ensure_workspace_ui_events(connection)

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
        event_name = str(event or "error")
        created_at = time.time()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO ui_events (
                    event_type, thread_id, graph_id, payload_json, created_at
                ) VALUES (?, ?, '', '{}', ?)
                """,
                (event_name, tid, created_at),
            )
            seq = int(cursor.lastrowid)
            envelope = ThreadEventEnvelope(
                seq=seq,
                event=event_name,
                thread_id=tid,
                message_id=str(message_id or ""),
                status=str(status or ""),
                created_at=created_at,
                data=dict(data or {}),
            )
            connection.execute(
                "UPDATE ui_events SET payload_json = ? WHERE event_id = ?",
                (
                    json.dumps(
                        envelope.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    seq,
                ),
            )
            connection.execute(
                """
                DELETE FROM ui_events
                WHERE graph_id = ''
                  AND thread_id = ?
                  AND event_id NOT IN (
                      SELECT event_id
                      FROM ui_events
                      WHERE graph_id = '' AND thread_id = ?
                      ORDER BY event_id DESC
                      LIMIT 5000
                  )
                """,
                (tid, tid),
            )
        self._persist_observation(tid, envelope)
        with self._condition:
            self._condition.notify_all()
        return envelope

    def replay(self, thread_id: str, *, last_seq: int = 0, limit: int = 1000) -> list[ThreadEventEnvelope]:
        tid = _safe_id(thread_id, label="thread_id")
        capped_limit = min(5000, max(1, int(limit or 1000)))
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM ui_events
                WHERE graph_id = '' AND thread_id = ? AND event_id > ?
                ORDER BY event_id ASC
                LIMIT ?
                """,
                (tid, max(0, int(last_seq or 0)), capped_limit),
            ).fetchall()
        events: list[ThreadEventEnvelope] = []
        for row in rows:
            try:
                events.append(
                    ThreadEventEnvelope.model_validate(
                        json.loads(str(row["payload_json"]))
                    )
                )
            except Exception:
                continue
        return events

    def replay_through(
        self,
        thread_id: str,
        *,
        through_seq: int,
        limit: int = 5000,
    ) -> list[ThreadEventEnvelope]:
        """Return recent events through a cursor for reconnect state recovery."""

        tid = _safe_id(thread_id, label="thread_id")
        capped_limit = min(5000, max(1, int(limit or 5000)))
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM (
                    SELECT event_id, payload_json
                    FROM ui_events
                    WHERE graph_id = '' AND thread_id = ? AND event_id <= ?
                    ORDER BY event_id DESC
                    LIMIT ?
                )
                ORDER BY event_id ASC
                """,
                (tid, max(0, int(through_seq or 0)), capped_limit),
            ).fetchall()
        events: list[ThreadEventEnvelope] = []
        for row in rows:
            try:
                events.append(
                    ThreadEventEnvelope.model_validate(
                        json.loads(str(row["payload_json"]))
                    )
                )
            except Exception:
                continue
        return events

    def latest_seq(self, thread_id: str) -> int:
        tid = _safe_id(thread_id, label="thread_id")
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                """
                SELECT MAX(event_id) AS event_id
                FROM ui_events
                WHERE graph_id = '' AND thread_id = ?
                """,
                (tid,),
            ).fetchone()
        return int(row["event_id"] or 0) if row is not None else 0

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
                # The local condition makes same-process delivery immediate.
                # Periodic polling is still required for other workers.
                self._condition.wait(timeout=min(0.5, remaining))

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
