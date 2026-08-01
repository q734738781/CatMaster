from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catmaster.storage import connect_workspace_db
from catmaster.tools.base import system_root

from .thread_models import MessagePart, ThreadMessage, ThreadRecord, ThreadStatus, utc_ts

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{3,120}$")
_THREAD_MESSAGES_SCHEMA_VERSION = 2
_MESSAGE_PAGE_LIMIT = 200
_DELTA_FLUSH_INTERVAL_SECONDS = 0.075
logger = logging.getLogger(__name__)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _safe_id(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_ID_RE.match(text):
        raise ValueError(f"Invalid {label}: {value!r}")
    return text


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _completed_tool_call_ids_from_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in payload.get("parts") or []:
        if not isinstance(part, dict) or str(part.get("type") or "") != "tool-call":
            continue
        meta = part.get("meta") if isinstance(part.get("meta"), dict) else {}
        call_id = str(
            part.get("tool_call_id")
            or meta.get("tool_call_id")
            or ""
        ).strip()
        if not call_id or call_id in seen:
            continue
        status = str(part.get("status") or "").strip().lower()
        has_output = (
            ("output" in part and part.get("output") not in (None, ""))
            or ("output" in meta and meta.get("output") not in (None, ""))
        )
        if status in {"completed", "failed"} or has_output:
            seen.add(call_id)
            out.append(call_id)
    return out


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


@dataclass(frozen=True)
class ThreadMessagePage:
    messages: list[ThreadMessage]
    next_cursor: str
    has_more: bool
    total_count: int


def _message_cursor(row_id: int, message_id: str) -> str:
    raw = f"{int(row_id)}:{message_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_message_cursor(value: str) -> tuple[int, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Message cursor is empty.")
    try:
        padded = text + ("=" * (-len(text) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        row_text, message_id = decoded.split(":", 1)
        row_id = int(row_text)
    except Exception as exc:
        raise ValueError("Message cursor is invalid.") from exc
    if row_id <= 0 or not _SAFE_ID_RE.match(message_id):
        raise ValueError("Message cursor is invalid.")
    return row_id, message_id


class ThreadStore:
    """Workspace-scoped thread persistence.

    Thread descriptors remain small atomic JSON files. Messages live in
    ``metadata/workspace.sqlite`` so newest-page reads and streaming updates do
    not scan or rewrite the full conversation. Existing JSONL logs are imported
    once and retained as untouched migration evidence.
    """

    def __init__(self, *, workspace: Path | str, workspace_id: str = "") -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_id = str(workspace_id or self.workspace.name).strip() or self.workspace.name
        self.root = system_root(self.workspace) / "threads"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pending_delta_messages: dict[tuple[str, str], ThreadMessage] = {}
        self._delta_timers: dict[tuple[str, str], threading.Timer] = {}
        self._init_message_schema()
        self._migrate_legacy_chat_sessions()
        self._import_existing_message_logs()

    def thread_dir(self, thread_id: str) -> Path:
        return self.root / _safe_id(thread_id, label="thread_id")

    def thread_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "thread.json"

    def messages_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "messages.jsonl"

    def interrupts_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "interrupts.jsonl"

    def create_thread(
        self,
        *,
        title: str = "",
        entrypoint: str = "research",
        thread_id: str = "",
        deepagent_thread_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> ThreadRecord:
        with self._lock:
            tid = _safe_id(thread_id, label="thread_id") if thread_id else new_id("thread")
            existing = self.thread_path(tid)
            if existing.exists():
                return self.get_thread(tid)
            now = utc_ts()
            record = ThreadRecord(
                thread_id=tid,
                workspace_id=self.workspace_id,
                deepagent_thread_id=str(deepagent_thread_id or tid),
                title=str(title or "").strip(),
                entrypoint=str(entrypoint or "research").strip() or "research",
                created_at=now,
                updated_at=now,
                meta=dict(meta or {}),
            )
            self.thread_dir(tid).mkdir(parents=True, exist_ok=True)
            self.messages_path(tid).touch(exist_ok=True)
            _atomic_write_json(existing, _model_dump(record))
            return record

    def get_thread(self, thread_id: str) -> ThreadRecord:
        path = self.thread_path(thread_id)
        if not path.exists():
            raise KeyError(f"Thread not found: {thread_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("workspace_id", self.workspace_id)
        return ThreadRecord.model_validate(payload)

    def list_threads(self) -> list[ThreadRecord]:
        with self._lock:
            records: list[ThreadRecord] = []
            for item in self.root.iterdir():
                if not item.is_dir():
                    continue
                path = item / "thread.json"
                if not path.exists():
                    continue
                try:
                    records.append(ThreadRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
                except Exception:
                    continue
            records.sort(key=lambda record: float(record.updated_at or 0.0), reverse=True)
            return records

    def update_thread(self, thread_id: str, **updates: Any) -> ThreadRecord:
        with self._lock:
            record = self.get_thread(thread_id)
            payload = _model_dump(record)
            for key, value in updates.items():
                if value is not None:
                    payload[key] = value
            payload["updated_at"] = utc_ts()
            updated = ThreadRecord.model_validate(payload)
            _atomic_write_json(self.thread_path(thread_id), _model_dump(updated))
            return updated

    def append_message(self, message: ThreadMessage) -> ThreadMessage:
        with self._lock:
            record = self.get_thread(message.thread_id)
            payload = _model_dump(message)
            with connect_workspace_db(self.workspace) as connection:
                connection.execute(
                    """
                    INSERT INTO thread_messages (
                        thread_id, message_id, created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, message_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (
                        message.thread_id,
                        message.id,
                        float(message.created_at),
                        float(message.updated_at),
                        _dump_json(payload),
                    ),
                )
                self._replace_completed_tool_call_rows(
                    connection,
                    thread_id=message.thread_id,
                    message_id=message.id,
                    payload=payload,
                )
            title = record.title
            if not title and message.role == "user":
                title = self._title_from_message(message)
            self.update_thread(
                message.thread_id,
                title=title,
                active_message_id=message.id if message.status in {"streaming", "created"} else record.active_message_id,
            )
            return message

    def list_messages(self, thread_id: str) -> list[ThreadMessage]:
        tid = _safe_id(thread_id, label="thread_id")
        with self._lock:
            self._flush_thread_deltas_locked(tid)
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM thread_messages
                WHERE thread_id = ?
                ORDER BY row_id ASC
                """,
                (tid,),
            ).fetchall()
        return self._messages_from_rows(rows)

    def list_current_turn_messages(self, thread_id: str) -> list[ThreadMessage]:
        """Read the latest user turn without depending on message-part pagination."""

        tid = _safe_id(thread_id, label="thread_id")
        with self._lock:
            self._flush_thread_deltas_locked(tid)
        with connect_workspace_db(self.workspace) as connection:
            latest_user = connection.execute(
                """
                SELECT row_id
                FROM thread_messages
                WHERE thread_id = ? AND message_role = 'user'
                ORDER BY row_id DESC
                LIMIT 1
                """,
                (tid,),
            ).fetchone()
            start_row_id = int(latest_user["row_id"]) if latest_user is not None else 0
            rows = connection.execute(
                """
                SELECT payload_json
                FROM thread_messages
                WHERE thread_id = ? AND row_id >= ?
                ORDER BY row_id ASC
                """,
                (tid, start_row_id),
            ).fetchall()
        return self._messages_from_rows(rows)

    def completed_tool_call_ids(
        self,
        thread_id: str,
        *,
        exclude_message_id: str = "",
    ) -> set[str]:
        """Return the minimal completed-tool projection for a thread.

        This query reads only the dedicated ID index. It must remain independent
        of message payload size because a checkpoint snapshot can replay old
        ToolMessages when a new turn starts.
        """

        tid = _safe_id(thread_id, label="thread_id")
        excluded = (
            _safe_id(exclude_message_id, label="message_id")
            if str(exclude_message_id or "").strip()
            else ""
        )
        with self._lock:
            self._flush_thread_deltas_locked(tid)
        where = "thread_id = ?"
        params: list[Any] = [tid]
        if excluded:
            where += " AND message_id != ?"
            params.append(excluded)
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                f"""
                SELECT tool_call_id
                FROM thread_completed_tool_calls
                WHERE {where}
                ORDER BY tool_call_id
                """,
                tuple(params),
            )
            return {
                str(row["tool_call_id"])
                for row in rows
                if str(row["tool_call_id"] or "").strip()
            }

    def latest_message_id(self, thread_id: str, *, role: str = "") -> str:
        """Return one newest message ID without deserializing its payload."""

        tid = _safe_id(thread_id, label="thread_id")
        normalized_role = str(role or "").strip().lower()
        with self._lock:
            self._flush_thread_deltas_locked(tid)
        where = "thread_id = ?"
        params: list[Any] = [tid]
        if normalized_role:
            where += " AND message_role = ?"
            params.append(normalized_role)
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                f"""
                SELECT message_id
                FROM thread_messages
                WHERE {where}
                ORDER BY row_id DESC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
        return str(row["message_id"] or "") if row is not None else ""

    def latest_assistant_run(self, thread_id: str) -> tuple[str, str]:
        """Return the newest assistant message/run pair from indexed scalars."""

        tid = _safe_id(thread_id, label="thread_id")
        with self._lock:
            self._flush_thread_deltas_locked(tid)
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                """
                SELECT message_id, message_run_id
                FROM thread_messages
                WHERE thread_id = ?
                  AND message_role = 'assistant'
                  AND message_run_id != ''
                ORDER BY row_id DESC
                LIMIT 1
                """,
                (tid,),
            ).fetchone()
        if row is None:
            return "", ""
        return str(row["message_id"] or ""), str(row["message_run_id"] or "")

    def list_messages_page(
        self,
        thread_id: str,
        *,
        before: str = "",
        limit: int = 50,
    ) -> ThreadMessagePage:
        """Read a stable newest-first window and return it in display order."""

        tid = _safe_id(thread_id, label="thread_id")
        capped_limit = min(_MESSAGE_PAGE_LIMIT, max(1, int(limit or 50)))
        with self._lock:
            self._flush_thread_deltas_locked(tid)
        before_row_id: int | None = None
        if before:
            before_row_id, cursor_message_id = _decode_message_cursor(before)
            with connect_workspace_db(self.workspace) as connection:
                cursor_row = connection.execute(
                    """
                    SELECT message_id
                    FROM thread_messages
                    WHERE thread_id = ? AND row_id = ?
                    """,
                    (tid, before_row_id),
                ).fetchone()
            if cursor_row is None or str(cursor_row["message_id"]) != cursor_message_id:
                raise ValueError("Message cursor does not belong to this thread.")

        where = "thread_id = ?"
        params: list[Any] = [tid]
        if before_row_id is not None:
            where += " AND row_id < ?"
            params.append(before_row_id)
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                f"""
                SELECT row_id, message_id, payload_json
                FROM thread_messages
                WHERE {where}
                ORDER BY row_id DESC
                LIMIT ?
                """,
                (*params, capped_limit + 1),
            ).fetchall()
            total_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM thread_messages WHERE thread_id = ?",
                    (tid,),
                ).fetchone()["count"]
            )

        has_more = len(rows) > capped_limit
        visible_rows = list(rows[:capped_limit])
        visible_rows.reverse()
        messages = self._messages_from_rows(visible_rows)
        next_cursor = ""
        if has_more and visible_rows:
            oldest = visible_rows[0]
            next_cursor = _message_cursor(int(oldest["row_id"]), str(oldest["message_id"]))
        return ThreadMessagePage(
            messages=messages,
            next_cursor=next_cursor,
            has_more=has_more,
            total_count=total_count,
        )

    def get_message(self, thread_id: str, message_id: str) -> ThreadMessage | None:
        tid = _safe_id(thread_id, label="thread_id")
        mid = _safe_id(message_id, label="message_id")
        with self._lock:
            self._flush_message_deltas_locked(tid, mid)
            return self._get_message_from_db(tid, mid)

    def _get_message_from_db(self, thread_id: str, message_id: str) -> ThreadMessage | None:
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM thread_messages
                WHERE thread_id = ? AND message_id = ?
                """,
                (thread_id, message_id),
            ).fetchone()
        if row is None:
            return None
        try:
            return ThreadMessage.model_validate(json.loads(str(row["payload_json"])))
        except Exception:
            return None

    def update_message(self, thread_id: str, message_id: str, **updates: Any) -> ThreadMessage:
        with self._lock:
            tid = _safe_id(thread_id, label="thread_id")
            mid = _safe_id(message_id, label="message_id")
            self._flush_message_deltas_locked(tid, mid)
            message = self._get_message_from_db(tid, mid)
            if message is None:
                raise KeyError(f"Message not found: {message_id}")
            payload = _model_dump(message)
            for key, value in updates.items():
                if value is not None:
                    payload[key] = value
            payload["updated_at"] = utc_ts()
            found = ThreadMessage.model_validate(payload)
            found_payload = _model_dump(found)
            with connect_workspace_db(self.workspace) as connection:
                cursor = connection.execute(
                    """
                    UPDATE thread_messages
                    SET updated_at = ?, payload_json = ?
                    WHERE thread_id = ? AND message_id = ?
                    """,
                    (
                        float(found.updated_at),
                        _dump_json(found_payload),
                        found.thread_id,
                        found.id,
                    ),
                )
                if int(cursor.rowcount or 0) != 1:
                    raise KeyError(f"Message not found: {message_id}")
                self._replace_completed_tool_call_rows(
                    connection,
                    thread_id=found.thread_id,
                    message_id=found.id,
                    payload=found_payload,
                )
            self.update_thread(thread_id)
            return found

    def append_part(self, thread_id: str, message_id: str, part: MessagePart | dict[str, Any]) -> ThreadMessage:
        message = self.get_message(thread_id, message_id)
        if message is None:
            raise KeyError(f"Message not found: {message_id}")
        part_model = MessagePart.model_validate(part)
        parts = list(message.parts)
        parts.append(part_model)
        return self.update_message(thread_id, message_id, parts=parts)

    def update_part(self, thread_id: str, message_id: str, part_id: str, **updates: Any) -> ThreadMessage:
        message = self.get_message(thread_id, message_id)
        if message is None:
            raise KeyError(f"Message not found: {message_id}")
        parts: list[MessagePart] = []
        found = False
        for part in message.parts:
            payload = _model_dump(part)
            if payload.get("id") == part_id:
                payload.update({key: value for key, value in updates.items() if value is not None})
                found = True
            parts.append(MessagePart.model_validate(payload))
        if not found:
            raise KeyError(f"Part not found: {part_id}")
        return self.update_message(thread_id, message_id, parts=parts)

    def add_text_delta(self, thread_id: str, message_id: str, part_id: str, delta: str) -> ThreadMessage:
        tid = _safe_id(thread_id, label="thread_id")
        mid = _safe_id(message_id, label="message_id")
        pid = _safe_id(part_id, label="part_id")
        key = (tid, mid)
        with self._lock:
            message = self._pending_delta_messages.get(key)
            if message is None:
                message = self._get_message_from_db(tid, mid)
            if message is None:
                raise KeyError(f"Message not found: {message_id}")
            parts: list[MessagePart] = []
            found = False
            for part in message.parts:
                payload = _model_dump(part)
                if payload.get("id") == pid:
                    payload["text"] = str(payload.get("text") or "") + str(delta or "")
                    payload["status"] = "streaming"
                    found = True
                parts.append(MessagePart.model_validate(payload))
            if not found:
                parts.append(MessagePart(id=pid, type="text", text=str(delta or ""), status="streaming"))
            payload = _model_dump(message)
            payload["parts"] = [_model_dump(part) for part in parts]
            payload["status"] = "streaming"
            payload["updated_at"] = utc_ts()
            updated = ThreadMessage.model_validate(payload)
            self._pending_delta_messages[key] = updated
            if key not in self._delta_timers:
                timer = threading.Timer(
                    _DELTA_FLUSH_INTERVAL_SECONDS,
                    self._flush_message_deltas,
                    args=(tid, mid),
                )
                timer.daemon = True
                self._delta_timers[key] = timer
                timer.start()
            return updated

    def _flush_message_deltas(self, thread_id: str, message_id: str) -> None:
        try:
            with self._lock:
                self._flush_message_deltas_locked(thread_id, message_id)
        except Exception:
            logger.exception("Failed to flush buffered thread text deltas.")

    def _flush_message_deltas_locked(self, thread_id: str, message_id: str) -> None:
        key = (thread_id, message_id)
        timer = self._delta_timers.pop(key, None)
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        message = self._pending_delta_messages.pop(key, None)
        if message is None:
            return
        payload = _model_dump(message)
        with connect_workspace_db(self.workspace) as connection:
            cursor = connection.execute(
                """
                UPDATE thread_messages
                SET updated_at = ?, payload_json = ?
                WHERE thread_id = ? AND message_id = ?
                """,
                (
                    float(message.updated_at),
                    _dump_json(payload),
                    thread_id,
                    message_id,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise KeyError(f"Message not found: {message_id}")
            self._replace_completed_tool_call_rows(
                connection,
                thread_id=thread_id,
                message_id=message_id,
                payload=payload,
            )

    def _flush_thread_deltas_locked(self, thread_id: str) -> None:
        message_ids = [
            message_id
            for pending_thread_id, message_id in tuple(self._pending_delta_messages)
            if pending_thread_id == thread_id
        ]
        for message_id in message_ids:
            self._flush_message_deltas_locked(thread_id, message_id)

    def _rewrite_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
            for row in rows:
                handle.write(_dump_json(row) + "\n")
            temp_name = handle.name
        Path(temp_name).replace(path)

    @staticmethod
    def _replace_completed_tool_call_rows(
        connection: Any,
        *,
        thread_id: str,
        message_id: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            DELETE FROM thread_completed_tool_calls
            WHERE thread_id = ? AND message_id = ?
            """,
            (thread_id, message_id),
        )
        call_ids = _completed_tool_call_ids_from_payload(payload)
        if call_ids:
            connection.executemany(
                """
                INSERT OR IGNORE INTO thread_completed_tool_calls(
                    thread_id, message_id, tool_call_id
                ) VALUES (?, ?, ?)
                """,
                [(thread_id, message_id, call_id) for call_id in call_ids],
            )

    @staticmethod
    def _messages_from_rows(rows: list[Any]) -> list[ThreadMessage]:
        out: list[ThreadMessage] = []
        for row in rows:
            try:
                out.append(ThreadMessage.model_validate(json.loads(str(row["payload_json"]))))
            except Exception:
                continue
        return out

    def _init_message_schema(self) -> None:
        with connect_workspace_db(self.workspace) as connection:
            version_row = connection.execute(
                """
                SELECT version
                FROM schema_migrations
                WHERE component = 'thread_messages'
                """
            ).fetchone()
            previous_version = int(version_row["version"]) if version_row is not None else 0
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_messages (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(thread_id, message_id)
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_xinfo(thread_messages)"
                ).fetchall()
            }
            if "message_role" not in columns:
                connection.execute(
                    """
                    ALTER TABLE thread_messages
                    ADD COLUMN message_role TEXT
                    GENERATED ALWAYS AS (
                        COALESCE(json_extract(payload_json, '$.role'), '')
                    ) VIRTUAL
                    """
                )
            if "message_run_id" not in columns:
                connection.execute(
                    """
                    ALTER TABLE thread_messages
                    ADD COLUMN message_run_id TEXT
                    GENERATED ALWAYS AS (
                        COALESCE(json_extract(payload_json, '$.meta.run_id'), '')
                    ) VIRTUAL
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_thread_messages_page
                ON thread_messages(thread_id, row_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_thread_messages_role_newest
                ON thread_messages(thread_id, message_role, row_id DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_thread_messages_assistant_run
                ON thread_messages(
                    thread_id, message_role, message_run_id, row_id DESC
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_completed_tool_calls (
                    thread_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    PRIMARY KEY(thread_id, message_id, tool_call_id),
                    FOREIGN KEY(thread_id, message_id)
                        REFERENCES thread_messages(thread_id, message_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_thread_completed_tool_calls_thread
                ON thread_completed_tool_calls(thread_id, tool_call_id)
                """
            )
            if previous_version < 2:
                connection.execute("DELETE FROM thread_completed_tool_calls")
                rows = connection.execute(
                    """
                    SELECT thread_id, message_id, payload_json
                    FROM thread_messages
                    ORDER BY row_id
                    """
                )
                for row in rows:
                    try:
                        payload = json.loads(str(row["payload_json"]))
                    except Exception:
                        continue
                    self._replace_completed_tool_call_rows(
                        connection,
                        thread_id=str(row["thread_id"]),
                        message_id=str(row["message_id"]),
                        payload=payload if isinstance(payload, dict) else {},
                    )
            connection.execute(
                """
                INSERT INTO schema_migrations(component, version)
                VALUES ('thread_messages', ?)
                ON CONFLICT(component) DO UPDATE SET version=excluded.version
                """,
                (_THREAD_MESSAGES_SCHEMA_VERSION,),
            )

    def _import_existing_message_logs(self) -> None:
        """Import legacy per-thread JSONL once without mutating the source logs."""

        with self._lock:
            for item in self.root.iterdir():
                if not item.is_dir():
                    continue
                path = item / "messages.jsonl"
                if not path.is_file() or path.stat().st_size <= 0:
                    continue
                thread_id = item.name
                try:
                    _safe_id(thread_id, label="thread_id")
                except ValueError:
                    continue
                with connect_workspace_db(self.workspace) as connection:
                    existing = int(
                        connection.execute(
                            "SELECT COUNT(*) AS count FROM thread_messages WHERE thread_id = ?",
                            (thread_id,),
                        ).fetchone()["count"]
                    )
                    if existing:
                        continue
                    rows: list[tuple[Any, ...]] = []
                    message_payloads: list[dict[str, Any]] = []
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            text = line.strip()
                            if not text:
                                continue
                            try:
                                message = ThreadMessage.model_validate(json.loads(text))
                            except Exception:
                                continue
                            payload = _model_dump(message)
                            message_payloads.append(payload)
                            rows.append(
                                (
                                    message.thread_id,
                                    message.id,
                                    float(message.created_at),
                                    float(message.updated_at),
                                    _dump_json(payload),
                                )
                            )
                    if rows:
                        connection.executemany(
                            """
                            INSERT OR IGNORE INTO thread_messages (
                                thread_id, message_id, created_at, updated_at, payload_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            rows,
                        )
                        for payload in message_payloads:
                            self._replace_completed_tool_call_rows(
                                connection,
                                thread_id=str(payload.get("thread_id") or ""),
                                message_id=str(payload.get("id") or ""),
                                payload=payload,
                            )

    @staticmethod
    def _title_from_message(message: ThreadMessage) -> str:
        text = ""
        for part in message.parts:
            if part.type == "text":
                text = str(getattr(part, "text", "") or "").strip()
                if text:
                    break
        if len(text) > 64:
            text = text[:61].rstrip() + "..."
        return text or message.id

    def _migrate_legacy_chat_sessions(self) -> None:
        marker = self.root / ".chat_sessions_migrated"
        if marker.exists():
            return
        chat_root = system_root(self.workspace) / "chat_sessions"
        if not chat_root.exists() or not chat_root.is_dir():
            marker.write_text("no legacy chat sessions\n", encoding="utf-8")
            return
        with self._lock:
            for session_dir in sorted(path for path in chat_root.iterdir() if path.is_dir()):
                session_id = session_dir.name
                thread_id = f"thread_{session_id}"
                if self.thread_path(thread_id).exists():
                    continue
                session_payload = self._read_json(session_dir / "session.json")
                title = str(session_payload.get("title") or "").strip()
                record = self.create_thread(
                    title=title,
                    entrypoint="research",
                    thread_id=thread_id,
                    deepagent_thread_id=thread_id,
                    meta={"legacy_chat_session_id": session_id},
                )
                rows: list[dict[str, Any]] = []
                messages_path = session_dir / "messages.jsonl"
                if messages_path.exists():
                    for raw in messages_path.read_text(encoding="utf-8").splitlines():
                        try:
                            item = json.loads(raw)
                        except Exception:
                            continue
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role") or "assistant").strip().lower()
                        if role not in {"user", "assistant"}:
                            role = "assistant"
                        content = str(item.get("content") or "").strip()
                        if not content:
                            continue
                        msg_id = str(item.get("message_id") or new_id("msg"))
                        message = ThreadMessage(
                            id=msg_id,
                            thread_id=record.thread_id,
                            role=role,  # type: ignore[arg-type]
                            status="completed",
                            created_at=float(item.get("created_at") or utc_ts()),
                            updated_at=float(item.get("created_at") or utc_ts()),
                            parts=[MessagePart(id=new_id("part_text"), type="text", text=content, status="completed")],
                            meta={
                                "legacy_kind": str(item.get("kind") or "chat"),
                                "run_id": str(item.get("source_run_id") or ""),
                            },
                        )
                        rows.append(_model_dump(message))
                self._rewrite_jsonl(self.messages_path(record.thread_id), rows)
                self.update_thread(record.thread_id, status=ThreadStatus.IDLE)
            marker.write_text("migrated\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}


__all__ = ["ThreadStore", "new_id"]
