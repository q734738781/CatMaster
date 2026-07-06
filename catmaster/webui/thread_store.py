from __future__ import annotations

import json
import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from catmaster.tools.base import system_root

from .thread_models import MessagePart, ThreadMessage, ThreadRecord, ThreadStatus, utc_ts

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{3,120}$")


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


class ThreadStore:
    """Workspace-scoped JSON-first thread persistence."""

    def __init__(self, *, workspace: Path | str, workspace_id: str = "") -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_id = str(workspace_id or self.workspace.name).strip() or self.workspace.name
        self.root = system_root(self.workspace) / "threads"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migrate_legacy_chat_sessions()

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
            with self.messages_path(message.thread_id).open("a", encoding="utf-8") as handle:
                handle.write(_dump_json(payload) + "\n")
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
        path = self.messages_path(thread_id)
        if not path.exists():
            return []
        out: list[ThreadMessage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    out.append(ThreadMessage.model_validate(payload))
                except Exception:
                    continue
        return out

    def get_message(self, thread_id: str, message_id: str) -> ThreadMessage | None:
        for message in self.list_messages(thread_id):
            if message.id == message_id:
                return message
        return None

    def update_message(self, thread_id: str, message_id: str, **updates: Any) -> ThreadMessage:
        with self._lock:
            messages = self.list_messages(thread_id)
            found: ThreadMessage | None = None
            rewritten: list[dict[str, Any]] = []
            for message in messages:
                if message.id == message_id:
                    payload = _model_dump(message)
                    for key, value in updates.items():
                        if value is not None:
                            payload[key] = value
                    payload["updated_at"] = utc_ts()
                    found = ThreadMessage.model_validate(payload)
                    rewritten.append(_model_dump(found))
                else:
                    rewritten.append(_model_dump(message))
            if found is None:
                raise KeyError(f"Message not found: {message_id}")
            self._rewrite_jsonl(self.messages_path(thread_id), rewritten)
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
        message = self.get_message(thread_id, message_id)
        if message is None:
            raise KeyError(f"Message not found: {message_id}")
        parts: list[MessagePart] = []
        found = False
        for part in message.parts:
            payload = _model_dump(part)
            if payload.get("id") == part_id:
                payload["text"] = str(payload.get("text") or "") + str(delta or "")
                found = True
            parts.append(MessagePart.model_validate(payload))
        if not found:
            parts.append(MessagePart(id=part_id, type="text", text=str(delta or ""), status="streaming"))
        return self.update_message(thread_id, message_id, parts=parts, status="streaming")

    def _rewrite_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
            for row in rows:
                handle.write(_dump_json(row) + "\n")
            temp_name = handle.name
        Path(temp_name).replace(path)

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
