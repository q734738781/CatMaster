from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from catmaster.tools.base import system_root


def _utc_ts() -> float:
    return time.time()


def _message_to_chat(message: Dict[str, Any]) -> Dict[str, Any]:
    role = str(message.get("role") or "assistant")
    content = str(message.get("content") or "").strip()
    return {
        "role": role,
        "content": content,
        "kind": str(message.get("kind") or "chat"),
        "created_at": message.get("created_at"),
        "source_run_id": str(message.get("source_run_id") or ""),
    }


def _is_conversation_message(message: Dict[str, Any]) -> bool:
    kind = str(message.get("kind") or "chat").strip() or "chat"
    return kind in {"chat", "run_result"}


@dataclass
class ChatHistoryPack:
    session_id: str
    summary_text: str
    recent_messages: List[Dict[str, Any]]


class ChatSessionStore:
    def __init__(self, *, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = system_root(self.workspace) / "chat_sessions"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def active_session_path(self) -> Path:
        return self.root / "active_session.txt"

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def session_json_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def messages_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "messages.jsonl"

    def _new_session_id(self) -> str:
        return f"chat_{uuid.uuid4().hex[:12]}"

    def get_active_session_id(self) -> str:
        if self.active_session_path.exists():
            try:
                value = self.active_session_path.read_text(encoding="utf-8").strip()
            except Exception:
                value = ""
            if value:
                self.ensure_session(value)
                return value
        session_id = self._new_session_id()
        self.create_session(session_id=session_id)
        self.set_active_session_id(session_id)
        return session_id

    def set_active_session_id(self, session_id: str) -> str:
        self.ensure_session(session_id)
        self.active_session_path.write_text(session_id, encoding="utf-8")
        return session_id

    def create_session(self, *, session_id: Optional[str] = None) -> str:
        sid = str(session_id or self._new_session_id()).strip()
        session_dir = self.session_dir(sid)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = self.session_json_path(sid)
        if not session_path.exists():
            payload = {
                "session_id": sid,
                "created_at": _utc_ts(),
                "updated_at": _utc_ts(),
                "message_count": 0,
                "summary_text": "",
                "title": "",
            }
            session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        messages_path = self.messages_path(sid)
        if not messages_path.exists():
            messages_path.write_text("", encoding="utf-8")
        return sid

    def ensure_session(self, session_id: str) -> str:
        return self.create_session(session_id=session_id)

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self.session_json_path(session_id)
        if not path.exists():
            self.ensure_session(session_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _write_session(self, session_id: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["session_id"] = session_id
        payload["updated_at"] = _utc_ts()
        self.session_json_path(session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        active_id = self.get_active_session_id()
        for session_dir in sorted(
            [path for path in self.root.iterdir() if path.is_dir()],
            key=lambda item: item.name,
            reverse=True,
        ):
            payload = self.load_session(session_dir.name)
            message_count = int(payload.get("message_count") or 0)
            title = str(payload.get("title") or "").strip()
            summary = str(payload.get("summary_text") or "").strip()
            sessions.append(
                {
                    "session_id": session_dir.name,
                    "title": title or self._default_session_title(session_dir.name, summary=summary),
                    "message_count": message_count,
                    "updated_at": float(payload.get("updated_at") or 0.0),
                    "is_active": session_dir.name == active_id,
                }
            )
        return sessions

    def create_active_session(self) -> str:
        session_id = self.create_session()
        self.set_active_session_id(session_id)
        return session_id

    def list_messages(self, session_id: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        path = self.messages_path(session_id)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        if limit is not None and limit > 0:
            lines = lines[-limit:]
        out: List[Dict[str, Any]] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        kind: str = "chat",
        source_run_id: str = "",
        source_prompt_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        text = str(content or "").strip()
        if not text:
            return None
        sid = self.ensure_session(session_id)
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        payload: Dict[str, Any] = {
            "message_id": message_id,
            "role": str(role or "assistant"),
            "kind": str(kind or "chat"),
            "content": text,
            "created_at": _utc_ts(),
        }
        if source_run_id:
            payload["source_run_id"] = source_run_id
        if source_prompt_id:
            payload["source_prompt_id"] = source_prompt_id
        if isinstance(meta, dict) and meta:
            payload["meta"] = meta
        with self.messages_path(sid).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        session = self.load_session(sid)
        session["message_count"] = int(session.get("message_count") or 0) + 1
        conversation_messages = [item for item in self.list_messages(sid) if _is_conversation_message(item)]
        session["summary_text"] = self._build_summary_text(conversation_messages)
        if not str(session.get("title") or "").strip() and _is_conversation_message(payload):
            session["title"] = self._default_session_title(sid, summary=session["summary_text"], latest_text=text)
        self._write_session(sid, session)
        return message_id

    def ensure_prompt_message(
        self,
        session_id: str,
        *,
        prompt_id: str,
        title: str,
        body: str,
        meta_text: str = "",
        run_id: str = "",
    ) -> Optional[str]:
        pid = str(prompt_id or "").strip()
        if not pid:
            return None
        for message in self.list_messages(session_id):
            if str(message.get("source_prompt_id") or "") == pid:
                return str(message.get("message_id") or "")
        content = str(title or "").strip()
        body_text = str(body or "").strip()
        if body_text:
            content = f"{content}\n\n{body_text}" if content else body_text
        meta = str(meta_text or "").strip()
        if meta:
            content = f"{content}\n\n{meta}" if content else meta
        return self.append_message(
            session_id,
            role="assistant",
            content=content,
            kind="hitl",
            source_run_id=run_id,
            source_prompt_id=pid,
        )

    def chat_messages(self, session_id: str, *, limit: Optional[int] = None) -> List[Dict[str, str]]:
        messages = [item for item in self.list_messages(session_id) if _is_conversation_message(item)]
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        return [_message_to_chat(item) for item in messages]

    def build_history_pack(
        self,
        session_id: str,
        *,
        recent_messages: int = 8,
    ) -> ChatHistoryPack:
        sid = self.ensure_session(session_id)
        session = self.load_session(sid)
        messages = [item for item in self.list_messages(sid) if _is_conversation_message(item)]
        summary_text = self._build_summary_text(messages)
        recent = messages[-recent_messages:] if recent_messages > 0 else []
        return ChatHistoryPack(session_id=sid, summary_text=summary_text, recent_messages=recent)

    @staticmethod
    def _default_session_title(session_id: str, *, summary: str = "", latest_text: str = "") -> str:
        seed = " ".join(str(latest_text or summary or "").split()).strip()
        if seed:
            if len(seed) > 56:
                seed = seed[:53].rstrip() + "..."
            return seed
        return session_id

    @staticmethod
    def _build_summary_text(messages: List[Dict[str, Any]], *, keep_recent: int = 8, max_chars: int = 2400) -> str:
        if len(messages) <= keep_recent:
            return ""
        earlier = messages[:-keep_recent]
        if not earlier:
            return ""
        lines = ["Earlier chat session summary:"]
        for item in earlier[-12:]:
            role = str(item.get("role") or "assistant").capitalize()
            kind = str(item.get("kind") or "chat")
            content = " ".join(str(item.get("content") or "").split())
            if len(content) > 220:
                content = content[:217] + "..."
            prefix = f"{role}"
            if kind != "chat":
                prefix += f" ({kind})"
            lines.append(f"- {prefix}: {content}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 20].rstrip() + "\n...[truncated]"
        return text


__all__ = ["ChatHistoryPack", "ChatSessionStore"]
