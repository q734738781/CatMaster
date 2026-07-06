#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite-backed observability records for CatMaster runs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import hashlib
import json
import sqlite3
import time

from catmaster.runtime import observation_events as obs_events


OBSERVABILITY_DB_NAME = "observability.sqlite"
OBSERVABILITY_SCHEMA_VERSION = 2
LEGACY_TRACE_SOURCES = frozenset({"event_trace.jsonl", "tool_trace.jsonl", "patch_trace.jsonl"})


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, default=str, sort_keys=True)


def _json_loads(value: str) -> Any:
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return default


def _usage_reasoning_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    candidates = [
        usage.get("reasoning_tokens"),
        usage.get("reasoning"),
    ]
    for details_key in ("output_token_details", "completion_tokens_details", "output_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, dict):
            candidates.extend([details.get("reasoning"), details.get("reasoning_tokens")])
    return max(_to_int(value, 0) for value in candidates)


def _compact_text(value: Any, limit: int = 420) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _status_failed(value: Any, *, error: Any = None) -> bool:
    if error:
        return True
    status = str(value or "").strip().lower()
    if not status:
        return False
    return status not in {
        "ok",
        "done",
        "success",
        "completed",
        "running",
        "started",
        "pending",
        "queued",
    }


def _event_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _extract_agent_name(payload: Dict[str, Any]) -> str:
    for key in ("agent_name", "agent", "subagent", "lc_agent_name"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def _extract_callback_id(payload: Dict[str, Any], name: str) -> str:
    for key in ("callback_run_id", "llm_call_id", "toolcall_id", "call_id"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    if name.startswith("TOOL_"):
        return str(payload.get("tool_call_id") or "").strip()
    return ""


def _extract_parent_id(payload: Dict[str, Any]) -> str:
    for key in ("parent_callback_run_id", "parent_run_id", "parent_call_id", "parent_id"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def _event_summary(name: str, payload: Dict[str, Any]) -> str:
    if name == obs_events.LLM_CALL_END:
        tools = payload.get("tool_calls") if isinstance(payload.get("tool_calls"), list) else []
        bits = [
            str(payload.get("model") or "").strip(),
            _compact_text(payload.get("reasoning_text") or payload.get("text_preview") or "", 180),
        ]
        if tools:
            bits.append("tools=" + ", ".join(str(item) for item in tools[:5]))
        return " | ".join(item for item in bits if item)
    if name in {obs_events.TOOL_CALL_START, obs_events.TOOL_CALL_END, obs_events.TOOL_RAW_INPUT, obs_events.TOOL_RAW_OUTPUT}:
        return " | ".join(
            item
            for item in [
                str(payload.get("tool") or payload.get("tool_name") or "").strip(),
                str(payload.get("status") or "").strip(),
                _compact_text(payload.get("error") or payload.get("highlights") or payload.get("params_compact") or "", 180),
            ]
            if item
        )
    if name == obs_events.RUN_STATE_CHANGE:
        return " | ".join(
            item
            for item in [
                str(payload.get("status") or "").strip(),
                str(payload.get("phase") or "").strip(),
                _compact_text(payload.get("text_preview") or payload.get("summary") or "", 180),
            ]
            if item
        )
    return _compact_text(
        payload.get("summary_snippet")
        or payload.get("text_preview")
        or payload.get("goal")
        or payload.get("status")
        or payload.get("error")
        or "",
        220,
    )


@dataclass(frozen=True)
class ObservabilityStore:
    """Run-local SQLite store for events, raw callback data, and state changes."""

    run_dir: Path

    @property
    def db_path(self) -> Path:
        return Path(self.run_dir).expanduser().resolve() / OBSERVABILITY_DB_NAME

    def db_exists(self) -> bool:
        return self.db_path.exists()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _ensure_schema(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_ui_event(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        payload = _event_payload(event)
        self.record_event(
            source="ui_event",
            channel="ui",
            name=str(event.get("name") or "EVENT"),
            category=str(event.get("category") or ""),
            ts=_to_float(event.get("ts"), time.time()),
            seq=_to_int(event.get("seq"), 0) or None,
            run_id=str(event.get("run_id") or ""),
            task_id=str(event.get("task_id") or ""),
            step_id=event.get("step_id") if isinstance(event.get("step_id"), int) else None,
            thread_id=str(event.get("thread_id") or payload.get("thread_id") or ""),
            message_id=str(event.get("message_id") or payload.get("message_id") or ""),
            part_id=str(event.get("part_id") or payload.get("part_id") or ""),
            payload=payload,
        )

    def record_trace_record(self, trace_name: str, record: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            return
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else dict(record)
        name = str(record.get("event") or "").strip()
        if not name:
            if "tool_name" in record:
                name = "TOOL_TRACE"
            elif trace_name:
                name = Path(trace_name).stem.upper()
            else:
                name = "TRACE_RECORD"
        self.record_event(
            source=str(trace_name or "trace"),
            channel="legacy_trace",
            name=name,
            category="trace",
            ts=_coerce_trace_ts(record.get("ts")),
            seq=None,
            run_id=str(record.get("run_id") or payload.get("run_id") or ""),
            task_id=str(record.get("task_id") or payload.get("task_id") or ""),
            step_id=record.get("step_id") if isinstance(record.get("step_id"), int) else None,
            thread_id=str(record.get("thread_id") or payload.get("thread_id") or ""),
            message_id=str(record.get("message_id") or payload.get("message_id") or ""),
            part_id=str(record.get("part_id") or payload.get("part_id") or ""),
            payload=payload,
        )

    def record_raw_callback(
        self,
        name: str,
        *,
        payload: Dict[str, Any],
        category: str,
        ts: Optional[float] = None,
        run_id: str = "",
        task_id: str = "",
        step_id: Optional[int] = None,
    ) -> None:
        self.record_event(
            source="langchain_callback",
            channel="callback",
            name=str(name or "CALLBACK_RECORD"),
            category=str(category or "callback"),
            ts=float(ts or time.time()),
            seq=None,
            run_id=run_id,
            task_id=task_id,
            step_id=step_id,
            thread_id=str(payload.get("thread_id") or ""),
            message_id=str(payload.get("message_id") or ""),
            part_id=str(payload.get("part_id") or ""),
            payload=payload,
        )

    def record_run_state(self, payload: Dict[str, Any], *, reason: str = "write") -> None:
        if not isinstance(payload, dict):
            return
        normalized = _json_safe(payload)
        state_hash = hashlib.sha256(_json_dumps(normalized).encode("utf-8")).hexdigest()[:16]
        state_payload = {
            "reason": str(reason or "write"),
            "state_hash": state_hash,
            **(normalized if isinstance(normalized, dict) else {}),
        }
        self.record_event(
            source="run_state",
            channel="state",
            name=obs_events.RUN_STATE_CHANGE,
            category="state",
            ts=time.time(),
            seq=None,
            run_id=str(Path(self.run_dir).name),
            task_id="",
            step_id=None,
            thread_id=str(state_payload.get("thread_id") or ""),
            message_id=str(state_payload.get("message_id") or ""),
            part_id=str(state_payload.get("part_id") or ""),
            payload=state_payload,
        )

    def record_chat_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        kind: str = "chat",
        source_run_id: str = "",
        source_prompt_id: str = "",
        message_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = str(content or "")
        if not text.strip():
            return
        payload = {
            "session_id": str(session_id or ""),
            "message_id": str(message_id or ""),
            "role": str(role or ""),
            "kind": str(kind or "chat"),
            "content": text,
            "source_run_id": str(source_run_id or ""),
            "source_prompt_id": str(source_prompt_id or ""),
            "meta": meta if isinstance(meta, dict) else {},
        }
        self.record_event(
            source="chat_session",
            channel="chat",
            name=obs_events.CHAT_MESSAGE,
            category="chat",
            ts=time.time(),
            seq=None,
            run_id=str(source_run_id or Path(self.run_dir).name),
            task_id="",
            step_id=None,
            thread_id=str(meta.get("thread_id") or "") if isinstance(meta, dict) else "",
            message_id=str(message_id or ""),
            part_id="",
            payload=payload,
        )

    def record_thread_event(self, event: Any) -> None:
        """Persist a WebUI thread stream event in the canonical observation table."""
        if hasattr(event, "model_dump"):
            payload = event.model_dump(mode="json")
        elif isinstance(event, dict):
            payload = dict(event)
        else:
            return
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        self.record_event(
            source="thread_event",
            channel="thread",
            name=str(payload.get("event") or "thread.event"),
            category="thread",
            ts=_to_float(payload.get("created_at"), time.time()),
            seq=_to_int(payload.get("seq"), 0) or None,
            run_id=str(data.get("run_id") or payload.get("run_id") or Path(self.run_dir).name),
            task_id=str(data.get("task_id") or ""),
            step_id=None,
            thread_id=str(payload.get("thread_id") or data.get("thread_id") or ""),
            message_id=str(payload.get("message_id") or data.get("message_id") or ""),
            part_id=str(data.get("part_id") or data.get("text_part_id") or ""),
            payload=payload,
        )

    def record_event(
        self,
        *,
        source: str,
        channel: str = "",
        name: str,
        category: str,
        ts: float,
        seq: Optional[int],
        run_id: str,
        task_id: str,
        step_id: Optional[int],
        payload: Dict[str, Any],
        thread_id: str = "",
        message_id: str = "",
        part_id: str = "",
    ) -> None:
        self._insert_event(
            source=source,
            channel=channel,
            name=name,
            category=category,
            ts=ts,
            seq=seq,
            run_id=run_id,
            task_id=task_id,
            step_id=step_id,
            thread_id=thread_id,
            message_id=message_id,
            part_id=part_id,
            payload=payload,
        )

    def last_ui_event_seq(self) -> int:
        if not self.db_exists():
            return 0
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(seq) AS max_seq
                FROM observation_events
                WHERE source = 'ui_event' AND seq IS NOT NULL
                """
            ).fetchone()
        if row is None:
            return 0
        return _to_int(row["max_seq"], 0)

    def read_ui_events_page(
        self,
        *,
        limit: int = 200,
        before_seq: int = 0,
        after_seq: int = 0,
    ) -> Optional[Dict[str, Any]]:
        if not self.db_exists():
            return None
        capped_limit = min(1000, max(1, int(limit or 200)))
        query_limit = capped_limit + 1
        if after_seq > 0:
            sql = """
                SELECT * FROM observation_events
                WHERE source = 'ui_event' AND seq IS NOT NULL AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
            """
            params: tuple[Any, ...] = (int(after_seq), query_limit)
            reverse_rows = False
        elif before_seq > 0:
            sql = """
                SELECT * FROM observation_events
                WHERE source = 'ui_event' AND seq IS NOT NULL AND seq < ?
                ORDER BY seq DESC
                LIMIT ?
            """
            params = (int(before_seq), query_limit)
            reverse_rows = True
        else:
            sql = """
                SELECT * FROM observation_events
                WHERE source = 'ui_event' AND seq IS NOT NULL
                ORDER BY seq DESC
                LIMIT ?
            """
            params = (query_limit,)
            reverse_rows = True
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        has_more = len(rows) > capped_limit
        page_rows = rows[:capped_limit]
        if reverse_rows:
            page_rows = list(reversed(page_rows))
        events = [_row_to_ui_event(row) for row in page_rows]
        return {
            "events": events,
            "has_more": has_more,
            "min_seq": int(events[0].get("seq") or 0) if events else 0,
            "max_seq": int(events[-1].get("seq") or 0) if events else 0,
        }

    def read_thread_events_page(self, thread_id: str, *, last_seq: int = 0, limit: int = 1000) -> list[Dict[str, Any]]:
        if not self.db_exists():
            return []
        capped_limit = min(5000, max(1, int(limit or 1000)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM observation_events
                WHERE channel = 'thread' AND thread_id = ? AND seq IS NOT NULL AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (str(thread_id or ""), int(last_seq or 0), capped_limit),
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def read_thread_events(self, thread_id: str, *, last_seq: int = 0, limit: int = 1000) -> list[Dict[str, Any]]:
        return self.read_thread_events_page(thread_id, last_seq=last_seq, limit=limit)

    def read_events_page(
        self,
        *,
        limit: int = 400,
        before_id: int = 0,
        after_id: int = 0,
        channel: str = "",
        category: str = "",
        names: Optional[Iterable[str]] = None,
        run_id: str = "",
        thread_id: str = "",
        agent_name: str = "",
        tool: str = "",
        include_legacy_trace_records: bool = False,
    ) -> Dict[str, Any]:
        if not self.db_exists():
            return {"events": [], "has_more": False, "min_id": 0, "max_id": 0}
        capped_limit = min(5000, max(1, int(limit or 400)))
        clauses: list[str] = []
        params: list[Any] = []
        if not include_legacy_trace_records:
            placeholders = ", ".join("?" for _ in LEGACY_TRACE_SOURCES)
            clauses.append(f"source NOT IN ({placeholders})")
            params.extend(_legacy_trace_source_values())
        if channel:
            clauses.append("channel = ?")
            params.append(str(channel))
        if category:
            clauses.append("category = ?")
            params.append(str(category))
        clean_names = [str(name).strip() for name in list(names or []) if str(name).strip()]
        if clean_names:
            placeholders = ", ".join("?" for _ in clean_names)
            clauses.append(f"name IN ({placeholders})")
            params.extend(clean_names)
        if run_id:
            clauses.append("run_id = ?")
            params.append(str(run_id))
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(str(thread_id))
        if agent_name:
            clauses.append("agent_name = ?")
            params.append(str(agent_name))
        if tool:
            clauses.append("tool = ?")
            params.append(str(tool))
        if after_id > 0:
            clauses.append("id > ?")
            params.append(int(after_id))
            order_sql = "ORDER BY id ASC"
            reverse_rows = False
        else:
            if before_id > 0:
                clauses.append("id < ?")
                params.append(int(before_id))
            order_sql = "ORDER BY id DESC"
            reverse_rows = True
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM observation_events
                {where_sql}
                {order_sql}
                LIMIT ?
                """,
                (*params, capped_limit + 1),
            ).fetchall()
        has_more = len(rows) > capped_limit
        page_rows = rows[:capped_limit]
        if reverse_rows:
            page_rows = list(reversed(page_rows))
        events = [_row_to_event(row) for row in page_rows]
        return {
            "events": events,
            "has_more": has_more,
            "min_id": int(events[0].get("id") or 0) if events else 0,
            "max_id": int(events[-1].get("id") or 0) if events else 0,
        }

    def latest_thread_event_seq(self, thread_id: str) -> int:
        if not self.db_exists():
            return 0
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(seq) AS max_seq
                FROM observation_events
                WHERE channel = 'thread' AND thread_id = ? AND seq IS NOT NULL
                """,
                (str(thread_id or ""),),
            ).fetchone()
        if row is None:
            return 0
        return _to_int(row["max_seq"], 0)

    def list_tool_names(
        self,
        *,
        limit: int = 64,
        event_names: Optional[Iterable[str]] = None,
        include_legacy_trace_records: bool = False,
    ) -> list[str]:
        if not self.db_exists():
            return []
        capped_limit = min(500, max(1, int(limit or 64)))
        source_filter = _source_filter_clause(prefix="AND", include_legacy_trace_records=include_legacy_trace_records)
        params: list[Any] = list(_legacy_trace_source_values() if not include_legacy_trace_records else ())
        clean_names = [str(name).strip() for name in list(event_names or []) if str(name).strip()]
        name_filter = ""
        if clean_names:
            placeholders = ", ".join("?" for _ in clean_names)
            name_filter = f"AND name IN ({placeholders})"
            params.extend(clean_names)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT tool
                FROM observation_events
                WHERE tool IS NOT NULL AND tool != ''
                {source_filter}
                {name_filter}
                ORDER BY id ASC
                LIMIT 5000
                """,
                tuple(params),
            ).fetchall()
        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row["tool"] or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
            if len(out) >= capped_limit:
                break
        return out

    def read_snapshot(self, *, limit: int = 400, include_legacy_trace_records: bool = False) -> Dict[str, Any]:
        capped = min(2000, max(1, int(limit or 400)))
        source_filter_sql, source_filter_params = _legacy_trace_source_filter(include_legacy_trace_records)
        with self._connect() as conn:
            total = _scalar_int(
                conn,
                f"SELECT COUNT(*) FROM observation_events {source_filter_sql}",
                source_filter_params,
            )
            rows = [
                _row_to_event(row)
                for row in conn.execute(
                    f"""
                    SELECT * FROM observation_events
                    {source_filter_sql}
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (*source_filter_params, capped),
                ).fetchall()
            ]
            rows.reverse()
            all_call_rows = [
                _row_to_event(row)
                for row in conn.execute(
                    f"""
                    SELECT * FROM observation_events
                    WHERE name IN (
                        'LLM_CALL_START', 'LLM_CALL_END', 'LLM_ERROR',
                        'TOOL_CALL_START', 'TOOL_CALL_END',
                        'LLM_RAW_REQUEST', 'LLM_RAW_RESPONSE',
                        'TOOL_RAW_INPUT', 'TOOL_RAW_OUTPUT'
                    )
                    {_source_filter_clause(prefix="AND", include_legacy_trace_records=include_legacy_trace_records)}
                    ORDER BY id ASC
                    LIMIT 5000
                    """,
                    source_filter_params,
                ).fetchall()
            ]
            llm_rows = [
                _row_to_event(row)
                for row in conn.execute(
                    f"""
                    SELECT * FROM observation_events
                    WHERE name IN ('LLM_CALL_END', 'LLM_ERROR')
                    {_source_filter_clause(prefix="AND", include_legacy_trace_records=include_legacy_trace_records)}
                    ORDER BY id ASC
                    LIMIT 5000
                    """,
                    source_filter_params,
                ).fetchall()
            ]
            tool_rows = [
                _row_to_event(row)
                for row in conn.execute(
                    f"""
                    SELECT * FROM observation_events
                    WHERE name IN ('TOOL_CALL_END', 'TOOL_RAW_OUTPUT')
                    {_source_filter_clause(prefix="AND", include_legacy_trace_records=include_legacy_trace_records)}
                    ORDER BY id ASC
                    LIMIT 5000
                    """,
                    source_filter_params,
                ).fetchall()
            ]
            decision_rows = [
                _row_to_event(row)
                for row in conn.execute(
                    f"""
                    SELECT * FROM observation_events
                    WHERE name IN ('LLM_CALL_END', 'TASK_DECISION')
                    {_source_filter_clause(prefix="AND", include_legacy_trace_records=include_legacy_trace_records)}
                    ORDER BY id DESC
                    LIMIT 120
                    """,
                    source_filter_params,
                ).fetchall()
            ]
            decision_rows.reverse()
            task_rows = [
                _row_to_event(row)
                for row in conn.execute(
                    f"""
                    SELECT * FROM observation_events
                    WHERE name IN (
                        'TASKS_COMPILED', 'TASK_START', 'TASK_SUMMARY', 'TASK_END',
                        'RUN_START', 'RUN_END', 'RUN_PAUSED', 'RUN_STATE_CHANGE',
                        'TOOL_CALL_START', 'TOOL_RAW_INPUT'
                    )
                    {_source_filter_clause(prefix="AND", include_legacy_trace_records=include_legacy_trace_records)}
                    ORDER BY id ASC
                    LIMIT 5000
                    """,
                    source_filter_params,
                ).fetchall()
            ]
            by_name = {
                str(row["name"]): int(row["n"])
                for row in conn.execute(
                    f"SELECT name, COUNT(*) AS n FROM observation_events {source_filter_sql} GROUP BY name ORDER BY n DESC",
                    source_filter_params,
                ).fetchall()
            }
            bounds = conn.execute(
                f"SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM observation_events {source_filter_sql}",
                source_filter_params,
            ).fetchone()
        metrics = _build_metrics(
            total=total,
            by_name=by_name,
            first_ts=_to_float(bounds["first_ts"]) if bounds else 0.0,
            last_ts=_to_float(bounds["last_ts"]) if bounds else 0.0,
            llm_rows=llm_rows,
            tool_rows=tool_rows,
        )
        return {
            "db_path": str(self.db_path),
            "metrics": metrics,
            "events": rows,
            "trace_tree": _build_trace_tree(all_call_rows),
            "decisions": _build_decisions(decision_rows),
            "task_state": _build_task_state(task_rows),
            "raw_logs": {
                "events": rows,
                "total_events": total,
            },
        }

    def read_run_snapshot(self, *, limit: int = 400, include_legacy_trace_records: bool = False) -> Dict[str, Any]:
        return self.read_snapshot(limit=limit, include_legacy_trace_records=include_legacy_trace_records)

    def read_metrics(self, *, include_legacy_trace_records: bool = False) -> Dict[str, Any]:
        return self.read_snapshot(limit=1, include_legacy_trace_records=include_legacy_trace_records).get("metrics", {})

    def _insert_event(
        self,
        *,
        source: str,
        channel: str,
        name: str,
        category: str,
        ts: float,
        seq: Optional[int],
        run_id: str,
        task_id: str,
        step_id: Optional[int],
        thread_id: str,
        message_id: str,
        part_id: str,
        payload: Dict[str, Any],
    ) -> None:
        payload = payload if isinstance(payload, dict) else {"value": payload}
        callback_id = _extract_callback_id(payload, name)
        parent_id = _extract_parent_id(payload)
        duration_ms = _duration_ms(name, payload)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observation_events (
                    ts, seq, source, channel, category, name,
                    run_id, task_id, step_id, thread_id, message_id, part_id,
                    agent_name, callback_run_id, parent_callback_run_id,
                    node, model, tool, status, duration_ms, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(ts or time.time()),
                    seq,
                    str(source or ""),
                    str(channel or ""),
                    str(category or ""),
                    str(name or "EVENT"),
                    str(run_id or payload.get("run_id") or ""),
                    str(task_id or payload.get("task_id") or ""),
                    step_id,
                    str(thread_id or payload.get("thread_id") or ""),
                    str(message_id or payload.get("message_id") or ""),
                    str(part_id or payload.get("part_id") or ""),
                    _extract_agent_name(payload),
                    callback_id,
                    parent_id,
                    str(payload.get("node") or ""),
                    str(payload.get("model") or ""),
                    str(payload.get("tool") or payload.get("tool_name") or ""),
                    str(payload.get("status") or payload.get("tool_status") or ""),
                    duration_ms,
                    _json_dumps(payload),
                ),
            )

    def import_legacy_jsonl(self, *, include_ui_events: bool = True, include_legacy_trace_records: bool = False) -> int:
        imported = 0
        try:
            if include_ui_events:
                for row in _iter_jsonl(Path(self.run_dir) / "ui_events.jsonl"):
                    self.record_ui_event(row)
                    imported += 1
            if include_legacy_trace_records:
                for trace_name in ("event_trace.jsonl", "tool_trace.jsonl", "patch_trace.jsonl"):
                    for row in _iter_jsonl(Path(self.run_dir) / trace_name):
                        self.record_trace_record(trace_name, row)
                        imported += 1
        except Exception:
            return imported
        return imported


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            seq INTEGER,
            source TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            run_id TEXT,
            task_id TEXT,
            step_id INTEGER,
            thread_id TEXT,
            message_id TEXT,
            part_id TEXT,
            agent_name TEXT,
            callback_run_id TEXT,
            parent_callback_run_id TEXT,
            node TEXT,
            model TEXT,
            tool TEXT,
            status TEXT,
            duration_ms INTEGER,
            payload_json TEXT NOT NULL
        )
        """
    )
    _ensure_columns(
        conn,
        "observation_events",
        {
            "channel": "TEXT NOT NULL DEFAULT ''",
            "thread_id": "TEXT",
            "message_id": "TEXT",
            "part_id": "TEXT",
        },
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observation_ts ON observation_events(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observation_name ON observation_events(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observation_source ON observation_events(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observation_channel_thread_seq ON observation_events(channel, thread_id, seq)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observation_run ON observation_events(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observation_callback ON observation_events(callback_run_id)")
    current_version = _to_int(conn.execute("PRAGMA user_version").fetchone()[0], 0)
    if current_version < OBSERVABILITY_SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {OBSERVABILITY_SCHEMA_VERSION}")


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _legacy_trace_source_values() -> tuple[str, ...]:
    return tuple(sorted(LEGACY_TRACE_SOURCES))


def _legacy_trace_source_filter(include_legacy_trace_records: bool) -> tuple[str, tuple[str, ...]]:
    if include_legacy_trace_records:
        return "", ()
    placeholders = ", ".join("?" for _ in LEGACY_TRACE_SOURCES)
    return f"WHERE source NOT IN ({placeholders})", _legacy_trace_source_values()


def _source_filter_clause(*, prefix: str, include_legacy_trace_records: bool) -> str:
    if include_legacy_trace_records:
        return ""
    placeholders = ", ".join("?" for _ in LEGACY_TRACE_SOURCES)
    return f"{prefix} source NOT IN ({placeholders})"


def _scalar_int(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return 0
    return _to_int(row[0], 0)


def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
    payload = _json_loads(str(row["payload_json"] or "{}"))
    payload = payload if isinstance(payload, dict) else {"value": payload}
    return {
        "id": int(row["id"]),
        "ts": float(row["ts"] or 0.0),
        "seq": row["seq"],
        "source": str(row["source"] or ""),
        "channel": str(row["channel"] or ""),
        "category": str(row["category"] or ""),
        "name": str(row["name"] or ""),
        "run_id": str(row["run_id"] or ""),
        "task_id": str(row["task_id"] or ""),
        "step_id": row["step_id"],
        "thread_id": str(row["thread_id"] or ""),
        "message_id": str(row["message_id"] or ""),
        "part_id": str(row["part_id"] or ""),
        "agent_name": str(row["agent_name"] or ""),
        "callback_run_id": str(row["callback_run_id"] or ""),
        "parent_callback_run_id": str(row["parent_callback_run_id"] or ""),
        "node": str(row["node"] or ""),
        "model": str(row["model"] or ""),
        "tool": str(row["tool"] or ""),
        "status": str(row["status"] or ""),
        "duration_ms": row["duration_ms"],
        "summary": _event_summary(str(row["name"] or ""), payload),
        "payload": payload,
    }


def _row_to_ui_event(row: sqlite3.Row) -> Dict[str, Any]:
    payload = _json_loads(str(row["payload_json"] or "{}"))
    payload = payload if isinstance(payload, dict) else {"value": payload}
    event: Dict[str, Any] = {
        "ts": float(row["ts"] or 0.0),
        "level": "info",
        "category": str(row["category"] or ""),
        "name": str(row["name"] or ""),
        "payload": payload,
        "run_id": str(row["run_id"] or "") or None,
        "task_id": str(row["task_id"] or "") or None,
        "step_id": row["step_id"],
        "thread_id": str(row["thread_id"] or "") or None,
        "message_id": str(row["message_id"] or "") or None,
        "part_id": str(row["part_id"] or "") or None,
        "seq": int(row["seq"] or 0),
    }
    if not event["run_id"]:
        event.pop("run_id", None)
    if not event["task_id"]:
        event.pop("task_id", None)
    if event["step_id"] is None:
        event.pop("step_id", None)
    if not event["thread_id"]:
        event.pop("thread_id", None)
    if not event["message_id"]:
        event.pop("message_id", None)
    if not event["part_id"]:
        event.pop("part_id", None)
    return event


def _duration_ms(name: str, payload: Dict[str, Any]) -> Optional[int]:
    for key in ("elapsed_ms", "duration_ms"):
        value = payload.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return max(0, value)
        try:
            return max(0, int(value))
        except Exception:
            pass
    if name in {"TOOL_RAW_OUTPUT", "TOOL_CALL_END"}:
        started = _to_float(payload.get("started_ts"), 0.0)
        ended = _to_float(payload.get("ended_ts"), 0.0)
        if started > 0 and ended >= started:
            return int((ended - started) * 1000)
    return None


def _coerce_trace_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return time.time()
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []

    def _gen() -> Iterator[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        yield payload
        except Exception:
            return

    return _gen()


def _build_metrics(
    *,
    total: int,
    by_name: Dict[str, int],
    first_ts: float,
    last_ts: float,
    llm_rows: List[Dict[str, Any]],
    tool_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    llm_latency_ms: list[int] = []
    llm_errors = 0
    models: Dict[str, int] = {}
    agents: Dict[str, int] = {}
    counted_llm_rows: list[Dict[str, Any]] = []
    seen_llm_calls: set[str] = set()
    for row in llm_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        call_key = str(row.get("callback_run_id") or payload.get("callback_run_id") or "").strip()
        if not call_key:
            call_key = f"{row.get('name')}:{row.get('id')}"
        if call_key in seen_llm_calls:
            continue
        seen_llm_calls.add(call_key)
        counted_llm_rows.append(row)
    for row in counted_llm_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if row.get("name") == "LLM_ERROR":
            llm_errors += 1
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        input_tokens += _to_int(usage.get("input_tokens"), 0)
        output_tokens += _to_int(usage.get("output_tokens"), 0)
        reasoning_tokens += _usage_reasoning_tokens(usage)
        elapsed = _to_int(payload.get("elapsed_ms"), 0)
        if elapsed > 0:
            llm_latency_ms.append(elapsed)
        model = str(row.get("model") or payload.get("model") or "").strip()
        if model:
            models[model] = models.get(model, 0) + 1
        agent = str(row.get("agent_name") or payload.get("agent_name") or "").strip()
        if agent:
            agents[agent] = agents.get(agent, 0) + 1
    tool_calls = 0
    tool_failures = 0
    tool_latency_ms: list[int] = []
    tools: Dict[str, int] = {}
    seen_tool_calls: set[str] = set()
    for row in tool_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        name = str(row.get("name") or "")
        if name == "TOOL_RAW_OUTPUT" and row.get("status") == "":
            continue
        if name in {"TOOL_CALL_END", "TOOL_RAW_OUTPUT"}:
            call_key = str(row.get("callback_run_id") or payload.get("callback_run_id") or "").strip()
            if not call_key:
                call_key = f"{name}:{row.get('id')}"
            if call_key in seen_tool_calls:
                continue
            seen_tool_calls.add(call_key)
            tool_calls += 1
            if _status_failed(row.get("status") or payload.get("status") or payload.get("tool_status"), error=payload.get("error")):
                tool_failures += 1
            duration = _to_int(row.get("duration_ms"), 0)
            if duration > 0:
                tool_latency_ms.append(duration)
            tool = str(row.get("tool") or payload.get("tool") or payload.get("tool_name") or "").strip()
            if tool:
                tools[tool] = tools.get(tool, 0) + 1
    llm_calls = len([row for row in counted_llm_rows if row.get("name") == "LLM_CALL_END"])
    total_calls = llm_calls + llm_errors + tool_calls
    failed_calls = llm_errors + tool_failures
    return {
        "total_events": total,
        "duration_sec": round(max(0.0, last_ts - first_ts), 3) if first_ts and last_ts else 0.0,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "event_counts": by_name,
        "llm_calls": llm_calls,
        "llm_errors": llm_errors,
        "tool_calls": tool_calls,
        "tool_failures": tool_failures,
        "error_rate": round((failed_calls / total_calls), 4) if total_calls else 0.0,
        "avg_llm_latency_ms": int(sum(llm_latency_ms) / len(llm_latency_ms)) if llm_latency_ms else 0,
        "avg_tool_latency_ms": int(sum(tool_latency_ms) / len(tool_latency_ms)) if tool_latency_ms else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "models": _top_counts(models),
        "agents": _top_counts(agents),
        "tools": _top_counts(tools),
    }


def _top_counts(values: Dict[str, int], *, limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {"name": name, "count": count}
        for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _build_trace_tree(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    order = 0
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        event_name = str(row.get("name") or "")
        callback_id = str(row.get("callback_run_id") or "").strip()
        if not callback_id:
            callback_id = f"{event_name.lower()}:{row.get('id')}"
        node_type = "llm" if "LLM" in event_name else "tool" if "TOOL" in event_name else "event"
        display_name = str(row.get("tool") or row.get("model") or payload.get("tool") or payload.get("tool_name") or payload.get("model") or event_name)
        node = nodes.get(callback_id)
        if node is None:
            order += 1
            node = {
                "id": callback_id,
                "parent_id": str(row.get("parent_callback_run_id") or "").strip(),
                "type": node_type,
                "name": display_name,
                "agent_name": str(row.get("agent_name") or payload.get("agent_name") or ""),
                "status": "running" if event_name.endswith("_START") or event_name.endswith("_INPUT") or event_name == "LLM_RAW_REQUEST" else "",
                "start_ts": float(row.get("ts") or 0.0),
                "end_ts": 0.0,
                "duration_ms": 0,
                "summary": "",
                "order": order,
            }
            nodes[callback_id] = node
        if event_name.endswith("_START") or event_name in {"LLM_RAW_REQUEST", "TOOL_RAW_INPUT"}:
            node["start_ts"] = min(float(node.get("start_ts") or row.get("ts") or 0.0), float(row.get("ts") or 0.0))
            if not node.get("summary"):
                node["summary"] = str(row.get("summary") or "")
        else:
            node["end_ts"] = max(float(node.get("end_ts") or 0.0), float(row.get("ts") or 0.0))
            status = str(row.get("status") or payload.get("status") or payload.get("tool_status") or "").strip()
            if not status and event_name == "LLM_CALL_END":
                status = "success"
            if not status and event_name == "LLM_ERROR":
                status = "error"
            if status:
                node["status"] = status
            node["summary"] = str(row.get("summary") or node.get("summary") or "")
            duration = _to_int(row.get("duration_ms"), 0)
            if duration > 0:
                node["duration_ms"] = duration
    for node in nodes.values():
        if not node.get("duration_ms") and node.get("start_ts") and node.get("end_ts"):
            node["duration_ms"] = int(max(0.0, float(node["end_ts"]) - float(node["start_ts"])) * 1000)
        if not node.get("status"):
            node["status"] = "running" if not node.get("end_ts") else "success"
    depths: Dict[str, int] = {}

    def _depth(node_id: str, seen: Optional[set[str]] = None) -> int:
        if node_id in depths:
            return depths[node_id]
        seen = set(seen or set())
        if node_id in seen:
            return 0
        seen.add(node_id)
        parent = str(nodes.get(node_id, {}).get("parent_id") or "")
        if not parent or parent not in nodes:
            depths[node_id] = 0
        else:
            depths[node_id] = min(12, _depth(parent, seen) + 1)
        return depths[node_id]

    flat = []
    for node_id, node in nodes.items():
        flat.append({**node, "depth": _depth(node_id)})
    flat.sort(key=lambda item: (float(item.get("start_ts") or 0.0), int(item.get("order") or 0)))
    return {"nodes": flat[-300:], "root_count": len([item for item in flat if not item.get("parent_id") or item.get("parent_id") not in nodes])}


def _build_decisions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        name = str(row.get("name") or "")
        if name == "TASK_DECISION":
            decisions.append(
                {
                    "ts": row.get("ts"),
                    "agent_name": row.get("agent_name") or _extract_agent_name(payload),
                    "reason": str(payload.get("reason") or ""),
                    "decision": " | ".join(str(payload.get(key) or "").strip() for key in ("action", "method") if str(payload.get(key) or "").strip()),
                    "evidence": _compact_text(payload.get("params_compact") or payload.get("summary") or "", 360),
                }
            )
            continue
        if name == "LLM_CALL_END":
            tools = payload.get("tool_calls") if isinstance(payload.get("tool_calls"), list) else []
            reasoning = str(payload.get("reasoning_text") or "").strip()
            preview = str(payload.get("text_preview") or "").strip()
            if not reasoning and not preview and not tools:
                continue
            if tools:
                decision = "Tool plan: " + ", ".join(str(item) for item in tools[:8])
            else:
                decision = _compact_text(preview, 260)
            decisions.append(
                {
                    "ts": row.get("ts"),
                    "agent_name": row.get("agent_name") or _extract_agent_name(payload),
                    "model": str(payload.get("model") or row.get("model") or ""),
                    "reason": reasoning,
                    "decision": decision,
                    "evidence": _compact_text(preview, 520),
                }
            )
    return decisions[-80:]


def _build_task_state(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    timeline: List[Dict[str, Any]] = []
    todos: List[Dict[str, Any]] = []
    plan_revision_count = 0
    seen_plan_updates: set[str] = set()
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        name = str(row.get("name") or "")
        if name in {"TOOL_CALL_START", "TOOL_RAW_INPUT"}:
            tool_name = str(payload.get("tool") or payload.get("tool_name") or row.get("tool") or "")
            if tool_name != "write_todos":
                continue
            parsed = _extract_todos(payload)
            call_key = str(row.get("callback_run_id") or payload.get("callback_run_id") or "").strip()
            if not call_key:
                call_key = f"{name}:{row.get('id')}"
            if parsed:
                todos = parsed
                if call_key not in seen_plan_updates:
                    seen_plan_updates.add(call_key)
                    plan_revision_count += 1
            timeline.append(
                {
                    "ts": row.get("ts"),
                    "name": "TODO_PLAN_UPDATE",
                    "status": "updated",
                    "summary": f"{len(parsed)} todo rows" if parsed else str(row.get("summary") or "todo update"),
                }
            )
            continue
        if name == "RUN_STATE_CHANGE":
            timeline.append(
                {
                    "ts": row.get("ts"),
                    "name": name,
                    "status": str(payload.get("status") or ""),
                    "phase": str(payload.get("phase") or ""),
                    "summary": _event_summary(name, payload),
                }
            )
            continue
        timeline.append(
            {
                "ts": row.get("ts"),
                "name": name,
                "task_id": str(row.get("task_id") or ""),
                "status": str(payload.get("status") or payload.get("outcome") or ""),
                "summary": str(row.get("summary") or ""),
            }
        )
    return {
        "timeline": timeline[-200:],
        "todos": todos,
        "plan_revision_count": plan_revision_count,
    }


def _extract_todos(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = payload.get("params_full")
    if not isinstance(params, (dict, list)):
        raw = payload.get("raw_params")
        params = raw if isinstance(raw, (dict, list)) else params
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {}
    candidates: Any
    if isinstance(params, dict):
        candidates = params.get("todos") or params.get("items") or params.get("tasks") or []
    else:
        candidates = params
    if not isinstance(candidates, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict):
            text = str(item.get("content") or item.get("task") or item.get("text") or "").strip()
            status = str(item.get("status") or "pending").strip() or "pending"
        else:
            text = str(item or "").strip()
            status = "pending"
        if text:
            rows.append({"content": text, "status": status})
    return rows


__all__ = ["OBSERVABILITY_DB_NAME", "OBSERVABILITY_SCHEMA_VERSION", "ObservabilityStore"]
