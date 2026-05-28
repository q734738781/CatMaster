from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.usage_stats import summarize_usage_from_metadata, write_usage_summary_from_metadata
from catmaster.ui import Reporter
from catmaster.ui.events import UIEvent, make_event

from .live_state import apply_event, new_live_state


def _clone_prompt(prompt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    payload = prompt.get("payload") if isinstance(prompt.get("payload"), dict) else {}
    copied = dict(prompt)
    copied["payload"] = dict(payload)
    return copied


def _copy_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in (
        "input_tokens",
        "input_uncached_tokens",
        "input_cached_tokens",
        "input_cache_read_tokens",
        "input_cache_write_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, int):
            out[key] = value
    input_details = usage.get("input_token_details") if isinstance(usage.get("input_token_details"), dict) else {}
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    if "input_cached_tokens" not in out:
        value = input_details.get("cache_read", prompt_details.get("cache_read"))
        if isinstance(value, bool):
            out["input_cached_tokens"] = int(value)
        elif isinstance(value, int):
            out["input_cached_tokens"] = value
    if "input_cache_read_tokens" not in out and "input_cached_tokens" in out:
        out["input_cache_read_tokens"] = out["input_cached_tokens"]
    if "input_cache_write_tokens" not in out:
        value = input_details.get("cache_write", input_details.get("cache_creation", prompt_details.get("cache_creation")))
        if isinstance(value, bool):
            out["input_cache_write_tokens"] = int(value)
        elif isinstance(value, int):
            out["input_cache_write_tokens"] = value
    if "input_uncached_tokens" not in out:
        out["input_uncached_tokens"] = max(
            0,
            int(out.get("input_tokens") or 0)
            - int(out.get("input_cached_tokens") or 0)
            - int(out.get("input_cache_write_tokens") or 0),
        )
    if "reasoning_tokens" not in out:
        output_details = usage.get("output_token_details") if isinstance(usage.get("output_token_details"), dict) else {}
        completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        value = output_details.get("reasoning")
        if not isinstance(value, int):
            value = completion_details.get("reasoning_tokens")
        if isinstance(value, bool):
            out["reasoning_tokens"] = int(value)
        elif isinstance(value, int):
            out["reasoning_tokens"] = value
    return out


def _merge_usage_payload(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if isinstance(value, dict):
            current = merged.get(key)
            merged[key] = _merge_usage_payload(current if isinstance(current, dict) else {}, value)
            continue
        if isinstance(value, bool):
            merged[key] = int(bool(merged.get(key, 0))) + int(value)
            continue
        if isinstance(value, int):
            merged[key] = int(merged.get(key, 0) or 0) + value
            continue
        if isinstance(value, float):
            merged[key] = float(merged.get(key, 0.0) or 0.0) + value
            continue
        if value is not None:
            merged[key] = value
    return merged


class PromptBroker:
    """In-memory HITL prompt broker.

    The previous file-backed prompt channel was removed from the hot path.
    This broker keeps the blocking request/response contract for the runner,
    but all synchronization stays in memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: Optional[Dict[str, Any]] = None
        self._responses: Dict[str, str] = {}

    def set_store_dir(self, path: Path) -> None:
        _ = path
        return

    def request_prompt(self, kind: str, payload: Dict[str, Any]) -> str:
        prompt_id = uuid.uuid4().hex
        with self._cond:
            self._pending = {
                "prompt_id": prompt_id,
                "kind": kind,
                "payload": dict(payload or {}),
                "created_at": time.time(),
            }
            self._cond.notify_all()
            while prompt_id not in self._responses:
                self._cond.wait(timeout=0.5)
            text = self._responses.pop(prompt_id, "")
            if self._pending and self._pending.get("prompt_id") == prompt_id:
                self._pending = None
            self._cond.notify_all()
            return text

    def submit(self, prompt_id: str, text: str) -> bool:
        with self._cond:
            if not self._pending or self._pending.get("prompt_id") != prompt_id:
                return False
            self._responses[prompt_id] = text
            self._cond.notify_all()
            return True

    def submit_persisted(self, prompt_id: str, text: str) -> bool:
        return self.submit(prompt_id, text)

    def get_pending(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return _clone_prompt(self._pending)


class WebReporter(Reporter):
    def __init__(self, *, broker: PromptBroker | None = None, max_events: int = 4000) -> None:
        self._broker = broker
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._run_dir: Optional[Path] = None
        self._final_summary: Optional[str] = None
        self._live_state: Dict[str, Any] = new_live_state()
        self._llm_state: Dict[str, Any] = {
            "model": "",
            "phase": "",
            "status": "idle",
            "text": "",
            "reasoning_text": "",
            "usage": {},
            "elapsed_ms": 0,
            "last_updated_ts": 0.0,
        }
        self._graph_state: Dict[str, Any] = {
            "node": "",
            "message_count": 0,
            "tool_calls": [],
            "text_preview": "",
            "last_updated_ts": 0.0,
        }
        self._usage_totals: Dict[str, Any] = {}
        self._usage_metadata_by_model: Dict[str, Dict[str, Any]] = {}
        self._call_counts_by_model: Dict[str, int] = {}
        self._usage_metadata_by_role: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._call_counts_by_role: Dict[str, int] = {}

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def is_live(self) -> bool:
        return True

    def wants_graph_streaming(self) -> bool:
        return False

    def set_run_dir(self, run_dir: Path) -> None:
        resolved = Path(run_dir).expanduser().resolve()
        with self._cond:
            self._run_dir = resolved
            self._seq = max(int(self._seq or 0), self._last_persisted_event_seq(resolved))
            self._live_state = new_live_state(run_id=resolved.name)
            self._usage_totals = {}
            self._usage_metadata_by_model = {}
            self._call_counts_by_model = {}
            self._usage_metadata_by_role = {}
            self._call_counts_by_role = {}
            self._cond.notify_all()

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
        name = str(payload.get("name") or "")
        if name in {"LLM_REASONING_DELTA", "LLM_TOKEN_DELTA"}:
            if name == "LLM_REASONING_DELTA":
                self._append_observability_event(payload)
            return
        with self._cond:
            self._seq += 1
            payload["seq"] = self._seq
            self._apply_live_updates(payload)
            self._events.append(payload)
            self._append_observability_event(payload)
            self._cond.notify_all()

    def get_events_since(self, last_seq: int) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            events = [event for event in self._events if int(event.get("seq") or 0) > int(last_seq or 0)]
            new_seq = int(events[-1]["seq"]) if events else int(last_seq or 0)
            return events, new_seq

    def wait_for_events_since(self, last_seq: int, *, timeout_s: float = 15.0) -> Tuple[List[Dict[str, Any]], int]:
        deadline = time.time() + max(0.1, float(timeout_s))
        with self._cond:
            while True:
                events = [event for event in self._events if int(event.get("seq") or 0) > int(last_seq or 0)]
                if events:
                    return events, int(events[-1]["seq"])
                remaining = deadline - time.time()
                if remaining <= 0:
                    return [], int(last_seq or 0)
                self._cond.wait(timeout=min(remaining, 0.5))

    def recent_events(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._events)[-limit:]

    def prompt_proposal_feedback(self, *, todo: List[str], proposal_description: str) -> str:
        if self._broker is None:
            return ""
        payload = {
            "todo": list(todo),
            "proposal_description": proposal_description or "",
        }
        return self._request_prompt("proposal_review", payload)

    def prompt_hitl_feedback(self, *, report_text: str, report_path: str) -> str:
        if self._broker is None:
            return ""
        payload = {
            "report_text": report_text or "",
            "report_path": report_path or "",
        }
        return self._request_prompt("hitl", payload)

    def prompt_interrupt_feedback(self, *, guidance: str, run_id: str, phase: str) -> str:
        if self._broker is None:
            return ""
        payload = {
            "guidance": guidance or "",
            "run_id": run_id or "",
            "phase": phase or "",
        }
        return self._request_prompt("interrupt_feedback", payload)

    def submit_prompt(self, prompt_id: str, text: str) -> bool:
        if self._broker is None:
            return False
        submitted = self._broker.submit(prompt_id, text)
        if submitted:
            self.emit(
                make_event(
                    "PROMPT_RESOLVED",
                    category="prompt",
                    payload={"prompt_id": str(prompt_id or ""), "text_len": len(str(text or ""))},
                    run_id=self._live_state.get("run_id") or None,
                )
            )
        return submitted

    def get_prompt(self) -> Optional[Dict[str, Any]]:
        if self._broker is None:
            return None
        return self._broker.get_pending()

    def show_final_summary(self, summary: str) -> None:
        with self._cond:
            self._final_summary = summary
            self._cond.notify_all()

    def get_final_summary(self) -> Optional[str]:
        with self._lock:
            return self._final_summary

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            run_dir = str(self._run_dir) if self._run_dir is not None else ""
            return {
                "seq": self._seq,
                "run_dir": run_dir,
                "run_name": Path(run_dir).name if run_dir else "",
                "live_state": dict(self._live_state),
                "llm": {
                    "model": str(self._llm_state.get("model") or ""),
                    "phase": str(self._llm_state.get("phase") or ""),
                    "status": str(self._llm_state.get("status") or "idle"),
                    "text": str(self._llm_state.get("text") or ""),
                    "reasoning_text": str(self._llm_state.get("reasoning_text") or ""),
                    "usage": dict(self._llm_state.get("usage") or {}),
                    "elapsed_ms": int(self._llm_state.get("elapsed_ms") or 0),
                    "last_updated_ts": float(self._llm_state.get("last_updated_ts") or 0.0),
                },
                "graph": {
                    "node": str(self._graph_state.get("node") or ""),
                    "message_count": int(self._graph_state.get("message_count") or 0),
                    "tool_calls": list(self._graph_state.get("tool_calls") or []),
                    "text_preview": str(self._graph_state.get("text_preview") or ""),
                    "last_updated_ts": float(self._graph_state.get("last_updated_ts") or 0.0),
                },
                "prompt": _clone_prompt(self._broker.get_pending()) if self._broker is not None else None,
                "final_summary": str(self._final_summary or ""),
                "usage_totals": dict(self._usage_totals),
                "recent_events": list(self._events)[-120:],
            }

    def _request_prompt(self, kind: str, payload: Dict[str, Any]) -> str:
        if self._broker is None:
            return ""
        text_holder: Dict[str, str] = {}

        def _waiter() -> None:
            text_holder["text"] = self._broker.request_prompt(kind, payload)

        thread = threading.Thread(target=_waiter, daemon=True)
        thread.start()
        while thread.is_alive():
            pending = self._broker.get_pending()
            if pending is not None:
                self.emit(
                    make_event(
                        "PROMPT_REQUESTED",
                        category="prompt",
                        payload={
                            "prompt_id": str(pending.get("prompt_id") or ""),
                            "kind": str(kind or ""),
                        },
                        run_id=self._live_state.get("run_id") or None,
                    )
                )
                break
            time.sleep(0.01)
        thread.join()
        return text_holder.get("text", "")

    def _apply_live_updates(self, event: Dict[str, Any]) -> None:
        if not self._live_state.get("run_id"):
            run_id = str(event.get("run_id") or "")
            if not run_id and self._run_dir is not None:
                run_id = self._run_dir.name
            if run_id:
                self._live_state["run_id"] = run_id

        apply_event(self._live_state, event)

        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        name = str(event.get("name") or "")
        now_ts = float(event.get("ts") or time.time())

        if name == "LLM_CALL_START":
            self._llm_state.update(
                {
                    "model": str(payload.get("model") or ""),
                    "phase": str(payload.get("phase") or ""),
                    "status": "running",
                    "text": "",
                    "reasoning_text": "",
                    "usage": {},
                    "elapsed_ms": 0,
                    "last_updated_ts": now_ts,
                }
            )
        elif name == "LLM_CALL_END":
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            compact_usage = _copy_usage(usage)
            final_text = str(payload.get("text_preview") or "").strip()
            self._llm_state.update(
                {
                    "model": str(payload.get("model") or self._llm_state.get("model") or ""),
                    "phase": str(payload.get("phase") or self._llm_state.get("phase") or ""),
                    "status": "completed",
                    "text": final_text,
                    "reasoning_text": str(payload.get("reasoning_text") or self._llm_state.get("reasoning_text") or ""),
                    "usage": compact_usage,
                    "elapsed_ms": int(payload.get("elapsed_ms") or 0),
                    "last_updated_ts": now_ts,
                }
            )
            tool_calls = payload.get("tool_calls") if isinstance(payload.get("tool_calls"), list) else []
            if final_text or tool_calls:
                self._graph_state.update(
                    {
                        "tool_calls": [str(item) for item in tool_calls if str(item).strip()],
                        "text_preview": final_text,
                        "last_updated_ts": now_ts,
                    }
                )
            model_name = str(payload.get("model") or self._llm_state.get("model") or "").strip()
            agent_name = str(payload.get("agent_name") or "").strip()
            if model_name and usage:
                self._usage_metadata_by_model[model_name] = _merge_usage_payload(
                    self._usage_metadata_by_model.get(model_name) if isinstance(self._usage_metadata_by_model.get(model_name), dict) else {},
                    usage,
                )
                self._call_counts_by_model[model_name] = int(self._call_counts_by_model.get(model_name, 0)) + 1
                if agent_name:
                    role_bucket = self._usage_metadata_by_role.setdefault(agent_name, {})
                    role_bucket[model_name] = _merge_usage_payload(
                        role_bucket.get(model_name) if isinstance(role_bucket.get(model_name), dict) else {},
                        usage,
                    )
                    self._call_counts_by_role[agent_name] = int(self._call_counts_by_role.get(agent_name, 0)) + 1
                self._usage_totals = summarize_usage_from_metadata(
                    self._usage_metadata_by_model,
                    run_dir=self._run_dir,
                    call_counts_by_model=self._call_counts_by_model,
                    usage_metadata_by_role=self._usage_metadata_by_role,
                    call_counts_by_role=self._call_counts_by_role,
                )
                self._persist_usage_summary()
        elif name == "GRAPH_NODE_UPDATE":
            tool_calls = payload.get("tool_calls") if isinstance(payload.get("tool_calls"), list) else []
            self._graph_state.update(
                {
                    "node": str(payload.get("node") or ""),
                    "message_count": int(payload.get("message_count") or 0),
                    "tool_calls": [str(item) for item in tool_calls if str(item).strip()],
                    "text_preview": str(payload.get("text_preview") or ""),
                    "last_updated_ts": now_ts,
                }
            )
        elif name == "RUN_END":
            self._llm_state.update(
                {
                    "status": "idle",
                    "text": "",
                    "reasoning_text": "",
                    "usage": {},
                    "elapsed_ms": 0,
                    "last_updated_ts": now_ts,
                }
            )
            self._graph_state.update(
                {
                    "tool_calls": [],
                    "text_preview": "",
                    "last_updated_ts": now_ts,
                }
            )

    def _append_observability_event(self, event: Dict[str, Any]) -> None:
        run_dir = self._run_dir
        if run_dir is None:
            return
        try:
            ObservabilityStore(Path(run_dir)).record_ui_event(event)
        except Exception:
            return

    @staticmethod
    def _last_persisted_event_seq(run_dir: Path) -> int:
        store = ObservabilityStore(Path(run_dir))
        if store.db_exists():
            try:
                return store.last_ui_event_seq()
            except Exception:
                return 0
        path = Path(run_dir) / "ui_events.jsonl"
        if not path.exists():
            return 0
        try:
            store.read_snapshot(limit=1)
            sqlite_seq = store.last_ui_event_seq()
            if sqlite_seq > 0:
                return sqlite_seq
        except Exception:
            pass
        last_seq = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        item = json.loads(text)
                    except Exception:
                        continue
                    try:
                        last_seq = max(last_seq, int(item.get("seq") or index))
                    except Exception:
                        last_seq = max(last_seq, index)
        except Exception:
            return 0
        return last_seq

    def _persist_usage_summary(self) -> None:
        run_dir = self._run_dir
        if run_dir is None or not self._usage_metadata_by_model:
            return
        try:
            write_usage_summary_from_metadata(
                run_dir,
                usage_metadata=self._usage_metadata_by_model,
                call_counts_by_model=self._call_counts_by_model,
                usage_metadata_by_role=self._usage_metadata_by_role,
                call_counts_by_role=self._call_counts_by_role,
                append=False,
            )
        except Exception:
            return


__all__ = ["PromptBroker", "WebReporter"]
