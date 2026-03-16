from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Tuple

_TASK_TRIGGER_EVENTS = {
    "TASK_START",
    "TASK_SUMMARY",
    "TASK_END",
    "TASK_JOURNAL_APPEND",
    "MEMORY_MERGE_DONE",
}

_TOOL_TRIGGER_EVENTS = {
    "TOOL_CALL_START",
    "TOOL_CALL_END",
    "TOOL_VALIDATE_FAILED",
    "TOOL_CALL_INTERRUPTED",
}


def new_live_state(run_id: str = "") -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "unknown",
        "todo_items": [],
        "todo_rows": [],
        "current_task_id": "",
        "current_task_goal": "",
        "current_phase": "",
        "current_node": "",
        "active_toolcall": None,
        "recent_toolcalls": [],
        "last_task_summary": None,
        "llm": {
            "model": "",
            "phase": "",
            "status": "idle",
            "text": "",
            "reasoning_text": "",
            "usage": {},
            "elapsed_ms": 0,
        },
        "progress": {
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "needs_intervention": 0,
            "total": 0,
        },
        "journal_recent": [],
        "recent_events": [],
        "live_summary": {},
        "last_updated_ts": 0.0,
        "_task_outcomes": {},
        "_seen_task_ids": [],
        "_tool_events_since_summary": 0,
        "_last_summary_ts": 0.0,
    }


def apply_events(
    state: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
    *,
    max_recent_toolcalls: int = 20,
    max_recent_events: int = 80,
    max_journal_items: int = 5,
) -> Tuple[Dict[str, Any], bool]:
    changed = False
    for event in events:
        if apply_event(
            state,
            event,
            max_recent_toolcalls=max_recent_toolcalls,
            max_recent_events=max_recent_events,
            max_journal_items=max_journal_items,
        ):
            changed = True
    return state, changed


def apply_event(
    state: Dict[str, Any],
    event: Dict[str, Any],
    *,
    max_recent_toolcalls: int = 20,
    max_recent_events: int = 80,
    max_journal_items: int = 5,
) -> bool:
    if not isinstance(event, dict):
        return False
    name = str(event.get("name") or "")
    payload = event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    task_id = str(event.get("task_id") or "")
    step_id = event.get("step_id")
    ts = _event_ts(event)
    changed = False

    if name in {"RUN_INIT_DONE"}:
        state["status"] = "starting"
        changed = True
    elif name == "PROPOSAL_START":
        state["current_phase"] = "planning"
        state["status"] = "running"
        changed = True
    elif name in {"PROPOSAL_REVIEW_WAIT_INPUT", "PROPOSAL_REVIEW_SHOW"}:
        state["current_phase"] = "waiting_human"
        state["status"] = "awaiting_human_feedback"
        changed = True
    elif name == "RUN_WAITING_INPUT":
        state["current_phase"] = "waiting_human"
        state["status"] = "awaiting_human_feedback"
        changed = True
    elif name == "RUN_INPUT_RECEIVED":
        state["current_phase"] = "executing"
        state["status"] = "running"
        changed = True
    elif name == "TASKS_COMPILED":
        total = payload.get("n_tasks")
        if isinstance(total, int) and total > 0:
            state["progress"]["total"] = max(int(state["progress"].get("total", 0)), total)
            _recompute_pending(state)
            changed = True
    elif name == "TASK_START":
        state["status"] = "running"
        state["current_phase"] = "executing"
        state["current_task_id"] = task_id
        state["current_task_goal"] = str(payload.get("goal") or "")
        _mark_seen_task(state, task_id)
        _recompute_pending(state)
        changed = True
    elif name == "TASK_SUMMARIZE_START":
        state["current_phase"] = "summarizing"
        state["current_task_id"] = task_id or state.get("current_task_id", "")
        changed = True
    elif name == "TASK_SUMMARY":
        state["current_phase"] = "summarizing"
        state["current_task_id"] = task_id or state.get("current_task_id", "")
        state["last_task_summary"] = {
            "task_id": task_id,
            "outcome": str(payload.get("outcome") or ""),
            "summary_snippet": str(payload.get("summary_snippet") or ""),
            "artifacts": payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else [],
            "ops_counts": payload.get("ops_counts") if isinstance(payload.get("ops_counts"), dict) else {},
            "ts": ts,
        }
        changed = True
    elif name == "TASK_END":
        state["current_phase"] = "executing"
        state["current_task_id"] = task_id or state.get("current_task_id", "")
        outcome = str(payload.get("outcome") or "")
        if task_id:
            _mark_seen_task(state, task_id)
            _set_task_outcome(state, task_id, outcome)
        if outcome == "failure":
            state["status"] = "error"
        elif outcome == "needs_intervention":
            state["status"] = "needs_intervention"
        else:
            state["status"] = "running"
        changed = True
    elif name == "TOOL_CALL_START":
        state["status"] = "running"
        state["current_phase"] = "executing"
        state["current_task_id"] = task_id or state.get("current_task_id", "")
        todo_rows = _normalize_todos(payload.get("params_full"))
        if str(payload.get("tool") or "").strip() == "write_todos" and todo_rows:
            state["todo_rows"] = todo_rows
            state["todo_items"] = [str(item.get("content") or "").strip() for item in todo_rows if str(item.get("content") or "").strip()]
        active = {
            "task_id": task_id,
            "step_id": step_id if isinstance(step_id, int) else None,
            "tool": str(payload.get("tool") or ""),
            "toolcall_id": str(payload.get("toolcall_id") or ""),
            "params_compact": str(payload.get("params_compact") or ""),
            "params_full": payload.get("params_full"),
            "started_ts": ts,
            "elapsed_sec": 0,
            "status": "running",
        }
        state["active_toolcall"] = active
        changed = True
    elif name == "GRAPH_NODE_UPDATE":
        node = str(payload.get("node") or "").strip()
        if node:
            state["current_node"] = node
            if node not in {"proposal", "director", "task", "summarize"}:
                state["current_phase"] = node
        changed = True
    elif name == "LLM_CALL_START":
        state["status"] = "running"
        llm = state.get("llm") if isinstance(state.get("llm"), dict) else {}
        llm.update(
            {
                "model": str(payload.get("model") or ""),
                "phase": str(payload.get("phase") or ""),
                "status": "running",
                "text": "",
                "reasoning_text": "",
                "usage": {},
                "elapsed_ms": 0,
            }
        )
        state["llm"] = llm
        changed = True
    elif name == "LLM_REASONING_DELTA":
        llm = state.get("llm") if isinstance(state.get("llm"), dict) else {}
        llm["status"] = "running"
        llm["model"] = str(payload.get("model") or llm.get("model") or "")
        llm["phase"] = str(payload.get("phase") or llm.get("phase") or "")
        llm["reasoning_text"] = str(llm.get("reasoning_text") or "") + str(payload.get("text") or "")
        state["llm"] = llm
        changed = True
    elif name == "LLM_TOKEN_DELTA":
        llm = state.get("llm") if isinstance(state.get("llm"), dict) else {}
        llm["status"] = "running"
        llm["model"] = str(payload.get("model") or llm.get("model") or "")
        llm["phase"] = str(payload.get("phase") or llm.get("phase") or "")
        llm["text"] = str(llm.get("text") or "") + str(payload.get("text") or "")
        state["llm"] = llm
        changed = True
    elif name == "LLM_CALL_END":
        llm = state.get("llm") if isinstance(state.get("llm"), dict) else {}
        final_text = str(llm.get("text") or "").strip()
        if not final_text:
            final_text = str(payload.get("text_preview") or "").strip()
        llm.update(
            {
                "model": str(payload.get("model") or llm.get("model") or ""),
                "phase": str(payload.get("phase") or llm.get("phase") or ""),
                "status": "completed",
                "text": final_text,
                "reasoning_text": str(payload.get("reasoning_text") or llm.get("reasoning_text") or ""),
                "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
                "elapsed_ms": int(payload.get("elapsed_ms") or 0),
            }
        )
        state["llm"] = llm
        changed = True
    elif name == "INTERRUPT_REQUESTED":
        state["status"] = "interrupting"
        state["current_phase"] = "interrupting"
        changed = True
    elif name == "INTERRUPT_ACKED":
        state["status"] = "interrupting"
        phase = str(payload.get("phase") or "")
        state["current_phase"] = f"interrupt:{phase}" if phase else "interrupting"
        changed = True
    elif name == "TOOL_VALIDATE_FAILED":
        active = state.get("active_toolcall")
        if isinstance(active, dict):
            active["status"] = "validation_failed"
            changed = True
    elif name == "TOOL_CALL_INTERRUPTED":
        active = state.get("active_toolcall")
        if isinstance(active, dict):
            active["status"] = "interrupted"
        state["status"] = "interrupted_paused"
        state["current_phase"] = "interrupted"
        changed = True
    elif name == "TOOL_CALL_END":
        active = state.get("active_toolcall")
        toolcall_id = str(payload.get("toolcall_id") or "")
        ended_ts = ts
        ended_record: Dict[str, Any] = {
            "task_id": task_id,
            "step_id": step_id if isinstance(step_id, int) else None,
            "tool": str(payload.get("tool") or ""),
            "status": str(payload.get("status") or ""),
            "highlights": str(payload.get("highlights") or ""),
            "toolcall_id": toolcall_id,
            "input_ref": str(payload.get("input_ref") or ""),
            "output_ref": str(payload.get("output_ref") or ""),
            "params_compact": "",
            "started_ts": ended_ts,
            "ended_ts": ended_ts,
            "duration_sec": 0,
        }
        if isinstance(active, dict):
            active_id = str(active.get("toolcall_id") or "")
            if not toolcall_id or not active_id or toolcall_id == active_id:
                started_ts = _to_float(active.get("started_ts"), default=ended_ts)
                ended_record["started_ts"] = started_ts
                ended_record["duration_sec"] = max(0, int(ended_ts - started_ts))
                ended_record["params_compact"] = str(active.get("params_compact") or "")
                state["active_toolcall"] = None
                changed = True
        if not ended_record["params_compact"]:
            ended_record["params_compact"] = str(payload.get("params_compact") or "")
        _push_limited(state, "recent_toolcalls", ended_record, max_recent_toolcalls)
        changed = True
    elif name == "MEMORY_MERGE_DONE":
        state["current_phase"] = "executing"
        changed = True
    elif name == "TASK_JOURNAL_APPEND":
        entry = {
            "task_id": task_id,
            "outcome": str(payload.get("outcome") or ""),
            "summary_snippet": str(payload.get("summary_snippet") or ""),
            "artifacts": payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else [],
            "ts": ts,
        }
        _push_limited(state, "journal_recent", entry, max_journal_items)
        changed = True
    elif name == "FINAL_SUMMARY_START":
        state["current_phase"] = "finalizing"
        changed = True
    elif name == "RUN_END":
        status = str(payload.get("status") or "").strip().lower()
        state["status"] = status or state.get("status", "done")
        if status == "interrupted_paused":
            state["current_phase"] = "paused"
        elif status == "awaiting_human_feedback":
            state["current_phase"] = "waiting_human"
        else:
            state["current_phase"] = "finalizing"
        state["active_toolcall"] = None
        state["recent_toolcalls"] = []
        llm = state.get("llm") if isinstance(state.get("llm"), dict) else {}
        llm.update(
            {
                "status": "idle",
                "text": "",
                "reasoning_text": "",
                "usage": {},
                "elapsed_ms": 0,
            }
        )
        state["llm"] = llm
        graph = state.get("graph") if isinstance(state.get("graph"), dict) else {}
        graph.update(
            {
                "tool_calls": [],
                "text_preview": "",
            }
        )
        state["graph"] = graph
        changed = True
    elif name == "RUN_PAUSED":
        state["status"] = "interrupted_paused"
        phase = str(payload.get("phase") or "")
        state["current_phase"] = f"paused:{phase}" if phase else "paused"
        changed = True

    if changed:
        state["last_updated_ts"] = ts

    _push_limited(state, "recent_events", _compact_event(event), max_recent_events)
    return changed


def should_refresh_live_summary(
    state: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
    *,
    min_interval_s: int,
    tool_event_batch: int,
) -> bool:
    now = time.time()
    names = [str((event or {}).get("name") or "") for event in events]
    if not names:
        return False
    has_task_trigger = any(name in _TASK_TRIGGER_EVENTS for name in names)
    tool_count = sum(1 for name in names if name in _TOOL_TRIGGER_EVENTS)
    state["_tool_events_since_summary"] = int(state.get("_tool_events_since_summary", 0)) + tool_count
    if not has_task_trigger and int(state.get("_tool_events_since_summary", 0)) < max(1, tool_event_batch):
        return False
    last_summary_ts = _to_float(state.get("_last_summary_ts"), default=0.0)
    if now - last_summary_ts < max(1, int(min_interval_s)):
        return False
    state["_last_summary_ts"] = now
    state["_tool_events_since_summary"] = 0
    return True


def compact_live_state_for_llm(
    state: Dict[str, Any],
    *,
    max_events: int,
    max_params_chars: int,
    max_journal_items: int,
) -> Dict[str, Any]:
    active = state.get("active_toolcall")
    active_copy: Dict[str, Any] | None = None
    if isinstance(active, dict):
        active_copy = dict(active)
        params_full = active_copy.get("params_full")
        active_copy["params_full"] = _trim_text(_to_json_text(params_full), max_params_chars)

    return {
        "run_id": state.get("run_id", ""),
        "status": state.get("status", "unknown"),
        "todo_items": list(state.get("todo_items", [])[-12:]),
        "current_task_id": state.get("current_task_id", ""),
        "current_task_goal": state.get("current_task_goal", ""),
        "current_phase": state.get("current_phase", ""),
        "progress": state.get("progress", {}),
        "active_toolcall": active_copy,
        "recent_toolcalls": list(state.get("recent_toolcalls", [])[-8:]),
        "last_task_summary": state.get("last_task_summary"),
        "journal_recent": list(state.get("journal_recent", [])[-max_journal_items:]),
        "recent_events": list(state.get("recent_events", [])[-max_events:]),
    }


def _recompute_pending(state: Dict[str, Any]) -> None:
    total = int(state.get("progress", {}).get("total", 0))
    outcomes = state.get("_task_outcomes")
    if not isinstance(outcomes, dict):
        outcomes = {}
        state["_task_outcomes"] = outcomes

    completed = sum(1 for value in outcomes.values() if value == "success")
    failed = sum(1 for value in outcomes.values() if value == "failure")
    needs_intervention = sum(1 for value in outcomes.values() if value == "needs_intervention")

    state["progress"]["completed"] = completed
    state["progress"]["failed"] = failed
    state["progress"]["needs_intervention"] = needs_intervention
    state["progress"]["pending"] = max(0, total - completed - failed - needs_intervention)


def _normalize_todos(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        todos = value.get("todos")
    else:
        todos = None
    if not isinstance(todos, list):
        return []
    rows: list[dict[str, str]] = []
    for item in todos:
        if isinstance(item, dict):
            content = str(item.get("content") or "").strip()
            status = str(item.get("status") or "pending").strip() or "pending"
        else:
            content = str(item or "").strip()
            status = "pending"
        if not content:
            continue
        rows.append({"content": content, "status": status})
    return rows


def _mark_seen_task(state: Dict[str, Any], task_id: str) -> None:
    if not task_id:
        return
    seen = state.get("_seen_task_ids")
    if not isinstance(seen, list):
        seen = []
        state["_seen_task_ids"] = seen
    if task_id not in seen:
        seen.append(task_id)
    state["progress"]["total"] = max(int(state["progress"].get("total", 0)), len(seen))


def _set_task_outcome(state: Dict[str, Any], task_id: str, outcome: str) -> None:
    outcomes = state.get("_task_outcomes")
    if not isinstance(outcomes, dict):
        outcomes = {}
        state["_task_outcomes"] = outcomes
    normalized = outcome if outcome in {"success", "failure", "needs_intervention"} else ""
    if normalized:
        outcomes[task_id] = normalized
    _recompute_pending(state)


def _compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    name = str(event.get("name") or "")
    payload = event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    compact_payload: Dict[str, Any] = {}
    for key in ("tool", "status", "outcome", "summary_snippet", "goal", "reason", "highlights", "params_compact"):
        if key in payload:
            compact_payload[key] = payload.get(key)
    return {
        "ts": event.get("ts"),
        "name": name,
        "task_id": event.get("task_id"),
        "step_id": event.get("step_id"),
        "payload": compact_payload,
    }


def _event_ts(event: Dict[str, Any]) -> float:
    return _to_float(event.get("ts"), default=time.time())


def _to_float(value: Any, *, default: float) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default


def _to_json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _push_limited(state: Dict[str, Any], key: str, item: Dict[str, Any], limit: int) -> None:
    bucket = state.get(key)
    if not isinstance(bucket, list):
        bucket = []
        state[key] = bucket
    bucket.append(item)
    if len(bucket) > limit:
        del bucket[:-limit]


__all__ = [
    "new_live_state",
    "apply_event",
    "apply_events",
    "should_refresh_live_summary",
    "compact_live_state_for_llm",
]
