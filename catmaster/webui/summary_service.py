from __future__ import annotations

from datetime import datetime, timezone
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from catmaster.agents.llm_utils import llm_text
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model

_SUMMARY_FILE = "ui_summary.json"
_SUMMARY_LOCK = threading.Lock()
_SUMMARY_LLM: Any = None
_SUMMARY_LLM_INIT_ATTEMPTED = False


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def summary_path(run_dir: Path) -> Path:
    return run_dir / _SUMMARY_FILE


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    path = summary_path(run_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def summarize_run(run_dir: Path, *, run_error: Optional[str] = None) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        return {}

    meta = _load_json(run_dir / "meta.json")
    events = _read_events(run_dir / "ui_events.jsonl")
    final_report_excerpt = _read_text_excerpt(run_dir / "reports" / "FINAL_REPORT.md", max_chars=6000)

    fallback = _fallback_summary(
        run_dir=run_dir,
        meta=meta,
        events=events,
        final_report_excerpt=final_report_excerpt,
        run_error=run_error,
    )

    llm_payload = _llm_summary(meta=meta, events=events, final_report_excerpt=final_report_excerpt)
    merged = _merge_summary(fallback, llm_payload)
    merged["generated_at"] = _utcnow()
    merged["run_id"] = run_dir.name
    merged["project_space"] = str(meta.get("workspace") or "")
    merged["workspace"] = merged["project_space"]

    path = summary_path(run_dir)
    try:
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return merged


def snapshot_summary(run_dir: Path) -> Dict[str, Any]:
    cached = load_run_summary(run_dir)
    if cached:
        task_state_status = _status_from_task_state(run_dir)
        if task_state_status and str(cached.get("status") or "").strip().lower() != task_state_status:
            updated = dict(cached)
            updated["status"] = task_state_status
            headline = str(updated.get("headline") or "").strip()
            if headline:
                parts = [part.strip() for part in headline.split("|")]
                if len(parts) >= 2:
                    parts[1] = task_state_status
                    updated["headline"] = " | ".join(parts)
            updated["generated_at"] = _utcnow()
            try:
                summary_path(run_dir).write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            return updated
        return cached
    meta = _load_json(run_dir / "meta.json")
    events = _read_events(run_dir / "ui_events.jsonl")
    final_report_excerpt = _read_text_excerpt(run_dir / "reports" / "FINAL_REPORT.md", max_chars=4000)
    fallback = _fallback_summary(
        run_dir=run_dir,
        meta=meta,
        events=events,
        final_report_excerpt=final_report_excerpt,
        run_error=None,
    )
    fallback["run_id"] = run_dir.name
    fallback["project_space"] = str(meta.get("workspace") or "")
    fallback["workspace"] = fallback["project_space"]
    fallback["generated_at"] = _utcnow()
    return fallback


def _merge_summary(base: Dict[str, Any], llm_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not llm_payload:
        return base
    merged = dict(base)
    headline = str(llm_payload.get("headline") or "").strip()
    summary = str(llm_payload.get("summary") or "").strip()
    next_actions = llm_payload.get("next_actions")
    if headline:
        merged["headline"] = headline
    if summary:
        merged["summary"] = summary
    if isinstance(next_actions, list):
        cleaned = [str(item).strip() for item in next_actions if str(item).strip()]
        if cleaned:
            merged["next_actions"] = cleaned[:3]
    merged["source"] = "llm"
    return merged


def _fallback_summary(
    *,
    run_dir: Path,
    meta: Dict[str, Any],
    events: List[Dict[str, Any]],
    final_report_excerpt: str,
    run_error: Optional[str],
) -> Dict[str, Any]:
    model_name = str(meta.get("model_name") or "")
    run_status = _infer_status(events, run_error=run_error, run_dir=run_dir)
    last_event = events[-1] if events else {}
    last_name = str(last_event.get("name") or "")

    headline_bits = [run_dir.name, run_status]
    if model_name:
        headline_bits.append(model_name)
    headline = " | ".join([part for part in headline_bits if part])

    summary_parts: List[str] = []
    if run_error:
        summary_parts.append(f"Run ended with error: {run_error}")
    elif last_name:
        summary_parts.append(f"Last event: {last_name}")
    if final_report_excerpt:
        first_line = final_report_excerpt.strip().splitlines()[0] if final_report_excerpt.strip() else ""
        if first_line:
            summary_parts.append(f"Report: {first_line[:180]}")

    if not summary_parts:
        summary_parts.append("Run summary is not available yet.")

    next_actions = _rule_next_actions(run_status=run_status, has_report=bool(final_report_excerpt), has_error=bool(run_error))

    return {
        "headline": headline,
        "summary": " ".join(summary_parts),
        "next_actions": next_actions,
        "status": run_status,
        "source": "rule",
    }


def _rule_next_actions(*, run_status: str, has_report: bool, has_error: bool) -> List[str]:
    if has_error or run_status == "error":
        return [
            "Open event/tool traces and locate the first failed step.",
            "Fix inputs or tool parameters, then rerun from the same project space.",
            "If needed, start a new run with a narrower prompt scope.",
        ]
    if run_status in {"awaiting_human_feedback"}:
        return [
            "Open this run and provide HITL feedback in the prompt panel.",
            "If prompt panel is empty, refresh Monitor after selecting this run.",
            "Resume execution after feedback submission.",
        ]
    if run_status in {"running", "starting"}:
        return [
            "Keep this run selected and monitor new events.",
            "Avoid switching project spaces in the same page while the run is active.",
            "Prepare follow-up instructions for the next HITL prompt.",
        ]
    if has_report:
        return [
            "Review FINAL_REPORT.md and confirm if deliverables are complete.",
            "Create a next-step prompt to refine missing parts.",
            "Archive this run and start a focused follow-up run if needed.",
        ]
    return [
        "Inspect run artifacts and traces for missing outputs.",
        "Run again with clearer acceptance criteria.",
        "Capture key observations in the next prompt.",
    ]


def _infer_status(events: List[Dict[str, Any]], *, run_error: Optional[str], run_dir: Path) -> str:
    if run_error:
        return "error"
    for event in reversed(events):
        if str(event.get("name") or "") != "RUN_END":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").strip().lower()
            if status:
                return status
        return "done"
    status = _status_from_task_state(run_dir)
    if status:
        return status
    if events:
        return "running"
    return "unknown"


def _terminal_status_from_task_state(run_dir: Path) -> str:
    status = _status_from_task_state(run_dir)
    if status in {"done", "failure", "error", "needs_intervention", "interrupted_paused", "awaiting_human_feedback"}:
        return status
    return ""


def _status_from_task_state(run_dir: Path) -> str:
    task_state = _load_json(run_dir / "task_state.json")
    if not isinstance(task_state, dict):
        return ""
    status = str(task_state.get("status") or "").strip().lower()
    return status


def _read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines()[-300:]:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
    except Exception:
        return []
    return out


def _read_text_excerpt(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return text[:max_chars]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _llm_summary(*, meta: Dict[str, Any], events: List[Dict[str, Any]], final_report_excerpt: str) -> Dict[str, Any]:
    llm = _get_summary_llm()
    if llm is None:
        return {}

    prompt = {
        "task": "Summarize a CatMaster run for monitoring UI.",
        "requirements": {
            "output": "JSON object",
            "keys": ["headline", "summary", "next_actions"],
            "next_actions": "array of 1-3 imperative short bullets",
            "language": "English",
            "max_summary_chars": 320,
        },
        "run_meta": meta,
        "recent_events": events[-60:],
        "final_report_excerpt": final_report_excerpt[:4000],
    }

    text = ""
    try:
        resp = llm.invoke(json.dumps(prompt, ensure_ascii=False))
        text = llm_text(resp).strip()
    except Exception:
        return {}

    return _extract_json_object(text)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get_summary_llm() -> Any:
    global _SUMMARY_LLM
    global _SUMMARY_LLM_INIT_ATTEMPTED
    if _SUMMARY_LLM is not None:
        return _SUMMARY_LLM
    with _SUMMARY_LOCK:
        if _SUMMARY_LLM is not None:
            return _SUMMARY_LLM
        if _SUMMARY_LLM_INIT_ATTEMPTED:
            return None
        _SUMMARY_LLM_INIT_ATTEMPTED = True
        try:
            profile = LLMProfile.from_env_or_file()
            _SUMMARY_LLM = build_chat_model(profile.config_for_role("summary"))
        except Exception:
            _SUMMARY_LLM = None
        return _SUMMARY_LLM


__all__ = [
    "load_run_summary",
    "snapshot_summary",
    "summarize_run",
    "summary_path",
]
