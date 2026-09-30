from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_SUMMARY_FILE = "ui_summary.json"
_RUN_STATE_FILE = "run_state.json"


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
    task_state = _load_json(run_dir / _RUN_STATE_FILE)
    summary = _fallback_summary(
        run_dir=run_dir,
        meta=meta,
        task_state=task_state,
        run_error=run_error,
    )
    summary["generated_at"] = _utcnow()
    summary["run_id"] = run_dir.name
    summary["project_space"] = str(meta.get("workspace") or "")
    summary["workspace"] = summary["project_space"]
    try:
        summary_path(run_dir).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return summary


def snapshot_summary(run_dir: Path) -> Dict[str, Any]:
    cached = load_run_summary(run_dir)
    task_state_status = _status_from_task_state(run_dir)
    if cached:
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
    return summarize_run(run_dir)


def _fallback_summary(
    *,
    run_dir: Path,
    meta: Dict[str, Any],
    task_state: Dict[str, Any],
    run_error: Optional[str],
) -> Dict[str, Any]:
    model_name = str(meta.get("model_name") or "")
    has_result = bool(str(task_state.get("final_answer") or "").strip() or str(task_state.get("summary") or "").strip())
    run_status = _infer_status(task_state=task_state, run_error=run_error, has_result=has_result)

    headline_bits = [run_dir.name, run_status]
    if model_name:
        headline_bits.append(model_name)
    headline = " | ".join([part for part in headline_bits if part])

    summary_parts: List[str] = []
    if run_error:
        summary_parts.append(f"Run ended with error: {run_error}")

    current_work = str(task_state.get("text_preview") or "").strip()
    if current_work and run_status in {"running", "starting", "drafting", "planning", "executing"}:
        summary_parts.append(f"Current work: {current_work}")

    task_summary = _extract_task_summary(task_state)
    if task_summary:
        summary_parts.append(task_summary)

    if not summary_parts:
        summary_parts.append("Run summary is not available yet.")

    next_actions = _rule_next_actions(
        run_status=run_status,
        has_result=has_result,
        has_error=bool(run_error),
    )

    return {
        "headline": headline,
        "summary": " ".join(summary_parts),
        "next_actions": next_actions,
        "status": run_status,
        "source": "rule",
    }


def _extract_task_summary(task_state: Dict[str, Any]) -> str:
    summary = str(task_state.get("summary") or "").strip()
    if summary:
        return summary[:240]
    goal = str(task_state.get("text_preview") or "").strip()
    if goal:
        return f"Current focus: {goal[:180]}"
    return ""


def _rule_next_actions(*, run_status: str, has_result: bool, has_error: bool) -> List[str]:
    if has_error or run_status == "error":
        return [
            "Open run_state and traces to locate the first failing step.",
            "Fix the failing input or tool configuration, then rerun.",
            "If needed, restart with a narrower task scope.",
        ]
    if run_status in {"awaiting_human_feedback"}:
        return [
            "Resume the selected run and provide the required feedback.",
            "Keep the workspace unchanged until the paused run is closed out.",
            "Start a new run only if the paused run is no longer needed.",
        ]
    if run_status in {"running", "starting", "drafting", "planning", "executing"}:
        return [
            "Keep this run selected and watch the live tracker.",
            "Avoid refreshing large detail panes until the run settles.",
            "Prepare follow-up instructions if a HITL prompt appears.",
        ]
    if has_result:
        return [
            "Review the recorded result in run_state and confirm deliverables.",
            "Start a follow-up run only for missing or failed items.",
            "Archive key outputs into the project workspace.",
        ]
    return [
        "Inspect run_state and artifacts for missing outputs.",
        "Run again with clearer acceptance criteria if needed.",
        "Capture the next action in the follow-up prompt.",
    ]


def _infer_status(*, task_state: Dict[str, Any], run_error: Optional[str], has_result: bool) -> str:
    if run_error:
        return "error"
    status = str(task_state.get("status") or "").strip().lower()
    if status:
        return status
    if has_result:
        return "done"
    return "unknown"


def _status_from_task_state(run_dir: Path) -> str:
    task_state = _load_json(run_dir / _RUN_STATE_FILE)
    return str(task_state.get("status") or "").strip().lower()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = ["load_run_summary", "snapshot_summary", "summarize_run", "summary_path"]
