from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from catmaster.runtime.run_ledger.models import RunSearchBlob


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _squash(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _collect_task_goals(task_state: Dict[str, Any], *, limit: int = 10) -> List[str]:
    rows = task_state.get("tasks")
    if not isinstance(rows, list):
        return []
    out: List[str] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        for key in ("goal", "task_detail", "title", "task"):
            val = _squash(str(item.get(key) or ""))
            if val:
                out.append(val)
                break
        if len(out) >= limit:
            break
    return out


def _extract_tool_name(payload: Dict[str, Any]) -> str:
    for key in ("tool_name", "name", "tool"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inner = payload.get("payload")
    if isinstance(inner, dict):
        for key in ("tool_name", "name", "tool"):
            value = inner.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _collect_tool_names(tool_trace_path: Path, *, limit: int = 64) -> List[str]:
    if not tool_trace_path.exists():
        return []
    out: List[str] = []
    seen: set[str] = set()
    try:
        lines = tool_trace_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        name = _extract_tool_name(payload)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out


def _collect_artifact_paths(run_dir: Path, task_state: Dict[str, Any], *, limit: int = 40) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    reports_dir = run_dir / "reports"
    if reports_dir.exists():
        for child in sorted(reports_dir.iterdir(), key=lambda p: p.name):
            if not child.is_file():
                continue
            rel = child.relative_to(run_dir)
            text = str(rel).replace("\\", "/")
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                return out

    observations = task_state.get("observations")
    if isinstance(observations, list):
        for item in observations:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if not isinstance(value, str):
                    continue
                if not (key.endswith("_rel") or key.endswith("_path") or "path" in key.lower()):
                    continue
                text = value.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(text)
                if len(out) >= limit:
                    return out
    return out


def _answer_summary(task_state: Dict[str, Any], *, limit: int = 800) -> str:
    summary = _squash(str(task_state.get("summary") or ""))
    return summary[:limit]


def build_run_search_blob(run_dir: Path, *, max_chars: int = 5500) -> RunSearchBlob:
    run_root = Path(run_dir).expanduser().resolve()
    meta = _safe_json(run_root / "meta.json")
    task_state = _safe_json(run_root / "task_state.json")

    request = _squash(str(task_state.get("user_request") or meta.get("user_request") or ""))
    answer_summary = _answer_summary(task_state)
    task_goals = _collect_task_goals(task_state)
    tool_names = _collect_tool_names(run_root / "tool_trace.jsonl")
    artifact_paths = _collect_artifact_paths(run_root, task_state)

    lines: List[str] = [
        f"run_id={_squash(str(meta.get('run_id') or run_root.name))}",
        "",
        "[request]",
        request or "(empty)",
        "",
        "[answer_summary]",
        answer_summary or "(empty)",
    ]
    if task_goals:
        lines.append("")
        lines.append("[task_goals]")
        lines.extend(f"- {item}" for item in task_goals)
    if tool_names:
        lines.append("")
        lines.append("[tools]")
        lines.extend(f"- {item}" for item in tool_names)
    if artifact_paths:
        lines.append("")
        lines.append("[artifacts]")
        lines.extend(f"- {item}" for item in artifact_paths)

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n...[truncated]"

    return RunSearchBlob(
        run_id=_squash(str(meta.get("run_id") or run_root.name)),
        request=request,
        answer_summary=answer_summary,
        task_goals=task_goals,
        tool_names=tool_names,
        artifact_paths=artifact_paths,
        search_blob_text=text,
    )


__all__ = ["build_run_search_blob"]
