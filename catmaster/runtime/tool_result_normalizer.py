from __future__ import annotations

import json
from typing import Any, Mapping

CONTROL_TOOL_NAMES: set[str] = {
    "task_finish",
    "task_fail",
    "proposal_finish",
    "proposal_fail",
    "director_decide",
}

_FAILURE_STATUSES: set[str] = {
    "failed",
    "error",
    "validation_failed",
    "cancelled",
    "timeout",
    "timed_out",
}
_SUCCESS_STATUSES: set[str] = {
    "success",
    "ok",
    "done",
    "validated",
}


def _snippet(text: Any, limit: int = 500) -> str:
    if text is None:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _is_message_like(value: Any) -> bool:
    return hasattr(value, "content") and hasattr(value, "name")


def _parse_message_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return dict(content)
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
            return {"data": loaded}
        except Exception:
            return {"raw_text": text}
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
                continue
            alt = item.get("content")
            if isinstance(alt, str) and alt.strip():
                parts.append(alt)
        joined = "\n".join(parts).strip()
        if not joined:
            return {}
        try:
            loaded = json.loads(joined)
            if isinstance(loaded, dict):
                return loaded
            return {"data": loaded}
        except Exception:
            return {"raw_text": joined}
    return {"raw_text": str(content)}


def is_control_tool_name(tool_name: str | None) -> bool:
    return str(tool_name or "").strip() in CONTROL_TOOL_NAMES


def normalize_tool_result(
    raw: Any,
    *,
    tool_name: str | None = None,
    is_control_tool: bool | None = None,
) -> dict[str, Any]:
    parsed: dict[str, Any]
    status_hint = ""

    if isinstance(raw, Mapping):
        parsed = dict(raw)
    elif _is_message_like(raw):
        parsed = _parse_message_content(getattr(raw, "content", None))
        parsed.setdefault("tool_name", str(getattr(raw, "name", "") or ""))
        status_hint = str(getattr(raw, "status", "") or "").strip().lower()
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            parsed = {}
        else:
            try:
                loaded = json.loads(text)
                parsed = loaded if isinstance(loaded, dict) else {"data": loaded}
            except Exception:
                parsed = {"raw_text": text}
    elif raw is None:
        parsed = {}
    else:
        parsed = {"raw_text": str(raw)}

    if isinstance(parsed.get("toolresult"), Mapping):
        parsed = dict(parsed.get("toolresult") or {})

    resolved_tool_name = str(parsed.get("tool_name") or tool_name or "").strip()
    if not resolved_tool_name and _is_message_like(raw):
        resolved_tool_name = str(getattr(raw, "name", "") or "").strip()
    parsed["tool_name"] = resolved_tool_name

    if is_control_tool is None:
        is_control_tool = is_control_tool_name(resolved_tool_name)

    raw_text = str(parsed.get("raw_text") or "").strip()
    parse_failed = bool(raw_text) and "status" not in parsed and "error" not in parsed
    if parse_failed:
        parsed["status"] = "failed"
        parsed["error"] = _snippet(raw_text, 500)

    status_raw = str(parsed.get("status") or "").strip().lower()
    error_text = _snippet(parsed.get("error"), 500).strip()

    if is_control_tool:
        if status_raw in _FAILURE_STATUSES:
            normalized_status = "failed"
        else:
            normalized_status = "control"
    else:
        if status_raw in _FAILURE_STATUSES:
            normalized_status = "failed"
        elif status_raw in _SUCCESS_STATUSES:
            normalized_status = "success"
        elif error_text:
            normalized_status = "failed"
        elif status_hint == "error":
            normalized_status = "failed"
        elif status_raw:
            # Unknown status values should never be silently treated as success.
            normalized_status = "failed"
        else:
            meaningful_payload = bool(parsed) and not (
                len(parsed) == 1 and "tool_name" in parsed
            )
            normalized_status = "success" if meaningful_payload else "failed"

    data_raw = parsed.get("data")
    if isinstance(data_raw, Mapping):
        normalized_data = dict(data_raw)
    elif data_raw is None:
        normalized_data = {}
    else:
        normalized_data = {"value": data_raw}

    warnings_raw = parsed.get("warnings")
    if isinstance(warnings_raw, list):
        normalized_warnings = [str(item) for item in warnings_raw if item is not None]
    elif warnings_raw in (None, ""):
        normalized_warnings = []
    else:
        normalized_warnings = [str(warnings_raw)]

    normalized_error = parsed.get("error")
    if normalized_error is not None:
        normalized_error = str(normalized_error)
    if normalized_status == "failed" and not normalized_error:
        if raw_text:
            normalized_error = _snippet(raw_text, 500)
        elif status_hint == "error":
            normalized_error = "Tool message reported status=error."
        else:
            normalized_error = "Tool reported failure."

    normalized = dict(parsed)
    normalized["status"] = normalized_status
    normalized["tool_name"] = resolved_tool_name
    normalized["data"] = normalized_data
    normalized["warnings"] = normalized_warnings
    normalized["error"] = normalized_error
    return normalized


def to_tool_message_status(payload: Mapping[str, Any]) -> str:
    status_raw = str(payload.get("status") or "").strip().lower()
    if status_raw in {"failed", "error"}:
        return "error"
    if status_raw in {"success", "control"}:
        return "success"
    if payload.get("error"):
        return "error"
    return "success"


__all__ = [
    "CONTROL_TOOL_NAMES",
    "is_control_tool_name",
    "normalize_tool_result",
    "to_tool_message_status",
]

