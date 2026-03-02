from __future__ import annotations

from typing import Any, Mapping
import json

from langchain_core.messages import ToolMessage

from catmaster.runtime.tool_output_adapter import content_to_text


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _snippet(text: str, limit: int = 320) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _artifact_summary(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        keys = [str(k) for k in list(artifact.keys())[:20]]
        counts: dict[str, int] = {}
        for key, value in artifact.items():
            if isinstance(value, list):
                counts[str(key)] = len(value)
        summary: dict[str, Any] = {"kind": "mapping", "keys": keys}
        if counts:
            summary["list_counts"] = counts
        return summary
    if isinstance(artifact, list):
        return {"kind": "list", "count": len(artifact)}
    return {"kind": type(artifact).__name__}


def project_tool_observation(output: Any, *, tool_name: str | None = None) -> dict[str, Any]:
    resolved_name = str(tool_name or "").strip()
    status = "success"
    error_text = ""
    content: Any = ""
    artifact: Any = None
    raw_kind = type(output).__name__
    warnings: list[str] = []
    highlights: list[str] = []
    offload_refs: list[str] = []

    if isinstance(output, ToolMessage):
        raw_kind = "tool_message"
        resolved_name = str(getattr(output, "name", "") or resolved_name)
        status = str(getattr(output, "status", "success") or "success")
        content = getattr(output, "content", "")
        artifact = getattr(output, "artifact", None)
    elif isinstance(output, tuple) and len(output) == 2:
        raw_kind = "content_and_artifact"
        content, artifact = output
    elif isinstance(output, Mapping):
        raw_kind = "mapping"
        data = dict(output)
        resolved_name = str(data.get("tool_name") or resolved_name)
        raw_status = str(data.get("status") or "").strip().lower()
        if raw_status in {"failed", "error", "validation_failed", "timeout", "cancelled", "timed_out"}:
            status = "error"
        content = data.get("content") or data.get("summary") or ""
        if not content:
            try:
                content = json.dumps(_json_safe(data), ensure_ascii=False)
            except Exception:
                content = str(data)
        artifact = data.get("artifact") if "artifact" in data else data
        if data.get("error"):
            error_text = str(data.get("error") or "")
            status = "error"
    else:
        content = output

    content_text = content_to_text(content)
    if status == "error" and not error_text:
        error_text = _snippet(content_text, 600)

    if isinstance(artifact, Mapping):
        warnings_raw = artifact.get("warnings")
        if isinstance(warnings_raw, list):
            warnings = [str(v) for v in warnings_raw if v is not None]
        elif warnings_raw:
            warnings = [str(warnings_raw)]
        highlights_raw = artifact.get("highlights")
        if isinstance(highlights_raw, list):
            highlights = [str(v) for v in highlights_raw if v is not None]
        offload_raw = artifact.get("offload_refs")
        if isinstance(offload_raw, list):
            offload_refs = [str(v) for v in offload_raw if v is not None]

    return {
        "tool_name": resolved_name,
        "tool_status": "error" if status == "error" else "success",
        "content_text": content_text,
        "content_preview": _snippet(content_text),
        "artifact_summary": _artifact_summary(artifact),
        "highlights": highlights,
        "warnings": warnings,
        "error": error_text,
        "offload_refs": offload_refs,
        "raw_kind": raw_kind,
    }


__all__ = ["project_tool_observation"]
