from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from .models import PublicField, TruncationInfo


_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|password|passwd|secret|token|authorization|cookie|environment|env_vars?)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.-])(?:/[A-Za-z0-9_.@+ -]+){2,}|[A-Za-z]:\\(?:[^\\\s]+\\)+[^\\\s]+")
_INTERNAL_SUFFIX_RE = re.compile(r"(?:^|[_\s-])(worker|tool)$", re.IGNORECASE)
_OPAQUE_AGENT_SOURCE_RE = re.compile(
    r"(?:^|[\s:/.])tools?(?:[\s:/.]|$)|[0-9a-f]{8}(?:[\s-]+[0-9a-f]{4}){3}[\s-]+[0-9a-f]{12}",
    re.IGNORECASE,
)
_AGENT_LABELS = {
    "general-purpose": "Research assistant",
    "litreview_agent": "Literature review",
    "research_specialist": "Research",
    "materials_worker": "Materials",
    "ml_worker": "Machine learning",
    "dynamics_worker": "Dynamics",
    "writing_worker_agent": "Writing",
    "writing_polisher_agent": "Writing",
    "peer_review_worker_agent": "Review",
}


def encode_public_cursor(kind: str, identity: str, position: str | int) -> str:
    payload = json.dumps(
        {"v": 1, "kind": kind, "id": identity, "position": position},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_public_cursor(value: str, *, kind: str, identity: str) -> str | int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Cursor is empty.")
    try:
        padded = text + ("=" * (-len(text) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Cursor is invalid.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != 1
        or payload.get("kind") != kind
        or payload.get("id") != identity
        or not isinstance(payload.get("position"), (str, int))
        or isinstance(payload.get("position"), bool)
    ):
        raise ValueError("Cursor does not belong to this content.")
    return payload["position"]


def compact_text(value: Any, *, limit: int | None = 320) -> str:
    text = " ".join(str(value or "").split())
    if limit is None or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def humanize_identifier(value: Any, *, fallback: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"^(?:mcp__|tool__)", "", text)
    text = _INTERNAL_SUFFIX_RE.sub("", text)
    text = re.sub(r"[_./:-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else fallback


def humanize_agent_name(value: Any, *, fallback: str = "Specialist") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    normalized = text.lower()
    if normalized in _AGENT_LABELS:
        return _AGENT_LABELS[normalized]
    for name, label in _AGENT_LABELS.items():
        if name in normalized:
            return label
    if _OPAQUE_AGENT_SOURCE_RE.search(text):
        return fallback
    return humanize_identifier(text, fallback=fallback)


def display_path(value: Any, *, workspace: Path | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if workspace is not None:
            try:
                return str(candidate.resolve().relative_to(workspace.resolve())).replace("\\", "/")
            except Exception:
                pass
        return f"…/{candidate.name}" if candidate.name else "external path"
    normalized = text.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def redact_internal_text(
    value: Any,
    *,
    workspace: Path | None = None,
    limit: int | None = 500,
) -> str:
    text = compact_text(value, limit=limit)
    if workspace is not None:
        text = text.replace(str(workspace), "workspace")
    return _ABSOLUTE_PATH_RE.sub(lambda match: display_path(match.group(0), workspace=workspace), text)


def scalar_text(value: Any, *, workspace: Path | None = None, limit: int = 220) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value
        if "/" in text or "\\" in text:
            text = display_path(text, workspace=workspace)
        return redact_internal_text(text, workspace=workspace, limit=limit)
    return ""


def safe_scalar_fields(
    payload: Any,
    *,
    workspace: Path | None = None,
    preferred_keys: tuple[str, ...] = (),
    limit: int = 6,
    include_remaining: bool = True,
) -> list[PublicField]:
    if not isinstance(payload, dict):
        return []
    ordered_keys = list(preferred_keys)
    if include_remaining:
        ordered_keys.extend(key for key in payload if key not in ordered_keys)
    fields: list[PublicField] = []
    seen: set[str] = set()
    for raw_key in ordered_keys:
        key = str(raw_key)
        if key in seen or key not in payload or _SENSITIVE_KEY_RE.search(key):
            continue
        seen.add(key)
        if key.startswith("_") or key in {
            "thread_id",
            "run_id",
            "message_id",
            "tool_call_id",
            "submission_hash",
            "params_full",
            "raw_params",
            "metadata",
            "content",
            "prompt",
            "code",
            "script",
            "payload",
            "input",
            "output",
        }:
            continue
        value = scalar_text(payload.get(key), workspace=workspace)
        if not value:
            continue
        fields.append(PublicField(label=humanize_identifier(key), value=value))
        if len(fields) >= limit:
            break
    return fields


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text or text[:1] not in "[{":
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def truncate_text(
    text: Any,
    *,
    limit: int,
    full_content_ref: str,
) -> tuple[str, TruncationInfo]:
    value = str(text or "")
    truncated = len(value) > limit
    shown = value[:limit] if truncated else value
    return shown, TruncationInfo(
        shown_count=len(shown),
        total_count=len(value),
        truncated=truncated,
        next_cursor=str(len(shown)) if truncated else "",
        full_content_ref=full_content_ref if truncated else "",
        unit="characters",
    )
