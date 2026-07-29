from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

from .common import compact_text, humanize_identifier, redact_internal_text
from .models import PublicAction, PublicField, PublicPart


_HTML_MARKER_RE = re.compile(r"<!doctype\s+html|<html(?:\s|>)|<body(?:\s|>)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTTP_STATUS_RE = re.compile(r"\b([45]\d{2})\b(?:\s*[-:]\s*|\s+)?([A-Za-z][A-Za-z ]{1,60})?")
_HTTP_STATUS_LABELS = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def _status_code(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        code = int(value)
    except (TypeError, ValueError):
        return 0
    return code if 400 <= code <= 599 else 0


def _validation_summary(value: Any, *, workspace: Path | None) -> str:
    rows = value if isinstance(value, list) else []
    details: list[str] = []
    for row in rows[:6]:
        if not isinstance(row, dict):
            continue
        raw_location = row.get("loc") or row.get("location") or []
        location_parts = (
            [
                str(item)
                for item in raw_location
                if str(item).strip().lower() not in {"body", "query", "path"}
            ]
            if isinstance(raw_location, (list, tuple))
            else [str(raw_location)]
        )
        location = " ".join(
            humanize_identifier(item, fallback=item)
            for item in location_parts
            if str(item).strip()
        )
        message = redact_internal_text(
            row.get("msg") or row.get("message") or row.get("detail"),
            workspace=workspace,
            limit=260,
        )
        if message:
            details.append(f"{location}: {message}" if location else message)
    if not details:
        return ""
    suffix = " Additional validation issues are available in developer diagnostics." if len(rows) > len(details) else ""
    return "The request could not be validated. " + "; ".join(details) + "." + suffix


def _html_error_summary(value: str) -> tuple[str, int]:
    match = _HTTP_STATUS_RE.search(unescape(_HTML_TAG_RE.sub(" ", value)))
    code = _status_code(match.group(1) if match else 0)
    if code:
        label = _HTTP_STATUS_LABELS.get(code, "")
        suffix = f" ({label})" if label else ""
        return f"An upstream service returned HTTP {code}{suffix}.", code
    return "An upstream service returned an HTML error page instead of a usable response.", 0


def _present_error_summary(
    value: Any,
    *,
    workspace: Path | None,
    depth: int = 0,
) -> tuple[str, int]:
    if depth > 4:
        return "CatMaster received an error that could not be displayed safely.", 0
    if isinstance(value, dict):
        code = _status_code(
            value.get("status_code")
            or value.get("statusCode")
            or value.get("http_status")
            or value.get("status")
        )
        detail = value.get("detail")
        validation = _validation_summary(detail, workspace=workspace)
        if not validation:
            validation = _validation_summary(value.get("errors"), workspace=workspace)
        if validation:
            return validation, code or 422
        for key in ("message", "error", "detail", "reason", "summary"):
            nested = value.get(key)
            if nested in (None, "", [], {}):
                continue
            summary, nested_code = _present_error_summary(
                nested,
                workspace=workspace,
                depth=depth + 1,
            )
            if summary:
                return summary, code or nested_code
        if code:
            label = _HTTP_STATUS_LABELS.get(code, "")
            return (
                f"The service returned HTTP {code}{f' ({label})' if label else ''}.",
                code,
            )
        return "CatMaster received a structured error that could not be displayed safely.", 0
    if isinstance(value, list):
        validation = _validation_summary(value, workspace=workspace)
        if validation:
            return validation, 422
        for item in value[:3]:
            summary, code = _present_error_summary(
                item,
                workspace=workspace,
                depth=depth + 1,
            )
            if summary:
                return summary, code
        return "CatMaster received an error list that could not be displayed safely.", 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "", 0
        if _HTML_MARKER_RE.search(text):
            return _html_error_summary(text)
        if text[:1] in "[{":
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if parsed is not None:
                return _present_error_summary(
                    parsed,
                    workspace=workspace,
                    depth=depth + 1,
                )
        return redact_internal_text(text, workspace=workspace, limit=700), 0
    if value is None:
        return "", 0
    return redact_internal_text(value, workspace=workspace, limit=700), 0


def project_error_part(
    *,
    part_id: str,
    summary: Any = "",
    error_code: Any = "",
    retry_safe: bool = False,
    diagnostics_ref: str = "",
    workspace: Path | None = None,
) -> PublicPart:
    safe_summary, http_status = _present_error_summary(summary, workspace=workspace)
    fields = [
        PublicField(label="Task state", value="Stopped"),
        PublicField(
            label="Try again",
            value=(
                "The same request can be submitted again."
                if retry_safe
                else "Review the inputs or diagnostics reference before submitting again."
            ),
        ),
    ]
    if http_status:
        fields.append(PublicField(label="HTTP status", value=str(http_status)))
    safe_code = compact_text(error_code, limit=120)
    if safe_code and all(character.isalnum() or character in "._-" for character in safe_code):
        fields.append(PublicField(label="Error code", value=safe_code, copy_value=safe_code))
    return PublicPart(
        id=part_id,
        type="error",
        status="failed",
        title="The task stopped before it completed",
        summary=safe_summary or "CatMaster could not complete this run.",
        fields=fields,
        actions=[
            PublicAction(
                id="focus_composer",
                label="Review and try again",
                kind="primary",
            )
        ],
        diagnostics_ref=diagnostics_ref,
    )


__all__ = ["project_error_part"]
