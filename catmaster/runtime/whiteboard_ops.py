#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whiteboard ops (UPSERT / DEPRECATE) validation and application.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import difflib
import hashlib
import json


TARGET_SECTIONS = {"Goal", "Key Facts", "Key Files", "Constraints", "Open Questions", "Journal"}
PLACEHOLDERS = {"- (none)", "- (empty)"}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def validate_whiteboard_ops(ops: Any) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(ops, list):
        return {"ok": False, "errors": ["whiteboard_ops must be a list"], "warnings": warnings}
    for idx, op in enumerate(ops, start=1):
        if not isinstance(op, dict):
            errors.append(f"ops[{idx}] must be an object")
            continue
        op_type = str(op.get("op", "")).strip().upper()
        section = str(op.get("section", "")).strip()
        if op_type not in {"UPSERT", "DEPRECATE"}:
            errors.append(f"ops[{idx}].op must be UPSERT or DEPRECATE")
        if section not in TARGET_SECTIONS:
            errors.append(f"ops[{idx}].section must be one of {sorted(TARGET_SECTIONS)}")
        if op_type == "UPSERT":
            if section == "Open Questions":
                if not op.get("text"):
                    errors.append(f"ops[{idx}] Open Questions UPSERT requires text")
            elif section in {"Key Facts", "Key Files", "Constraints"}:
                record_type = str(op.get("record_type", "")).strip().upper()
                if section == "Key Facts" and record_type != "FACT":
                    errors.append(f"ops[{idx}] Key Facts UPSERT requires record_type=FACT")
                if section == "Key Files" and record_type != "FILE":
                    errors.append(f"ops[{idx}] Key Files UPSERT requires record_type=FILE")
                if section == "Constraints" and record_type != "CONSTRAINT":
                    errors.append(f"ops[{idx}] Constraints UPSERT requires record_type=CONSTRAINT")
                if not op.get("id"):
                    errors.append(f"ops[{idx}] UPSERT requires id")
                if section == "Key Files":
                    if not op.get("path"):
                        errors.append(f"ops[{idx}] FILE UPSERT requires path")
                else:
                    if not op.get("text"):
                        errors.append(f"ops[{idx}] UPSERT requires text")
        if op_type == "DEPRECATE":
            if section not in {"Key Facts", "Key Files", "Constraints"}:
                errors.append(f"ops[{idx}] DEPRECATE is only valid for Key Facts/Key Files/Constraints")
            record_type = str(op.get("record_type", "")).strip().upper()
            if section == "Key Facts" and record_type != "FACT":
                errors.append(f"ops[{idx}] Key Facts DEPRECATE requires record_type=FACT")
            if section == "Key Files" and record_type != "FILE":
                errors.append(f"ops[{idx}] Key Files DEPRECATE requires record_type=FILE")
            if section == "Constraints" and record_type != "CONSTRAINT":
                errors.append(f"ops[{idx}] Constraints DEPRECATE requires record_type=CONSTRAINT")
            if not op.get("id"):
                errors.append(f"ops[{idx}] DEPRECATE requires id")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def apply_whiteboard_ops_text(whiteboard_text: str, ops: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    lines = whiteboard_text.splitlines()
    trailing_newline = whiteboard_text.endswith("\n")
    warnings: List[str] = []
    applied_ops: List[Dict[str, Any]] = []
    failed_ops: List[Dict[str, Any]] = []

    for op in ops:
        try:
            op_type = str(op.get("op", "")).strip().upper()
            section = str(op.get("section", "")).strip()
            bounds = _section_bounds(lines)
            if section not in bounds:
                raise ValueError(f"Missing section: {section}")
            if op_type == "UPSERT":
                _apply_upsert(lines, bounds, op, task_id)
                applied_ops.append(op)
            elif op_type == "DEPRECATE":
                deprecated = _apply_deprecate(lines, bounds, op, task_id, warnings)
                if deprecated:
                    applied_ops.append(op)
                else:
                    failed_ops.append(op)
            else:
                failed_ops.append(op)
                warnings.append(f"Unknown op type: {op_type}")
        except Exception as exc:
            failed_ops.append(op)
            warnings.append(f"Failed to apply op {op}: {exc}")

    updated = "\n".join(lines) + ("\n" if trailing_newline else "")
    return {
        "updated_text": updated,
        "applied_ops": applied_ops,
        "failed_ops": failed_ops,
        "warnings": warnings,
    }


def apply_whiteboard_ops_atomic(whiteboard_path: Path, ops: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    if not whiteboard_path.exists():
        return {"ok": False, "errors": [f"Whiteboard not found: {whiteboard_path}"]}
    try:
        before_text = whiteboard_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "errors": [f"Failed to read whiteboard: {exc}"]}
    before_hash = hashlib.sha256(before_text.encode("utf-8")).hexdigest()
    applied = apply_whiteboard_ops_text(before_text, ops, task_id)
    after_text = applied["updated_text"]
    after_hash = hashlib.sha256(after_text.encode("utf-8")).hexdigest()

    tmp_path = whiteboard_path.with_suffix(whiteboard_path.suffix + ".tmp")
    tmp_path.write_text(after_text, encoding="utf-8")
    tmp_path.replace(whiteboard_path)
    return {
        "ok": True,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "warnings": applied.get("warnings", []),
        "failed_ops": applied.get("failed_ops", []),
        "before_text": before_text,
        "after_text": after_text,
    }


def whiteboard_ops_persist(ops: List[Dict[str, Any]], metadata: Dict[str, Any], *, root: Path) -> Dict[str, Any]:
    run_id = metadata.get("run_id")
    task_id = metadata.get("task_id")
    attempt = metadata.get("attempt")
    if not run_id or not task_id or attempt is None:
        raise ValueError("whiteboard_ops_persist metadata requires run_id, task_id, and attempt")
    ops_dir = root / "whiteboard_ops" / run_id
    ops_dir.mkdir(parents=True, exist_ok=True)
    ops_path = ops_dir / f"{task_id}_{attempt}.json"
    ops_path.write_text(json.dumps(ops, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ops_path": str(ops_path)}


def persist_whiteboard_diff(
    before_text: str,
    after_text: str,
    metadata: Dict[str, Any],
    *,
    root: Path,
    whiteboard_path: Path,
) -> Dict[str, Any]:
    run_id = metadata.get("run_id")
    task_id = metadata.get("task_id")
    attempt = metadata.get("attempt")
    if not run_id or not task_id or attempt is None:
        raise ValueError("persist_whiteboard_diff metadata requires run_id, task_id, and attempt")
    diff = difflib.unified_diff(
        before_text.splitlines(),
        after_text.splitlines(),
        fromfile=str(Path("a") / whiteboard_path.relative_to(root.parent)),
        tofile=str(Path("b") / whiteboard_path.relative_to(root.parent)),
        lineterm="",
    )
    diff_text = "\n".join(diff) + "\n"
    diff_dir = root / "patches" / run_id
    diff_dir.mkdir(parents=True, exist_ok=True)
    diff_path = diff_dir / f"{task_id}_{attempt}.diff"
    diff_path.write_text(diff_text, encoding="utf-8")
    return {"diff_path": str(diff_path)}


def _section_bounds(lines: List[str]) -> Dict[str, Tuple[int, int]]:
    headers: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines):
        if line.startswith("### "):
            name = line[4:].strip()
        elif line.startswith("## "):
            name = line[3:].strip()
        else:
            continue
        if name in TARGET_SECTIONS:
            headers.append((name, idx))
    headers.sort(key=lambda item: item[1])
    bounds: Dict[str, Tuple[int, int]] = {}
    for i, (name, start) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(lines)
        bounds[name] = (start, end)
    return bounds


def _strip_placeholders(body: List[str]) -> List[str]:
    return [line for line in body if line.strip() not in PLACEHOLDERS]


def _apply_upsert(lines: List[str], bounds: Dict[str, Tuple[int, int]], op: Dict[str, Any], task_id: str) -> None:
    section = op.get("section")
    start, end = bounds[section]
    body = _strip_placeholders(lines[start + 1:end])
    now = _now_iso()

    if section == "Open Questions":
        text = str(op.get("text", "")).strip()
        record_type = str(op.get("record_type", "")).strip().upper()
        record_id = (op.get("id") or "").strip()
        if record_type == "QUESTION" or record_id:
            qid = record_id or _stable_id(text)
            new_line = f"QUESTION[{qid}]: {text} | added={now}"
        else:
            new_line = f"- {text}"
        if new_line not in body:
            body.append(new_line)
        lines[start + 1:end] = body
        return

    record_type = str(op.get("record_type", "")).strip().upper()
    record_id = _normalize_id(record_type, str(op.get("id", "")).strip())
    if section == "Key Facts":
        source = str(op.get("source") or "").strip()
        text = str(op.get("text", "")).strip()
        new_line = _build_fact_line(record_id, text, "active", source, now, task_id=task_id)
        _replace_or_append(body, f"FACT[{record_id}]:", new_line)
    elif section == "Key Files":
        path = str(op.get("path", "")).strip()
        kind = str(op.get("kind") or "output").strip()
        desc = _truncate_desc(str(op.get("description") or op.get("desc") or "").strip())
        new_line = _build_file_line(record_id, path, kind, desc, now)
        _replace_or_append(body, f"FILE[{record_id}]:", new_line)
    elif section == "Constraints":
        text = str(op.get("text", "")).strip()
        rationale = str(op.get("rationale") or op.get("reason") or f"TASK[{task_id}]").strip()
        new_line = _build_constraint_line(record_id, text, rationale, now)
        _replace_or_append(body, f"CONSTRAINT[{record_id}]:", new_line)
    lines[start + 1:end] = body


def _apply_deprecate(
    lines: List[str],
    bounds: Dict[str, Tuple[int, int]],
    op: Dict[str, Any],
    task_id: str,
    warnings: List[str],
) -> bool:
    section = op.get("section")
    start, end = bounds[section]
    body = lines[start + 1:end]
    record_type = str(op.get("record_type", "")).strip().upper()
    record_id = _normalize_id(record_type, str(op.get("id", "")).strip())
    now = _now_iso()
    reason = str(op.get("reason") or "").strip()
    superseded_by = str(op.get("superseded_by") or "").strip()

    prefix = f"{record_type}[{record_id}]:"
    for idx, line in enumerate(body):
        if line.strip().startswith(prefix):
            parsed = _parse_record_line(line)
            if not parsed:
                break
            _, _, main, attrs = parsed
            if record_type == "FACT":
                source = attrs.get("source") or ""
                updated = attrs.get("updated") or now
                new_line = _build_fact_line(
                    record_id,
                    main,
                    "deprecated",
                    source,
                    updated,
                    task_id=task_id,
                    deprecated_at=now,
                    reason=reason,
                    superseded_by=superseded_by,
                )
            elif record_type == "FILE":
                kind = attrs.get("kind") or "output"
                desc = attrs.get("desc") or attrs.get("description") or ""
                updated = attrs.get("updated") or now
                new_line = _build_file_line(
                    record_id,
                    main,
                    kind,
                    desc,
                    updated,
                    status="deprecated",
                    deprecated_at=now,
                    reason=reason,
                    superseded_by=superseded_by,
                )
            elif record_type == "CONSTRAINT":
                rationale = attrs.get("rationale") or f"TASK[{task_id}]"
                added = attrs.get("added") or now
                new_line = _build_constraint_line(
                    record_id,
                    main,
                    rationale,
                    added,
                    status="deprecated",
                    deprecated_at=now,
                    reason=reason,
                    superseded_by=superseded_by,
                )
            else:
                new_line = line
            body[idx] = new_line
            lines[start + 1:end] = body
            return True

    warnings.append(f"Attempted to deprecate {record_type}[{record_id}] but record not found.")
    _append_journal_note(lines, bounds, f"Attempted to deprecate {record_type}[{record_id}] but record not found.")
    return False


def _append_journal_note(lines: List[str], bounds: Dict[str, Tuple[int, int]], note: str) -> None:
    if "Journal" not in bounds:
        return
    start, end = bounds["Journal"]
    body = _strip_placeholders(lines[start + 1:end])
    timestamp = _now_iso()
    body.append(f"- {timestamp} NOTE {note}")
    lines[start + 1:end] = body


def _replace_or_append(body: List[str], prefix: str, new_line: str) -> None:
    for idx, line in enumerate(body):
        if line.strip().startswith(prefix):
            body[idx] = new_line
            return
    body.append(new_line)


def _build_fact_line(
    record_id: str,
    text: str,
    status: str,
    source: str,
    updated: str,
    *,
    task_id: str,
    deprecated_at: Optional[str] = None,
    reason: Optional[str] = None,
    superseded_by: Optional[str] = None,
) -> str:
    parts = [
        f"FACT[{record_id}]: {text}",
    ]
    if status == "deprecated":
        parts.append("status=deprecated")
    if source:
        cleaned = _sanitize_source(source, task_id)
        if cleaned:
            parts.append(f"source={cleaned}")
    if deprecated_at:
        parts.append(f"deprecated_at={deprecated_at}")
    if reason:
        parts.append(f"reason={reason}")
    if superseded_by:
        parts.append(f"superseded_by={superseded_by}")
    return " | ".join(parts)


def _build_file_line(
    record_id: str,
    path: str,
    kind: str,
    desc: str,
    updated: str,
    *,
    status: Optional[str] = None,
    deprecated_at: Optional[str] = None,
    reason: Optional[str] = None,
    superseded_by: Optional[str] = None,
) -> str:
    parts = [
        f"FILE[{record_id}]: {path}",
    ]
    if desc:
        parts.append(f"desc={desc}")
    if status:
        parts.append(f"status={status}")
    if deprecated_at:
        parts.append(f"deprecated_at={deprecated_at}")
    if reason:
        parts.append(f"reason={reason}")
    if superseded_by:
        parts.append(f"superseded_by={superseded_by}")
    return " | ".join(parts)


def _build_constraint_line(
    record_id: str,
    text: str,
    rationale: str,
    added: str,
    *,
    status: Optional[str] = None,
    deprecated_at: Optional[str] = None,
    reason: Optional[str] = None,
    superseded_by: Optional[str] = None,
) -> str:
    parts = [
        f"CONSTRAINT[{record_id}]: {text}",
        f"rationale={rationale}",
        f"added={added}",
    ]
    if status:
        parts.append(f"status={status}")
    if deprecated_at:
        parts.append(f"deprecated_at={deprecated_at}")
    if reason:
        parts.append(f"reason={reason}")
    if superseded_by:
        parts.append(f"superseded_by={superseded_by}")
    return " | ".join(parts)


def _parse_record_line(line: str) -> Optional[Tuple[str, str, str, Dict[str, str]]]:
    if "]:" not in line or "[" not in line:
        return None
    prefix, rest = line.split("]:", 1)
    record_type, record_id = prefix.split("[", 1)
    record_type = record_type.strip()
    record_id = record_id.strip()
    rest = rest.strip()
    if not rest:
        return record_type, record_id, "", {}
    parts = [part.strip() for part in rest.split(" | ")]
    main = parts[0]
    attrs: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        attrs[key.strip()] = value.strip()
    return record_type, record_id, main, attrs


def _stable_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"Q_{digest}"


def _normalize_id(record_type: str, record_id: str) -> str:
    if not record_id:
        return record_id
    record_type = record_type.upper()
    normalized = record_id.strip()
    prefix_bracket = f"{record_type}["
    while normalized.upper().startswith(prefix_bracket):
        if normalized.endswith("]"):
            normalized = normalized[len(prefix_bracket):-1]
        else:
            normalized = normalized[len(prefix_bracket):]
        normalized = normalized.strip()
    prefix_map = {
        "FACT": "FACT_",
        "FILE": "FILE_",
        "CONSTRAINT": "CONSTRAINT_",
    }
    prefix = prefix_map.get(record_type)
    if prefix and normalized.upper().startswith(prefix):
        normalized = normalized[len(prefix):]
    return normalized.strip().strip("[]")


def _sanitize_source(source: str, task_id: str) -> str:
    lowered = source.lower()
    if "local observations" in lowered or "step " in lowered:
        return ""
    if source.strip() == f"TASK[{task_id}]":
        return ""
    return source.strip()


def _truncate_desc(desc: str, limit: int = 160) -> str:
    if not desc:
        return ""
    if len(desc) <= limit:
        return desc
    return desc[:limit].rstrip() + "…"


def append_task_journal_entry(
    whiteboard_text: str,
    *,
    task_id: str,
    outcome: str,
    summary: str,
    artifacts: List[str],
    max_artifacts: int = 4,
) -> str:
    lines = whiteboard_text.splitlines()
    trailing_newline = whiteboard_text.endswith("\n")
    bounds = _section_bounds(lines)
    if "Journal" not in bounds:
        return whiteboard_text
    start, end = bounds["Journal"]
    body = _strip_placeholders(lines[start + 1:end])
    timestamp = _now_iso()
    short_summary = summary.strip().replace("\n", " ")
    if len(short_summary) > 200:
        short_summary = short_summary[:200].rstrip() + "…"
    trimmed = [path for path in artifacts if path][:max_artifacts]
    artifacts_text = ", ".join(trimmed)
    entry = f"- {timestamp} TASK[{task_id}] outcome={outcome} summary=\"{short_summary}\" artifacts=[{artifacts_text}]"
    body.append(entry)
    lines[start + 1:end] = body
    return "\n".join(lines) + ("\n" if trailing_newline else "")


__all__ = [
    "validate_whiteboard_ops",
    "apply_whiteboard_ops_text",
    "apply_whiteboard_ops_atomic",
    "append_task_journal_entry",
    "whiteboard_ops_persist",
    "persist_whiteboard_diff",
]
