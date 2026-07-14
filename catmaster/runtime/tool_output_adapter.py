from __future__ import annotations

"""Tool-return post-processor for LangChain `content_and_artifact` flow.

This module is intentionally strict:
- accepted raw returns: `(content, artifact)` or `ToolMessage`
- non-conforming returns raise `CatMasterToolExecutionError`

Post-processing responsibilities:
- model-visible projection of small file-reference fields
- large-field offload in artifact `data`
- whole-artifact offload when oversized
"""

from dataclasses import dataclass
from datetime import datetime
import json
import re
import traceback
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from langchain_core.messages import ToolMessage

from catmaster.runtime.tool_output_config import ToolOutputConfig, get_tool_output_config


_INPUT_ECHO_KEYS = {"raw_params", "tool_args", "validated_params"}
_MAX_INLINE_FILE_REFS_PER_LIST = 12
_FILE_REFERENCE_TOKENS = {
    "artifact",
    "artifacts",
    "bbl",
    "bib",
    "csv",
    "dependencies",
    "dir",
    "directory",
    "file",
    "files",
    "image",
    "images",
    "incar",
    "json",
    "kpoints",
    "log",
    "manifest",
    "metadata",
    "path",
    "paths",
    "pdf",
    "poscar",
    "receipt",
    "ref",
    "rel",
    "report",
    "root",
    "structure",
    "summary",
    "trajectory",
    "view",
    "views",
    "xyz",
}
_KNOWN_FILE_NAMES = {
    "CONTCAR",
    "INCAR",
    "KPOINTS",
    "OUTCAR",
    "POSCAR",
    "POTCAR",
    "vasprun.xml",
}
_KNOWN_FILE_SUFFIXES = {
    ".bib",
    ".cif",
    ".csv",
    ".docx",
    ".extxyz",
    ".gif",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".log",
    ".md",
    ".nbib",
    ".pdf",
    ".png",
    ".pptx",
    ".ris",
    ".tex",
    ".tif",
    ".tiff",
    ".traj",
    ".vasp",
    ".xlsx",
    ".xml",
    ".xyz",
    ".yaml",
    ".yml",
}


def _strip_input_echoes(artifact: dict[str, Any]) -> None:
    for key in _INPUT_ECHO_KEYS:
        artifact.pop(key, None)
    if isinstance(artifact.get("data"), Mapping):
        data = dict(artifact["data"])
        for key in _INPUT_ECHO_KEYS:
            data.pop(key, None)
        artifact["data"] = data


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    try:
        return json.dumps(_json_safe(content), ensure_ascii=False)
    except Exception:
        return str(content)


def _is_file_reference_key(key: Any) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", str(key or "").lower()) if token}
    return bool(tokens & _FILE_REFERENCE_TOKENS)


def _is_explicit_file_reference_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    return bool(tokens & {"dir", "directory", "file", "files", "path", "paths", "ref", "rel", "root"})


def _looks_like_file_reference(value: Any, *, explicit_key: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or len(text) > 4_096 or "\n" in text or "\r" in text:
        return False
    lowered = text.lower()
    if lowered.startswith(("data:", "http://", "https://")):
        return False
    if explicit_key:
        return True
    if text in _KNOWN_FILE_NAMES or "/" in text or "\\" in text:
        return True
    return Path(text).suffix.lower() in _KNOWN_FILE_SUFFIXES


def _file_reference_projection(data: Mapping[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
    refs: list[tuple[str, str]] = []
    large_lists: list[tuple[str, int]] = []

    def collect(value: Any, *, label: str, inherited_hint: bool = False) -> list[tuple[str, str]]:
        if isinstance(value, Mapping):
            found: list[tuple[str, str]] = []
            for key, child in value.items():
                child_label = f"{label}.{key}" if label else str(key)
                key_hint = _is_file_reference_key(key)
                if isinstance(child, (list, tuple)):
                    list_refs = collect(child, label=child_label, inherited_hint=key_hint or inherited_hint)
                    if len(list_refs) > _MAX_INLINE_FILE_REFS_PER_LIST:
                        large_lists.append((child_label, len(list_refs)))
                    else:
                        found.extend(list_refs)
                    continue
                found.extend(collect(child, label=child_label, inherited_hint=key_hint or inherited_hint))
            return found
        if isinstance(value, (list, tuple)):
            found: list[tuple[str, str]] = []
            for index, child in enumerate(value):
                found.extend(collect(child, label=f"{label}[{index}]", inherited_hint=inherited_hint))
            return found
        if _looks_like_file_reference(value, explicit_key=inherited_hint and _is_explicit_file_reference_key(label)):
            return [(label, str(value).strip())]
        if inherited_hint and _looks_like_file_reference(value):
            return [(label, str(value).strip())]
        return []

    refs.extend(collect(data, label=""))
    deduped: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for label, value in refs:
        if value in seen_values:
            continue
        seen_values.add(value)
        deduped.append((label, value))
    return deduped, large_lists


def _append_text_content(content: Any, extra_text: str) -> Any:
    note = str(extra_text or "").strip()
    if not note:
        return content
    if isinstance(content, str):
        base = content.rstrip()
        return f"{base}\n{note}" if base else note
    if isinstance(content, list):
        return [*content, {"type": "text", "text": note}]
    base = content_to_text(content).strip()
    return f"{base}\n{note}" if base else note


def _project_file_references_into_content(content: Any, data: Mapping[str, Any]) -> Any:
    refs, large_lists = _file_reference_projection(data)
    existing = content_to_text(content)
    missing_refs = [(label, value) for label, value in refs if value not in existing]
    if not missing_refs and not large_lists:
        return content

    lines: list[str] = []
    if missing_refs:
        lines.append("File references:")
        lines.extend(f"- {label}={value}" for label, value in missing_refs)
    if large_lists:
        lines.append("Large file lists omitted from model context:")
        lines.extend(
            f"- {label}: {count} file reference(s); inspect the reported manifest or output root if needed."
            for label, count in large_lists
        )
    return _append_text_content(content, "\n".join(lines))


def _snippet(text: str, limit: int = 400) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _encoded_length(value: Any) -> int:
    try:
        return len(json.dumps(_json_safe(value), ensure_ascii=False))
    except Exception:
        return len(str(value))


def _safe_tool_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip())
    token = token.strip("._")
    return token or "tool"


def _write_offload_payload(
    *,
    tool_name: str,
    payload: Mapping[str, Any],
    target_root: Path,
    config: ToolOutputConfig,
) -> str:
    token = _safe_tool_token(tool_name)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    nonce = uuid4().hex[:8]
    rel = Path(config.offload_dir_rel) / token / f"{token}_{ts}_{nonce}.json"
    full = (target_root / rel).resolve()
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rel.as_posix()


def _should_offload(value: Mapping[str, Any], config: ToolOutputConfig) -> bool:
    return _encoded_length(value) > config.offload_chars


def _should_offload_data_field(value: Any, config: ToolOutputConfig) -> bool:
    if value in (None, "", [], {}):
        return False
    return _encoded_length(value) > config.offload_chars


def _value_preview(value: Any, *, config: ToolOutputConfig) -> str:
    if isinstance(value, str):
        return _snippet(value, config.preview_chars)
    if isinstance(value, list):
        preview = json.dumps(_json_safe(value), ensure_ascii=False)
        return f"{len(value)} items; value={_snippet(preview, config.preview_chars)}"
    if isinstance(value, Mapping):
        preview = json.dumps(_json_safe(value), ensure_ascii=False)
        return f"{len(value)} keys; value={_snippet(preview, config.preview_chars)}"
    return _snippet(str(value), min(240, config.preview_chars))


def _offload_data_fields(
    *,
    tool_name: str,
    data: Mapping[str, Any],
    target_root: Path,
    config: ToolOutputConfig,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    updated = dict(data)
    refs: list[dict[str, str]] = []
    for field_name, field_value in list(data.items()):
        if not _should_offload_data_field(field_value, config):
            continue
        offload_ref = _write_offload_payload(
            tool_name=tool_name,
            payload={
                "tool_name": tool_name,
                "field": str(field_name),
                "value": _json_safe(field_value),
            },
            target_root=target_root,
            config=config,
        )
        refs.append({"field": str(field_name), "offload_ref": offload_ref})
        updated[field_name] = {
            "offload_ref": offload_ref,
            "preview": _value_preview(field_value, config=config),
        }
    return updated, refs


@dataclass
class CatMasterToolExecutionError(RuntimeError):
    tool_name: str
    public_message: str
    artifact: dict[str, Any]
    retryable: bool = False
    error_code: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.public_message


def tool_error_to_message(
    *,
    exc: Exception,
    tool_name: str,
    tool_call_id: str,
) -> ToolMessage:
    if isinstance(exc, CatMasterToolExecutionError):
        message = exc.public_message
        artifact = {
            **_json_safe(exc.artifact),
            "retryable": bool(exc.retryable),
            "error_code": str(exc.error_code or ""),
        }
    else:
        message = f"{type(exc).__name__}: {exc}"
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        artifact = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_summary": _snippet(tb_text, 1_200),
        }
    _strip_input_echoes(artifact)
    data = artifact.get("data")
    if isinstance(data, Mapping) and data:
        message = _project_file_references_into_content(message, data)
    return ToolMessage(
        content=message,
        artifact=artifact,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error",
    )


def adapt_tool_return(
    *,
    tool_name: str,
    raw_result: Any,
    workspace_files_root: Path | None = None,
    output_config: ToolOutputConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Normalize tool returns after execution and before ToolMessage emission.

    This is a post-processor, not a legacy compatibility bridge.
    """
    config = output_config or get_tool_output_config()
    files_root = (workspace_files_root or Path.cwd().resolve()).resolve()

    content: Any
    artifact: dict[str, Any]

    if isinstance(raw_result, ToolMessage):
        if raw_result.status == "error":
            raise CatMasterToolExecutionError(
                tool_name=tool_name,
                public_message=content_to_text(raw_result.content).strip() or f"{tool_name} failed.",
                artifact=_json_safe(raw_result.artifact)
                if isinstance(raw_result.artifact, Mapping)
                else {"raw_artifact": _json_safe(raw_result.artifact)},
                retryable=False,
                error_code="tool_message_error",
            )
        content = raw_result.content
        artifact = (
            dict(raw_result.artifact)
            if isinstance(raw_result.artifact, Mapping)
            else {"raw_artifact": _json_safe(raw_result.artifact)}
        )
    elif isinstance(raw_result, tuple):
        if len(raw_result) != 2:
            raise CatMasterToolExecutionError(
                tool_name=tool_name,
                public_message=(
                    "Tool response_format=content_and_artifact requires a 2-tuple "
                    "of (content, artifact)."
                ),
                artifact={"raw_value": _json_safe(raw_result)},
                error_code="invalid_return_tuple",
            )
        content, raw_artifact = raw_result
        if not isinstance(raw_artifact, Mapping):
            raise CatMasterToolExecutionError(
                tool_name=tool_name,
                public_message=f"Invalid artifact type: {type(raw_artifact).__name__}",
                artifact={"raw_artifact": _json_safe(raw_artifact)},
                error_code="invalid_artifact_type",
            )
        artifact = dict(raw_artifact)
    else:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"Invalid tool return type: {type(raw_result).__name__}. "
                "Expected tuple(content, artifact) or ToolMessage."
            ),
            artifact={"raw_value": _json_safe(raw_result)},
            retryable=False,
            error_code="invalid_return_type",
        )

    suppress_content_offload_ref = bool(artifact.pop("suppress_content_offload_ref", False))
    _strip_input_echoes(artifact)
    if content in (None, ""):
        content = f"{tool_name} completed."
    data = artifact.get("data")
    if isinstance(data, Mapping) and data:
        content = _project_file_references_into_content(content, data)
        normalized_data, field_refs = _offload_data_fields(
            tool_name=tool_name,
            data=data,
            target_root=files_root,
            config=config,
        )
        artifact["data"] = normalized_data
        if field_refs:
            artifact["field_offload_refs"] = field_refs
            highlights = list(artifact.get("highlights") or [])
            highlights.extend([f"{item['field']} offloaded: {item['offload_ref']}" for item in field_refs])
            artifact["highlights"] = highlights

    text = content_to_text(content)
    if _should_offload(artifact, config):
        offload_ref = _write_offload_payload(
            tool_name=tool_name,
            payload=artifact,
            target_root=files_root,
            config=config,
        )
        previous_artifact = dict(artifact)
        artifact = {
            "offload_refs": [offload_ref],
            "summary": _snippet(text or f"{tool_name} completed."),
        }
        for key in ("warnings", "highlights", "raw_kind", "field_offload_refs"):
            if key in previous_artifact:
                artifact[key] = _json_safe(previous_artifact.get(key))
        summary = content_to_text(content).strip() or f"{tool_name} completed."
        content = summary if suppress_content_offload_ref else f"{summary}\nOffload: {offload_ref}"

    return content, artifact


__all__ = [
    "CatMasterToolExecutionError",
    "adapt_tool_return",
    "content_to_text",
    "tool_error_to_message",
]
