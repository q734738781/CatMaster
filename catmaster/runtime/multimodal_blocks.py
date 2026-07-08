from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CURRENT_TURN_INLINE_LIMIT_BYTES = 32 * 1024 * 1024
DEFAULT_TEXT_ATTACHMENT_CHAR_LIMIT = 20_000

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".aiff", ".aac", ".ogg", ".flac"}
VIDEO_SUFFIXES = {".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv", ".3gpp"}
DOCUMENT_SUFFIXES = {".pdf", ".ppt", ".pptx"}
TEXT_SUFFIXES = {
    ".csv",
    ".css",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".markdown",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class ModelMultimodalCapability:
    images: bool = True
    pdfs: bool = True
    documents: bool = True
    audio: bool = False
    video: bool = False
    tool_results: bool = True
    current_turn_inline_limit_bytes: int = DEFAULT_CURRENT_TURN_INLINE_LIMIT_BYTES

    @classmethod
    def from_llm_config(cls, cfg: Any) -> "ModelMultimodalCapability":
        provider = str(getattr(cfg, "provider", "") or "").strip().lower()
        if provider in {"openai", "openrouter", "anthropic", "gemini", "langchain"}:
            capability = cls()
        else:
            capability = cls(images=False, pdfs=False, documents=False, audio=False, video=False, tool_results=False)

        overrides: dict[str, Any] = {}
        raw_top = getattr(cfg, "multimodal", None)
        if isinstance(raw_top, dict):
            overrides.update(raw_top)
        provider_options = getattr(cfg, "provider_options", None)
        if isinstance(provider_options, dict):
            for key in (provider, "multimodal"):
                raw = provider_options.get(key)
                if isinstance(raw, dict) and isinstance(raw.get("multimodal"), dict):
                    overrides.update(raw["multimodal"])
                elif key == "multimodal" and isinstance(raw, dict):
                    overrides.update(raw)
            raw_provider = provider_options.get(provider)
            if isinstance(raw_provider, dict):
                for key in ("images", "pdfs", "documents", "audio", "video", "tool_results", "current_turn_inline_limit_bytes"):
                    if key in raw_provider:
                        overrides[key] = raw_provider[key]

        return capability.with_overrides(overrides)

    def with_overrides(self, overrides: dict[str, Any]) -> "ModelMultimodalCapability":
        if not isinstance(overrides, dict) or not overrides:
            return self
        values = {
            "images": self.images,
            "pdfs": self.pdfs,
            "documents": self.documents,
            "audio": self.audio,
            "video": self.video,
            "tool_results": self.tool_results,
            "current_turn_inline_limit_bytes": self.current_turn_inline_limit_bytes,
        }
        for key in ("images", "pdfs", "documents", "audio", "video", "tool_results"):
            if key in overrides:
                values[key] = _to_bool(overrides.get(key), default=bool(values[key]))
        if "current_turn_inline_limit_bytes" in overrides:
            limit = _to_int(overrides.get("current_turn_inline_limit_bytes"))
            if limit is not None and limit > 0:
                values["current_turn_inline_limit_bytes"] = limit
        return ModelMultimodalCapability(**values)

    def supports_kind(self, kind: str) -> bool:
        normalized = str(kind or "").strip().lower()
        if normalized == "image":
            return self.images
        if normalized == "pdf":
            return self.pdfs
        if normalized == "document":
            return self.documents
        if normalized == "audio":
            return self.audio
        if normalized == "video":
            return self.video
        if normalized == "text":
            return True
        return False


@dataclass(frozen=True)
class PreparedAttachment:
    artifact_id: str
    workspace_path: str
    filename: str
    mime_type: str
    size_bytes: int
    kind: str
    current_turn_block: dict[str, Any] | None = None
    history_part: Any = None
    warnings: list[str] = field(default_factory=list)

    @property
    def sent_to_model(self) -> bool:
        return self.current_turn_block is not None

    @property
    def sent_as(self) -> str:
        block = self.current_turn_block
        if not isinstance(block, dict):
            return "stored_only"
        block_type = str(block.get("type") or "").strip().lower()
        if block_type == "text":
            return "text_excerpt"
        return block_type or "content_block"

    def sidecar(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "workspace_path": self.workspace_path,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "sent_to_model": self.sent_to_model,
            "sent_as": self.sent_as,
            "warnings": list(self.warnings),
        }


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None


def infer_attachment_kind(filename: str, mime_type: str = "") -> str:
    suffix = Path(str(filename or "")).suffix.lower()
    mime = str(mime_type or "").strip().lower()
    if mime.startswith("image/") or suffix in IMAGE_SUFFIXES:
        return "image"
    if mime == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if mime.startswith("audio/") or suffix in AUDIO_SUFFIXES:
        return "audio"
    if mime.startswith("video/") or suffix in VIDEO_SUFFIXES:
        return "video"
    if mime.startswith("text/") or suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    return "unsupported"


def guess_mime_type(filename: str, fallback: str = "") -> str:
    guessed = mimetypes.guess_type(str(filename or ""))[0]
    return str(fallback or guessed or "application/octet-stream")


def parse_data_url(data_url: str) -> tuple[str, bytes]:
    header, sep, encoded = str(data_url or "").partition(",")
    if not sep or not header.startswith("data:"):
        raise ValueError("attachment data must be a data URL")
    mime = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
    try:
        blob = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("attachment data is not valid base64") from exc
    return mime, blob


def file_to_content_block(path: Path, *, mime_type: str, kind: str, filename: str = "") -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    normalized_kind = str(kind or "").strip().lower()
    mime = str(mime_type or guess_mime_type(filename or path.name)).strip() or "application/octet-stream"
    if normalized_kind == "image":
        return {"type": "image", "base64": encoded, "mime_type": mime}
    if normalized_kind == "audio":
        return {"type": "audio", "base64": encoded, "mime_type": mime}
    if normalized_kind == "video":
        return {"type": "video", "base64": encoded, "mime_type": mime}
    return {
        "type": "file",
        "base64": encoded,
        "mime_type": mime,
        "filename": str(filename or path.name),
    }


def text_attachment_block(text: str, *, filename: str, workspace_path: str, limit: int = DEFAULT_TEXT_ATTACHMENT_CHAR_LIMIT) -> dict[str, str]:
    raw = str(text or "")
    excerpt = raw[: max(0, int(limit))]
    suffix = "\n[attachment text truncated]" if len(raw) > len(excerpt) else ""
    return {
        "type": "text",
        "text": (
            f"<attachment name=\"{filename}\" path=\"{workspace_path}\">\n"
            f"{excerpt}{suffix}\n"
            "</attachment>"
        ),
    }


def attachment_summary_text(user_text: str, attachments: list[PreparedAttachment]) -> str:
    text = str(user_text or "").strip() or "User submitted attachments."
    if not attachments:
        return text
    rows = ["", "Attached user files:"]
    for index, attachment in enumerate(attachments, start=1):
        status = f"sent as {attachment.sent_as}" if attachment.sent_to_model else "stored only"
        details = f"{attachment.kind}; {attachment.mime_type or 'unknown MIME'}; {attachment.size_bytes} bytes"
        rows.append(f"- {index}. `{attachment.filename}` at `{attachment.workspace_path}` ({details}; {status})")
        for warning in attachment.warnings:
            rows.append(f"  - warning: {warning}")
    rows.append("")
    rows.append("For later turns, reopen stored media with `read_file(file_path=...)` before inspecting visual or document content.")
    return text + "\n" + "\n".join(rows)


def build_turn_content(user_text: str, attachments: list[PreparedAttachment]) -> str | list[dict[str, Any]]:
    if not attachments:
        return str(user_text or "").strip()
    blocks: list[dict[str, Any]] = [{"type": "text", "text": attachment_summary_text(user_text, attachments)}]
    for attachment in attachments:
        if attachment.current_turn_block is not None:
            blocks.append(dict(attachment.current_turn_block))
    return blocks


def multimodal_prepare_summary(attachments: list[PreparedAttachment]) -> dict[str, Any]:
    sent = [item for item in attachments if item.sent_to_model]
    return {
        "attachment_count": len(attachments),
        "block_count": 1 + len(sent) if attachments else 0,
        "mime_summary": sorted({item.mime_type or "application/octet-stream" for item in attachments}),
        "store_paths": [item.workspace_path for item in attachments],
        "attachments": [item.sidecar() for item in attachments],
    }


__all__ = [
    "DEFAULT_CURRENT_TURN_INLINE_LIMIT_BYTES",
    "DEFAULT_TEXT_ATTACHMENT_CHAR_LIMIT",
    "ModelMultimodalCapability",
    "PreparedAttachment",
    "attachment_summary_text",
    "build_turn_content",
    "file_to_content_block",
    "guess_mime_type",
    "infer_attachment_kind",
    "multimodal_prepare_summary",
    "parse_data_url",
    "text_attachment_block",
]
