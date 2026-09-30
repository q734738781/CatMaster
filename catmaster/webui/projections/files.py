from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import display_path, redact_internal_text


def project_artifact(record: Any, *, workspace: Path | None = None) -> dict[str, Any]:
    raw = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record or {})
    path = display_path(raw.get("path"), workspace=workspace)
    return {
        "artifact_id": str(raw.get("artifact_id") or ""),
        "title": redact_internal_text(
            raw.get("title") or Path(path).name or "Artifact",
            workspace=workspace,
            limit=240,
        ),
        "summary": redact_internal_text(raw.get("summary"), workspace=workspace, limit=500),
        "path": path,
        "mime_type": str(raw.get("mime_type") or ""),
        "renderer": str(raw.get("renderer") or "file"),
        "created_at": float(raw.get("created_at") or 0.0),
        "updated_at": float(raw.get("updated_at") or 0.0),
        "preview_url": str(raw.get("preview_url") or ""),
        "download_url": str(raw.get("download_url") or ""),
    }


__all__ = ["project_artifact"]
