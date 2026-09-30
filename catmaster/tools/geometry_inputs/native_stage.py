from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class StageAssetMapping(BaseModel):
    """One workspace file copied to an explicit path inside a prepared stage."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(..., description="Workspace-relative source file path.")
    stage_path: str = Field(
        ...,
        description="Stage-relative destination path, for example inputs/coord.xyz or .CHRG.",
    )


def normalized_stage_path(raw: str, *, field: str = "stage_path") -> str:
    value = str(raw or "").strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or value in {".", ".."} or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a non-empty stage-relative path without parent traversal: {raw!r}")
    return path.as_posix()


def resolve_stage_assets(
    mappings: Iterable[StageAssetMapping],
    *,
    reserved_paths: Iterable[str] = (),
) -> list[tuple[Path, str]]:
    reserved = {normalized_stage_path(item, field="reserved path") for item in reserved_paths}
    resolved: list[tuple[Path, str]] = []
    destinations: set[str] = set()
    for mapping in mappings:
        source = resolve_workspace_path(mapping.source_path, must_exist=True)
        if not source.is_file():
            raise ValueError(f"source_path must name a file: {workspace_relpath(source)}")
        destination = normalized_stage_path(mapping.stage_path)
        if destination in reserved:
            raise ValueError(f"stage_path is reserved by the prepared-stage contract: {destination}")
        if destination in destinations:
            raise ValueError(f"Duplicate stage_path: {destination}")
        destinations.add(destination)
        resolved.append((source, destination))
    return resolved


def require_fresh_stage(stage_dir: Path, *, source_paths: Iterable[Path] = ()) -> None:
    resolved_stage = stage_dir.resolve()
    for source in source_paths:
        try:
            source.resolve().relative_to(resolved_stage)
        except ValueError:
            continue
        raise ValueError("output_root cannot contain a source file that will be staged")
    if stage_dir.exists():
        if not stage_dir.is_dir():
            raise ValueError(f"output_root is not a directory: {workspace_relpath(stage_dir)}")
        if any(stage_dir.iterdir()):
            raise ValueError(
                "output_root must be a fresh empty stage; use a new directory for changed scientific input"
            )


def copy_stage_assets(stage_dir: Path, assets: Iterable[tuple[Path, str]]) -> list[str]:
    copied: list[str] = []
    for source, relative in assets:
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative)
    return copied


def write_stage_manifest(stage_dir: Path, payload: dict[str, object]) -> Path:
    path = stage_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "StageAssetMapping",
    "copy_stage_assets",
    "normalized_stage_path",
    "require_fresh_stage",
    "resolve_stage_assets",
    "write_stage_manifest",
]
