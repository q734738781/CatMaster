from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

from .native_stage import (
    StageAssetMapping,
    copy_stage_assets,
    require_fresh_stage,
    resolve_stage_assets,
    write_stage_manifest,
)


class Cp2kPrepareInput(BaseModel):
    """[cp2k/prepare] Stage one complete agent-authored CP2K input and its explicit dependencies."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        ...,
        description="Workspace-relative path to the complete native CP2K input file that will become job.inp.",
    )
    output_root: str = Field(..., description="Fresh empty directory for the prepared CP2K stage.")
    asset_mappings: list[StageAssetMapping] = Field(
        default_factory=list,
        description=(
            "Explicit dependency files copied into the stage. Each item has source_path and stage_path; "
            "omit or pass [] when job.inp has no external file dependencies."
        ),
    )


def _tool_error(message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name="cp2k_prepare",
        public_message=str(message).strip(),
        artifact={"tool_name": "cp2k_prepare", "data": data or {}},
        error_code=error_code,
    )


def cp2k_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[cp2k/prepare] Stage one complete agent-authored CP2K input and its explicit dependencies."""
    try:
        params = Cp2kPrepareInput(**payload)
        source = resolve_workspace_path(params.input_path, must_exist=True)
        if not source.is_file():
            raise ValueError(f"input_path must name a file: {workspace_relpath(source)}")
        stage_dir = resolve_workspace_path(params.output_root)
        assets = resolve_stage_assets(
            params.asset_mappings,
            reserved_paths=("job.inp", "manifest.json"),
        )
        require_fresh_stage(stage_dir, source_paths=[source, *(item[0] for item in assets)])
        stage_dir.mkdir(parents=True, exist_ok=True)
        canonical_input = stage_dir / "job.inp"
        shutil.copy2(source, canonical_input)
        copied_assets = copy_stage_assets(stage_dir, assets)
        manifest = write_stage_manifest(
            stage_dir,
            {"input_file": "job.inp", "input_assets": copied_assets},
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _tool_error(
            f"cp2k_prepare failed: {exc}",
            data={"input_path": str(payload.get("input_path") or ""), "output_root": str(payload.get("output_root") or "")},
            error_code="cp2k_prepare_failed",
        )

    data = {
        "stage_path": workspace_relpath(stage_dir),
        "input_path": workspace_relpath(canonical_input),
        "manifest_path": workspace_relpath(manifest),
        "input_assets": copied_assets,
    }
    content = (
        "cp2k_prepare completed.\n"
        f"stage_path={data['stage_path']} input_path={data['input_path']}\n"
        f"manifest_path={data['manifest_path']} input_assets={len(copied_assets)}"
    )
    return content, {"tool_name": "cp2k_prepare", "data": data}


__all__ = ["Cp2kPrepareInput", "cp2k_prepare"]
