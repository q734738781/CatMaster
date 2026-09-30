from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

from .native_stage import (
    StageAssetMapping,
    copy_stage_assets,
    require_fresh_stage,
    resolve_stage_assets,
    write_stage_manifest,
)


class CrestPrepareInput(BaseModel):
    """[crest/prepare] Stage exact native CREST arguments and explicitly mapped input files."""

    model_config = ConfigDict(extra="forbid")

    output_root: str = Field(..., description="Fresh empty directory for the prepared CREST stage.")
    argv: list[str] = Field(
        default_factory=list,
        description=(
            "Complete ordered native tokens after the fixed CREST executable. A staged coordinate alone keeps "
            "the native default workflow; a TOML file may be the first token."
        ),
    )
    asset_mappings: list[StageAssetMapping] = Field(
        default_factory=list,
        description=(
            "Explicit source_path to stage_path file mappings used by argv or TOML. Nested files and selected "
            "dotfiles are supported; omit or pass [] when none are needed."
        ),
    )

    @field_validator("argv")
    @classmethod
    def _valid_argv(cls, value: list[str]) -> list[str]:
        if any("\x00" in token for token in value):
            raise ValueError("argv tokens cannot contain NUL characters")
        return value


def _tool_error(message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name="crest_prepare",
        public_message=str(message).strip(),
        artifact={"tool_name": "crest_prepare", "data": data or {}},
        error_code=error_code,
    )


def crest_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[crest/prepare] Stage exact native CREST arguments and explicitly mapped input files."""
    try:
        params = CrestPrepareInput(**payload)
        stage_dir = resolve_workspace_path(params.output_root)
        assets = resolve_stage_assets(params.asset_mappings, reserved_paths=("manifest.json",))
        require_fresh_stage(stage_dir, source_paths=[item[0] for item in assets])
        stage_dir.mkdir(parents=True, exist_ok=True)
        copied_assets = copy_stage_assets(stage_dir, assets)
        manifest = write_stage_manifest(
            stage_dir,
            {"argv": list(params.argv), "input_assets": copied_assets},
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _tool_error(
            f"crest_prepare failed: {exc}",
            data={"output_root": str(payload.get("output_root") or "")},
            error_code="crest_prepare_failed",
        )

    data = {
        "stage_path": workspace_relpath(stage_dir),
        "manifest_path": workspace_relpath(manifest),
        "argv": list(params.argv),
        "input_assets": copied_assets,
    }
    content = (
        "crest_prepare completed.\n"
        f"stage_path={data['stage_path']} manifest_path={data['manifest_path']}\n"
        f"argv={data['argv']} input_assets={copied_assets}"
    )
    return content, {"tool_name": "crest_prepare", "data": data}


__all__ = ["CrestPrepareInput", "crest_prepare"]
