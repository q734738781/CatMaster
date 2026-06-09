from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.tools.base import compact_records_for_artifact, resolve_workspace_path, workspace_relpath
from catmaster.tools.geometry_inputs.cp2k_common import (
    discover_structure_paths,
    normalize_settings,
    resolve_optional_workspace_file,
    safe_stage_name,
    tool_error,
    write_cp2k_stage,
    write_json,
)


class Cp2kAimdPrepareInput(BaseModel):
    """[cp2k/prepare] Prepare restart-aware CP2K AIMD stages for NVE, NVT, NPT, restart, or user-provided PLUMED metadynamics."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description="Structure file, structure directory, or prior result directory for restart.")
    output_root: str = Field(..., description="Output root for prepared CP2K AIMD stage directories.")
    recipe: Literal["nve", "nvt", "npt", "restart", "metadynamics_user_plumed"] = Field(
        ...,
        description="Controlled CP2K AIMD recipe.",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Restricted CP2K AIMD settings: timestep_fs, steps, temperature, ensemble, trajectory_stride, "
            "restart_stride, restart_file, plumed_input_path, plus standard DFT settings."
        ),
    )


def _cp2k_md_ensemble(recipe: str, settings: dict[str, Any]) -> str:
    if recipe == "nve":
        return "NVE"
    if recipe in {"nvt", "metadynamics_user_plumed"}:
        return "NVT"
    if recipe == "npt":
        return "NPT_I"
    if recipe == "restart":
        return str(settings.get("ensemble") or "NVT").upper()
    raise ValueError(f"Unsupported CP2K AIMD recipe: {recipe}")


def _cp2k_md_motion_lines(recipe: str, settings: dict[str, Any]) -> list[str]:
    ensemble = _cp2k_md_ensemble(recipe, settings)
    temperature = float(settings.get("temperature", 300.0))
    timestep_fs = float(settings.get("timestep_fs", 1.0))
    steps = int(settings.get("steps", 1000))
    trajectory_stride = int(settings.get("trajectory_stride", 10))
    restart_stride = int(settings.get("restart_stride", 100))
    energy_stride = int(settings.get("energy_stride", max(1, trajectory_stride)))
    restart_file = str(settings.get("restart_file") or "").strip()
    lines = [
        "&MOTION",
        "  &MD",
        f"    ENSEMBLE {ensemble}",
        f"    STEPS {steps}",
        f"    TIMESTEP [fs] {timestep_fs:.10g}",
        f"    TEMPERATURE {temperature:.10g}",
    ]
    if restart_file:
        lines.append(f"    RESTART_FILE_NAME {restart_file}")
    if ensemble.startswith("NVT"):
        lines.extend(
            [
                "    &THERMOSTAT",
                f"      TYPE {str(settings.get('thermostat') or 'NOSE').upper()}",
                "    &END THERMOSTAT",
            ]
        )
    if ensemble.startswith("NPT"):
        lines.extend(
            [
                "    &BAROSTAT",
                f"      PRESSURE {_float_text(settings.get('pressure', 1.01325))}",
                "    &END BAROSTAT",
            ]
        )
    lines.extend(
        [
            "    &PRINT",
            "      &ENERGY",
            "        &EACH",
            f"          MD {energy_stride}",
            "        &END EACH",
            "      &END ENERGY",
            "    &END PRINT",
        ]
    )
    lines.extend(
        [
            "  &END MD",
            "  &PRINT",
            "    &TRAJECTORY",
            "      FORMAT XYZ",
            "      &EACH",
            f"        MD {trajectory_stride}",
            "      &END EACH",
            "    &END TRAJECTORY",
            "    &RESTART",
            "      BACKUP_COPIES 2",
            "      &EACH",
            f"        MD {restart_stride}",
            "      &END EACH",
            "    &END RESTART",
            "  &END PRINT",
        ]
    )
    if recipe == "metadynamics_user_plumed":
        lines.extend(
            [
                "  &FREE_ENERGY",
                "    &METADYN",
                "      USE_PLUMED .TRUE.",
                "      PLUMED_INPUT_FILE ./plumed.dat",
                "    &END METADYN",
                "  &END FREE_ENERGY",
            ]
        )
    lines.append("&END MOTION")
    return lines


def _float_text(value: Any) -> str:
    return f"{float(value):.10g}"


def _restart_structure_source(input_path: Path, settings: dict[str, Any]) -> Path:
    explicit = str(settings.get("structure_file") or "").strip()
    if explicit:
        path = resolve_workspace_path(explicit, must_exist=True)
        if not path.is_file():
            raise ValueError(f"settings.structure_file is not a file: {workspace_relpath(path)}")
        return path
    if input_path.is_file():
        return input_path
    for name in ("final.xyz", "input.xyz", "structure.xyz", "CONTCAR", "POSCAR"):
        candidate = input_path / name
        if candidate.is_file():
            return candidate
    matches = discover_structure_paths(input_path)
    if matches:
        return matches[0]
    raise ValueError("Could not infer a restart structure; provide settings.structure_file.")


def _restart_file(input_path: Path, settings: dict[str, Any]) -> Path | None:
    explicit = resolve_optional_workspace_file(settings.get("restart_file"), tool_name="cp2k_aimd_prepare")
    if explicit is not None:
        return explicit
    if input_path.is_dir():
        candidates = sorted(input_path.glob("*RESTART*")) + sorted(input_path.glob("*.restart"))
        files = [path for path in candidates if path.is_file()]
        if files:
            return files[-1]
    return None


def cp2k_aimd_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[cp2k/prepare] Prepare CP2K AIMD stage directories without running CP2K."""
    tool_name = "cp2k_aimd_prepare"
    params = Cp2kAimdPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    settings = normalize_settings(params.settings, tool_name=tool_name)

    plumed_path = None
    if params.recipe == "metadynamics_user_plumed":
        plumed_path = resolve_optional_workspace_file(settings.get("plumed_input_path"), tool_name=tool_name)
        if plumed_path is None:
            tool_error(
                tool_name,
                "metadynamics_user_plumed requires settings.plumed_input_path.",
                data={"input_path_rel": workspace_relpath(input_path)},
                error_code="missing_plumed_input",
            )
    sources: list[Path]
    restart_path = None
    if params.recipe == "restart":
        restart_path = _restart_file(input_path, settings)
        if restart_path is not None:
            settings["restart_file"] = restart_path.name
        sources = [_restart_structure_source(input_path, settings)]
    else:
        sources = discover_structure_paths(input_path)
    if not sources:
        tool_error(
            tool_name,
            "No supported structure files found under input_path.",
            data={"input_path_rel": workspace_relpath(input_path)},
            error_code="no_structures",
        )

    records: list[dict[str, Any]] = []
    input_root = input_path if input_path.is_dir() and params.recipe != "restart" else None
    for source in sources:
        stage_dir = output_root if input_root is None else output_root / safe_stage_name(source, root=input_root)
        extra_files: dict[Path, str] = {}
        if restart_path is not None:
            extra_files[restart_path] = restart_path.name
        if plumed_path is not None:
            extra_files[plumed_path] = "plumed.dat"
        records.append(
            write_cp2k_stage(
                source_path=source,
                output_dir=stage_dir,
                recipe=params.recipe,
                run_type="MD",
                settings=dict(settings),
                motion_lines=_cp2k_md_motion_lines(params.recipe, settings),
                extra_files=extra_files,
            )
        )

    manifest = output_root / "cp2k_aimd_prepare_manifest.json"
    write_json(manifest, {"input_path_rel": workspace_relpath(input_path), "recipe": params.recipe, "records": records})
    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "recipe": params.recipe,
        "prepared_count": len(records),
        "manifest_rel": workspace_relpath(manifest),
        **compact_records_for_artifact(records, full_records_rel=workspace_relpath(manifest)),
    }
    content = (
        "cp2k_aimd_prepare completed.\n"
        f"recipe={params.recipe} prepared_count={len(records)} output_root_rel={data['output_root_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


__all__ = ["Cp2kAimdPrepareInput", "cp2k_aimd_prepare"]
