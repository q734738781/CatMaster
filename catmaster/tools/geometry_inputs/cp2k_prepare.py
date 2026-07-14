from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.tools.base import compact_records_for_artifact, resolve_workspace_path, workspace_relpath

from .cp2k_common import (
    band_motion_lines,
    cell_opt_motion_lines,
    dimer_motion_lines,
    discover_structure_paths,
    geo_opt_motion_lines,
    normalize_settings,
    read_atoms,
    resolve_optional_workspace_file,
    safe_stage_name,
    tool_error,
    vibrational_top_level_lines,
    write_cp2k_stage,
    write_json,
)


class Cp2kPrepareInput(BaseModel):
    """[cp2k/prepare] Prepare conventional CP2K DFT and path-refinement stages."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description="Structure file or directory of structures under the workspace files root.")
    output_root: str = Field(..., description="Output root for prepared CP2K stage directories.")
    recipe: Literal["sp", "geo_opt", "cell_opt", "freq", "dos", "neb", "dimer"] = Field(
        ...,
        description="Controlled CP2K conventional DFT recipe.",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Restricted CP2K settings override map: xc, basis_set, potential, charge, multiplicity, "
            "cutoff, rel_cutoff, eps_scf, max_scf, kpoints, periodic, cell_abc, dispersion, optimizer, "
            "max_iter, convergence tolerances, properties, and controlled NEB/dimer path settings."
        ),
    )


def _recipe_plan(recipe: str, settings: dict[str, Any]) -> tuple[str, list[str] | None, list[str] | None, bool]:
    if recipe == "sp":
        return "ENERGY_FORCE", None, None, False
    if recipe == "geo_opt":
        return "GEO_OPT", geo_opt_motion_lines(settings), None, False
    if recipe == "cell_opt":
        return "CELL_OPT", cell_opt_motion_lines(settings), None, True
    if recipe == "freq":
        return "VIBRATIONAL_ANALYSIS", None, vibrational_top_level_lines(settings), False
    if recipe == "dos":
        props = dict(settings.get("properties") or {})
        props["dos"] = True
        settings["properties"] = props
        return "ENERGY", None, None, False
    if recipe == "dimer":
        return "GEO_OPT", dimer_motion_lines(settings, vector_lines=_dimer_vector_lines(settings)), None, False
    raise ValueError(f"Unsupported CP2K recipe: {recipe}")


def _validate_same_atoms(paths: list[Path]) -> None:
    if len(paths) < 2:
        raise ValueError("CP2K NEB preparation requires at least two image structures.")
    first = read_atoms(paths[0])
    first_symbols = first.get_chemical_symbols()
    first_n = len(first)
    for path in paths[1:]:
        atoms = read_atoms(path)
        if len(atoms) != first_n or atoms.get_chemical_symbols() != first_symbols:
            raise ValueError(
                "CP2K NEB image structures must have identical atom counts and symbol ordering; "
                f"failed at {workspace_relpath(path)}."
            )


def _dimer_vector_lines(settings: dict[str, Any]) -> list[str] | None:
    vector_file = resolve_optional_workspace_file(settings.get("dimer_vector_file"), tool_name="cp2k_prepare")
    if vector_file is None:
        return None
    return vector_file.read_text(encoding="utf-8", errors="replace").splitlines()


def _prepare_neb_stage(
    *,
    sources: list[Path],
    output_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    _validate_same_atoms(sources)
    replica_names = [f"replica_{idx:03d}.xyz" for idx, _ in enumerate(sources)]
    extra_files = {source: name for source, name in zip(sources, replica_names)}
    return write_cp2k_stage(
        source_path=sources[0],
        output_dir=output_root,
        recipe="neb",
        run_type="BAND",
        settings=dict(settings),
        motion_lines=band_motion_lines(settings, replica_names),
        extra_files=extra_files,
    )


def cp2k_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[cp2k/prepare] Prepare CP2K stage directories without running CP2K."""
    tool_name = "cp2k_prepare"
    params = Cp2kPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    settings = normalize_settings(params.settings, tool_name=tool_name)

    sources = discover_structure_paths(input_path)
    if not sources:
        tool_error(
            tool_name,
            "No supported structure files found under input_path.",
            data={"input_path_rel": workspace_relpath(input_path)},
            error_code="no_structures",
        )
    if params.recipe == "neb":
        record = _prepare_neb_stage(sources=sources, output_root=output_root, settings=settings)
        manifest = output_root / "cp2k_prepare_manifest.json"
        write_json(manifest, {"input_path_rel": workspace_relpath(input_path), "recipe": params.recipe, "records": [record]})
        data = {
            "input_path_rel": workspace_relpath(input_path),
            "output_root_rel": workspace_relpath(output_root),
            "recipe": params.recipe,
            "prepared_count": 1,
            "manifest_rel": workspace_relpath(manifest),
            **compact_records_for_artifact([record], full_records_rel=workspace_relpath(manifest)),
        }
        content = (
            "cp2k_prepare completed.\n"
            f"recipe={params.recipe} prepared_count=1 output_root_rel={data['output_root_rel']}\n"
            f"manifest_rel={data['manifest_rel']}"
        )
        return content, {"tool_name": tool_name, "data": data}

    run_type, motion_lines, top_sections, stress_tensor = _recipe_plan(params.recipe, settings)
    records: list[dict[str, Any]] = []
    input_root: Path | None = input_path if input_path.is_dir() else None
    for source in sources:
        if input_root is None:
            stage_dir = output_root
        else:
            stage_dir = output_root / safe_stage_name(source, root=input_root)
        records.append(
            write_cp2k_stage(
                source_path=source,
                output_dir=stage_dir,
                recipe=params.recipe,
                run_type=run_type,
                settings=dict(settings),
                motion_lines=motion_lines,
                top_level_sections=top_sections,
                stress_tensor=stress_tensor,
            )
        )

    manifest = output_root / "cp2k_prepare_manifest.json"
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
        "cp2k_prepare completed.\n"
        f"recipe={params.recipe} prepared_count={len(records)} output_root_rel={data['output_root_rel']}\n"
        f"manifest_rel={data['manifest_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


__all__ = ["Cp2kPrepareInput", "cp2k_prepare"]
