from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Kpoints, Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath


def _error_message(message: str, *, data: Dict[str, Any] | None = None) -> str:
    lines = [str(message).strip()]
    if isinstance(data, dict):
        for key in (
            "structure_dir_rel",
            "structure_ref_rel",
            "input_rel",
            "output_rel",
            "output_dir_rel",
            "output_json_rel",
            "kpoints_rel",
            "metadata_rel",
        ):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            lines.append(f"{key}={value}")
    return "\n".join(lines)


def _collect_structure_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in {"POSCAR", "CONTCAR"}:
            files.append(path)
            continue
        if path.suffix.lower() in {".vasp", ".poscar", ".cif", ".xyz"}:
            files.append(path)
    return sorted(files, key=lambda p: str(p))


def _structure_id_from(rel_path: Path) -> str:
    return "__".join(rel_path.with_suffix("").parts)


def _default_structure_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".vasp", ".poscar", ".cif", ".xyz"}:
        return suffix
    return ".vasp"


def _write_structure(path: Path, structure: Structure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".vasp", ".poscar", ""}:
        Poscar(structure).write_file(str(path))
        return
    structure.to(filename=str(path))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_structure(structure_file: str) -> tuple[Path, Structure]:
    structure_path = resolve_workspace_path(structure_file, must_exist=True)
    return structure_path, Structure.from_file(structure_path)


def _symmetrized_groups(
    structure: Structure,
    *,
    symprec: float,
    angle_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    symm = analyzer.get_symmetrized_structure()
    groups: list[dict[str, Any]] = []
    index_to_group: dict[int, int] = {}
    wyckoffs = list(getattr(symm, "wyckoff_symbols", []) or [])
    for group_id, indices in enumerate(symm.equivalent_indices):
        representative_index = int(indices[0])
        site = structure[representative_index]
        for idx in indices:
            index_to_group[int(idx)] = group_id
        group = {
            "group_id": group_id,
            "species": site.species_string,
            "representative_index": representative_index,
            "equivalent_indices": [int(idx) for idx in indices],
            "frac_coords": [float(x) for x in site.frac_coords],
        }
        if group_id < len(wyckoffs):
            group["wyckoff"] = wyckoffs[group_id]
        groups.append(group)
    return groups, index_to_group


def _resolve_group_representative(
    structure: Structure,
    *,
    group_id: int,
    symprec: float,
    angle_tolerance: float,
) -> tuple[int, dict[str, Any]]:
    groups, _ = _symmetrized_groups(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    if group_id < 0 or group_id >= len(groups):
        raise ValueError(f"group_id out of range: {group_id}")
    group = groups[group_id]
    return int(group["representative_index"]), group


def _clone_structure_with_same_properties(structure: Structure, lattice_matrix: np.ndarray) -> Structure:
    return Structure(
        lattice=Lattice(lattice_matrix),
        species=[site.species for site in structure],
        coords=[site.frac_coords for site in structure],
        coords_are_cartesian=False,
        site_properties={key: list(values) for key, values in structure.site_properties.items()},
        charge=structure.charge,
    )


def _normalize_3x3_matrix(raw: Any) -> np.ndarray:
    arr = np.array(raw, dtype=float)
    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    if arr.shape != (3, 3):
        raise ValueError("strain/deformation matrix must be 3x3 or length-9.")
    return arr


def _strain_matrix_from_mode(mode: str, value: float) -> np.ndarray:
    matrix = np.eye(3)
    if mode == "isotropic":
        matrix += float(value) * np.eye(3)
        return matrix
    if mode == "xx":
        matrix[0, 0] += float(value)
        return matrix
    if mode == "yy":
        matrix[1, 1] += float(value)
        return matrix
    if mode == "zz":
        matrix[2, 2] += float(value)
        return matrix
    shear = float(value) / 2.0
    if mode == "xy":
        matrix[0, 1] += shear
        matrix[1, 0] += shear
        return matrix
    if mode == "xz":
        matrix[0, 2] += shear
        matrix[2, 0] += shear
        return matrix
    if mode == "yz":
        matrix[1, 2] += shear
        matrix[2, 1] += shear
        return matrix
    raise ValueError(f"Unsupported strain mode: {mode}")


def _manual_phonon_displacements(
    structure: Structure,
    *,
    supercell: list[int],
    displacement: float,
    plus_minus: bool,
    symprec: float,
    angle_tolerance: float,
) -> tuple[list[tuple[str, Structure]], dict[str, Any]]:
    groups, _ = _symmetrized_groups(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    supercell_structure = structure.copy()
    supercell_structure.make_supercell(np.diag([int(x) for x in supercell]))
    n_base = len(structure)
    directions = (("p", 1.0), ("m", -1.0)) if plus_minus else (("p", 1.0),)
    outputs: list[tuple[str, Structure]] = []
    axes = ("x", "y", "z")
    for group in groups:
        base_index = int(group["representative_index"])
        if base_index >= n_base:
            continue
        for axis_index, axis_label in enumerate(axes):
            for sign_label, sign in directions:
                displaced = supercell_structure.copy()
                displaced.translate_sites(
                    [base_index],
                    np.eye(3)[axis_index] * float(displacement) * sign,
                    frac_coords=False,
                    to_unit_cell=False,
                )
                label = f"site{base_index:03d}_{axis_label}_{sign_label}"
                outputs.append((label, displaced))
    metadata = {
        "generator": "manual_fallback",
        "representative_sites": [int(group["representative_index"]) for group in groups],
        "displacements_generated": len(outputs),
    }
    return outputs, metadata


class SupercellInput(BaseModel):
    """[structure/modeling] Create supercells from structure file(s) while preserving selective dynamics in POSCAR/VASP outputs.
    Provide exactly one of structure_file or structure_dir. When structure_dir is used, output_dir is required.
    Batch outputs are written to output_dir/<structure_id>.vasp and a summary JSON is written to
    output_dir/batch_supercell.json.
    """

    structure_file: Optional[str] = Field(
        None, description="Bulk structure file (POSCAR/CIF/etc.), workspace-relative."
    )
    structure_dir: Optional[str] = Field(
        None, description="Directory containing structure files for batch supercell generation."
    )
    supercell: list[int] = Field(..., min_length=3, max_length=3, description="Supercell replication [a,b,c].")
    output_path: Optional[str] = Field(
        None, description="Output structure path for single structure (workspace-relative, e.g., bulk_supercell.vasp)."
    )
    output_dir: Optional[str] = Field(
        None,
        description=(
            "Output directory for batch mode. Outputs are written as <output_dir>/<structure_id>.vasp where "
            "structure_id encodes the relative input path (without suffix) using '__'."
        ),
    )


class EnumerateUniqueSitesInput(BaseModel):
    """[structure/discovery] Enumerate symmetry-inequivalent atomic sites for one structure and optionally write a JSON report."""

    structure_file: str = Field(..., description="Input structure file (POSCAR/CIF/etc.), workspace-relative.")
    output_json: Optional[str] = Field(
        None,
        description="Optional JSON report path. Defaults to <input_stem>_unique_sites.json next to the input file.",
    )
    symprec: float = Field(0.01, gt=0.0, description="Symmetry tolerance in angstrom.")
    angle_tolerance: float = Field(5.0, gt=0.0, description="Angle tolerance in degrees.")


class CreateVacancyInput(BaseModel):
    """[defect/modeling] Remove one explicit site or generate one vacancy structure per symmetry-inequivalent site group."""

    structure_file: str = Field(..., description="Input structure file (POSCAR/CIF/etc.), workspace-relative.")
    mode: Literal["single", "one_per_group"] = Field(
        "single",
        description="single removes one site; one_per_group generates one vacancy structure for each inequivalent group.",
    )
    site_index: Optional[int] = Field(None, ge=0, description="0-based site index for single mode.")
    group_id: Optional[int] = Field(
        None,
        ge=0,
        description="Symmetry group_id from enumerate_unique_sites. In single mode this removes the representative site.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output structure path for single mode. Defaults to <input_stem>_vacancy_*.vasp.",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory for one_per_group mode. Defaults to <input_stem>_vacancies next to the input file.",
    )
    symprec: float = Field(0.01, gt=0.0, description="Symmetry tolerance in angstrom.")
    angle_tolerance: float = Field(5.0, gt=0.0, description="Angle tolerance in degrees.")

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "CreateVacancyInput":
        if self.mode == "single":
            if self.site_index is None and self.group_id is None:
                raise ValueError("single mode requires site_index or group_id.")
            if self.site_index is not None and self.group_id is not None:
                raise ValueError("single mode accepts only one of site_index or group_id.")
            return self
        if self.output_path is not None:
            raise ValueError("output_path is only valid in single mode.")
        if self.site_index is not None:
            raise ValueError("site_index is not used in one_per_group mode.")
        return self


class SubstituteSpeciesInput(BaseModel):
    """[defect/modeling] Replace one explicit site or generate one substitution structure per symmetry-inequivalent site group."""

    structure_file: str = Field(..., description="Input structure file (POSCAR/CIF/etc.), workspace-relative.")
    new_species: str = Field(..., description="Species symbol to place at the substituted site, e.g. 'Ni'.")
    mode: Literal["single", "one_per_group"] = Field(
        "single",
        description="single substitutes one site; one_per_group generates one substitution candidate per inequivalent group.",
    )
    site_index: Optional[int] = Field(None, ge=0, description="0-based site index for single mode.")
    group_id: Optional[int] = Field(
        None,
        ge=0,
        description="Symmetry group_id from enumerate_unique_sites. In single mode this substitutes the representative site.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output structure path for single mode. Defaults to <input_stem>_sub_<species>_*.vasp.",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory for one_per_group mode. Defaults to <input_stem>_sub_<species> next to the input file.",
    )
    symprec: float = Field(0.01, gt=0.0, description="Symmetry tolerance in angstrom.")
    angle_tolerance: float = Field(5.0, gt=0.0, description="Angle tolerance in degrees.")

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "SubstituteSpeciesInput":
        if self.mode == "single":
            if self.site_index is None and self.group_id is None:
                raise ValueError("single mode requires site_index or group_id.")
            if self.site_index is not None and self.group_id is not None:
                raise ValueError("single mode accepts only one of site_index or group_id.")
            return self
        if self.output_path is not None:
            raise ValueError("output_path is only valid in single mode.")
        if self.site_index is not None:
            raise ValueError("site_index is not used in one_per_group mode.")
        return self


class InsertInterstitialAtCoordsInput(BaseModel):
    """[defect/modeling] Insert one or more explicit interstitial atoms at user-specified fractional or Cartesian coordinates."""

    structure_file: str = Field(..., description="Input structure file (POSCAR/CIF/etc.), workspace-relative.")
    species: str = Field(..., description="Species symbol to insert, e.g. 'H' or 'Li'.")
    coords: list[list[float]] = Field(
        ...,
        min_length=1,
        description="One or more 3-component coordinate triplets. Single coordinate yields one structure; multiple coordinates yield a batch.",
    )
    coordinate_type: Literal["fractional", "cartesian"] = Field(
        "fractional",
        description="Whether coords are given in fractional or Cartesian coordinates.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output path for a single inserted structure. Required only when coords has one entry and output_dir is omitted.",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory for multi-coordinate mode. Defaults to <input_stem>_interstitials next to the input file.",
    )

    @model_validator(mode="after")
    def _validate_coords(self) -> "InsertInterstitialAtCoordsInput":
        for coords in self.coords:
            if len(coords) != 3:
                raise ValueError("Every coordinate must have exactly 3 components.")
        if len(self.coords) > 1 and self.output_path is not None:
            raise ValueError("output_path is only valid when exactly one coordinate is provided.")
        return self


class StrainMatrixSpec(BaseModel):
    label: Optional[str] = Field(None, description="Short label used in output filenames.")
    matrix: list[list[float]] = Field(..., description="3x3 deformation matrix F applied as lattice' = lattice @ F.")


class GenerateStrainedStructuresInput(BaseModel):
    """[bulk/modeling] Generate a strain set from explicit deformation matrices or from a mode/value grid."""

    structure_file: str = Field(..., description="Input bulk structure file (POSCAR/CIF/etc.), workspace-relative.")
    output_dir: str = Field(..., description="Output directory for strained structure files.")
    strain_matrices: list[StrainMatrixSpec] = Field(
        default_factory=list,
        description="Explicit deformation matrices. Each matrix is applied directly to the lattice.",
    )
    strain_modes: list[Literal["xx", "yy", "zz", "xy", "xz", "yz", "isotropic"]] = Field(
        default_factory=list,
        description="Convenience strain modes used together with strain_values.",
    )
    strain_values: list[float] = Field(
        default_factory=list,
        description="Strain amplitudes used together with strain_modes, e.g. [-0.01, 0.0, 0.01].",
    )

    @model_validator(mode="after")
    def _validate_specs(self) -> "GenerateStrainedStructuresInput":
        if self.strain_matrices:
            return self
        if self.strain_modes and self.strain_values:
            return self
        raise ValueError("Provide strain_matrices or both strain_modes and strain_values.")


class GenerateKpathInput(BaseModel):
    """[electronic/prepare] Generate a high-symmetry line-mode KPOINTS file from a relaxed bulk structure."""

    structure_file: str = Field(..., description="Relaxed bulk structure file (POSCAR/CIF/etc.), workspace-relative.")
    output_path: str = Field(..., description="Output KPOINTS path, workspace-relative.")
    output_json: Optional[str] = Field(
        None,
        description="Optional JSON summary path. Defaults to <output_path>.json.",
    )
    line_density: int = Field(20, ge=1, description="Number of divisions between high-symmetry points.")
    path_type: Literal["setyawan_curtarolo", "hinuma", "latimer_munro", "all"] = Field(
        "setyawan_curtarolo",
        description="High-symmetry path convention passed to pymatgen HighSymmKpath.",
    )
    primitive_standard: bool = Field(
        True,
        description="If true, standardize to pymatgen's primitive standard structure before building the path.",
    )
    symprec: float = Field(0.01, gt=0.0, description="Symmetry tolerance in angstrom.")
    angle_tolerance: float = Field(5.0, gt=0.0, description="Angle tolerance in degrees.")


class GeneratePhononDisplacementsInput(BaseModel):
    """[phonon/modeling] Generate finite-displacement supercell structures for a phonon force-constant workflow."""

    structure_file: str = Field(..., description="Input bulk structure file (POSCAR/CIF/etc.), workspace-relative.")
    output_dir: str = Field(..., description="Output directory for displaced supercell structures.")
    supercell: list[int] = Field(..., min_length=3, max_length=3, description="Supercell replication [a,b,c].")
    displacement: float = Field(0.01, gt=0.0, description="Displacement amplitude in angstrom.")
    plus_minus: bool = Field(
        True,
        description="If true, generate both positive and negative displacements when supported by the backend.",
    )
    symprec: float = Field(0.01, gt=0.0, description="Symmetry tolerance in angstrom.")
    angle_tolerance: float = Field(5.0, gt=0.0, description="Angle tolerance in degrees.")


def supercell(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[structure/modeling] Create supercells for one structure or a directory batch."""
    try:
        params = SupercellInput(**payload)
        if (params.structure_file is None) == (params.structure_dir is None):
            raise CatMasterToolExecutionError(
                tool_name="supercell",
                public_message=_error_message("Provide exactly one of structure_file or structure_dir."),
                artifact={"tool_name": "supercell", "data": {}},
                error_code="invalid_input_mode",
            )

        matrix = np.diag([int(x) for x in params.supercell])

        if params.structure_dir is not None:
            if params.output_dir is None:
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message("output_dir is required when structure_dir is provided."),
                    artifact={"tool_name": "supercell", "data": {}},
                    error_code="missing_output_dir",
                )
            structure_root = resolve_workspace_path(params.structure_dir, must_exist=True)
            if not structure_root.is_dir():
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message(
                        f"structure_dir is not a directory: {structure_root}",
                        data={"structure_dir_rel": workspace_relpath(structure_root)},
                    ),
                    artifact={
                        "tool_name": "supercell",
                        "data": {"structure_dir_rel": workspace_relpath(structure_root)},
                    },
                    error_code="invalid_structure_dir",
                )
            output_root = resolve_workspace_path(params.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            structures = _collect_structure_files(structure_root)
            if not structures:
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message(
                        "No structure files found in structure_dir.",
                        data={"structure_dir_rel": workspace_relpath(structure_root)},
                    ),
                    artifact={
                        "tool_name": "supercell",
                        "data": {"structure_dir_rel": workspace_relpath(structure_root)},
                    },
                    error_code="no_structures",
                )

            results = []
            errors = []
            for structure_path in structures:
                rel_path = structure_path.relative_to(structure_root)
                structure_id = _structure_id_from(rel_path)
                out_path = output_root / f"{structure_id}.vasp"
                try:
                    structure = Structure.from_file(structure_path)
                    structure.make_supercell(matrix)
                    _write_structure(out_path, structure)
                    results.append(
                        {
                            "input_rel": str(rel_path),
                            "structure_id": structure_id,
                            "output_rel": workspace_relpath(out_path),
                            "natoms": len(structure),
                        }
                    )
                except Exception as exc:
                    errors.append({"input_rel": str(rel_path), "error": str(exc)})

            batch_json = output_root / "batch_supercell.json"
            _write_json(batch_json, {"results": results, "errors": errors})
            data: dict[str, Any] = {
                "structure_dir_rel": workspace_relpath(structure_root),
                "output_dir_rel": workspace_relpath(output_root),
                "supercell": params.supercell,
                "structures_found": len(structures),
                "structures_processed": len(results),
                "batch_json_rel": workspace_relpath(batch_json),
                "errors_count": len(errors),
            }
            if errors:
                data.update(
                    compact_list_for_artifact(
                        errors,
                        count_key="errors_count",
                        inline_key="errors",
                        preview_key="errors_preview",
                        truncated_key="errors_truncated",
                        full_rel_key="errors_full_rel",
                        full_rel=workspace_relpath(batch_json),
                    )
                )
            first_output = results[0]["output_rel"] if results else ""
            lines = [
                "supercell completed.",
                f"structures_processed={len(results)} structures_found={len(structures)} errors_count={len(errors)}",
                f"output_dir_rel={data['output_dir_rel']}",
                f"batch_json_rel={data['batch_json_rel']}",
            ]
            if first_output:
                lines.append(f"first_output_rel={first_output}")
            return "\n".join(lines), {"tool_name": "supercell", "data": data}

        if params.output_path is None:
            raise CatMasterToolExecutionError(
                tool_name="supercell",
                public_message=_error_message("output_path is required when structure_file is provided."),
                artifact={"tool_name": "supercell", "data": {}},
                error_code="missing_output_path",
            )
        structure_path = resolve_workspace_path(params.structure_file, must_exist=True)
        out_path = resolve_workspace_path(params.output_path)

        structure = Structure.from_file(structure_path)
        structure.make_supercell(matrix)
        _write_structure(out_path, structure)

        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_rel": workspace_relpath(out_path),
            "supercell": params.supercell,
            "natoms": len(structure),
        }
        content = (
            "supercell completed.\n"
            f"input_rel={data['input_rel']} output_rel={data['output_rel']} natoms={data['natoms']}"
        )
        return content, {"tool_name": "supercell", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="supercell",
            public_message=_error_message(f"supercell failed: {exc}"),
            artifact={"tool_name": "supercell", "data": {}},
            error_code="supercell_failed",
        )


def enumerate_unique_sites(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[structure/discovery] Enumerate symmetry-inequivalent sites for one structure."""
    try:
        params = EnumerateUniqueSitesInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        groups, index_to_group = _symmetrized_groups(
            structure,
            symprec=params.symprec,
            angle_tolerance=params.angle_tolerance,
        )
        output_json = resolve_workspace_path(params.output_json) if params.output_json else (
            structure_path.parent / f"{structure_path.stem}_unique_sites.json"
        )
        payload_json = {
            "input_rel": workspace_relpath(structure_path),
            "natoms": len(structure),
            "symprec": params.symprec,
            "angle_tolerance": params.angle_tolerance,
            "groups": groups,
            "index_to_group": {str(k): int(v) for k, v in sorted(index_to_group.items())},
        }
        _write_json(output_json, payload_json)
        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_json_rel": workspace_relpath(output_json),
            "natoms": len(structure),
            **compact_list_for_artifact(
                groups,
                count_key="groups_count",
                inline_key="groups",
                preview_key="groups_preview",
                truncated_key="groups_truncated",
                full_rel_key="groups_full_rel",
                full_rel=workspace_relpath(output_json),
            ),
        }
        content = (
            "enumerate_unique_sites completed.\n"
            f"input_rel={data['input_rel']} groups_count={data['groups_count']} "
            f"output_json_rel={data['output_json_rel']}"
        )
        return content, {"tool_name": "enumerate_unique_sites", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="enumerate_unique_sites",
            public_message=_error_message(f"enumerate_unique_sites failed: {exc}"),
            artifact={"tool_name": "enumerate_unique_sites", "data": {}},
            error_code="enumerate_unique_sites_failed",
        )


def create_vacancy(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[defect/modeling] Create vacancy structures from one explicit site or one representative site per symmetry group."""
    try:
        params = CreateVacancyInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        suffix = _default_structure_suffix(structure_path)

        if params.mode == "single":
            if params.site_index is not None:
                target_index = int(params.site_index)
                if target_index < 0 or target_index >= len(structure):
                    raise ValueError(f"site_index out of range: {target_index}")
                group_payload = None
            else:
                target_index, group_payload = _resolve_group_representative(
                    structure,
                    group_id=int(params.group_id),
                    symprec=params.symprec,
                    angle_tolerance=params.angle_tolerance,
                )
            out_path = resolve_workspace_path(params.output_path) if params.output_path else (
                structure_path.parent / f"{structure_path.stem}_vacancy_idx{target_index}{suffix}"
            )
            modified = structure.copy()
            removed_species = modified[target_index].species_string
            modified.remove_sites([target_index])
            _write_structure(out_path, modified)
            data = {
                "input_rel": workspace_relpath(structure_path),
                "output_rel": workspace_relpath(out_path),
                "removed_index": target_index,
                "removed_species": removed_species,
            }
            if group_payload is not None:
                data["group"] = group_payload
            content = (
                "create_vacancy completed.\n"
                f"input_rel={data['input_rel']} output_rel={data['output_rel']} removed_index={target_index}"
            )
            return content, {"tool_name": "create_vacancy", "data": data}

        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
            structure_path.parent / f"{structure_path.stem}_vacancies"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        groups, _ = _symmetrized_groups(
            structure,
            symprec=params.symprec,
            angle_tolerance=params.angle_tolerance,
        )
        results: list[dict[str, Any]] = []
        for group in groups:
            target_index = int(group["representative_index"])
            modified = structure.copy()
            removed_species = modified[target_index].species_string
            modified.remove_sites([target_index])
            out_path = output_dir / f"group_{group['group_id']:03d}_idx{target_index}_{removed_species}{suffix}"
            _write_structure(out_path, modified)
            results.append(
                {
                    "group_id": int(group["group_id"]),
                    "representative_index": target_index,
                    "removed_species": removed_species,
                    "output_rel": workspace_relpath(out_path),
                }
            )
        batch_json = output_dir / "batch_create_vacancy.json"
        _write_json(
            batch_json,
            {
                "input_rel": workspace_relpath(structure_path),
                "mode": params.mode,
                "results": results,
            },
        )
        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_dir_rel": workspace_relpath(output_dir),
            "batch_json_rel": workspace_relpath(batch_json),
            "structures_generated": len(results),
        }
        content = (
            "create_vacancy completed.\n"
            f"input_rel={data['input_rel']} structures_generated={data['structures_generated']} "
            f"output_dir_rel={data['output_dir_rel']}\n"
            f"batch_json_rel={data['batch_json_rel']}"
        )
        return content, {"tool_name": "create_vacancy", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="create_vacancy",
            public_message=_error_message(f"create_vacancy failed: {exc}"),
            artifact={"tool_name": "create_vacancy", "data": {}},
            error_code="create_vacancy_failed",
        )


def substitute_species(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[defect/modeling] Create substitution structures from one explicit site or one representative site per symmetry group."""
    try:
        params = SubstituteSpeciesInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        suffix = _default_structure_suffix(structure_path)

        if params.mode == "single":
            if params.site_index is not None:
                target_index = int(params.site_index)
                if target_index < 0 or target_index >= len(structure):
                    raise ValueError(f"site_index out of range: {target_index}")
                group_payload = None
            else:
                target_index, group_payload = _resolve_group_representative(
                    structure,
                    group_id=int(params.group_id),
                    symprec=params.symprec,
                    angle_tolerance=params.angle_tolerance,
                )
            out_path = resolve_workspace_path(params.output_path) if params.output_path else (
                structure_path.parent / f"{structure_path.stem}_sub_{params.new_species}_idx{target_index}{suffix}"
            )
            modified = structure.copy()
            old_species = modified[target_index].species_string
            modified.replace(target_index, params.new_species)
            _write_structure(out_path, modified)
            data = {
                "input_rel": workspace_relpath(structure_path),
                "output_rel": workspace_relpath(out_path),
                "site_index": target_index,
                "old_species": old_species,
                "new_species": params.new_species,
            }
            if group_payload is not None:
                data["group"] = group_payload
            content = (
                "substitute_species completed.\n"
                f"input_rel={data['input_rel']} output_rel={data['output_rel']} site_index={target_index}"
            )
            return content, {"tool_name": "substitute_species", "data": data}

        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
            structure_path.parent / f"{structure_path.stem}_sub_{params.new_species}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        groups, _ = _symmetrized_groups(
            structure,
            symprec=params.symprec,
            angle_tolerance=params.angle_tolerance,
        )
        results: list[dict[str, Any]] = []
        for group in groups:
            target_index = int(group["representative_index"])
            modified = structure.copy()
            old_species = modified[target_index].species_string
            modified.replace(target_index, params.new_species)
            out_path = output_dir / f"group_{group['group_id']:03d}_idx{target_index}_{old_species}_to_{params.new_species}{suffix}"
            _write_structure(out_path, modified)
            results.append(
                {
                    "group_id": int(group["group_id"]),
                    "representative_index": target_index,
                    "old_species": old_species,
                    "new_species": params.new_species,
                    "output_rel": workspace_relpath(out_path),
                }
            )
        batch_json = output_dir / "batch_substitute_species.json"
        _write_json(
            batch_json,
            {
                "input_rel": workspace_relpath(structure_path),
                "mode": params.mode,
                "new_species": params.new_species,
                "results": results,
            },
        )
        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_dir_rel": workspace_relpath(output_dir),
            "batch_json_rel": workspace_relpath(batch_json),
            "structures_generated": len(results),
            "new_species": params.new_species,
        }
        content = (
            "substitute_species completed.\n"
            f"input_rel={data['input_rel']} structures_generated={data['structures_generated']} "
            f"output_dir_rel={data['output_dir_rel']}\n"
            f"batch_json_rel={data['batch_json_rel']}"
        )
        return content, {"tool_name": "substitute_species", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="substitute_species",
            public_message=_error_message(f"substitute_species failed: {exc}"),
            artifact={"tool_name": "substitute_species", "data": {}},
            error_code="substitute_species_failed",
        )


def insert_interstitial_at_coords(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[defect/modeling] Insert one or more explicit interstitial atoms at user-specified coordinates."""
    try:
        params = InsertInterstitialAtCoordsInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        suffix = _default_structure_suffix(structure_path)
        frac_mode = params.coordinate_type == "fractional"
        if len(params.coords) == 1:
            out_path = resolve_workspace_path(params.output_path) if params.output_path else (
                structure_path.parent / f"{structure_path.stem}_interstitial_{params.species}{suffix}"
            )
            modified = structure.copy()
            modified.append(params.species, np.array(params.coords[0], dtype=float), coords_are_cartesian=not frac_mode)
            _write_structure(out_path, modified)
            data = {
                "input_rel": workspace_relpath(structure_path),
                "output_rel": workspace_relpath(out_path),
                "species": params.species,
                "coordinate_type": params.coordinate_type,
                "coords": [float(x) for x in params.coords[0]],
            }
            content = (
                "insert_interstitial_at_coords completed.\n"
                f"input_rel={data['input_rel']} output_rel={data['output_rel']} species={params.species}"
            )
            return content, {"tool_name": "insert_interstitial_at_coords", "data": data}

        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
            structure_path.parent / f"{structure_path.stem}_interstitials"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        for idx, coords in enumerate(params.coords):
            modified = structure.copy()
            modified.append(params.species, np.array(coords, dtype=float), coords_are_cartesian=not frac_mode)
            out_path = output_dir / f"interstitial_{idx:03d}_{params.species}{suffix}"
            _write_structure(out_path, modified)
            results.append(
                {
                    "batch_index": idx,
                    "coords": [float(x) for x in coords],
                    "output_rel": workspace_relpath(out_path),
                }
            )
        batch_json = output_dir / "batch_insert_interstitials.json"
        _write_json(
            batch_json,
            {
                "input_rel": workspace_relpath(structure_path),
                "species": params.species,
                "coordinate_type": params.coordinate_type,
                "results": results,
            },
        )
        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_dir_rel": workspace_relpath(output_dir),
            "batch_json_rel": workspace_relpath(batch_json),
            "structures_generated": len(results),
            "species": params.species,
        }
        content = (
            "insert_interstitial_at_coords completed.\n"
            f"input_rel={data['input_rel']} structures_generated={data['structures_generated']} "
            f"output_dir_rel={data['output_dir_rel']}\n"
            f"batch_json_rel={data['batch_json_rel']}"
        )
        return content, {"tool_name": "insert_interstitial_at_coords", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="insert_interstitial_at_coords",
            public_message=_error_message(f"insert_interstitial_at_coords failed: {exc}"),
            artifact={"tool_name": "insert_interstitial_at_coords", "data": {}},
            error_code="insert_interstitial_at_coords_failed",
        )


def generate_strained_structures(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[bulk/modeling] Generate strained structure files from explicit deformation matrices or a mode/value grid."""
    try:
        params = GenerateStrainedStructuresInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        output_dir = resolve_workspace_path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = _default_structure_suffix(structure_path)

        specs: list[tuple[str, np.ndarray]] = []
        for idx, spec in enumerate(params.strain_matrices):
            label = (spec.label or f"explicit_{idx:03d}").strip()
            specs.append((label, _normalize_3x3_matrix(spec.matrix)))
        if params.strain_modes and params.strain_values:
            for mode in params.strain_modes:
                for value in params.strain_values:
                    label = f"{mode}_{value:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
                    specs.append((label, _strain_matrix_from_mode(mode, value)))
        if not specs:
            raise ValueError("No strain specifications generated from the provided inputs.")

        results: list[dict[str, Any]] = []
        for idx, (label, deformation) in enumerate(specs):
            strained = _clone_structure_with_same_properties(
                structure,
                np.dot(structure.lattice.matrix, deformation),
            )
            out_path = output_dir / f"strain_{idx:03d}_{label}{suffix}"
            _write_structure(out_path, strained)
            results.append(
                {
                    "batch_index": idx,
                    "label": label,
                    "matrix": [[float(x) for x in row] for row in deformation],
                    "output_rel": workspace_relpath(out_path),
                }
            )
        batch_json = output_dir / "batch_generate_strained_structures.json"
        _write_json(
            batch_json,
            {
                "input_rel": workspace_relpath(structure_path),
                "results": results,
            },
        )
        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_dir_rel": workspace_relpath(output_dir),
            "batch_json_rel": workspace_relpath(batch_json),
            "structures_generated": len(results),
        }
        content = (
            "generate_strained_structures completed.\n"
            f"input_rel={data['input_rel']} structures_generated={data['structures_generated']} "
            f"output_dir_rel={data['output_dir_rel']}\n"
            f"batch_json_rel={data['batch_json_rel']}"
        )
        return content, {"tool_name": "generate_strained_structures", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="generate_strained_structures",
            public_message=_error_message(f"generate_strained_structures failed: {exc}"),
            artifact={"tool_name": "generate_strained_structures", "data": {}},
            error_code="generate_strained_structures_failed",
        )


def generate_kpath(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[electronic/prepare] Generate a high-symmetry line-mode KPOINTS recommendation from a relaxed bulk structure."""
    try:
        params = GenerateKpathInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        analyzer = SpacegroupAnalyzer(structure, symprec=params.symprec, angle_tolerance=params.angle_tolerance)
        target_structure = analyzer.get_primitive_standard_structure() if params.primitive_standard else structure
        kpath = HighSymmKpath(
            target_structure,
            path_type=params.path_type,
            symprec=params.symprec,
            angle_tolerance=params.angle_tolerance,
        )
        kpoints = Kpoints.automatic_linemode(params.line_density, kpath)
        output_path = resolve_workspace_path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kpoints.write_file(str(output_path))
        output_json = resolve_workspace_path(params.output_json) if params.output_json else (
            output_path.with_suffix(output_path.suffix + ".json")
        )
        summary = {
            "input_rel": workspace_relpath(structure_path),
            "kpoints_rel": workspace_relpath(output_path),
            "primitive_standard": params.primitive_standard,
            "path_type": params.path_type,
            "line_density": params.line_density,
            "labels": {
                key: [float(x) for x in value]
                for key, value in (kpath.kpath.get("kpoints", {}) or {}).items()
            },
            "path": [list(segment) for segment in (kpath.kpath.get("path", []) or [])],
            "structure_formula": target_structure.composition.reduced_formula,
        }
        _write_json(output_json, summary)
        data = {
            "input_rel": workspace_relpath(structure_path),
            "kpoints_rel": workspace_relpath(output_path),
            "output_json_rel": workspace_relpath(output_json),
            "path_segments": len(summary["path"]),
            "path_type": params.path_type,
        }
        content = (
            "generate_kpath completed.\n"
            f"input_rel={data['input_rel']} kpoints_rel={data['kpoints_rel']} output_json_rel={data['output_json_rel']}"
        )
        return content, {"tool_name": "generate_kpath", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="generate_kpath",
            public_message=_error_message(f"generate_kpath failed: {exc}"),
            artifact={"tool_name": "generate_kpath", "data": {}},
            error_code="generate_kpath_failed",
        )


def generate_phonon_displacements(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[phonon/modeling] Generate finite-displacement supercell structures for phonon force calculations."""
    try:
        params = GeneratePhononDisplacementsInput(**payload)
        structure_path, structure = _load_structure(params.structure_file)
        output_dir = resolve_workspace_path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = _default_structure_suffix(structure_path)

        results: list[tuple[str, Structure]]
        metadata: dict[str, Any]
        try:
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms

            unitcell = PhonopyAtoms(
                symbols=[site.species_string for site in structure],
                scaled_positions=[site.frac_coords for site in structure],
                cell=structure.lattice.matrix,
            )
            phonon = Phonopy(unitcell, np.diag([int(x) for x in params.supercell]))
            phonon.generate_displacements(distance=params.displacement, is_plusminus=params.plus_minus)
            raw_supercells = getattr(phonon, "supercells_with_displacements", None)
            if raw_supercells is None:
                raw_supercells = phonon.get_supercells_with_displacements()
            results = []
            for idx, cell in enumerate(raw_supercells):
                displaced = Structure(
                    lattice=cell.cell,
                    species=list(cell.symbols),
                    coords=cell.scaled_positions,
                    coords_are_cartesian=False,
                )
                results.append((f"disp_{idx:03d}", displaced))
            metadata = {
                "generator": "phonopy",
                "displacements_generated": len(results),
            }
        except Exception:
            results, metadata = _manual_phonon_displacements(
                structure,
                supercell=params.supercell,
                displacement=params.displacement,
                plus_minus=params.plus_minus,
                symprec=params.symprec,
                angle_tolerance=params.angle_tolerance,
            )

        written: list[dict[str, Any]] = []
        for idx, (label, displaced) in enumerate(results):
            out_path = output_dir / f"{label}{suffix}"
            _write_structure(out_path, displaced)
            written.append(
                {
                    "batch_index": idx,
                    "label": label,
                    "output_rel": workspace_relpath(out_path),
                    "natoms": len(displaced),
                }
            )
        metadata_path = output_dir / "phonon_displacements.json"
        metadata_payload = {
            "input_rel": workspace_relpath(structure_path),
            "supercell": params.supercell,
            "displacement": params.displacement,
            "plus_minus": params.plus_minus,
            "results": written,
            **metadata,
        }
        _write_json(metadata_path, metadata_payload)
        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_dir_rel": workspace_relpath(output_dir),
            "metadata_rel": workspace_relpath(metadata_path),
            "structures_generated": len(written),
            "generator": metadata.get("generator"),
        }
        content = (
            "generate_phonon_displacements completed.\n"
            f"input_rel={data['input_rel']} structures_generated={data['structures_generated']} "
            f"output_dir_rel={data['output_dir_rel']}\n"
            f"metadata_rel={data['metadata_rel']}"
        )
        return content, {"tool_name": "generate_phonon_displacements", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="generate_phonon_displacements",
            public_message=_error_message(f"generate_phonon_displacements failed: {exc}"),
            artifact={"tool_name": "generate_phonon_displacements", "data": {}},
            error_code="generate_phonon_displacements_failed",
        )


__all__ = [
    "SupercellInput",
    "EnumerateUniqueSitesInput",
    "CreateVacancyInput",
    "SubstituteSpeciesInput",
    "InsertInterstitialAtCoordsInput",
    "GenerateStrainedStructuresInput",
    "GenerateKpathInput",
    "GeneratePhononDisplacementsInput",
    "supercell",
    "enumerate_unique_sites",
    "create_vacancy",
    "substitute_species",
    "insert_interstitial_at_coords",
    "generate_strained_structures",
    "generate_kpath",
    "generate_phonon_displacements",
]
