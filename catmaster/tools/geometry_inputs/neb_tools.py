from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from ase.geometry import find_mic
from ase.io import read as ase_read, write as ase_write
from pydantic import BaseModel, Field, field_validator, model_validator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath

from .vasp_inputs import StructWriter

_CELL_TOL = 1e-5
_NEB_MIN_PAIR_DISTANCE_WARNING_ANGSTROM = 0.8
_ELEMENT_MAP_INCAR_KEYS = {"MAGMOM", "LDAUU", "LDAUJ"}
_NEB_PROTECTED_KEYS_BASE = {"IBRION", "POTIM", "ICHAIN", "IMAGES", "IOPT", "ISYM", "LCLIMB"}
_STRUCTURE_FILE_SUFFIXES = {"", ".vasp", ".cif", ".xyz", ".poscar", ".contcar"}


def _element_map_error_message(key: str) -> str:
    return (
        f"{key} must be an element-map in this tool due to pymatgen constraints, "
        'e.g. {"Fe": 2.2} or {"O": 1}.'
    )


def _coerce_element_map_value(key: str, raw_val: Any | None) -> Any | None:
    if raw_val is None:
        return None
    if not isinstance(raw_val, dict):
        raise ValueError(_element_map_error_message(key))
    normalized: Dict[str, Any] = {}
    for sym_raw, value in raw_val.items():
        symbol = str(sym_raw).strip()
        if not symbol:
            raise ValueError(_element_map_error_message(key))
        normalized[symbol] = value
    return normalized


def _normalize_incar_patch(value: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_val in value.items():
        key = str(raw_key).strip().upper()
        if not key:
            raise ValueError("INCAR key must be a non-empty string.")
        if key in _ELEMENT_MAP_INCAR_KEYS:
            raw_val = _coerce_element_map_value(key, raw_val)
        normalized[key] = raw_val
    return normalized


class MakeNebGeometryInput(BaseModel):
    """[neb/modeling] Generate NEB interpolation geometries in nebmake.pl style."""

    initial_path: str | None = Field(None, description="Initial structure file (POSCAR/CONTCAR/.vasp/.cif).")
    final_path: str | None = Field(None, description="Final structure file (POSCAR/CONTCAR/.vasp/.cif).")
    input_root: str | None = Field(
        None,
        description="Optional batch root; each task subdir contains IS and FS structure files.",
    )
    output_root: str | None = Field(
        None,
        description="Batch output root used only with input_root.",
    )
    n_images: int = Field(
        ...,
        ge=1,
        description="Number of intermediate images (NI). Use neb-prepare skill for the image-count heuristic.",
    )
    output_dir: str = Field(
        "neb_images",
        description="Single-case output directory for a flat numbered image-file tree.",
    )
    interp_mode: str = Field(
        "direct",
        description="Output coordinate style when writing POSCAR: direct or cartesian.",
        pattern="^(direct|cartesian)$",
    )
    interp_method: str = Field(
        "linear",
        description="Interpolation method: linear or idpp.",
        pattern="^(linear|idpp)$",
    )
    mic: bool = Field(
        True,
        description="If true, interpolate using the periodic minimum image convention for endpoint displacements.",
    )
    min_pair_distance_warning_angstrom: float = Field(
        _NEB_MIN_PAIR_DISTANCE_WARNING_ANGSTROM,
        ge=0.0,
        description="Warn when any generated image has a minimum interatomic distance below this threshold.",
    )
    overwrite: bool = Field(False, description="If true, overwrite output_dir if it exists.")

    @model_validator(mode="after")
    def _validate_mode(self) -> "MakeNebGeometryInput":
        has_batch = bool(self.input_root)
        has_initial = bool(self.initial_path)
        has_final = bool(self.final_path)
        if has_batch:
            if has_initial or has_final:
                raise ValueError("Provide either initial_path+final_path or input_root, not both.")
            if not self.output_root:
                raise ValueError("Batch mode requires output_root.")
        else:
            if not (has_initial and has_final):
                raise ValueError("Single-case mode requires both initial_path and final_path.")
        return self


class VaspNebPrepareInput(BaseModel):
    """[neb/prepare] Prepare a canonical NEB VASP input tree from endpoint structures or an existing image tree."""

    initial_path: str | None = Field(None, description="Initial endpoint structure path.")
    final_path: str | None = Field(None, description="Final endpoint structure path.")
    input_root: str | None = Field(
        None,
        description="Optional batch root containing task subdirectories with one NEB image tree each.",
    )
    images_root: str | None = Field(
        None,
        description="Existing NEB image-tree root; flat numbered files or legacy numbered POSCAR dirs are accepted.",
    )
    output_root: str = Field(..., description="Target NEB job root. Image directories and root VASP files are written here.")
    n_images: int = Field(
        5,
        ge=1,
        description="Number of intermediate images when generating from endpoints.",
    )
    output_filename: str = Field("POSCAR", description="Filename for generated/copied image structures.")
    interp_mode: str = Field(
        "direct",
        description="Output coordinate style when writing generated POSCAR files: direct or cartesian.",
        pattern="^(direct|cartesian)$",
    )
    interp_method: str = Field(
        "linear",
        description="Interpolation method for endpoint mode: linear or idpp.",
        pattern="^(linear|idpp)$",
    )
    mic: bool = Field(
        True,
        description="If true, interpolate endpoint displacements with the periodic minimum image convention.",
    )
    min_pair_distance_warning_angstrom: float = Field(
        _NEB_MIN_PAIR_DISTANCE_WARNING_ANGSTROM,
        ge=0.0,
        description="Warn when any NEB image has a minimum interatomic distance below this threshold.",
    )
    overwrite: bool = Field(False, description="If true, replace output_root if it already exists.")
    regime: Literal["bulk", "slab", "gas"] = Field(
        "slab",
        description="Scientific regime for shared VASP support files: bulk, slab, or gas.",
    )
    k_product: int = Field(35, ge=1, description="Target k-mesh density for the root KPOINTS.")
    use_d3: bool = Field(False, description="Enable DFT-D3(BJ) correction.")
    use_dft_plus_u: bool = Field(False, description="Enable DFT+U baseline toggle.")
    enable_dipole: bool = Field(False, description="Enable dipole correction helper.")
    climb: bool = Field(False, description="Enable CI-NEB by setting LCLIMB=.TRUE..")
    iopt: int = Field(7, description="VTST IOPT setting. Allowed values: 7, 2, or 1.")
    ediff: float | None = Field(None, description="Optional explicit EDIFF override.")
    ediffg: float | None = Field(None, description="Optional explicit EDIFFG override.")
    spring: float | None = Field(None, description="Optional explicit SPRING override.")
    potim: float = Field(0.0, description="NEB POTIM value. Defaults to 0.0.")
    user_incar_patch: Dict[str, Any] = Field(
        default_factory=dict,
        description="Targeted INCAR patch object applied after canonical NEB overrides.",
    )
    patch_policy: Literal["safe", "force"] = Field(
        "safe",
        description="safe blocks overrides of protected NEB keys; force applies the patch after canonical NEB defaults.",
    )

    @field_validator("user_incar_patch")
    @classmethod
    def _validate_user_incar_patch(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_incar_patch(value)

    @model_validator(mode="after")
    def _validate_source_mode(self) -> "VaspNebPrepareInput":
        has_initial = bool(self.initial_path)
        has_final = bool(self.final_path)
        has_batch = bool(self.input_root)
        has_tree = bool(self.images_root)
        if sum(bool(flag) for flag in (has_batch, has_tree, has_initial or has_final)) != 1:
            raise ValueError(
                "Provide exactly one NEB source mode: either initial_path+final_path, images_root, or input_root."
            )
        if not has_batch and not has_tree and not (has_initial and has_final):
            raise ValueError("Endpoint mode requires both initial_path and final_path.")
        if self.iopt not in {7, 2, 1}:
            raise ValueError("iopt must be one of 7, 2, 1.")
        return self


@dataclass(frozen=True)
class NebImageSet:
    representative_atoms: Any
    image_dirs_rel: list[str]
    total_images: int
    intermediate_images: int
    warnings: list[str]
    geometry_check: dict[str, Any]


class EstimateNebImageCountInput(BaseModel):
    """[neb/modeling] Estimate NEB image count from endpoint displacements under periodic MIC."""

    initial_path: str = Field(..., description="Initial structure file path (POSCAR/CONTCAR/.vasp/.cif).")
    final_path: str = Field(..., description="Final structure file path (POSCAR/CONTCAR/.vasp/.cif).")
    mic: bool = Field(
        True,
        description="If true, compute endpoint displacements with the periodic minimum image convention.",
    )
    target_spacing_angstrom: float = Field(
        0.8,
        gt=0.0,
        description=(
            "Target path spacing in angstrom used for the practical heuristic "
            "ceil(sqrt(sum_i ||Δr_i||^2) / target_spacing_angstrom)."
        ),
    )


class RemapNebEndpointAtomsInput(BaseModel):
    """[neb/modeling] Reorder one endpoint onto the other by minimizing same-species displacement on the mobile subset."""

    initial_path: str = Field(..., description="Reference endpoint structure file path (POSCAR/CONTCAR/.vasp/.cif).")
    final_path: str = Field(..., description="Endpoint structure file to reorder onto the reference atom order.")
    output_path: str | None = Field(
        None,
        description=(
            "Workspace-relative output path for the reordered final structure. Defaults to "
            "mapped_structures/<final_stem>_mapped<suffix>. Selective dynamics are preserved for VASP-style outputs."
        ),
    )
    mic: bool = Field(
        True,
        description="If true, minimize squared endpoint displacements under the periodic minimum image convention.",
    )
    lock_if_current_displacement_below_angstrom: float = Field(
        0.5,
        ge=0.0,
        description=(
            "Mobile atoms whose current-order endpoint displacement is already below this threshold are locked and "
            "excluded from remapping. Default: 0.5 Å."
        ),
    )
    overwrite: bool = Field(False, description="If true, overwrite output_path if it already exists.")


def _read_atoms(path: Path):
    return ase_read(str(path))


def _validate_structures(initial, final) -> Optional[str]:
    if len(initial) != len(final):
        return "Initial and final structures have different atom counts."
    if not np.array_equal(initial.get_atomic_numbers(), final.get_atomic_numbers()):
        return "Initial and final structures have different element sequences."
    if not np.allclose(initial.cell.array, final.cell.array, rtol=_CELL_TOL, atol=_CELL_TOL):
        return "Initial and final lattices differ beyond tolerance."
    return None


def _validate_structure_pair_for_mapping(initial, final) -> Optional[str]:
    if len(initial) != len(final):
        return "Initial and final structures have different atom counts."
    if not np.array_equal(np.sort(initial.get_atomic_numbers()), np.sort(final.get_atomic_numbers())):
        return "Initial and final structures have different element counts."
    if not np.allclose(initial.cell.array, final.cell.array, rtol=_CELL_TOL, atol=_CELL_TOL):
        return "Initial and final lattices differ beyond tolerance."
    if not np.array_equal(np.asarray(initial.pbc, dtype=bool), np.asarray(final.pbc, dtype=bool)):
        return "Initial and final periodic-boundary-condition flags differ."
    return None


def _compute_endpoint_displacements(initial, final, *, mic: bool) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.asarray(final.get_positions() - initial.get_positions(), dtype=float)
    if mic:
        mic_vectors, mic_lengths = find_mic(deltas, cell=initial.cell, pbc=initial.pbc)
        return np.asarray(mic_vectors, dtype=float), np.asarray(mic_lengths, dtype=float)
    lengths = np.linalg.norm(deltas, axis=1)
    return deltas, lengths


def _build_neb_image_count_guidance(
    *,
    rss_displacement_angstrom: float,
    recommended_intermediate_images: int,
) -> list[str]:
    warnings: list[str] = []
    if recommended_intermediate_images < 4:
        warnings.append(
            "The displacement heuristic suggests fewer than 4 intermediate images. This may still be fine for a very "
            "local event, but 4-8 intermediate images is the usual practical range for routine NEB setup."
        )
    if rss_displacement_angstrom > 6.0:
        warnings.append(
            "The periodic path-scale estimate exceeds about 6 Å. Recheck whether the endpoints describe an overlong "
            "migration that should be remodeled into a shorter primitive hop."
        )
    if recommended_intermediate_images > 8:
        warnings.append(
            "The displacement heuristic suggests more than about 8 intermediate images. Reconsider whether this is "
            "one local elementary event or a longer path that should be decomposed."
        )
    return warnings


def _default_mapped_output_path(final_path: Path) -> Path:
    suffix = final_path.suffix or ".vasp"
    return resolve_workspace_path(f"mapped_structures/{final_path.stem}_mapped{suffix}")


def _pairwise_squared_displacements(
    initial_positions: np.ndarray,
    final_positions: np.ndarray,
    *,
    cell: Any,
    pbc: Any,
    mic: bool,
) -> np.ndarray:
    deltas = np.asarray(final_positions[None, :, :] - initial_positions[:, None, :], dtype=float)
    if mic:
        remapped, _ = find_mic(deltas.reshape(-1, 3), cell=cell, pbc=pbc)
        deltas = np.asarray(remapped, dtype=float).reshape(deltas.shape)
    return np.einsum("ijk,ijk->ij", deltas, deltas)


def _solve_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    size = int(cost_matrix.shape[0])
    if size <= 1:
        return np.zeros(size, dtype=int)
    preferences = np.argsort(cost_matrix, axis=1)
    assignment = _stable_local_assignment(cost_matrix, preferences)
    return _improve_assignment_by_swaps(cost_matrix, assignment)


def _stable_local_assignment(cost_matrix: np.ndarray, preferences: np.ndarray) -> np.ndarray:
    size = int(cost_matrix.shape[0])
    assignment = np.full(size, -1, dtype=int)
    owner_by_col = np.full(size, -1, dtype=int)
    next_choice = np.zeros(size, dtype=int)
    proposal_gap = []
    for row in range(size):
        ordered = cost_matrix[row, preferences[row]]
        gap = float(ordered[1] - ordered[0]) if size > 1 else float("inf")
        proposal_gap.append((gap, row))
    pending = [row for _gap, row in sorted(proposal_gap, reverse=True)]
    while pending:
        row = pending.pop(0)
        while assignment[row] < 0:
            choice_rank = int(next_choice[row])
            if choice_rank >= size:
                raise ValueError("Failed to find a complete local assignment.")
            col = int(preferences[row, choice_rank])
            next_choice[row] = choice_rank + 1
            current_owner = int(owner_by_col[col])
            if current_owner < 0:
                owner_by_col[col] = row
                assignment[row] = col
                break
            if float(cost_matrix[row, col]) + 1e-12 < float(cost_matrix[current_owner, col]):
                assignment[current_owner] = -1
                owner_by_col[col] = row
                assignment[row] = col
                pending.append(current_owner)
                break
    return assignment


def _improve_assignment_by_swaps(cost_matrix: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    improved = True
    size = int(len(assignment))
    while improved:
        improved = False
        for i in range(size):
            for j in range(i + 1, size):
                current = float(cost_matrix[i, assignment[i]] + cost_matrix[j, assignment[j]])
                swapped = float(cost_matrix[i, assignment[j]] + cost_matrix[j, assignment[i]])
                if swapped + 1e-12 < current:
                    assignment[i], assignment[j] = int(assignment[j]), int(assignment[i])
                    improved = True
    return assignment


def _infer_mobile_indices(input_path: Path, atoms: Any) -> tuple[list[int], str]:
    atom_count = int(len(atoms))
    try:
        poscar = Poscar.from_file(str(input_path))
    except Exception:
        poscar = None
    if poscar is not None:
        selective_dynamics = getattr(poscar, "selective_dynamics", None)
        if selective_dynamics is not None:
            mask = np.asarray(selective_dynamics, dtype=bool)
            if mask.shape == (atom_count, 3):
                active = [idx for idx, row in enumerate(mask) if bool(np.any(row))]
                return active, "selective_dynamics"

    raw_constraints = getattr(atoms, "constraints", None)
    if raw_constraints:
        constraints = raw_constraints if isinstance(raw_constraints, (list, tuple)) else [raw_constraints]
        fixed: set[int] = set()
        for constraint in constraints:
            if type(constraint).__name__ != "FixAtoms":
                continue
            indices = getattr(constraint, "index", None)
            if indices is None:
                continue
            fixed.update(int(idx) for idx in np.asarray(indices, dtype=int).reshape(-1))
        if fixed:
            return [idx for idx in range(atom_count) if idx not in fixed], "fixatoms_constraint"

    return list(range(atom_count)), "all_atoms"


def _build_reordered_final(
    initial,
    final,
    *,
    mic: bool,
    remap_indices: list[int],
) -> tuple[Any, np.ndarray]:
    initial_numbers = np.asarray(initial.get_atomic_numbers(), dtype=int)
    final_numbers = np.asarray(final.get_atomic_numbers(), dtype=int)
    assignment = np.arange(len(initial_numbers), dtype=int)
    mobile_idx_arr = np.asarray(sorted(int(idx) for idx in remap_indices), dtype=int)
    if mobile_idx_arr.size == 0:
        return final.copy(), assignment
    for atomic_number in np.unique(initial_numbers[mobile_idx_arr]):
        init_idx = mobile_idx_arr[initial_numbers[mobile_idx_arr] == atomic_number]
        final_idx = mobile_idx_arr[final_numbers[mobile_idx_arr] == atomic_number]
        cost_matrix = _pairwise_squared_displacements(
            np.asarray(initial.get_positions()[init_idx], dtype=float),
            np.asarray(final.get_positions()[final_idx], dtype=float),
            cell=initial.cell,
            pbc=initial.pbc,
            mic=mic,
        )
        local_assignment = _solve_assignment(cost_matrix)
        assignment[init_idx] = final_idx[local_assignment]
    reordered = final[assignment.tolist()]
    return reordered, assignment


def _write_structure_output(path: Path, atoms: Any) -> None:
    suffix = path.suffix.lower()
    if suffix in {"", ".vasp", ".poscar", ".contcar"}:
        ase_write(str(path), atoms, format="vasp", direct=True, vasp5=True)
        return
    ase_write(str(path), atoms)


def _write_reordered_structure_output(path: Path, atoms: Any, *, source_path: Path, assignment: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix not in {"", ".vasp", ".poscar", ".contcar"}:
        _write_structure_output(path, atoms)
        return
    try:
        poscar = Poscar.from_file(str(source_path))
    except Exception:
        _write_structure_output(path, atoms)
        return
    selective_dynamics = getattr(poscar, "selective_dynamics", None)
    reordered_sd = None
    if selective_dynamics is not None:
        sd_arr = np.asarray(selective_dynamics, dtype=bool)
        if sd_arr.shape == (len(assignment), 3):
            reordered_sd = sd_arr[np.asarray(assignment, dtype=int)].tolist()
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)
    Poscar(
        pmg_structure,
        comment=getattr(poscar, "comment", None),
        selective_dynamics=reordered_sd,
        sort_structure=False,
    ).write_file(str(path))


def remap_neb_endpoint_atoms(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[neb/modeling] Reorder the final endpoint onto the initial atom order on the mobile subset while leaving frozen atoms untouched."""
    params = RemapNebEndpointAtomsInput(**payload)
    init_path = resolve_workspace_path(params.initial_path, must_exist=True)
    final_path = resolve_workspace_path(params.final_path, must_exist=True)
    output_path = resolve_workspace_path(params.output_path) if params.output_path else _default_mapped_output_path(final_path)
    if output_path.exists() and not params.overwrite:
        _fail(
            "remap_neb_endpoint_atoms",
            message=f"output_path already exists: {workspace_relpath(output_path)}. Set overwrite=true to replace.",
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
                "output_dir": workspace_relpath(output_path),
            },
            error_code="output_exists",
        )

    try:
        initial = _read_atoms(init_path)
        final = _read_atoms(final_path)
    except Exception as exc:
        _fail(
            "remap_neb_endpoint_atoms",
            message=f"Failed to read initial/final structures: {exc}",
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
                "output_dir": workspace_relpath(output_path),
            },
            error_code="read_failed",
        )

    error = _validate_structure_pair_for_mapping(initial, final)
    if error:
        _fail(
            "remap_neb_endpoint_atoms",
            message=error,
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
                "output_dir": workspace_relpath(output_path),
            },
            error_code="invalid_neb_pair",
        )

    mobile_indices, mobile_source = _infer_mobile_indices(init_path, initial)
    if not mobile_indices:
        _fail(
            "remap_neb_endpoint_atoms",
            message="No mobile atoms were inferred from the reference structure; nothing can be remapped.",
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
                "output_dir": workspace_relpath(output_path),
            },
            error_code="no_mobile_atoms",
        )
    fixed_indices = [idx for idx in range(len(initial)) if idx not in set(mobile_indices)]
    if fixed_indices:
        initial_numbers = np.asarray(initial.get_atomic_numbers(), dtype=int)
        final_numbers = np.asarray(final.get_atomic_numbers(), dtype=int)
        if not np.array_equal(initial_numbers[fixed_indices], final_numbers[fixed_indices]):
            _fail(
                "remap_neb_endpoint_atoms",
                message=(
                    "Frozen atoms are excluded from remapping, but the fixed-index element sequence already differs "
                    "between initial and final. Reorder or regenerate the endpoints before using this tool."
                ),
                data={
                    "initial_rel": workspace_relpath(init_path),
                    "final_rel": workspace_relpath(final_path),
                    "output_dir": workspace_relpath(output_path),
                },
                error_code="fixed_atom_order_mismatch",
            )

    current_vectors, current_norms = _compute_endpoint_displacements(initial, final, mic=params.mic)
    initial_numbers = np.asarray(initial.get_atomic_numbers(), dtype=int)
    final_numbers = np.asarray(final.get_atomic_numbers(), dtype=int)
    locked_small_displacement_indices = [
        int(idx)
        for idx in mobile_indices
        if initial_numbers[idx] == final_numbers[idx]
        and float(current_norms[idx]) < float(params.lock_if_current_displacement_below_angstrom)
    ]
    remap_indices = [int(idx) for idx in mobile_indices if idx not in set(locked_small_displacement_indices)]
    reordered_final, assignment = _build_reordered_final(initial, final, mic=params.mic, remap_indices=remap_indices)
    reordered_vectors, reordered_norms = _compute_endpoint_displacements(initial, reordered_final, mic=params.mic)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_reordered_structure_output(output_path, reordered_final, source_path=final_path, assignment=assignment)

    mapping_changed = not np.array_equal(assignment, np.arange(len(assignment), dtype=int))
    data = {
        "initial_rel": workspace_relpath(init_path),
        "final_rel": workspace_relpath(final_path),
        "mapped_final_rel": workspace_relpath(output_path),
        "mic": bool(params.mic),
        "atom_count": int(len(initial)),
        "mapping_algorithm": "local_preference_matching_plus_swap_refinement",
        "mobile_atom_indices": [int(idx) for idx in mobile_indices],
        "mobile_atom_indices_source": mobile_source,
        "locked_small_displacement_indices": locked_small_displacement_indices,
        "lock_if_current_displacement_below_angstrom": float(params.lock_if_current_displacement_below_angstrom),
        "remap_candidate_indices": remap_indices,
        "element_sequence_matches_before_mapping": bool(
            np.array_equal(initial.get_atomic_numbers(), final.get_atomic_numbers())
        ),
        "mapping_changed": bool(mapping_changed),
        "assigned_final_indices_for_initial_order": [int(idx) for idx in assignment.tolist()],
        "rss_displacement_current_order_angstrom": float(np.linalg.norm(current_vectors.reshape(-1))),
        "rss_displacement_mapped_order_angstrom": float(np.linalg.norm(reordered_vectors.reshape(-1))),
        "max_atom_displacement_current_order_angstrom": float(np.max(current_norms)) if len(current_norms) else 0.0,
        "max_atom_displacement_mapped_order_angstrom": float(np.max(reordered_norms)) if len(reordered_norms) else 0.0,
        "per_atom_displacements_mapped_angstrom": [float(value) for value in reordered_norms.tolist()],
    }
    lines = [
        "remap_neb_endpoint_atoms completed.",
        f"mapped_final_rel={data['mapped_final_rel']}",
        f"mapping_changed={str(bool(mapping_changed)).lower()}",
        f"rss_displacement_current_order_angstrom={data['rss_displacement_current_order_angstrom']:.6f}",
        f"rss_displacement_mapped_order_angstrom={data['rss_displacement_mapped_order_angstrom']:.6f}",
        f"locked_small_displacement_count={len(locked_small_displacement_indices)}",
        f"mic={str(bool(params.mic)).lower()}",
    ]
    return _success("remap_neb_endpoint_atoms", content="\n".join(lines), data=data)


def estimate_neb_image_count(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[neb/modeling] Estimate a practical NEB image count from two endpoint structures under periodic MIC."""
    params = EstimateNebImageCountInput(**payload)
    init_path = resolve_workspace_path(params.initial_path, must_exist=True)
    final_path = resolve_workspace_path(params.final_path, must_exist=True)

    try:
        initial = _read_atoms(init_path)
        final = _read_atoms(final_path)
    except Exception as exc:
        _fail(
            "estimate_neb_image_count",
            message=f"Failed to read initial/final structures: {exc}",
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
            },
            error_code="read_failed",
        )

    error = _validate_structures(initial, final)
    if error:
        _fail(
            "estimate_neb_image_count",
            message=error,
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
            },
            error_code="invalid_neb_pair",
        )

    displacement_vectors, displacement_norms = _compute_endpoint_displacements(initial, final, mic=params.mic)
    rss_displacement = float(np.linalg.norm(displacement_vectors.reshape(-1)))
    recommended_images = max(1, int(math.ceil(rss_displacement / float(params.target_spacing_angstrom))))
    warnings = _build_neb_image_count_guidance(
        rss_displacement_angstrom=rss_displacement,
        recommended_intermediate_images=recommended_images,
    )
    data = {
        "initial_rel": workspace_relpath(init_path),
        "final_rel": workspace_relpath(final_path),
        "mic": bool(params.mic),
        "target_spacing_angstrom": float(params.target_spacing_angstrom),
        "atom_count": int(len(initial)),
        "rss_displacement_angstrom": rss_displacement,
        "max_atom_displacement_angstrom": float(np.max(displacement_norms)) if len(displacement_norms) else 0.0,
        "recommended_intermediate_images": recommended_images,
        "typical_intermediate_image_range": "4-8",
        "per_atom_displacements_angstrom": [float(value) for value in displacement_norms.tolist()],
    }
    lines = [
        "estimate_neb_image_count completed.",
        f"recommended_intermediate_images={recommended_images}",
        f"rss_displacement_angstrom={rss_displacement:.6f}",
        f"max_atom_displacement_angstrom={data['max_atom_displacement_angstrom']:.6f}",
        f"mic={str(bool(params.mic)).lower()} target_spacing_angstrom={float(params.target_spacing_angstrom):.3f}",
        "practical_guideline=Prefer about 4-8 intermediate images for routine local events.",
    ]
    return _success("estimate_neb_image_count", content="\n".join(lines), data=data, warnings=warnings)


def _find_named_structure_file(task_dir: Path, label: str) -> Path:
    target = str(label).strip().upper()
    matches: list[Path] = []
    for candidate in sorted(task_dir.iterdir()):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix not in _STRUCTURE_FILE_SUFFIXES:
            continue
        stem = candidate.stem.upper() if suffix else candidate.name.upper()
        if stem != target:
            continue
        matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f"No {target} structure file found in {workspace_relpath(task_dir)}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {target} structure files found in {workspace_relpath(task_dir)}")
    return matches[0]


def _build_images(
    initial,
    final,
    n_images: int,
    *,
    interp_method: str,
    mic: bool,
) -> Tuple[List[Any], List[str]]:
    try:
        from ase.mep import NEB
    except ImportError:  # pragma: no cover
        from ase.neb import NEB

    warnings: List[str] = []
    init_use = initial.copy()
    final_use = final.copy()
    images_atoms = [init_use]
    for _ in range(n_images):
        images_atoms.append(init_use.copy())
    images_atoms.append(final_use)

    neb = NEB(images_atoms, method="improvedtangent")
    neb.interpolate(method=interp_method, mic=bool(mic))
    return list(images_atoms), warnings


def _minimum_pair_distance_record(atoms: Any, *, image_label: str) -> dict[str, Any] | None:
    atom_count = int(len(atoms))
    if atom_count < 2:
        return None
    use_mic = bool(np.any(np.asarray(getattr(atoms, "pbc", False), dtype=bool)))
    try:
        distances = np.asarray(atoms.get_all_distances(mic=use_mic), dtype=float)
    except Exception:
        distances = np.asarray(atoms.get_all_distances(mic=False), dtype=float)
    if distances.shape != (atom_count, atom_count):
        return None
    np.fill_diagonal(distances, np.inf)
    flat_index = int(np.argmin(distances))
    distance = float(distances.reshape(-1)[flat_index])
    if not math.isfinite(distance):
        return None
    i, j = np.unravel_index(flat_index, distances.shape)
    symbols = atoms.get_chemical_symbols()
    return {
        "image": str(image_label),
        "distance_angstrom": distance,
        "atom_indices_0based": [int(i), int(j)],
        "atom_indices_1based": [int(i) + 1, int(j) + 1],
        "elements": [str(symbols[int(i)]), str(symbols[int(j)])],
    }


def _check_neb_image_distances(
    images: list[Any],
    *,
    threshold_angstrom: float,
    image_labels: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    threshold = float(threshold_angstrom)
    labels = image_labels or [f"{idx:02d}" for idx in range(len(images))]
    records: list[dict[str, Any]] = []
    for idx, atoms in enumerate(images):
        label = labels[idx] if idx < len(labels) else f"{idx:02d}"
        record = _minimum_pair_distance_record(atoms, image_label=label)
        if record is not None:
            records.append(record)
    shortest = min(records, key=lambda item: float(item["distance_angstrom"])) if records else None
    short_records = [record for record in records if float(record["distance_angstrom"]) < threshold]
    warnings: list[str] = []
    if short_records:
        shortest_short = min(short_records, key=lambda item: float(item["distance_angstrom"]))
        atom_pair = "-".join(str(idx) for idx in shortest_short["atom_indices_1based"])
        elem_pair = "-".join(str(symbol) for symbol in shortest_short["elements"])
        warnings.append(
            f"NEB image geometry warning: {len(short_records)} image(s) have minimum interatomic distance below "
            f"{threshold:.2f} Angstrom; shortest is {float(shortest_short['distance_angstrom']):.3f} Angstrom "
            f"in image {shortest_short['image']} between atoms {atom_pair} ({elem_pair}). This often indicates "
            "interpolated atom overlap; recheck endpoint atom mapping, use remap_neb_endpoint_atoms for same-species "
            "permutations, try idpp interpolation, or remodel the path as a shorter primitive hop."
        )
    check = {
        "min_pair_distance_warning_threshold_angstrom": threshold,
        "min_pair_distance_angstrom": float(shortest["distance_angstrom"]) if shortest else None,
        "min_pair_distance_image": str(shortest["image"]) if shortest else None,
        "min_pair_distance_atoms_1based": list(shortest["atom_indices_1based"]) if shortest else [],
        "min_pair_distance_elements": list(shortest["elements"]) if shortest else [],
        "short_distance_count": len(short_records),
        "short_distance_records": short_records,
    }
    return check, warnings


def _success(
    tool_name: str,
    *,
    content: str,
    data: dict[str, Any],
    warnings: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    return content, artifact


def _fail(
    tool_name: str,
    *,
    message: str,
    data: dict[str, Any] | None = None,
    error_code: str = "",
) -> None:
    details: list[str] = [str(message).strip()]
    if isinstance(data, dict):
        for key in (
            "initial_rel",
            "final_rel",
            "images_root_rel",
            "output_root_rel",
            "output_dir",
            "output_incar_path",
            "diff_json_rel",
        ):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            details.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(details),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def make_neb_geometry(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[neb/modeling] Generate interpolated NEB image geometries from initial and final structures."""
    params = MakeNebGeometryInput(**payload)
    if params.input_root:
        return _make_neb_geometry_batch(params)
    init_path = resolve_workspace_path(params.initial_path, must_exist=True)
    final_path = resolve_workspace_path(params.final_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_dir)

    try:
        initial = _read_atoms(init_path)
        final = _read_atoms(final_path)
    except Exception as exc:
        _fail(
            "make_neb_geometry",
            message=f"Failed to read initial/final structures: {exc}",
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
            },
            error_code="read_failed",
        )
    error = _validate_structures(initial, final)
    if error:
        _fail(
            "make_neb_geometry",
            message=error,
            data={
                "initial_rel": workspace_relpath(init_path),
                "final_rel": workspace_relpath(final_path),
            },
            error_code="invalid_neb_pair",
        )
    warnings: List[str] = []

    if output_root.exists():
        if output_root.is_file():
            _fail(
                "make_neb_geometry",
                message=f"output_dir is a file: {output_root}",
                error_code="invalid_output_dir",
            )
        if not params.overwrite:
            _fail(
                "make_neb_geometry",
                message=f"output_dir already exists: {output_root}. Set overwrite=true to replace.",
                error_code="output_dir_exists",
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images, interp_warnings = _build_images(
        initial,
        final,
        params.n_images,
        interp_method=params.interp_method,
        mic=params.mic,
    )
    warnings.extend(interp_warnings)
    geometry_check, geometry_warnings = _check_neb_image_distances(
        images,
        threshold_angstrom=params.min_pair_distance_warning_angstrom,
    )
    warnings.extend(geometry_warnings)

    image_files: List[str] = []
    for idx, atoms in enumerate(images):
        out_path = output_root / f"{idx:02d}.vasp"
        ase_write(
            str(out_path),
            atoms,
            format="vasp",
            direct=(params.interp_mode == "direct"),
            vasp5=True,
        )
        image_files.append(workspace_relpath(out_path))

    summary_path = output_root / "neb_geometry_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "output_dir": workspace_relpath(output_root),
                "num_intermediate_images": params.n_images,
                "num_total_images": params.n_images + 2,
                "image_files": image_files,
                "geometry_check": geometry_check,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    data = {
        "output_dir": workspace_relpath(output_root),
        "num_intermediate_images": params.n_images,
        "num_total_images": params.n_images + 2,
        "summary_json_rel": workspace_relpath(summary_path),
        "geometry_check": geometry_check,
        **compact_list_for_artifact(
            image_files,
            count_key="image_files_count",
            inline_key="image_files",
            preview_key="image_files_preview",
            truncated_key="image_files_truncated",
            full_rel_key="image_files_full_rel",
            full_rel=workspace_relpath(summary_path),
            max_inline=20,
        ),
    }
    lines = [
        "make_neb_geometry completed.",
        f"num_total_images={data['num_total_images']} num_intermediate_images={params.n_images}",
        f"output_dir={data['output_dir']}",
        f"summary_json_rel={data['summary_json_rel']}",
        f"min_pair_distance_angstrom={geometry_check['min_pair_distance_angstrom']}",
        f"short_distance_count={geometry_check['short_distance_count']}",
    ]
    if image_files:
        lines.append(f"first_image_file={image_files[0]}")
        lines.append(f"last_image_file={image_files[-1]}")
    return _success("make_neb_geometry", content="\n".join(lines), data=data, warnings=warnings)


def _make_neb_geometry_batch(params: MakeNebGeometryInput) -> tuple[str, dict[str, Any]]:
    input_root = resolve_workspace_path(str(params.input_root), must_exist=True)
    output_root = resolve_workspace_path(str(params.output_root))
    if not input_root.is_dir():
        _fail("make_neb_geometry", message=f"input_root is not a directory: {input_root}", error_code="invalid_input_root")
    if output_root.exists():
        if output_root.is_file():
            _fail("make_neb_geometry", message=f"output_root is a file: {output_root}", error_code="invalid_output_root")
        if not params.overwrite:
            _fail("make_neb_geometry", message=f"output_root already exists: {output_root}. Set overwrite=true to replace.", error_code="output_root_exists")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    stray_files = sorted(path.name for path in input_root.iterdir() if path.is_file())
    if stray_files:
        _fail(
            "make_neb_geometry",
            message=f"Batch input_root must contain only task directories. Unexpected files: {', '.join(stray_files[:5])}",
            data={"output_root_rel": workspace_relpath(output_root)},
            error_code="invalid_batch_root",
        )
    if not task_dirs:
        _fail("make_neb_geometry", message="Batch input_root contains no task directories.", error_code="invalid_batch_root")

    task_results: list[dict[str, Any]] = []
    batch_warnings: list[str] = []
    for task_dir in task_dirs:
        if any(path.is_dir() for path in task_dir.iterdir()):
            _fail(
                "make_neb_geometry",
                message=f"Nested directories are forbidden inside batch NEB geometry task dirs: {workspace_relpath(task_dir)}",
                error_code="nested_task_forbidden",
            )
        init_path = _find_named_structure_file(task_dir, "IS")
        final_path = _find_named_structure_file(task_dir, "FS")
        task_output_root = output_root / task_dir.name
        task_params = params.model_copy(update={"initial_path": workspace_relpath(init_path), "final_path": workspace_relpath(final_path), "input_root": None, "output_root": None, "output_dir": workspace_relpath(task_output_root)})
        _content, artifact = make_neb_geometry(task_params.model_dump(exclude_none=True))
        task_results.append({"task_id": task_dir.name, **artifact["data"]})
        for warning in artifact.get("warnings", []) or []:
            batch_warnings.append(f"{task_dir.name}: {warning}")

    batch_summary_path = output_root / "batch_summary.json"
    batch_summary = {
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "task_count": len(task_results),
        "tasks": task_results,
        "warnings": batch_warnings,
    }
    batch_summary_path.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    data = {
        "input_root_rel": workspace_relpath(input_root),
        "output_root_rel": workspace_relpath(output_root),
        "task_count": len(task_results),
        "batch_summary_rel": workspace_relpath(batch_summary_path),
    }
    content = (
        "make_neb_geometry completed.\n"
        f"task_count={len(task_results)} output_root_rel={data['output_root_rel']} "
        f"batch_summary_rel={data['batch_summary_rel']}"
    )
    return _success("make_neb_geometry", content=content, data=data, warnings=batch_warnings)


def _strip_incar_comment(line: str) -> str:
    for sep in ("!", "#"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.strip()


def _parse_incar(path: Path) -> Tuple[List[str], Dict[str, str]]:
    order: List[str] = []
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = _strip_incar_comment(raw)
        if not line:
            continue
        if "=" in line:
            key, val = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
        key = key.strip().upper()
        val = val.strip()
        values[key] = val
        if key not in order:
            order.append(key)
    return order, values


def _format_incar_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.16g}"
    if isinstance(value, (list, tuple)):
        return " ".join(_format_incar_value(v) for v in value)
    if value is None:
        return ""
    return str(value).strip()


def _find_image_structure_file(image_dir: Path) -> Path:
    for name in ("POSCAR", "CONTCAR"):
        candidate = image_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(image_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in {".vasp", ".cif", ".xyz"}:
            return candidate
    raise FileNotFoundError(f"No structure file found in {image_dir}")


def _parse_flat_image_file_name(path: Path) -> int | None:
    if not path.is_file():
        return None
    stem = path.stem
    if not stem.isdigit():
        return None
    if path.suffix.lower() not in {".vasp", ".cif", ".xyz", ".poscar"}:
        return None
    return int(stem)


def _classify_single_image_tree(path: Path) -> tuple[list[tuple[int, Path]], str] | None:
    if not path.is_dir():
        return None
    numbered_dirs = sorted(
        [child for child in path.iterdir() if child.is_dir() and child.name.isdigit()],
        key=lambda child: int(child.name),
    )
    numbered_files = sorted(
        [(idx, child) for child in path.iterdir() if (idx := _parse_flat_image_file_name(child)) is not None],
        key=lambda item: item[0],
    )
    if numbered_dirs and numbered_files:
        return None
    if not numbered_dirs and not numbered_files:
        return None
    image_sources = [(int(child.name), child) for child in numbered_dirs] if numbered_dirs else numbered_files
    found = [idx for idx, _ in image_sources]
    expected = list(range(found[0], found[-1] + 1))
    if found != expected or found[0] != 0:
        return None
    return image_sources, ("legacy_dir" if numbered_dirs else "flat_file")


def _discover_image_tree_tasks(input_root: Path) -> tuple[list[Path], bool]:
    single = _classify_single_image_tree(input_root)
    if single is not None:
        return [input_root], True
    child_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    stray_files = sorted(path.name for path in input_root.iterdir() if path.is_file())
    if stray_files:
        raise ValueError(
            f"Batch NEB input_root must contain only task directories. Unexpected files: {', '.join(stray_files[:5])}"
        )
    if not child_dirs:
        raise ValueError("input_root is neither a NEB image tree nor a batch directory of NEB tasks.")
    for child in child_dirs:
        if _classify_single_image_tree(child) is None:
            raise ValueError(
                f"Each child of batch input_root must be one NEB image tree with flat files or legacy image dirs: {workspace_relpath(child)}"
            )
    return child_dirs, False


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


class VaspNebWriter:
    """Prepare a NEB job root while keeping geometry generation as a separate reusable primitive."""

    def __init__(self) -> None:
        self._vasp_writer = StructWriter()

    def prepare(self, params: VaspNebPrepareInput) -> tuple[str, dict[str, Any]]:
        if params.input_root:
            return self._prepare_batch(params)
        output_root = resolve_workspace_path(params.output_root)
        self._prepare_output_root(output_root=output_root, overwrite=params.overwrite)

        image_set = (
            self._materialize_from_image_tree(params=params, output_root=output_root)
            if params.images_root
            else self._materialize_from_endpoints(params=params, output_root=output_root)
        )

        representative_structure = AseAtomsAdaptor.get_structure(image_set.representative_atoms)
        support_info = self._write_neb_support_files(
            params=params,
            output_root=output_root,
            structure=representative_structure,
            intermediate_images=image_set.intermediate_images,
        )
        data = {
            "output_root_rel": workspace_relpath(output_root),
            "image_dirs": image_set.image_dirs_rel,
            "num_total_images": image_set.total_images,
            "num_intermediate_images": image_set.intermediate_images,
            "geometry_check": image_set.geometry_check,
            "output_incar_path": support_info["output_incar_path"],
            "diff_json_rel": support_info["diff_json_rel"],
            "protected_incar_keys": support_info["protected_incar_keys"],
            "applied_overrides": support_info["applied_overrides"],
            "regime": params.regime,
            "k_product": params.k_product,
        }
        lines = [
            "vasp_neb_prepare completed.",
            f"num_total_images={image_set.total_images} num_intermediate_images={image_set.intermediate_images}",
            f"output_root_rel={data['output_root_rel']}",
            f"output_incar_path={data['output_incar_path']}",
            f"min_pair_distance_angstrom={image_set.geometry_check['min_pair_distance_angstrom']}",
            f"short_distance_count={image_set.geometry_check['short_distance_count']}",
        ]
        if params.images_root:
            lines.append(
                "note=image-tree mode does not infer endpoint energies; copy the original relax OUTCAR files into the endpoint image directories before NEB analysis."
            )
        if data["diff_json_rel"]:
            lines.append(f"diff_json_rel={data['diff_json_rel']}")
        return _success(
            "vasp_neb_prepare",
            content="\n".join(lines),
            data=data,
            warnings=image_set.warnings,
        )

    def _prepare_batch(self, params: VaspNebPrepareInput) -> tuple[str, dict[str, Any]]:
        input_root = resolve_workspace_path(str(params.input_root), must_exist=True)
        output_root = resolve_workspace_path(params.output_root)
        if not input_root.is_dir():
            _fail("vasp_neb_prepare", message=f"input_root is not a directory: {input_root}", error_code="invalid_input_root")
        self._prepare_output_root(output_root=output_root, overwrite=params.overwrite)
        try:
            task_dirs, _single_task_mode = _discover_image_tree_tasks(input_root)
        except Exception as exc:
            _fail(
                "vasp_neb_prepare",
                message=str(exc),
                data={"output_root_rel": workspace_relpath(output_root)},
                error_code="invalid_batch_root",
            )
        task_results: list[dict[str, Any]] = []
        batch_warnings: list[str] = []
        for task_dir in task_dirs:
            task_payload = params.model_dump(exclude_none=True)
            task_payload.pop("input_root", None)
            task_payload["images_root"] = workspace_relpath(task_dir)
            task_payload["output_root"] = workspace_relpath(output_root / task_dir.name)
            _content, artifact = self.prepare(VaspNebPrepareInput(**task_payload))
            task_results.append({"task_id": task_dir.name, **artifact["data"]})
            for warning in artifact.get("warnings", []) or []:
                batch_warnings.append(f"{task_dir.name}: {warning}")
        batch_summary_path = output_root / "batch_summary.json"
        batch_summary = {
            "input_root_rel": workspace_relpath(input_root),
            "output_root_rel": workspace_relpath(output_root),
            "task_count": len(task_results),
            "tasks": task_results,
        }
        batch_summary_path.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        data = {
            "input_root_rel": workspace_relpath(input_root),
            "output_root_rel": workspace_relpath(output_root),
            "task_count": len(task_results),
            "batch_summary_rel": workspace_relpath(batch_summary_path),
        }
        content = (
            "vasp_neb_prepare completed.\n"
            f"task_count={len(task_results)} output_root_rel={data['output_root_rel']} "
            f"batch_summary_rel={data['batch_summary_rel']}\n"
            "note=batch image-tree mode does not infer endpoint energies; copy the original relax OUTCAR files into the endpoint image directories before NEB analysis."
        )
        return _success("vasp_neb_prepare", content=content, data=data, warnings=batch_warnings)

    @staticmethod
    def _prepare_output_root(*, output_root: Path, overwrite: bool) -> None:
        if output_root.exists():
            if output_root.is_file():
                _fail(
                    "vasp_neb_prepare",
                    message=f"output_root is a file: {output_root}",
                    error_code="invalid_output_root",
                )
            if not overwrite:
                _fail(
                    "vasp_neb_prepare",
                    message=f"output_root already exists: {output_root}. Set overwrite=true to replace.",
                    data={"output_root_rel": workspace_relpath(output_root)},
                    error_code="output_root_exists",
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    def _materialize_from_endpoints(self, *, params: VaspNebPrepareInput, output_root: Path) -> NebImageSet:
        init_path = resolve_workspace_path(str(params.initial_path), must_exist=True)
        final_path = resolve_workspace_path(str(params.final_path), must_exist=True)
        try:
            initial = _read_atoms(init_path)
            final = _read_atoms(final_path)
        except Exception as exc:
            _fail(
                "vasp_neb_prepare",
                message=f"Failed to read initial/final structures: {exc}",
                data={
                    "initial_rel": workspace_relpath(init_path),
                    "final_rel": workspace_relpath(final_path),
                    "output_root_rel": workspace_relpath(output_root),
                },
                error_code="read_failed",
            )
        error = _validate_structures(initial, final)
        if error:
            _fail(
                "vasp_neb_prepare",
                message=error,
                data={
                    "initial_rel": workspace_relpath(init_path),
                    "final_rel": workspace_relpath(final_path),
                    "output_root_rel": workspace_relpath(output_root),
                },
                error_code="invalid_neb_pair",
            )

        images, warnings = _build_images(
            initial,
            final,
            params.n_images,
            interp_method=params.interp_method,
            mic=params.mic,
        )
        geometry_check, geometry_warnings = _check_neb_image_distances(
            images,
            threshold_angstrom=params.min_pair_distance_warning_angstrom,
        )
        warnings.extend(geometry_warnings)
        image_dirs = self._write_images(
            output_root=output_root,
            images=images,
            output_filename=params.output_filename,
            interp_mode=params.interp_mode,
        )
        warnings.extend(
            self._copy_endpoint_support_files(
                init_path=init_path,
                final_path=final_path,
                output_root=output_root,
                total_images=len(images),
            )
        )
        return NebImageSet(
            representative_atoms=images[0],
            image_dirs_rel=image_dirs,
            total_images=len(images),
            intermediate_images=params.n_images,
            warnings=warnings,
            geometry_check=geometry_check,
        )

    def _materialize_from_image_tree(self, *, params: VaspNebPrepareInput, output_root: Path) -> NebImageSet:
        images_root = resolve_workspace_path(str(params.images_root), must_exist=True)
        if not images_root.is_dir():
            _fail(
                "vasp_neb_prepare",
                message=f"images_root is not a directory: {workspace_relpath(images_root)}",
                data={
                    "images_root_rel": workspace_relpath(images_root),
                    "output_root_rel": workspace_relpath(output_root),
                },
                error_code="invalid_images_root",
            )
        classified = _classify_single_image_tree(images_root)
        if classified is None:
            _fail(
                "vasp_neb_prepare",
                message="images_root must be one image tree with contiguous numbering from 00 using either flat image files or legacy image directories.",
                data={
                    "images_root_rel": workspace_relpath(images_root),
                    "output_root_rel": workspace_relpath(output_root),
                },
                error_code="invalid_image_tree",
            )
        image_sources, _layout = classified
        if len(image_sources) < 2:
            _fail(
                "vasp_neb_prepare",
                message="images_root must contain at least endpoint images such as 00/01 or 00.vasp/01.vasp.",
                data={
                    "images_root_rel": workspace_relpath(images_root),
                    "output_root_rel": workspace_relpath(output_root),
                },
                error_code="invalid_image_tree",
            )

        image_dirs_rel: list[str] = []
        images_atoms: list[Any] = []
        image_labels: list[str] = []
        representative_atoms = None
        warnings: list[str] = []
        for idx, src in image_sources:
            dst_dir = output_root / f"{idx:02d}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                struct_path = _find_image_structure_file(src) if src.is_dir() else src
            except Exception as exc:
                _fail(
                    "vasp_neb_prepare",
                    message=f"Failed to locate image structure under {workspace_relpath(src)}: {exc}",
                    data={
                        "images_root_rel": workspace_relpath(images_root),
                        "output_root_rel": workspace_relpath(output_root),
                    },
                    error_code="invalid_image_tree",
                )
            atoms = _read_atoms(struct_path)
            if representative_atoms is None:
                representative_atoms = atoms
            else:
                error = _validate_structures(representative_atoms, atoms)
                if error:
                    _fail(
                        "vasp_neb_prepare",
                        message=f"Image {idx:02d} is inconsistent with image 00: {error}",
                        data={
                            "images_root_rel": workspace_relpath(images_root),
                            "output_root_rel": workspace_relpath(output_root),
                        },
                        error_code="invalid_image_tree",
                    )
            images_atoms.append(atoms)
            image_labels.append(f"{idx:02d}")
            out_path = dst_dir / params.output_filename
            ase_write(
                str(out_path),
                atoms,
                format="vasp",
                direct=(params.interp_mode == "direct"),
                vasp5=True,
            )
            image_dirs_rel.append(workspace_relpath(dst_dir))

        geometry_check, geometry_warnings = _check_neb_image_distances(
            images_atoms,
            threshold_angstrom=params.min_pair_distance_warning_angstrom,
            image_labels=image_labels,
        )
        warnings.extend(geometry_warnings)
        warnings.extend(self._copy_required_image_tree_outcars(images_root=images_root, output_root=output_root, total_images=len(image_dirs_rel)))

        if representative_atoms is None:  # pragma: no cover
            raise AssertionError("representative_atoms should have been set")
        return NebImageSet(
            representative_atoms=representative_atoms,
            image_dirs_rel=image_dirs_rel,
            total_images=len(image_dirs_rel),
            intermediate_images=max(0, len(image_dirs_rel) - 2),
            warnings=warnings,
            geometry_check=geometry_check,
        )

    @staticmethod
    def _write_images(
        *,
        output_root: Path,
        images: list[Any],
        output_filename: str,
        interp_mode: str,
    ) -> list[str]:
        image_dirs_rel: list[str] = []
        for idx, atoms in enumerate(images):
            img_dir = output_root / f"{idx:02d}"
            img_dir.mkdir(parents=True, exist_ok=True)
            out_path = img_dir / output_filename
            ase_write(
                str(out_path),
                atoms,
                format="vasp",
                direct=(interp_mode == "direct"),
                vasp5=True,
            )
            image_dirs_rel.append(workspace_relpath(img_dir))
        return image_dirs_rel

    @staticmethod
    def _copy_endpoint_support_files(
        *,
        init_path: Path,
        final_path: Path,
        output_root: Path,
        total_images: int,
    ) -> list[str]:
        warnings: list[str] = []
        first_dir = output_root / "00"
        last_dir = output_root / f"{max(0, total_images - 1):02d}"
        initial_outcar = init_path.parent / "OUTCAR"
        final_outcar = final_path.parent / "OUTCAR"
        if not _copy_if_exists(initial_outcar, first_dir / "OUTCAR"):
            warnings.append(
                "initial endpoint OUTCAR not found. "
                f"Copy the original relax OUTCAR into {workspace_relpath(first_dir / 'OUTCAR')} "
                f"before NEB analysis. Missing source path: {workspace_relpath(initial_outcar)}"
            )
        if not _copy_if_exists(final_outcar, last_dir / "OUTCAR"):
            warnings.append(
                "final endpoint OUTCAR not found. "
                f"Copy the original relax OUTCAR into {workspace_relpath(last_dir / 'OUTCAR')} "
                f"before NEB analysis. Missing source path: {workspace_relpath(final_outcar)}"
            )
        return warnings

    @staticmethod
    def _copy_required_image_tree_outcars(
        *,
        images_root: Path,
        output_root: Path,
        total_images: int,
    ) -> list[str]:
        if total_images < 2:
            return []
        first_dir = output_root / "00"
        last_dir = output_root / f"{max(0, total_images - 1):02d}"
        return [
            "initial endpoint OUTCAR not provided for image-tree input. "
            f"Copy the original relax OUTCAR into {workspace_relpath(first_dir / 'OUTCAR')} before NEB analysis.",
            "final endpoint OUTCAR not provided for image-tree input. "
            f"Copy the original relax OUTCAR into {workspace_relpath(last_dir / 'OUTCAR')} before NEB analysis.",
        ]

    def _write_neb_support_files(
        self,
        *,
        params: VaspNebPrepareInput,
        output_root: Path,
        structure: Any,
        intermediate_images: int,
    ) -> dict[str, Any]:
        self._vasp_writer.write_vasp_inputs(
            structure=structure,
            output_dir=output_root,
            preset="static",
            regime=params.regime,  # type: ignore[arg-type]
            k_product=params.k_product,
            use_d3=params.use_d3,
            use_dft_plus_u=params.use_dft_plus_u,
            enable_dipole=params.enable_dipole,
            user_incar_patch={},
            patch_policy="safe",
        )

        incar_path = output_root / "INCAR"
        order, template_vals = _parse_incar(incar_path)
        canonical_overrides = self._canonical_neb_overrides(
            params=params,
            intermediate_images=intermediate_images,
        )
        final_vals: Dict[str, Any] = dict(template_vals)
        final_vals.update(canonical_overrides)

        protected_keys = set(_NEB_PROTECTED_KEYS_BASE)
        if params.spring is not None:
            protected_keys.add("SPRING")

        self._apply_user_patch(
            final_vals=final_vals,
            user_patch=params.user_incar_patch,
            protected_keys=protected_keys,
            patch_policy=params.patch_policy,
        )

        template_str = {k: _format_incar_value(v) for k, v in template_vals.items()}
        final_str = {k: _format_incar_value(v) for k, v in final_vals.items()}
        changed_keys = {
            key
            for key in set(template_str) | set(final_str)
            if template_str.get(key) != final_str.get(key)
        }
        diff = {
            key: {"old": template_str.get(key), "new": final_str.get(key)}
            for key in sorted(changed_keys)
        }
        diff_path = output_root / "neb_incar_patch.json"
        diff_path.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")

        lines: List[str] = []
        for key in order:
            if key in final_str:
                lines.append(f"{key} = {final_str[key]}")
        for key in sorted(set(final_str) - set(order)):
            lines.append(f"{key} = {final_str[key]}")
        incar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "output_incar_path": workspace_relpath(incar_path),
            "diff_json_rel": workspace_relpath(diff_path),
            "applied_overrides": {k: final_str[k] for k in sorted(changed_keys) if k in final_str},
            "protected_incar_keys": sorted(protected_keys),
        }

    @staticmethod
    def _canonical_neb_overrides(
        *,
        params: VaspNebPrepareInput,
        intermediate_images: int,
    ) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {
            "IBRION": 3,
            "POTIM": params.potim,
            "ICHAIN": 0,
            "IMAGES": intermediate_images,
            "IOPT": params.iopt,
            "ISYM": 0,
            "LCLIMB": params.climb,
            "LWAVE": False,
            "LCHARG": False,
        }
        if params.ediff is not None:
            overrides["EDIFF"] = params.ediff
        if params.ediffg is not None:
            overrides["EDIFFG"] = params.ediffg
        if params.spring is not None:
            overrides["SPRING"] = params.spring
        return overrides

    @staticmethod
    def _apply_user_patch(
        *,
        final_vals: Dict[str, Any],
        user_patch: Dict[str, Any],
        protected_keys: set[str],
        patch_policy: str,
    ) -> None:
        for raw_key, raw_value in (user_patch or {}).items():
            key = str(raw_key).strip().upper()
            if not key:
                raise ValueError("INCAR key must be a non-empty string.")
            if patch_policy == "safe" and key in protected_keys:
                current_value = final_vals.get(key)
                if raw_value != current_value:
                    raise ValueError(
                        f"user_incar_patch attempts to override protected NEB INCAR key {key} under patch_policy='safe'. "
                        "Use patch_policy='force' if you really need to replace NEB-critical defaults."
                    )
            if raw_value is None:
                final_vals.pop(key, None)
            else:
                final_vals[key] = raw_value


def vasp_neb_prepare(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[neb/prepare] Prepare a canonical VASP NEB job tree from endpoints or an existing image tree."""
    params = VaspNebPrepareInput(**payload)
    writer = VaspNebWriter()
    try:
        return writer.prepare(params)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        output_root = resolve_workspace_path(params.output_root)
        _fail(
            "vasp_neb_prepare",
            message=f"vasp_neb_prepare failed: {exc}",
            data={"output_root_rel": workspace_relpath(output_root)},
            error_code="neb_prepare_failed",
        )
        raise AssertionError("unreachable")


__all__ = [
    "EstimateNebImageCountInput",
    "MakeNebGeometryInput",
    "RemapNebEndpointAtomsInput",
    "VaspNebPrepareInput",
    "estimate_neb_image_count",
    "make_neb_geometry",
    "remap_neb_endpoint_atoms",
    "vasp_neb_prepare",
]
