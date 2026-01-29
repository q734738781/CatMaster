"""
Slab generation and editing helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from catmaster.tools.base import resolve_workspace_path, workspace_relpath, create_tool_output


class SlabBuildInput(BaseModel):
    """Build slabs for all terminations of a Miller index from bulk structure(s).

    Provide exactly one of bulk_structure or bulk_dir. When bulk_dir is used, output_root is required.
    Batch outputs are written under output_root/<slab_id>/..., where slab_id encodes the relative input path
    (without suffix) using '__'.
    """

    bulk_structure: Optional[str] = Field(
        None,
        description="Bulk structure file (POSCAR/CIF/etc.), workspace-relative.",
    )
    bulk_dir: Optional[str] = Field(
        None,
        description="Directory containing bulk structure files for batch slab building.",
    )
    miller_index: List[int] = Field(..., min_length=3, max_length=3, description="Miller index [h,k,l].")
    output_root: Optional[str] = Field(
        None,
        description=(
            "Directory to write the slab structures. Required for bulk_dir; defaults to 'slabs' for single file. "
            "For bulk_dir, outputs are written under output_root/<slab_id>/..., where slab_id encodes the relative "
            "input path (without suffix) using '__'."
        ),
    )
    slab_thickness: float = Field(12.0, ge=0.0, description="Target slab thickness (Å).")
    vacuum_thickness: float = Field(15.0, ge=0.0, description="Vacuum thickness (Å).")
    supercell: List[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Supercell replication [a,b,c] for slab structure after generation.")
    get_symmetry_slab: bool = Field(False, description="If true, returned slabs will ensure top and bottom surfaces are identical. Use it for surface energy calculation.")
    orthogonal: bool = Field(False, description="If true, convert each slab to an orthogonal c-oriented cell. Useful for grain boundary generation.")
    lll_reduce: bool = Field(
        False,
        description="Apply LLL reduction during slab generation. Not recommended to use unless necessary.",
    )


class FixAtomsByLayersInput(BaseModel):
    """Fix (freeze) the bottom N atomic layers of slab structure(s).

    Provide exactly one of structure_ref or structure_dir. When structure_dir is used, output_dir is required.
    Batch outputs are written to output_dir/<slab_id>.vasp and a summary JSON is written to
    output_dir/batch_fix_atoms_by_layers.json.
    """

    structure_ref: Optional[str] = Field(None, description="Slab structure file to modify (POSCAR/CIF).")
    structure_dir: Optional[str] = Field(None, description="Directory containing slab structure files to modify.")
    output_path: Optional[str] = Field(None, description="Output structure path (workspace-relative) for single file.")
    output_dir: Optional[str] = Field(
        None,
        description=(
            "Output directory for batch mode. Outputs are written as <output_dir>/<slab_id>.vasp where slab_id "
            "encodes the relative input path (without suffix) using '__'."
        ),
    )
    freeze_layers: int = Field(
        ...,
        ge=0,
        description="Number of bottom layers to freeze. Use layer_tol to group layers.",
    )
    centralize: bool = Field(False, description="Recentre slab along c before applying constraints.")
    layer_tol: float = Field(
        0.2,
        gt=0.0,
        description="Layer grouping tolerance in Å. 0.2 is suitable for most cases except for highly tilted slabs.",
    )


class ZRange(BaseModel):
    z_min: float = Field(..., description="Lower bound (Å) in Cartesian coordinates.")
    z_max: float = Field(..., description="Upper bound (Å) in Cartesian coordinates.")


class FixAtomsByHeightInput(BaseModel):
    """Fix (freeze) atoms within specified z ranges of slab structure(s).

    Provide exactly one of structure_ref or structure_dir. When structure_dir is used, output_dir is required.
    Batch outputs are written to output_dir/<slab_id>.vasp and a summary JSON is written to
    output_dir/batch_fix_atoms_by_height.json.
    """

    structure_ref: Optional[str] = Field(None, description="Slab structure file to modify (POSCAR/CIF).")
    structure_dir: Optional[str] = Field(None, description="Directory containing slab structure files to modify.")
    output_path: Optional[str] = Field(None, description="Output structure path (workspace-relative) for single file.")
    output_dir: Optional[str] = Field(
        None,
        description=(
            "Output directory for batch mode. Outputs are written as <output_dir>/<slab_id>.vasp where slab_id "
            "encodes the relative input path (without suffix) using '__'."
        ),
    )
    z_ranges: List[ZRange] = Field(
        ...,
        min_length=1,
        description="Ranges in Å in Cartesian coordinates; atoms in these z ranges are frozen.",
    )
    centralize: bool = Field(False, description="Recentre slab along c after applying constraints.")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in {"POSCAR", "CONTCAR"}:
            files.append(path)
            continue
        if path.suffix.lower() in {".vasp", ".poscar", ".cif"}:
            files.append(path)
    return sorted(files, key=lambda p: str(p))


def _slab_id_from(rel_path: Path) -> str:
    return "__".join(rel_path.with_suffix("").parts)


def _build_slab_single(
    bulk_path: Path,
    output_root: Path,
    *,
    miller: tuple[int, int, int],
    slab_thickness: float,
    vacuum_thickness: float,
    supercell: tuple[int, int, int],
    get_symmetry: bool,
    orthogonal: bool,
    lll_reduce: bool,
) -> Dict[str, object]:
    structure = Structure.from_file(bulk_path)
    structure = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=2.0).get_refined_structure()
    gen = SlabGenerator(
        initial_structure=structure,
        miller_index=miller,
        min_slab_size=slab_thickness,
        min_vacuum_size=vacuum_thickness,
        center_slab=True,
        lll_reduce=lll_reduce,
    )
    slabs = gen.get_slabs(symmetrize=get_symmetry)
    if orthogonal:
        slabs = [s.get_orthogonal_c_slab() for s in slabs]
    if not slabs:
        raise ValueError("SlabGenerator returned no slabs")

    _ensure_dir(output_root)
    base = bulk_path.stem
    emitted = []
    for term_index, slab in enumerate(slabs):
        slab_copy = slab.copy()
        if supercell != (1, 1, 1):
            slab_copy.make_supercell(np.diag(supercell))

        fname = f"{base}_h{miller[0]}{miller[1]}{miller[2]}_t{term_index}.vasp"
        out_path = output_root / fname
        slab_copy.to(fmt="poscar", filename=out_path)

        a, b, _ = slab_copy.lattice.matrix
        surface_area = float(np.linalg.norm(np.cross(a, b)))

        emitted.append(
            {
                "termination_index": term_index,
                "slab_structure_rel": workspace_relpath(out_path),
                "surface_area": surface_area,
                "natoms": len(slab_copy),
            }
        )

    return {
        "total_terminations": len(emitted),
        "terminations": emitted,
    }


def build_slab(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Build slabs for all terminations of a given Miller index. Each termination is written
    as a separate POSCAR under output_root. Supercell expansion is applied to every termination.
    """
    params = SlabBuildInput(**payload)
    if (params.bulk_structure is None) == (params.bulk_dir is None):
        return create_tool_output(
            "build_slab",
            success=False,
            error="Provide exactly one of bulk_structure or bulk_dir.",
        )
    miller = tuple(int(x) for x in params.miller_index)
    slab_thickness = float(params.slab_thickness)
    vacuum_thickness = float(params.vacuum_thickness)
    supercell = tuple(int(x) for x in params.supercell)
    get_symmetry = bool(params.get_symmetry_slab)
    orthogonal = bool(params.orthogonal)
    lll_reduce = bool(params.lll_reduce)

    if len(miller) != 3:
        return create_tool_output("build_slab", success=False, error="miller_index must have 3 integers")

    if params.bulk_dir is not None:
        if params.output_root is None:
            return create_tool_output(
                "build_slab",
                success=False,
                error="output_root is required when bulk_dir is provided.",
            )
        bulk_root = resolve_workspace_path(params.bulk_dir, must_exist=True)
        if not bulk_root.is_dir():
            return create_tool_output(
                "build_slab",
                success=False,
                error=f"bulk_dir is not a directory: {bulk_root}",
            )
        structures = _collect_structure_files(bulk_root)
        if not structures:
            return create_tool_output(
                "build_slab",
                success=False,
                error="No bulk structure files found in bulk_dir.",
            )
        output_root = resolve_workspace_path(params.output_root)
        _ensure_dir(output_root)

        results = []
        errors = []
        for bulk_path in structures:
            rel_path = bulk_path.relative_to(bulk_root)
            slab_id = _slab_id_from(rel_path)
            out_dir = output_root / slab_id
            try:
                result = _build_slab_single(
                    bulk_path,
                    out_dir,
                    miller=miller,
                    slab_thickness=slab_thickness,
                    vacuum_thickness=vacuum_thickness,
                    supercell=supercell,
                    get_symmetry=get_symmetry,
                    orthogonal=orthogonal,
                    lll_reduce=lll_reduce,
                )
                results.append(
                    {
                        "bulk_rel": str(rel_path),
                        "slab_id": slab_id,
                        "output_dir_rel": workspace_relpath(out_dir),
                        **result,
                    }
                )
            except Exception as exc:
                errors.append({"bulk_rel": str(rel_path), "error": str(exc)})

        data = {
            "bulk_dir_rel": workspace_relpath(bulk_root),
            "output_root_rel": workspace_relpath(output_root),
            "miller_index": list(miller),
            "supercell": list(supercell),
            "slab_thickness": slab_thickness,
            "vacuum_thickness": vacuum_thickness,
            "get_symmetry_slab": get_symmetry,
            "orthogonal": orthogonal,
            "lll_reduce": lll_reduce,
            "structures_found": len(structures),
            "structures_built": len(results),
            "results": results,
            "errors": errors,
        }
        return create_tool_output("build_slab", success=True, data=data)

    bulk_path = resolve_workspace_path(params.bulk_structure, must_exist=True)
    output_root = resolve_workspace_path(params.output_root or "slabs")
    try:
        result = _build_slab_single(
            bulk_path,
            output_root,
            miller=miller,
            slab_thickness=slab_thickness,
            vacuum_thickness=vacuum_thickness,
            supercell=supercell,
            get_symmetry=get_symmetry,
            orthogonal=orthogonal,
            lll_reduce=lll_reduce,
        )
    except Exception as exc:
        return create_tool_output("build_slab", success=False, error=str(exc))

    data = {
        "miller_index": list(miller),
        "total_terminations": result["total_terminations"],
        "supercell": list(supercell),
        "slab_thickness": slab_thickness,
        "vacuum_thickness": vacuum_thickness,
        "get_symmetry_slab": get_symmetry,
        "orthogonal": orthogonal,
        "lll_reduce": lll_reduce,
        "terminations": result["terminations"],
    }
    return create_tool_output("build_slab", success=True, data=data)

def _bin_z_layers(z: np.ndarray, layer_tol: float) -> np.ndarray:
    """
    Assign an integer layer bin id to each z coordinate using tolerance-based binning.

    Design goals:
      - Translation-robust: binning is relative to zmin, not absolute 0.
      - Deterministic: return integer bin ids (avoid float comparisons in sets).
      - Tolerance semantics: points within ~layer_tol/2 tend to fall into same bin.

    Parameters
    ----------
    z : (N,) array-like
        Cartesian z coordinates (Å).
    layer_tol : float
        Layer tolerance in Å (must be > 0).

    Returns
    -------
    bin_id : (N,) np.ndarray of int
        Integer layer id for each atom. Bottom-most atoms are near bin 0.
    """
    z = np.asarray(z, dtype=float)
    tol = float(layer_tol)
    if tol <= 0.0:
        raise ValueError("layer_tol must be > 0")

    z0 = float(z.min())
    # “round to nearest bin” but expressed in an integer-stable way
    # Equivalent to round((z - z0)/tol) with half-up behavior.
    bin_id = np.floor((z - z0) / tol + 0.5).astype(int)
    return bin_id

def _center_of_mass(slab: Structure) -> np.ndarray:
    """
    Calculate the center of mass of a (periodic) Structure in Cartesian coordinates (Å).

    Notes
    -----
    - For disordered sites, masses are weighted by occupancy.
    - For sites with species lacking atomic_mass (e.g., DummySpecie), falls back to geometric center.
    - For periodic structures, COM depends on the chosen image (i.e., current coordinates); for slab
      centralization along z this is typically acceptable.
    """
    coords = np.asarray(slab.cart_coords, dtype=float)

    masses = []
    for site in slab.sites:
        m = 0.0
        ok = False
        # site.species is a Composition-like mapping: {Species/Element: occupancy}
        for sp, occu in site.species.items():
            try:
                am = float(sp.atomic_mass)  # Element/Species usually has atomic_mass
            except Exception:
                am = float("nan")
            if np.isfinite(am) and am > 0:
                m += float(occu) * am
                ok = True
        masses.append(m if ok else float("nan"))

    masses = np.asarray(masses, dtype=float)

    # If any mass is NaN or total mass is non-positive, fall back to geometric center
    if (not np.all(np.isfinite(masses))) or (float(np.nansum(masses)) <= 0.0):
        return coords.mean(axis=0)

    total_mass = float(masses.sum())
    return (coords * masses[:, None]).sum(axis=0) / total_mass

def _fix_atoms_by_layers_single(
    structure_ref: Path,
    output_path: Path,
    *,
    freeze_layers: int,
    centralize: bool,
    layer_tol: float,
) -> Dict[str, object]:
    slab = Structure.from_file(structure_ref)
    if centralize:
        z_target = float(slab.lattice.matrix[2][2]) / 2
        z_current = _center_of_mass(slab)[2]
        slab.translate_sites(range(len(slab)), [0, 0, z_target - z_current], frac_coords=False)

    coords = slab.cart_coords[:, 2]
    bin_ids = _bin_z_layers(coords, layer_tol)
    unique_bins = sorted(set(bin_ids.tolist()))
    if freeze_layers < 0 or freeze_layers > len(unique_bins):
        raise ValueError("freeze_layers is more than layers counted")
    freeze_bins = set(unique_bins[:freeze_layers])
    relax_mask = [b not in freeze_bins for b in bin_ids]

    slab.add_site_property("selective_dynamics", [[m, m, m] for m in relax_mask])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slab.to(fmt="poscar", filename=output_path)

    return {
        "input_rel": workspace_relpath(structure_ref),
        "output_rel": workspace_relpath(output_path),
        "relaxed_atoms": int(np.sum(relax_mask)),
        "frozen_atoms": int(len(relax_mask) - np.sum(relax_mask)),
        "frozen_layers": int(freeze_layers),
        "total_layers": int(len(unique_bins)),
    }


def fix_atoms_by_layers(payload: Dict[str, object]) -> Dict[str, object]:
    params = FixAtomsByLayersInput(**payload)
    if (params.structure_ref is None) == (params.structure_dir is None):
        return create_tool_output(
            "fix_atoms_by_layers",
            success=False,
            error="Provide exactly one of structure_ref or structure_dir.",
        )

    freeze_layers = int(params.freeze_layers)
    centralize = bool(params.centralize)
    layer_tol = float(params.layer_tol)

    if params.structure_dir is not None:
        if params.output_dir is None:
            return create_tool_output(
                "fix_atoms_by_layers",
                success=False,
                error="output_dir is required when structure_dir is provided.",
            )
        structure_root = resolve_workspace_path(params.structure_dir, must_exist=True)
        if not structure_root.is_dir():
            return create_tool_output(
                "fix_atoms_by_layers",
                success=False,
                error=f"structure_dir is not a directory: {structure_root}",
            )
        output_root = resolve_workspace_path(params.output_dir)
        _ensure_dir(output_root)
        structures = _collect_structure_files(structure_root)
        if not structures:
            return create_tool_output(
                "fix_atoms_by_layers",
                success=False,
                error="No structure files found in structure_dir.",
            )

        results = []
        errors = []
        for structure_path in structures:
            rel_path = structure_path.relative_to(structure_root)
            slab_id = _slab_id_from(rel_path)
            output_path = output_root / f"{slab_id}.vasp"
            try:
                result = _fix_atoms_by_layers_single(
                    structure_path,
                    output_path,
                    freeze_layers=freeze_layers,
                    centralize=centralize,
                    layer_tol=layer_tol,
                )
                results.append(
                    {
                        "input_rel": str(rel_path),
                        "slab_id": slab_id,
                        "output_rel": workspace_relpath(output_path),
                        "relaxed_atoms": result["relaxed_atoms"],
                        "frozen_atoms": result["frozen_atoms"],
                    }
                )
            except Exception as exc:
                errors.append({"input_rel": str(rel_path), "error": str(exc)})

        batch_json = output_root / "batch_fix_atoms_by_layers.json"
        try:
            batch_json.write_text(
                json.dumps({"results": results, "errors": errors}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        data = {
            "structure_dir_rel": workspace_relpath(structure_root),
            "output_dir_rel": workspace_relpath(output_root),
            "freeze_layers": freeze_layers,
            "layer_tol": layer_tol,
            "centralize": centralize,
            "structures_found": len(structures),
            "structures_processed": len(results),
            "batch_json_rel": workspace_relpath(batch_json) if batch_json.exists() else None,
            "errors_count": len(errors),
        }
        return create_tool_output("fix_atoms_by_layers", success=True, data=data)

    if params.output_path is None:
        return create_tool_output(
            "fix_atoms_by_layers",
            success=False,
            error="output_path is required when structure_ref is provided.",
        )
    structure_ref = resolve_workspace_path(params.structure_ref, must_exist=True)
    output_path = resolve_workspace_path(params.output_path)
    try:
        data = _fix_atoms_by_layers_single(
            structure_ref,
            output_path,
            freeze_layers=freeze_layers,
            centralize=centralize,
            layer_tol=layer_tol,
        )
    except Exception as exc:
        return create_tool_output("fix_atoms_by_layers", success=False, error=str(exc))
    return create_tool_output("fix_atoms_by_layers", success=True, data=data)


def _fix_atoms_by_height_single(
    structure_ref: Path,
    output_path: Path,
    *,
    z_ranges: list[tuple[float, float]],
    centralize: bool,
) -> Dict[str, object]:
    slab = Structure.from_file(structure_ref)
    coords = slab.cart_coords[:, 2]
    freeze_mask = [any(zmin <= z <= zmax for zmin, zmax in z_ranges) for z in coords]
    relax_mask = [not f for f in freeze_mask]
    slab.add_site_property("selective_dynamics", [[m, m, m] for m in relax_mask])

    if centralize:
        z_target = float(slab.lattice.matrix[2][2]) / 2
        z_current = _center_of_mass(slab)[2]
        slab.translate_sites(range(len(slab)), [0, 0, z_target - z_current], frac_coords=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    slab.to(fmt="poscar", filename=output_path)

    return {
        "input_rel": workspace_relpath(structure_ref),
        "output_rel": workspace_relpath(output_path),
        "relaxed_atoms": int(np.sum(relax_mask)),
        "frozen_atoms": int(len(relax_mask) - np.sum(relax_mask)),
        "z_ranges": [[float(zmin), float(zmax)] for zmin, zmax in z_ranges],
    }


def fix_atoms_by_height(payload: Dict[str, object]) -> Dict[str, object]:
    params = FixAtomsByHeightInput(**payload)
    if (params.structure_ref is None) == (params.structure_dir is None):
        return create_tool_output(
            "fix_atoms_by_height",
            success=False,
            error="Provide exactly one of structure_ref or structure_dir.",
        )
    if not params.z_ranges:
        return create_tool_output(
            "fix_atoms_by_height",
            success=False,
            error="z_ranges must not be empty.",
        )
    z_ranges = [(float(item.z_min), float(item.z_max)) for item in params.z_ranges]
    centralize = bool(params.centralize)

    for zmin, zmax in z_ranges:
        if zmin >= zmax:
            return create_tool_output(
                "fix_atoms_by_height",
                success=False,
                error="Each z_range must satisfy z_min < z_max",
            )

    if params.structure_dir is not None:
        if params.output_dir is None:
            return create_tool_output(
                "fix_atoms_by_height",
                success=False,
                error="output_dir is required when structure_dir is provided.",
            )
        structure_root = resolve_workspace_path(params.structure_dir, must_exist=True)
        if not structure_root.is_dir():
            return create_tool_output(
                "fix_atoms_by_height",
                success=False,
                error=f"structure_dir is not a directory: {structure_root}",
            )
        output_root = resolve_workspace_path(params.output_dir)
        _ensure_dir(output_root)
        structures = _collect_structure_files(structure_root)
        if not structures:
            return create_tool_output(
                "fix_atoms_by_height",
                success=False,
                error="No structure files found in structure_dir.",
            )

        results = []
        errors = []
        for structure_path in structures:
            rel_path = structure_path.relative_to(structure_root)
            slab_id = _slab_id_from(rel_path)
            output_path = output_root / f"{slab_id}.vasp"
            try:
                result = _fix_atoms_by_height_single(
                    structure_path,
                    output_path,
                    z_ranges=z_ranges,
                    centralize=centralize,
                )
                results.append(
                    {
                        "input_rel": str(rel_path),
                        "slab_id": slab_id,
                        "output_rel": workspace_relpath(output_path),
                        "relaxed_atoms": result["relaxed_atoms"],
                        "frozen_atoms": result["frozen_atoms"],
                    }
                )
            except Exception as exc:
                errors.append({"input_rel": str(rel_path), "error": str(exc)})

        batch_json = output_root / "batch_fix_atoms_by_height.json"
        try:
            batch_json.write_text(
                json.dumps({"results": results, "errors": errors}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        data = {
            "structure_dir_rel": workspace_relpath(structure_root),
            "output_dir_rel": workspace_relpath(output_root),
            "z_ranges": [[float(zmin), float(zmax)] for zmin, zmax in z_ranges],
            "centralize": centralize,
            "structures_found": len(structures),
            "structures_processed": len(results),
            "batch_json_rel": workspace_relpath(batch_json) if batch_json.exists() else None,
            "errors_count": len(errors),
        }
        return create_tool_output("fix_atoms_by_height", success=True, data=data)

    if params.output_path is None:
        return create_tool_output(
            "fix_atoms_by_height",
            success=False,
            error="output_path is required when structure_ref is provided.",
        )
    structure_ref = resolve_workspace_path(params.structure_ref, must_exist=True)
    output_path = resolve_workspace_path(params.output_path)
    try:
        data = _fix_atoms_by_height_single(
            structure_ref,
            output_path,
            z_ranges=z_ranges,
            centralize=centralize,
        )
    except Exception as exc:
        return create_tool_output("fix_atoms_by_height", success=False, error=str(exc))
    return create_tool_output("fix_atoms_by_height", success=True, data=data)


__all__ = [
    "SlabBuildInput",
    "FixAtomsByLayersInput",
    "ZRange",
    "FixAtomsByHeightInput",
    "build_slab",
    "fix_atoms_by_layers",
    "fix_atoms_by_height",
]
