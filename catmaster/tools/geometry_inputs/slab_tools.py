"""
Slab generation and editing helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from catmaster.tools.base import resolve_workspace_path, workspace_relpath, create_tool_output


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_slab(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Build slabs for all terminations of a given Miller index. Each termination is written
    as a separate POSCAR under output_root. Supercell expansion is applied to every termination.
    """
    bulk_path = resolve_workspace_path(str(payload["bulk_structure"]), must_exist=True)
    miller = tuple(int(x) for x in payload.get("miller_index", []))
    slab_thickness = float(payload.get("slab_thickness", 12.0))
    vacuum_thickness = float(payload.get("vacuum_thickness", 15.0))
    supercell = tuple(int(x) for x in payload.get("supercell", (1, 1, 1)))
    get_symmetry = bool(payload.get("get_symmetry_slab", False))
    orthogonal = bool(payload.get("orthogonal", True))
    output_root = resolve_workspace_path(str(payload.get("output_root", "slabs")))

    if len(miller) != 3:
        return create_tool_output("build_slab", success=False, error="miller_index must have 3 integers")

    structure = Structure.from_file(bulk_path)
    structure = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=2.0).get_refined_structure()
    gen = SlabGenerator(
        initial_structure=structure,
        miller_index=miller,
        min_slab_size=slab_thickness,
        min_vacuum_size=vacuum_thickness,
        center_slab=True,
    )
    slabs = gen.get_slabs(symmetrize=get_symmetry)
    if orthogonal:
        slabs = [s.get_orthogonal_c_slab() for s in slabs]
    if not slabs:
        return create_tool_output("build_slab", success=False, error="SlabGenerator returned no slabs")

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

    data = {
        "miller_index": list(miller),
        "total_terminations": len(emitted),
        "supercell": list(supercell),
        "slab_thickness": slab_thickness,
        "vacuum_thickness": vacuum_thickness,
        "get_symmetry_slab": get_symmetry,
        "orthogonal": orthogonal,
        "terminations": emitted
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

def set_selective_dynamics(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Add selective dynamics flags: freeze bottom layers or everything below a relax_thickness.
    """
    structure_ref = resolve_workspace_path(str(payload["structure_ref"]), must_exist=True)
    output_path = resolve_workspace_path(str(payload["output_path"]))
    freeze_layers = payload.get("freeze_layers")
    relax_thickness = payload.get("relax_thickness")
    centralize = bool(payload.get("centralize", False))
    layer_tol = float(payload.get("layer_tol", 0.2))

    if (freeze_layers is None) == (relax_thickness is None):
        return create_tool_output("set_selective_dynamics", success=False, error="Provide exactly one of freeze_layers or relax_thickness.")

    slab = Structure.from_file(structure_ref)
    if centralize:
        # Translate the slab to z/2
        z_target = float(slab.lattice.matrix[2][2]) / 2
        z_current = _center_of_mass(slab)[2]
        slab.translate_sites(range(len(slab)), [0, 0, z_target - z_current], frac_coords=False)

    coords = slab.cart_coords[:, 2]
    zmax = float(coords.max())

    if relax_thickness is not None:
        relax_thickness = float(relax_thickness)
        threshold = zmax - relax_thickness
        relax_mask = [z >= threshold for z in coords]
        mode = "relax_thickness"
    else:
        bin_ids = _bin_z_layers(coords, layer_tol)
        unique_bins = sorted(set(bin_ids.tolist()))
        n_freeze = int(freeze_layers)
        if n_freeze < 0 or n_freeze > len(unique_bins):
            return create_tool_output("set_selective_dynamics", success=False, error="freeze_layers is more than layers counted")
        freeze_bins = set(unique_bins[:n_freeze])
        relax_mask = [b not in freeze_bins for b in bin_ids]
        mode = "freeze_layers"

    slab.add_site_property("selective_dynamics", [[m, m, m] for m in relax_mask])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slab.to(fmt="poscar", filename=output_path)

    data = {
        "input_rel": workspace_relpath(structure_ref),
        "output_rel": workspace_relpath(output_path),
        "relaxed_atoms": int(np.sum(relax_mask)),
        "frozen_atoms": int(len(relax_mask) - np.sum(relax_mask)),
        "mode": mode,
    }
    return create_tool_output("set_selective_dynamics", success=True, data=data)


__all__ = ["build_slab", "set_selective_dynamics"]
