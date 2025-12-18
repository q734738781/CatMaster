"""
Lightweight slab generation and post-processing without external core deps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

from catmaster.tools.base import resolve_workspace_path, workspace_relpath


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cut_slabs(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Generate slabs for given Miller indices using pymatgen.SlabGenerator.
    """
    structure_file = resolve_workspace_path(str(payload["structure_file"]), must_exist=True)
    compound_name = (payload.get("compound_name") or structure_file.stem).strip() or structure_file.stem
    miller_list_raw: Iterable[Iterable[int]] = payload.get("miller_list") or []
    miller_list: List[Tuple[int, int, int]] = [tuple(int(x) for x in item) for item in miller_list_raw]
    min_slab_size = float(payload.get("min_slab_size", 12.0))
    min_vacuum_size = float(payload.get("min_vacuum_size", 15.0))
    relax_thickness = float(payload.get("relax_thickness", 5.0))
    output_root = resolve_workspace_path(str(payload["output_root"]))
    get_symmetry_slab = bool(payload.get("get_symmetry_slab", False))
    fix_bottom = bool(payload.get("fix_bottom", True))  # kept for compatibility; unused here

    if not miller_list:
        raise ValueError("miller_list must contain at least one facet.")

    structure = Structure.from_file(structure_file)

    emitted: List[Path] = []
    for hkl in miller_list:
        gen = SlabGenerator(
            initial_structure=structure,
            miller_index=hkl,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=True,
            in_unit_planes=True,
            reorient_lattice=True,
        )
        slabs = gen.get_slabs(symmetrize=get_symmetry_slab)
        facet_dir = output_root / compound_name / f"{hkl[0]}{hkl[1]}{hkl[2]}"
        _ensure_dir(facet_dir)
        for idx, slab in enumerate(slabs):
            fname = facet_dir / f"{compound_name}_{hkl[0]}{hkl[1]}{hkl[2]}_{idx}.vasp"
            slab.to(fmt="poscar", filename=fname)
            emitted.append(fname)

    return {
        "compound": compound_name,
        "facets": miller_list,
        "output_root_rel": workspace_relpath(output_root),
        "generated_rel": [workspace_relpath(p) for p in emitted],
    }


def fix_slab(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Simple post-process: optionally centralize slab and set selective dynamics to fix bottom region.
    """
    input_path = resolve_workspace_path(str(payload["input_path"]), must_exist=True)
    output_dir = resolve_workspace_path(str(payload["output_dir"]))
    relax_thickness = float(payload.get("relax_thickness", 5.0))
    fix_bottom = bool(payload.get("fix_bottom", True))
    centralize = bool(payload.get("centralize", False))

    _ensure_dir(output_dir)

    inputs: List[Path] = []
    if input_path.is_dir():
        inputs = sorted([p for p in input_path.glob("*.vasp")])
    else:
        inputs = [input_path]

    emitted: List[Path] = []
    for src in inputs:
        slab = Structure.from_file(src)
        if centralize:
            slab.translate_sites(range(len(slab)), [0, 0, -slab.center_of_mass[2]], frac_coords=False)

        if fix_bottom:
            coords = slab.cart_coords
            zmin = coords[:, 2].min()
            mobile_mask = [(z - zmin) >= relax_thickness for z in coords[:, 2]]
            slab.add_site_property("selective_dynamics", [[m, m, m] for m in mobile_mask])

        dest = output_dir / src.name
        slab.to(fmt="poscar", filename=dest)
        emitted.append(dest)

    return {
        "source_rel": workspace_relpath(input_path),
        "output_dir_rel": workspace_relpath(output_dir),
        "generated_rel": [workspace_relpath(p) for p in emitted],
    }


__all__ = ["cut_slabs", "fix_slab"]
