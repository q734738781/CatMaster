from __future__ import annotations

from typing import Any

import numpy as np

from .models import (
    MakeSupercellRequest,
    SetCellRequest,
    TransformRequest,
)
from .serialization import derived_summary, snapshot_from_structure, snapshot_to_structure

SOURCE_INDEX_PROPERTY = "_catmaster_source_index"


def _space_group(structure: Any, *, symprec: float, angle_tolerance: float) -> dict[str, Any]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
        return {
            "symbol": analyzer.get_space_group_symbol(),
            "number": int(analyzer.get_space_group_number()),
        }
    except Exception:
        return {"symbol": "Unresolved", "number": 0}


def make_supercell(request: MakeSupercellRequest) -> dict[str, Any]:
    structure = snapshot_to_structure(request.input)
    before_count = len(structure)
    work = structure.copy()
    if SOURCE_INDEX_PROPERTY in work.site_properties:
        work.remove_site_property(SOURCE_INDEX_PROPERTY)
    work.add_site_property(SOURCE_INDEX_PROPERTY, list(range(before_count)))
    work.make_supercell(request.params.matrix)
    mapping = [int(value) for value in work.site_properties.get(SOURCE_INDEX_PROPERTY, [])]
    if SOURCE_INDEX_PROPERTY in work.site_properties:
        work.remove_site_property(SOURCE_INDEX_PROPERTY)
    snapshot = snapshot_from_structure(work, fmt=request.input.format)
    return {
        "kind": "snapshot",
        "snapshot": snapshot.model_dump(mode="json"),
        "summary": derived_summary(snapshot),
        "warnings": [],
        "atom_mapping": mapping,
        "change": {
            "before_atoms": before_count,
            "after_atoms": len(work),
            "determinant": int(round(abs(np.linalg.det(np.asarray(request.params.matrix, dtype=float))))),
        },
    }


def set_cell(request: SetCellRequest) -> dict[str, Any]:
    from pymatgen.core import Lattice, Structure

    structure = snapshot_to_structure(request.input)
    new_lattice = Lattice(np.asarray(request.params.matrix, dtype=float))
    coords_are_cartesian = request.params.keep == "cartesian"
    coordinates = structure.cart_coords if coords_are_cartesian else structure.frac_coords
    rebuilt = Structure(
        new_lattice,
        [site.species for site in structure],
        coordinates,
        coords_are_cartesian=coords_are_cartesian,
        site_properties={key: list(values) for key, values in structure.site_properties.items()},
        labels=list(structure.labels),
    )
    snapshot = snapshot_from_structure(rebuilt, fmt=request.input.format)
    return {
        "kind": "snapshot",
        "snapshot": snapshot.model_dump(mode="json"),
        "summary": derived_summary(snapshot),
        "warnings": [
            "Changing the cell can break the original symmetry; the workbench will not repair it automatically."
        ],
        "atom_mapping": list(range(len(rebuilt))),
        "change": {
            "keep": request.params.keep,
            "before_volume": float(structure.volume),
            "after_volume": float(rebuilt.volume),
        },
    }


def transform_structure(request: TransformRequest) -> dict[str, Any]:
    if request.operation == "make_supercell":
        return make_supercell(request)
    if request.operation == "set_cell":
        return set_cell(request)
    if request.operation in {"primitive", "conventional", "standardize", "symmetrize"}:
        from .symmetry import symmetry_transform

        return symmetry_transform(request)
    if request.operation == "slab_candidates":
        from .surfaces import slab_candidates

        return slab_candidates(request)
    if request.operation == "defect_candidates":
        from .defects import defect_candidates

        return defect_candidates(request)
    if request.operation == "adsorption_candidates":
        from .adsorption import adsorption_candidates

        return adsorption_candidates(request)
    if request.operation == "molecule_conformers":
        from .molecules import molecule_conformers

        return molecule_conformers(request)
    if request.operation == "molecule_refresh":
        from .molecules import molecule_refresh

        return molecule_refresh(request)
    if request.operation == "molecule_from_viewer":
        from .molecules import molecule_from_viewer

        return molecule_from_viewer(request)
    raise ValueError(f"Unsupported structure operation: {request.operation}")


__all__ = ["make_supercell", "set_cell", "transform_structure", "_space_group"]
