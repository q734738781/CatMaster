from __future__ import annotations

from typing import Any

from .operations import _space_group
from .serialization import derived_summary, snapshot_from_structure, snapshot_to_structure


def _equivalent_groups(structure: Any, *, symprec: float, angle_tolerance: float) -> list[list[int]]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    symmetrized = analyzer.get_symmetrized_structure()
    return [[int(index) for index in group] for group in symmetrized.equivalent_indices]


def _mapping_by_species_and_distance(before: Any, after: Any) -> list[int]:
    """Return a useful selection mapping without pretending it is bijective."""
    mapping: list[int] = []
    before_cart = before.cart_coords
    for site in after:
        compatible = [
            index
            for index, original in enumerate(before)
            if original.species == site.species
        ]
        if not compatible:
            mapping.append(-1)
            continue
        distances = [
            float(min(before.lattice.get_all_distances([before[index].frac_coords], [before.lattice.get_fractional_coords(site.coords)])[0]))
            if before.lattice.volume > 0
            else float(((before_cart[index] - site.coords) ** 2).sum() ** 0.5)
            for index in compatible
        ]
        mapping.append(int(compatible[min(range(len(distances)), key=distances.__getitem__)]))
    return mapping


def symmetry_transform(request: Any) -> dict[str, Any]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = snapshot_to_structure(request.input)
    params = request.params
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=float(params.symprec),
        angle_tolerance=float(params.angle_tolerance),
    )
    before_group = _space_group(
        structure,
        symprec=params.symprec,
        angle_tolerance=params.angle_tolerance,
    )
    equivalent_groups = _equivalent_groups(
        structure,
        symprec=params.symprec,
        angle_tolerance=params.angle_tolerance,
    )

    if request.operation == "primitive":
        transformed = analyzer.get_primitive_standard_structure(
            international_monoclinic=True,
            keep_site_properties=True,
        )
    elif request.operation == "conventional":
        transformed = analyzer.get_conventional_standard_structure(
            international_monoclinic=True,
            keep_site_properties=True,
        )
    elif request.operation == "standardize":
        transformed = analyzer.get_refined_structure(keep_site_properties=True)
    else:
        # Refinement moves sites onto the detected symmetry while preserving the
        # existing conventional choice more closely than cell standardization.
        transformed = analyzer.get_refined_structure(keep_site_properties=True)

    after_group = _space_group(
        transformed,
        symprec=params.symprec,
        angle_tolerance=params.angle_tolerance,
    )
    snapshot = snapshot_from_structure(transformed, fmt=request.input.format)
    warnings: list[str] = []
    if len(transformed) != len(structure):
        warnings.append(
            f"The proposed {request.operation} representation changes the site count from "
            f"{len(structure)} to {len(transformed)}; inspect the preview before applying it."
        )
    return {
        "kind": "snapshot",
        "snapshot": snapshot.model_dump(mode="json"),
        "summary": derived_summary(snapshot),
        "warnings": warnings,
        "atom_mapping": _mapping_by_species_and_distance(structure, transformed),
        "change": {
            "operation": request.operation,
            "symprec": float(params.symprec),
            "angle_tolerance": float(params.angle_tolerance),
            "before_space_group": before_group,
            "after_space_group": after_group,
            "before_atoms": len(structure),
            "after_atoms": len(transformed),
            "equivalent_groups": equivalent_groups,
        },
    }


__all__ = ["symmetry_transform"]
