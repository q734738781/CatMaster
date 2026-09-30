from __future__ import annotations

from typing import Any

from .serialization import derived_summary, snapshot_from_structure, snapshot_to_structure


def symmetry_site_groups(structure: Any, *, symprec: float, angle_tolerance: float) -> list[list[int]]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    symmetrized = SpacegroupAnalyzer(
        structure,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    ).get_symmetrized_structure()
    return [[int(index) for index in indices] for indices in symmetrized.equivalent_indices]


def apply_site_defect(
    structure: Any,
    *,
    kind: str,
    site_index: int,
    new_species: str = "",
) -> Any:
    if site_index < 0 or site_index >= len(structure):
        raise ValueError(f"site_index {site_index} is outside the structure.")
    candidate = structure.copy()
    if kind == "vacancy":
        candidate.remove_sites([site_index])
    elif kind == "substitution":
        if not str(new_species).strip():
            raise ValueError("new_species is required for a substitution.")
        candidate.replace(site_index, str(new_species).strip())
    else:
        raise ValueError(f"Unsupported site defect kind: {kind}")
    return candidate


def insert_interstitial(
    structure: Any,
    *,
    species: str,
    coordinates: list[float],
    coordinate_type: str,
) -> Any:
    if len(coordinates) != 3:
        raise ValueError("Interstitial coordinates must contain three values.")
    candidate = structure.copy()
    candidate.append(
        str(species).strip(),
        [float(value) for value in coordinates],
        coords_are_cartesian=coordinate_type == "cartesian",
    )
    return candidate


def generate_defect_candidates(
    structure: Any,
    *,
    kind: str,
    new_species: str = "",
    site_index: int = -1,
    coordinates: list[float] | None = None,
    coordinate_type: str = "fractional",
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> list[dict[str, Any]]:
    if kind == "interstitial":
        candidate = insert_interstitial(
            structure,
            species=new_species,
            coordinates=coordinates or [],
            coordinate_type=coordinate_type,
        )
        snapshot = snapshot_from_structure(candidate, fmt="poscar")
        return [
            {
                "candidate_index": 0,
                "label": f"{new_species} interstitial",
                "site_index": len(candidate) - 1,
                "equivalent_indices": [],
                "snapshot": snapshot.model_dump(mode="json"),
                "summary": derived_summary(snapshot),
            }
        ]

    groups = symmetry_site_groups(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if site_index >= 0:
        if site_index >= len(structure):
            raise ValueError(f"site_index {site_index} is outside the structure.")
        groups = [next((group for group in groups if site_index in group), [site_index])]
        representatives = [site_index]
    else:
        representatives = [group[0] for group in groups]

    candidates: list[dict[str, Any]] = []
    for candidate_index, (target, group) in enumerate(zip(representatives, groups)):
        original_species = structure[target].species_string
        candidate = apply_site_defect(
            structure,
            kind=kind,
            site_index=target,
            new_species=new_species,
        )
        if kind == "vacancy":
            label = f"Remove {original_species} at site {target}"
        elif kind == "substitution":
            label = f"{original_species} → {new_species} at site {target}"
        else:
            raise ValueError(f"Unsupported defect kind: {kind}")
        snapshot = snapshot_from_structure(candidate, fmt="poscar")
        candidates.append(
            {
                "candidate_index": candidate_index,
                "label": label,
                "site_index": int(target),
                "equivalent_indices": [int(index) for index in group],
                "snapshot": snapshot.model_dump(mode="json"),
                "summary": derived_summary(snapshot),
            }
        )
    return candidates


def defect_candidates(request: Any) -> dict[str, Any]:
    structure = snapshot_to_structure(request.input)
    params = request.params
    candidates = generate_defect_candidates(
        structure,
        kind=params.kind,
        new_species=params.new_species.strip(),
        site_index=params.site_index,
        coordinates=params.coordinates,
        coordinate_type=params.coordinate_type,
        symprec=params.symprec,
        angle_tolerance=params.angle_tolerance,
    )
    return {
        "kind": "candidates",
        "candidate_type": params.kind,
        "candidates": candidates,
        "warnings": [],
        "change": {
            "candidate_count": len(candidates),
            "symprec": float(params.symprec),
            "angle_tolerance": float(params.angle_tolerance),
        },
    }


__all__ = [
    "apply_site_defect",
    "defect_candidates",
    "generate_defect_candidates",
    "insert_interstitial",
    "symmetry_site_groups",
]
