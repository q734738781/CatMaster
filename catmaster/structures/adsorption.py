from __future__ import annotations

from typing import Any

import numpy as np

from .serialization import (
    derived_summary,
    snapshot_from_structure,
    snapshot_to_molecule,
    snapshot_to_structure,
)


def _rdkit_to_pymatgen(molecule: Any):
    from pymatgen.core import Molecule

    if molecule.GetNumConformers() == 0:
        raise ValueError("Adsorbate needs a 3D conformer before it can be placed.")
    conformer = molecule.GetConformer()
    species = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    coordinates = [
        [
            float(conformer.GetAtomPosition(index).x),
            float(conformer.GetAtomPosition(index).y),
            float(conformer.GetAtomPosition(index).z),
        ]
        for index in range(molecule.GetNumAtoms())
    ]
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    return Molecule(species, coordinates, charge=charge)


def enumerate_adsorption_sites(
    slab: Any,
    *,
    distance: float,
    site_kinds: list[str],
) -> list[dict[str, Any]]:
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder

    sites = AdsorbateSiteFinder(slab).find_adsorption_sites(distance=float(distance))
    rows: list[dict[str, Any]] = []
    for kind in site_kinds:
        for site_number, coordinate in enumerate(sites.get(kind, [])):
            rows.append(
                {
                    "label": f"{kind}_{site_number}",
                    "kind": kind,
                    "cart_coords": [float(item) for item in coordinate],
                }
            )
    return rows


def place_adsorbate_at_site(
    slab: Any,
    adsorbate_molecule: Any,
    site_cartesian: list[float] | tuple[float, float, float] | np.ndarray,
    *,
    reorient: bool = False,
) -> Any:
    """Place one pymatgen Molecule while preserving slab mobility semantics."""
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder

    structure = AdsorbateSiteFinder(slab).add_adsorbate(
        adsorbate_molecule,
        np.asarray(site_cartesian, dtype=float),
        translate=True,
        reorient=bool(reorient),
    )
    slab_constraints = slab.site_properties.get("selective_dynamics", [])
    if slab_constraints and len(slab_constraints) == len(slab):
        if "selective_dynamics" in structure.site_properties:
            structure.remove_site_property("selective_dynamics")
        structure.add_site_property(
            "selective_dynamics",
            [list(map(bool, flags)) for flags in slab_constraints]
            + [[True, True, True] for _ in range(len(adsorbate_molecule))],
        )
    return structure


def generate_adsorption_candidates(
    slab: Any,
    adsorbate: Any,
    *,
    distance: float,
    site_kinds: list[str],
    reorient: bool,
    orientation_euler_deg: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[dict[str, Any]]:
    from pymatgen.core import Molecule

    sites = enumerate_adsorption_sites(
        slab,
        distance=distance,
        site_kinds=site_kinds,
    )
    adsorbate_molecule = _rdkit_to_pymatgen(adsorbate)
    angles = np.radians(np.asarray(orientation_euler_deg, dtype=float))
    if np.any(np.abs(angles) > 1e-12):
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        rotation = (
            np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        )
        coordinates = np.asarray(adsorbate_molecule.cart_coords, dtype=float)
        coordinates = (coordinates - coordinates.mean(axis=0)) @ rotation.T
        adsorbate_molecule = Molecule(
            [site.species for site in adsorbate_molecule],
            coordinates,
            charge=adsorbate_molecule.charge,
            spin_multiplicity=adsorbate_molecule.spin_multiplicity,
        )
    candidates: list[dict[str, Any]] = []
    for site_row in sites:
        kind = site_row["kind"]
        site_number = int(str(site_row["label"]).rsplit("_", 1)[-1])
        coordinate = site_row["cart_coords"]
        structure = place_adsorbate_at_site(
            slab,
            adsorbate_molecule,
            coordinate,
            reorient=bool(reorient),
        )
        snapshot = snapshot_from_structure(structure, fmt="poscar")
        candidates.append(
            {
                "candidate_index": len(candidates),
                "label": f"{kind.title()} site {site_number + 1}",
                "site_kind": kind,
                "site_cartesian": [float(item) for item in coordinate],
                "orientation_euler_deg": [float(item) for item in orientation_euler_deg],
                "adsorbate_indices": list(range(len(slab), len(structure))),
                "snapshot": snapshot.model_dump(mode="json"),
                "summary": derived_summary(snapshot),
            }
        )
    return candidates


def adsorption_candidates(request: Any) -> dict[str, Any]:
    from .models import MoleculePayload, MoleculeSnapshot

    slab = snapshot_to_structure(request.input)
    adsorbate = snapshot_to_molecule(
        MoleculeSnapshot(format="mol", payload=MoleculePayload(molblock=request.params.adsorbate_molblock))
    )
    candidates = generate_adsorption_candidates(
        slab,
        adsorbate,
        distance=request.params.distance,
        site_kinds=list(request.params.site_kinds),
        reorient=request.params.reorient,
        orientation_euler_deg=request.params.orientation_euler_deg,
    )
    if not candidates:
        raise ValueError("No adsorption sites were found for the selected site types.")
    return {
        "kind": "candidates",
        "candidate_type": "adsorption",
        "candidates": candidates,
        "warnings": [],
        "change": {
            "candidate_count": len(candidates),
            "height": float(request.params.distance),
            "orientation_euler_deg": [float(item) for item in request.params.orientation_euler_deg],
        },
    }


__all__ = [
    "adsorption_candidates",
    "enumerate_adsorption_sites",
    "generate_adsorption_candidates",
    "place_adsorbate_at_site",
]
