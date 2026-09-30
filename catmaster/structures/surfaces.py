from __future__ import annotations

from typing import Any

import numpy as np

from .serialization import derived_summary, snapshot_from_structure, snapshot_to_structure


def _surface_composition(slab: Any, *, top: bool) -> str:
    from pymatgen.core import Composition

    cartesian_z = np.asarray(slab.cart_coords, dtype=float)[:, 2]
    edge = float(cartesian_z.max() if top else cartesian_z.min())
    mask = cartesian_z >= edge - 1.25 if top else cartesian_z <= edge + 1.25
    species = [slab[index].species for index in np.flatnonzero(mask)]
    composition: dict[str, float] = {}
    for site_species in species:
        for element, amount in site_species.items():
            composition[str(element)] = composition.get(str(element), 0.0) + float(amount)
    return Composition(composition).formula if composition else "—"


def generate_slab_candidates(
    structure: Any,
    *,
    miller_index: list[int],
    min_slab_size: float,
    min_vacuum_size: float,
    center_slab: bool,
    symmetrize: bool,
    orthogonal: bool,
    lll_reduce: bool,
    surface_supercell: list[list[int]],
) -> list[dict[str, Any]]:
    from pymatgen.core.surface import SlabGenerator

    generator = SlabGenerator(
        initial_structure=structure,
        miller_index=tuple(int(item) for item in miller_index),
        min_slab_size=float(min_slab_size),
        min_vacuum_size=float(min_vacuum_size),
        center_slab=bool(center_slab),
        lll_reduce=bool(lll_reduce),
    )
    slabs = generator.get_slabs(symmetrize=bool(symmetrize))
    candidates: list[dict[str, Any]] = []
    for index, slab in enumerate(slabs):
        candidate = slab.get_orthogonal_c_slab() if orthogonal else slab.copy()
        if surface_supercell != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            candidate.make_supercell(surface_supercell)
        vector_a, vector_b = candidate.lattice.matrix[:2]
        area = float(np.linalg.norm(np.cross(vector_a, vector_b)))
        snapshot = snapshot_from_structure(candidate, fmt="poscar")
        candidates.append(
            {
                "candidate_index": index,
                "label": f"Termination {index + 1}",
                "snapshot": snapshot.model_dump(mode="json"),
                "summary": derived_summary(snapshot),
                "surface_area": area,
                "top_composition": _surface_composition(candidate, top=True),
                "bottom_composition": _surface_composition(candidate, top=False),
                "shift": float(getattr(slab, "shift", 0.0)),
            }
        )
    return candidates


def slab_candidates(request: Any) -> dict[str, Any]:
    structure = snapshot_to_structure(request.input)
    params = request.params
    candidates = generate_slab_candidates(
        structure,
        miller_index=params.miller_index,
        min_slab_size=params.min_slab_size,
        min_vacuum_size=params.min_vacuum_size,
        center_slab=params.center_slab,
        symmetrize=params.symmetrize,
        orthogonal=params.orthogonal,
        lll_reduce=params.lll_reduce,
        surface_supercell=params.surface_supercell,
    )
    if not candidates:
        raise ValueError("No slab terminations were generated for these settings.")
    return {
        "kind": "candidates",
        "candidate_type": "slab",
        "candidates": candidates,
        "warnings": [],
        "change": {
            "miller_index": params.miller_index,
            "termination_count": len(candidates),
        },
    }


__all__ = ["generate_slab_candidates", "slab_candidates"]
