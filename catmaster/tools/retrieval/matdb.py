"""
Materials Project API wrapper used in Stage 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mp_api.client import MPRester
from pymatgen.core import Structure


@dataclass(slots=True)
class MatdbHit:
    material_id: str
    formula: str
    energy_above_hull: Optional[float]
    formation_energy_per_atom: Optional[float]
    e_above_hull: Optional[float] = None
    task_id: Optional[str] = None


class MaterialsDBAdapter:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self.client = MPRester(api_key) if api_key else MPRester()

    def get_structures(self, identifier: str, output_dir: Path) -> List[Path]:
        """Download structures for the given material id or formula."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if identifier.startswith("mp-"):
            structures: List[Structure] = [
                self.client.get_structure_by_material_id(identifier)
            ]
        else:
            structures = self.client.get_structures(identifier) or []
        paths: List[Path] = []
        for idx, struct in enumerate(structures):
            if isinstance(struct, dict):
                struct = Structure.from_dict(struct)
            fid = f"{identifier.replace(' ', '_')}_{idx:03d}.cif"
            path = output_dir / fid
            struct.to(fmt="cif", filename=str(path))
            paths.append(path)
        return paths

    def search(self, criteria: Dict, properties: Optional[Iterable[str]] = None) -> List[MatdbHit]:
        allowed = {
            "material_ids",
            "formula",
            "chemsys",
            "elements",
        }
        search_kwargs: Dict[str, any] = {}
        for key, value in criteria.items():
            if key not in allowed:
                raise ValueError(f"Unsupported Summary search key: {key}")
            search_kwargs[key] = value

        fields = list(properties) if properties else [
            "material_id",
            "formula_pretty",
            "energy_above_hull",
            "formation_energy_per_atom",
        ]

        docs = self.client.summary.search(fields=fields, **search_kwargs)
        hits: List[MatdbHit] = []
        for doc in docs:
            hits.append(
                MatdbHit(
                    material_id=doc.material_id,
                    formula=doc.formula_pretty,
                    energy_above_hull=getattr(doc, "energy_above_hull", None),
                    formation_energy_per_atom=getattr(doc, "formation_energy_per_atom", None),
                    e_above_hull=getattr(doc, "energy_above_hull", None),
                    task_id=getattr(doc, "task_id", None),
                )
            )
        return hits


def save_hits(hits: Iterable[MatdbHit], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(hit) for hit in hits], fh, indent=2, ensure_ascii=False)


def query(
    criteria: Dict,
    properties: Optional[Iterable[str]] = None,
    structures_dir: Path = Path("structures"),
    api_key: Optional[str] = None,
) -> Dict[str, object]:
    adapter = MaterialsDBAdapter(api_key=api_key)
    hits = adapter.search(criteria=criteria, properties=properties)
    structures_path = Path(structures_dir)
    structures_path.mkdir(parents=True, exist_ok=True)
    hits_path = (
        structures_path.parent / "matdb_hits.json"
        if structures_path.name == "structures"
        else structures_path / "matdb_hits.json"
    )
    save_hits(hits, hits_path)

    structure_paths: List[str] = []
    if hits:
        def energy(hit: MatdbHit) -> float:
            return hit.energy_above_hull if hit.energy_above_hull is not None else float("inf")

        best_hit = min(hits, key=energy)
        structure_paths = [p.as_posix() for p in adapter.get_structures(best_hit.material_id, structures_path)]

    return {
        "count": len(hits),
        "hits_path": hits_path.as_posix(),
        "structures": structure_paths,
        "provider": "materials_project",
        "api_version": getattr(adapter.client, "api_version", None),
    }


