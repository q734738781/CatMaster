"""
Materials Project retrieval tools exposed to LLM agents.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from pymatgen.core.structure import Structure
from mp_api.client import MPRester
from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class MPSearchMaterialsInput(BaseModel):
    """Search Materials Project by formula/chemsys/elements with optional filters."""

    query: str = Field(..., description="Formula or chemsys (e.g., 'LiFePO4' or 'Li-Fe-P-O') or mp-id.")
    limit: int = Field(50, ge=1, description="Maximum number of hits to return (default 50).")
    band_gap_min: Optional[float] = Field(None, ge=0, description="Minimum band gap (eV) if specified.")
    band_gap_max: Optional[float] = Field(None, ge=0, description="Maximum band gap (eV) if specified.")
    e_above_hull_max: Optional[float] = Field(None, ge=0, description="Maximum energy above hull (eV/atom) if specified.")
    nsites_max: Optional[int] = Field(None, ge=1, description="Maximum number of sites (filters out very large cells) if specified.")


class MPDownloadStructureInput(BaseModel):
    """Download a structure from Materials Project into the workspace."""

    mp_id: str = Field(..., description="Materials Project ID, e.g., mp-149.")
    fmt: str = Field("poscar", pattern="^(poscar|cif|json)$", description="Output format: poscar|cif|json.")
    output_dir: str = Field("retrieval/mp", description="Workspace-relative directory to save the structure.")


@dataclass(slots=True)
class MatdbHit:
    material_id: str
    formula: str
    energy_above_hull: Optional[float]
    formation_energy_per_atom: Optional[float]
    band_gap: Optional[float]
    nsites: Optional[int]
    volume: Optional[float]
    density: Optional[float]


def _mpr() -> MPRester:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable is not set.")
    return MPRester(api_key, monty_decode=True, use_document_model=True)


def mp_search_materials(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Search Materials Project by formula/chemsys/elements and return concise hit list.
    Args:
        query: Formula or chemsys (e.g., 'LiFePO4' or 'Li-Fe-P-O') or mp-id.
        limit: Maximum number of hits to return.
        band_gap_min: Minimum band gap (eV).
        band_gap_max: Maximum band gap (eV).
        e_above_hull_max: Maximum energy above hull (eV/atom).
        nsites_max: Maximum number of sites (filters out very large cells).
        api_key: Materials Project API key (optional if env provides).
    Returns:
        List of hits with material_id, formula, energy_above_hull, formation_energy_per_atom, band_gap, nsites, volume, and density.
    """
    params = MPSearchMaterialsInput(**payload)
    try:
        client = _mpr()
    except Exception as exc:
        return create_tool_output("mp_search_materials", success=False, error=str(exc))

    # Build search kwargs
    criteria: Dict[str, Any] = {}
    q = params.query.strip()
    if q.startswith("mp-"):
        criteria["material_ids"] = [q]
    elif "-" in q:
        criteria["chemsys"] = q
    else:
        criteria["formula"] = q

    fields = [
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "nsites",
        "volume",
        "density",
    ]

    docs = client.summary.search(fields=fields, **criteria)

    hits: List[MatdbHit] = []
    for doc in docs:
        # Apply optional filters
        if params.band_gap_min is not None and (doc.band_gap or 0.0) < params.band_gap_min:
            continue
        if params.band_gap_max is not None and (doc.band_gap or 0.0) > params.band_gap_max:
            continue
        if params.e_above_hull_max is not None and (doc.energy_above_hull or 0.0) > params.e_above_hull_max:
            continue
        if params.nsites_max is not None and (doc.nsites or 0) > params.nsites_max:
            continue

        hits.append(
            MatdbHit(
                material_id=doc.material_id,
                formula=doc.formula_pretty,
                energy_above_hull=getattr(doc, "energy_above_hull", None),
                formation_energy_per_atom=getattr(doc, "formation_energy_per_atom", None),
                band_gap=getattr(doc, "band_gap", None),
                nsites=getattr(doc, "nsites", None),
                volume=getattr(doc, "volume", None),
                density=getattr(doc, "density", None),
            )
        )

    # Sort by energy above hull then band gap for stability-first ranking
    hits.sort(key=lambda h: (h.energy_above_hull if h.energy_above_hull is not None else 1e9,
                             h.band_gap if h.band_gap is not None else 0.0))

    limited = hits[: params.limit]
    return create_tool_output(
        "mp_search_materials",
        success=True,
        data={
            "count": len(hits),
            "returned": len(limited),
            "hits": [
                {
                    "mp_id": h.material_id,
                    "formula": h.formula,
                    "e_above_hull": h.energy_above_hull,
                    "formation_energy_per_atom": h.formation_energy_per_atom,
                    "band_gap": h.band_gap,
                    "nsites": h.nsites,
                    "volume": h.volume,
                    "density": h.density,
                }
                for h in limited
            ],
            "truncated": len(hits) > len(limited),
        },
    )


def mp_download_structure(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Download a single structure from Materials Project and write it under the workspace.
    Args:
        mp_id: Materials Project ID, e.g., mp-149.
        fmt: Output format: poscar|cif|json.
        output_dir: Directory to write the structure.
    Returns:
        Path to the written structure.
    """
    params = MPDownloadStructureInput(**payload)
    try:
        client = _mpr()
    except Exception as exc:
        return create_tool_output("mp_download_structure", success=False, error=str(exc))

    out_dir = resolve_workspace_path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = params.fmt.lower()
    ext = {"poscar": "vasp", "cif": "cif", "json": "json"}.get(fmt, fmt)
    out_path = out_dir / f"{params.mp_id}.{ext}"

    try:
        structure = client.get_structure_by_material_id(params.mp_id)
        if isinstance(structure, dict):
            # Use pymatgen json structure load
            structure = Structure.from_dict(structure)
    except Exception as exc:  # pragma: no cover - remote call
        return create_tool_output(
            "mp_download_structure",
            success=False,
            error=str(exc),
        )

    if fmt == "json":
        out_path.write_text(json.dumps(structure.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        structure.to(fmt=fmt, filename=str(out_path))

    meta = {
        "formula": structure.composition.reduced_formula,
        "natoms": len(structure)
    }

    return create_tool_output(
        "mp_download_structure",
        success=True,
        data={
            "mp_id": params.mp_id,
            "format": fmt,
            "structure_rel": workspace_relpath(out_path),
            "output_dir_rel": workspace_relpath(out_dir),
            "metadata": meta,
        },
    )


__all__ = [
    "MPSearchMaterialsInput",
    "MPDownloadStructureInput",
    "mp_search_materials",
    "mp_download_structure",
]
