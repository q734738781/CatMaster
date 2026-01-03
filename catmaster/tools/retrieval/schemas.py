from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


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


__all__ = [
    "MPSearchMaterialsInput",
    "MPDownloadStructureInput",
]
