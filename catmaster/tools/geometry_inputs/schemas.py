from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SlabCutInput(BaseModel):
    structure_file: str = Field(..., description="Bulk structure file (POSCAR/CIF).")
    compound_name: Optional[str] = Field(None, description="Name used as folder prefix.")
    miller_list: List[List[int]] = Field(..., description="List of Miller indices, e.g., [[1,1,1],[1,0,0]].")
    min_slab_size: float = Field(12.0, ge=1.0, description="Min slab thickness (Å).")
    min_vacuum_size: float = Field(15.0, ge=1.0, description="Min vacuum thickness (Å).")
    relax_thickness: float = Field(5.0, ge=0.0, description="Thickness to relax (Å).")
    output_root: str = Field(..., description="Root directory to write slabs.")
    get_symmetry_slab: bool = Field(False, description="Generate symmetry-distinct slabs.")
    fix_bottom: bool = Field(True, description="Fix bottom region during relaxation.")


class SlabFixInput(BaseModel):
    input_path: str = Field(..., description="A slab file or folder containing slabs.")
    output_dir: str = Field(..., description="Directory to write fixed slabs.")
    relax_thickness: float = Field(5.0, ge=0.0, description="Thickness to relax (Å).")
    fix_bottom: bool = Field(True, description="Fix bottom region during relaxation.")
    centralize: bool = Field(False, description="Centralize slab along c axis.")


class AdsorbatePlacementInput(BaseModel):
    config: Dict[str, Any] = Field(..., description="Config dict forwarded to adsorbate placement generator.")


class MPRelaxPrepareInput(BaseModel):
    """Prepare MPRelax-style VASP input sets for molecular/bulk/slab relaxations."""

    input_path: str = Field(..., description="Path to structure file (POSCAR/CIF/XYZ/JSON) or directory")
    output_root: str = Field(..., description="Root directory where MPRelax input sets will be created")
    calc_type: str = Field("bulk", description="Calculation type: 'gas', 'bulk', or 'slab', will affect k-point and ISIF settings")
    k_product: int = Field(20, ge=1, description="K-point density product (use 1 for molecules)")
    user_incar_settings: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "INCAR overrides on top of MPRelaxSet defaults (pymatgen format). "
            'Specify MAGMOM as {"element": value}, not a list.'
        ),
    )


__all__ = [
    "SlabCutInput",
    "SlabFixInput",
    "AdsorbatePlacementInput",
    "MPRelaxPrepareInput",
]
