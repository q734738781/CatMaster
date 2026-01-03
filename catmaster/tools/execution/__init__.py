from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from catmaster.tools.execution.mace_dispatch import mace_relax, mace_relax_batch
from catmaster.tools.execution.vasp_dispatch import vasp_execute, vasp_execute_batch

logger = logging.getLogger(__name__)


class MaceRelaxInput(BaseModel):
    """Submit a single MACE relaxation."""

    structure_file: str = Field(..., description="Input structure file with lattice information (Supports POSCAR/CIF, xyz files are NOT supported).")
    fmax: float = Field(0.03, gt=0, description="Force threshold for relaxation in eV/Å.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: Optional[str] = Field(None, description="MACE model name; defaults from router config (medium-mpa-0).")
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.") 

class MaceRelaxBatchInput(BaseModel):
    """Submit multiple MACE relaxations in one DPDispatcher submission. Preferred tool for batch MACE relaxations."""

    structure_files: list[str] = Field(..., description="List of structure files with lattice (POSCAR/CIF).")
    fmax: float = Field(0.03, gt=0, description="Force threshold for relaxation in eV/Å.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: Optional[str] = Field(None, description="MACE model name; defaults from router config (medium-mpa-0).")
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.") 


class VaspExecuteInput(BaseModel):
    """Submit a single VASP run."""

    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    check_interval: int = Field(30, description="Polling interval seconds")

class VaspExecuteBatchInput(BaseModel):
    """Submit multiple VASP runs in one DPDispatcher submission. Preferred tool for batch VASP runs."""

    input_dirs: list[str] = Field(..., description="List of VASP input directories.")
    check_interval: int = Field(30, description="Polling interval seconds")



__all__ = [
    "MaceRelaxInput",
    "MaceRelaxBatchInput",
    "mace_relax",
    "mace_relax_batch",
    "VaspExecuteInput",
    "VaspExecuteBatchInput",
    "vasp_execute",
    "vasp_execute_batch",
]
