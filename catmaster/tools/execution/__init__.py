from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from catmaster.tools.execution.mace_dispatch import mace_relax
from catmaster.tools.execution.vasp_dispatch import vasp_execute

logger = logging.getLogger(__name__)


class MaceRelaxInput(BaseModel):
    """Submit a single MACE relaxation (resource routed automatically)."""

    structure_file: str = Field(..., description="Input structure file.")
    work_dir: str = Field(".", description="Local working directory containing inputs.")
    fmax: float = Field(0.05, gt=0, description="Force threshold for relaxation.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    model: Optional[str] = Field(None, description="MACE model name; defaults from router config.")
    work_base: Optional[str] = Field(None, description="Override work_base; default uses work_dir name.")
    check_interval: int = Field(30, description="Polling interval in seconds when waiting.")


class VaspExecuteInput(BaseModel):
    """Submit a single VASP run (resource routed automatically)."""

    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    check_interval: int = Field(30, description="Polling interval seconds")



__all__ = [
    "MaceRelaxInput",
    "mace_relax",
    "VaspExecuteInput",
    "vasp_execute",
]
