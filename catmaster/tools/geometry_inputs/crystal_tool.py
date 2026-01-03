from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure

from catmaster.tools.base import resolve_workspace_path, workspace_relpath, create_tool_output


class SupercellInput(BaseModel):
    """Create a supercell from a bulk structure."""

    structure_file: str = Field(..., description="Bulk structure file (POSCAR/CIF/etc.), workspace-relative.")
    supercell: List[int] = Field(..., min_length=3, max_length=3, description="Supercell replication [a,b,c].")
    output_path: str = Field(..., description="Output structure path (workspace-relative, e.g., bulk_supercell.vasp).")


def supercell(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a supercell from a bulk structure and write it to output_path.
    """
    try:
        params = SupercellInput(**payload)
        structure_path = resolve_workspace_path(params.structure_file, must_exist=True)
        out_path = resolve_workspace_path(params.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        structure = Structure.from_file(structure_path)
        matrix = np.diag(params.supercell)
        structure.make_supercell(matrix)
        structure.to(fmt="poscar", filename=str(out_path))

        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_rel": workspace_relpath(out_path),
            "supercell": params.supercell,
            "natoms": len(structure),
        }
        return create_tool_output("supercell", success=True, data=data)
    except Exception as exc:
        return create_tool_output("supercell", success=False, error=str(exc))


__all__ = ["SupercellInput", "supercell"]
