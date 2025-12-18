from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from ase import Atoms
from ase.io import write

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class MoleculeCreateInput(BaseModel):
    """
    Explicit molecule builder from user-specified symbols and Cartesian coordinates.
    """

    molname: str = Field(..., description="Label for the molecule (used in metadata only).")
    symbols: List[str] = Field(..., description="Chemical symbols for each atom, e.g., ['O','O'].")
    coordinates: List[List[float]] = Field(..., description="Cartesian coordinates (Å), shape = N x 3.")
    output_path: str = Field(..., description="Destination VASP file path (e.g., POSCAR or *.vasp).")
    box_size: Optional[List[float]] = Field(
        None,
        description="Optional [a, b, c] cell lengths in Å; if provided, periodic box is set with PBC=True.",
    )

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.symbols) != len(self.coordinates):
            raise ValueError("symbols and coordinates must have the same length")
        for row in self.coordinates:
            if len(row) != 3:
                raise ValueError("each coordinate entry must have length 3")
        if self.box_size and len(self.box_size) != 3:
            raise ValueError("box_size must have exactly three elements [a, b, c]")
        return self


def create_molecule(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a molecule from explicit atomic positions and write a VASP-format file.

    Returns: dict {structure_file_rel, molname, formula, natoms, box_size}
    """
    params = MoleculeCreateInput(**payload)

    atoms = Atoms(symbols=params.symbols, positions=params.coordinates)
    if params.box_size:
        atoms.set_cell(params.box_size)
        atoms.set_pbc(True)

    output_path = resolve_workspace_path(params.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms, format="vasp")

    return create_tool_output(
        tool_name="create_molecule",
        success=True,
        data={
            "structure_file_rel": workspace_relpath(output_path),
            "molname": params.molname,
            "formula": atoms.get_chemical_formula(),
            "natoms": len(atoms),
            "box_size": params.box_size,
        },
    )


__all__ = ["MoleculeCreateInput", "create_molecule"]
