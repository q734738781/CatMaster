from __future__ import annotations

import math
import os
from io import StringIO
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field
from ase import Atoms
from ase.io import write, read

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class MoleculeFromSmilesInput(BaseModel):
    """
    Build a 3D molecule from a SMILES string using RDKit. Specially, pay attention to the formal charge of the SMILES string, RDkit will check the charge and add Hs to the molecule to make it neutral.
    """

    smiles: str = Field(..., description="SMILES string for the molecule.")
    name: Optional[str] = Field(None, description="Optional base name for output files; defaults to formula.")
    output_path: str = Field("mol", description="Workspace-relative path prefix; files will be <prefix>.xyz and/or <prefix>.vasp.")
    box_padding: float = Field(10.0, ge=0.0, description="Padding (Å) added around the molecule to make a cubic box for POSCAR. Box lattice is around twice the padding value (10A Padding -> 20A Box).")
    fmt: str = Field("poscar", pattern="^(poscar|xyz|both)$", description="Output format: poscar (with PBC box), xyz (no PBC), or both.")


def _build_conformer(smiles: str):
    """Return RDKit Mol with embedded 3D coords; raise if fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - dependency import
        raise ImportError("RDKit is required for create_molecule_from_smiles") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("ETKDG embedding failed")

    # Optimize geometry
    ff_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    if ff_status != 0:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    return mol


def _mol_to_ase(mol) -> Atoms:
    """Convert RDKit Mol with conformer to ASE Atoms via XYZ block."""
    from rdkit import Chem
    block = Chem.MolToXYZBlock(mol)
    atoms = read(StringIO(block), format="xyz")
    return atoms


def create_molecule_from_smiles(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a molecule from SMILES, generate 3D coords, and write XYZ + (optional) POSCAR with padded box.

    Returns: tool output with paths and basic metadata.
    """
    try:
        params = MoleculeFromSmilesInput(**payload)
        mol = _build_conformer(params.smiles)
        atoms = _mol_to_ase(mol)
    except Exception as exc:
        return create_tool_output(
            tool_name="create_molecule_from_smiles",
            success=False,
            error=str(exc),
        )

    # Derive name/formula
    formula = atoms.get_chemical_formula()
    base = (params.name or formula).replace(" ", "_")

    prefix_path = resolve_workspace_path(params.output_path)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    xyz_path = None
    poscar_path = None
    box = None

    if params.fmt in {"xyz", "both"}:
        xyz_path = prefix_path.with_suffix(".xyz")
        write(xyz_path, atoms, format="xyz")

    if params.fmt in {"poscar", "both"}:
        coords = atoms.get_positions()
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = maxs - mins
        max_span = float(np.max(span))
        padding = float(params.box_padding)
        box_len = max_span + 2 * padding if max_span > 0 else max(1.0, 2 * padding)
        box = [box_len, box_len, box_len]

        # Shift to center in box
        center = (mins + maxs) / 2.0
        shift = np.array([box_len / 2.0, box_len / 2.0, box_len / 2.0]) - center
        atoms_shifted = atoms.copy()
        atoms_shifted.set_positions(coords + shift)
        atoms_shifted.set_cell(box)
        atoms_shifted.set_pbc(True)

        poscar_path = prefix_path.with_suffix(".vasp")
        write(poscar_path, atoms_shifted, format="vasp")

    return create_tool_output(
        tool_name="create_molecule_from_smiles",
        success=True,
        data={
            "smiles": params.smiles,
            "formula": formula,
            "natoms": len(atoms),
            "xyz_file_rel": workspace_relpath(xyz_path) if xyz_path else None,
            "poscar_file_rel": workspace_relpath(poscar_path) if poscar_path else None,
            "box_size": box,
        },
    )


__all__ = ["MoleculeFromSmilesInput", "create_molecule_from_smiles"]
