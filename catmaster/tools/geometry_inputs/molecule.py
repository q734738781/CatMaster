from __future__ import annotations

import math
import os
from io import StringIO
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from ase import Atoms
from ase.io import write, read

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.structures.molecules import generate_conformers
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from .molecular_qchem import _optimize_conformer


class MoleculeFromSmilesInput(BaseModel):
    """
    [molecule/modeling] Build a 3D molecule from a SMILES string using RDKit. Pay attention to formal charge in the SMILES string; RDKit checks charge and adds H atoms as needed.
    """

    smiles: str = Field(..., description="SMILES string for the molecule.")
    name: str = Field("", description="Output basename when output_path is omitted; defaults to the molecular formula.")
    output_path: str = Field("", description="Explicit workspace-relative output prefix; takes precedence over name. The selected format supplies the suffix.")
    optimize: str = Field("mmff", pattern="^(mmff|uff|none)$", description="Requested cleanup method; no automatic change of force field. Failure retains embedded coordinates with explicit partial status.")
    random_seed: int = Field(42, description="RDKit ETKDGv3 embedding seed.")
    max_iterations: int = Field(500, ge=1, description="Force-field iteration limit.")
    overwrite: bool = Field(False, description="Allow replacing selected output files; otherwise an existing destination is an error.")
    box_padding: float = Field(10.0, ge=0.0, description="Padding (Å) added around the molecule to make a cubic box for POSCAR. Box lattice is around twice the padding value (10A Padding -> 20A Box).")
    fmt: str = Field(
        "poscar",
        pattern="^(poscar|xyz|both)$",
        description=(
            "Choose one output format: poscar (with PBC box) or xyz (no PBC). "
            "Use both only when the user explicitly requests both formats or a downstream interface requires them."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_null(cls, values):
        if isinstance(values, dict):
            return {key: "" if key in {"name", "output_path"} and value is None else value for key, value in values.items()}
        return values


def _build_conformer(smiles: str, random_seed: int = 42):
    """Return RDKit Mol with embedded 3D coords; raise if fails."""
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - dependency import
        raise ImportError("RDKit is required for create_molecule_from_smiles") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    candidates = generate_conformers(
        mol,
        count=1,
        random_seed=random_seed,
        optimize="none",
        prune_rms_threshold=0.0,
    )
    return candidates[0][0]


def _mol_to_ase(mol) -> Atoms:
    """Convert RDKit Mol with conformer to ASE Atoms via XYZ block."""
    from rdkit import Chem
    block = Chem.MolToXYZBlock(mol)
    atoms = read(StringIO(block), format="xyz")
    return atoms


def create_molecule_from_smiles(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    [molecule/modeling] Create a molecule from SMILES, generate 3D coords, and write XYZ plus optional POSCAR with a padded box.

    Returns: tool output with paths and basic metadata.
    """
    try:
        params = MoleculeFromSmilesInput(**payload)
        mol = _build_conformer(params.smiles, params.random_seed)
        energy, optimization_state, optimization_error = _optimize_conformer(mol, 0, params.optimize, params.max_iterations)
        atoms = _mol_to_ase(mol)
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="create_molecule_from_smiles",
            public_message=f"Failed to build molecule from SMILES: {exc}",
            artifact={
                "tool_name": "create_molecule_from_smiles",
                "data": {"smiles": str(payload.get("smiles") or "")},
            },
            error_code="molecule_build_failed",
        )

    # Derive name/formula
    formula = atoms.get_chemical_formula()
    base = (params.name or formula).replace(" ", "_")

    prefix_path = resolve_workspace_path(params.output_path or base)
    suffixes = [".xyz", ".vasp"] if params.fmt == "both" else [".xyz" if params.fmt == "xyz" else ".vasp"]
    if not params.overwrite:
        for suffix in suffixes:
            if prefix_path.with_suffix(suffix).exists():
                raise ValueError(f"Output exists: {workspace_relpath(prefix_path.with_suffix(suffix))}; choose a new output_path or overwrite=true")
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

    data = {
        "smiles": params.smiles,
        "formula": formula,
        "natoms": len(atoms),
        "xyz_file_rel": workspace_relpath(xyz_path) if xyz_path else None,
        "poscar_file_rel": workspace_relpath(poscar_path) if poscar_path else None,
        "box_size": box,
        "optimization_state": optimization_state,
        "optimization_error": optimization_error,
        "energy_kcal_mol": energy,
    }
    content = (
        f"create_molecule_from_smiles {'partial' if optimization_state in {'failed', 'not_converged'} else 'completed'}.\n"
        f"optimization_state={optimization_state} {optimization_error}\n"
        f"formula={formula} natoms={len(atoms)}\n"
        f"xyz_file_rel={data['xyz_file_rel']} poscar_file_rel={data['poscar_file_rel']}"
    )
    return content, {
        "tool_name": "create_molecule_from_smiles",
        "data": data,
    }


__all__ = ["MoleculeFromSmilesInput", "create_molecule_from_smiles"]
