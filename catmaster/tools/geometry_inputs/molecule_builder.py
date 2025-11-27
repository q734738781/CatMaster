"""
Molecular structure creation and manipulation tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time

from ase import Atoms
from ase.io import write
from ase.data import atomic_numbers, covalent_radii
from ase.build import molecule as ase_molecule

from catmaster.tools.base import create_tool_output


def create_molecule(
    molecule: str,
    output_path: str,
    bond_length: Optional[float] = None,
    box_size: Optional[List[float]] = None,
    center: bool = True,
) -> Dict[str, Any]:
    """
    Create a molecular structure and save to file.
    
    Args:
        molecule: Molecule formula (e.g., 'O2', 'H2O', 'CO2', 'CH4') or SMILES string
        output_path: Output file path for structure (VASP format. File extension should be .vasp)
        bond_length: Optional bond length override for diatomic molecules (Angstrom)
        box_size: Box dimensions [x, y, z] in Angstrom. If None, auto-sized to 15, 16, 17 Angstrom
        center: Whether to center molecule in the box
    
    Returns:
        dict: {
            "structure_file": path to created structure,
            "formula": molecular formula,
            "natoms": number of atoms,
            "mass": molecular mass in amu,
            "box_size": actual box size used,
            "meta_file": path to metadata JSON (if bonds detected)
        }
    """
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to build molecule
    atoms = None
    
    # Handle common diatomic molecules with custom bond lengths
    if molecule.upper() in ['O2', 'N2', 'H2', 'F2', 'CL2'] and bond_length:
        # Create diatomic molecule manually
        symbol = molecule[0]
        if box_size is None:
            box_size = [15.0, 16.0, 17.0]
        
        positions = [
            [box_size[0]/2, box_size[1]/2, box_size[2]/2 - bond_length/2],
            [box_size[0]/2, box_size[1]/2, box_size[2]/2 + bond_length/2]
        ]
        atoms = Atoms(molecule.upper(), positions=positions, cell=box_size, pbc=True)
        
    else:
        # Use ASE's molecule database
        try:
            atoms = ase_molecule(molecule)
        except Exception:
            # Try direct formula
            try:
                atoms = Atoms(molecule)
            except Exception:
                raise ValueError(f"Could not create molecule: {molecule}")
        
        # Put in box
        if box_size is None:
            box_size = [15.0, 16.0, 17.0]
        
        atoms.set_cell(box_size)
        atoms.set_pbc(True)
        
        if center:
            atoms.center()
    
    # Write structure in format based on file extension
    file_ext = output_file.suffix.lower()
    if file_ext in ['.vasp']:
        write(str(output_file), atoms, format='vasp')
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}. Only .vasp is supported.")
    
    # Detect bonds and create metadata
    meta_data = {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "symbols": atoms.get_chemical_symbols(),
    }
    
    # Simple bond detection for small molecules
    if len(atoms) <= 50:
        entities = _detect_molecules(atoms)
        if entities:
            meta_data["entities"] = entities
    
    # Save metadata
    meta_file = output_file.with_suffix('.meta.json')
    meta_file.write_text(json.dumps(meta_data, indent=2), encoding='utf-8')
    
    # Calculate molecular mass
    masses = atoms.get_masses()
    total_mass = float(masses.sum())
    
    # Use standardized output format
    return create_tool_output(
        tool_name="create_molecule",
        success=True,
        data={
            "structure_file": str(output_file),
            "formula": atoms.get_chemical_formula(),
            "natoms": len(atoms),
            "mass": total_mass,
            "box_size": list(box_size),
            "meta_file": str(meta_file)
        }
    )


def _detect_molecules(atoms: Atoms) -> List[Dict[str, Any]]:
    """
    Simple molecule detection based on covalent radii.
    Returns list of molecular entities with bonds.
    """
    from scipy.spatial.distance import pdist, squareform
    
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Calculate distance matrix
    dist_matrix = squareform(pdist(positions))
    
    # Detect bonds (distance < sum of covalent radii * 1.2)
    bonds = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            r_cov_i = covalent_radii[atomic_numbers[symbols[i]]]
            r_cov_j = covalent_radii[atomic_numbers[symbols[j]]]
            if dist_matrix[i, j] < (r_cov_i + r_cov_j) * 1.2:
                bonds.append({
                    "i": i,
                    "j": j,
                    "r0": float(dist_matrix[i, j]),
                    "symbols": f"{symbols[i]}-{symbols[j]}"
                })
    
    if not bonds:
        return []
    
    # For now, treat all atoms as one molecule
    entity = {
        "role": "molecule",
        "global_indices": list(range(len(atoms))),
        "local_bonds": bonds
    }
    
    return [entity]

