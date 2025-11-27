from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MoleculeCreateInput(BaseModel):
    molecule: str = Field(..., description="Molecule formula (e.g., 'O2', 'H2O') or name")
    output_path: str = Field(..., description="Output file path for structure. Supposed to be VASP format. File extension should be .vasp")
    bond_length: Optional[float] = Field(None, description="Bond length for diatomic molecules (Angstrom)")
    box_size: Optional[List[float]] = Field(None, description="Box dimensions [x,y,z] in Angstrom")
    center: bool = Field(True, description="Whether to center molecule in box")


def create_molecule(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a molecular structure and save to file.
    
    Pydantic Args Schema: MoleculeCreateInput
    Returns: dict {structure_file, formula, natoms, mass, box_size, meta_file}
    """
    from .molecule_builder import create_molecule as create_mol_impl
    return create_mol_impl(**payload)


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


def slab_cut(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cut slabs for given facets and write POSCARs with selective dynamics metadata.

    Pydantic Args Schema: SlabCutInput
    Returns: dict {compound, facets, output_root, generated[]}
    """
    from .slab_tools import cut_slabs
    return cut_slabs(payload)


class SlabFixInput(BaseModel):
    input_path: str = Field(..., description="A slab file or folder containing slabs.")
    output_dir: str = Field(..., description="Directory to write fixed slabs.")
    relax_thickness: float = Field(5.0, ge=0.0, description="Thickness to relax (Å).")
    fix_bottom: bool = Field(True, description="Fix bottom region during relaxation.")
    centralize: bool = Field(False, description="Centralize slab along c axis.")


def slab_fix(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process slab(s): add selective dynamics, optional centralization, and write VASP POSCARs.

    Pydantic Args Schema: SlabFixInput
    Returns: dict {source, output_dir, generated[]}
    """
    from .slab_tools import fix_slab
    return fix_slab(payload)


class AdsorbatePlacementInput(BaseModel):
    config: Dict[str, Any] = Field(..., description="Config dict forwarded to adsorbate placement generator.")


def adsorbate_placements(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate candidate adsorbate placements according to a configuration.

    Pydantic Args Schema: AdsorbatePlacementInput
    Returns: dict {schema_version, output_dir, num_candidates, candidate_paths[]}
    """
    from .adsorbate_placements import generate_placements
    return generate_placements(config)


class GasFillInput(BaseModel):
    input: str = Field(..., description="Slab POSCAR/CIF path.")
    output: Optional[str] = Field(None, description="Output VASP path; defaults to input.with_h2.vasp")
    temperature: float = Field(673.0, description="Temperature in K.")
    pressure_mpa: float = Field(1.0, description="Pressure in MPa.")
    buffer_A: float = Field(2.0, description="Buffer from slab surfaces (Å).")
    bond_A: float = Field(0.74, description="H–H bond length (Å).")
    replicate: Optional[List[int]] = Field(None, description="Replicate slab in x,y, e.g., [2,2].")
    targetN: Optional[float] = Field(None, description="Target expected molecule count (if replicate not set).")
    max_rep: int = Field(16, ge=1, description="Max replication for automatic choice.")
    rounding: str = Field("round", description="Rounding mode: round|ceil|floor|poisson.")
    min_molecules: int = Field(0, ge=0, description="Minimum molecules to place.")
    seed: int = Field(42, description="Random seed.")
    summary_json: Optional[str] = Field(None, description="Optional JSON summary output path.")
def gas_fill(**config: Any) -> Dict[str, Any]:
    """
    Insert H2 molecules into the slab vacuum region using ideal gas estimation and distance constraints.

    Pydantic Args Schema: GasFillInput
    Returns: dict with placement summary and output path
    """
    from .gas_fill import fill_h2
    return fill_h2(config)


class VaspPrepareInput(BaseModel):
    """Input parameters for VASP input preparation."""
    input_path: str = Field(..., description="Path to structure file (POSCAR/CIF/XYZ/JSON) or directory")
    output_root: str = Field(..., description="Root directory where VASP input sets will be created")
    calc_type: str = Field("bulk", description="Calculation type: 'gas', 'bulk', or 'slab'")
    k_product: int = Field(20, ge=1, description="K-point density product (use 1 for molecules)")
    user_incar_settings: Optional[Dict[str, Any]] = Field(None, description=f"""
    INCAR parameter overrides as dict in PyMatGen's format as overrides on top of MPRelaxSet defaults. 
    No need to provide parameters already set and not required to override.
    Specially, the MAGMOM should be specified as {{"element": "value"}}, not provided as a list.""")


def vasp_prepare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare normalized VASP input sets (INCAR/KPOINTS/POSCAR/POTCAR) for each structure.

    Pydantic Args Schema: VaspPrepareInput
    Returns: dict {calc_type, k_product, structures_processed, outputs[]}
    """
    from .vasp_prepare import prepare_vasp_inputs
    return prepare_vasp_inputs(payload)
