"""
VASP input writer using pymatgen input sets.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet


class StructWriter:
    """
    VASP input writer using MPRelaxSet as base.
    """
    
    def write_vasp_inputs(
        self, 
        structure: Structure,
        output_dir: Path,
        calc_type: str = "bulk",
        k_product: int = 20,
        user_incar_overrides: Dict[str, Any] = None,
        run_template: Optional[Path] = None,
    ) -> None:
        """
        Write VASP input files for a structure using MPRelaxSet.
        
        Args:
            structure: Pymatgen Structure object
            output_dir: Directory to write inputs
            calc_type: Type of calculation (gas/bulk/slab)
            k_product: K-point density product
            user_incar_overrides: User INCAR overrides
            run_template: Optional run.yaml template path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get calculation-specific user_incar_settings
        calc_incar_settings = self._get_calc_type_settings(calc_type)
        
        # Merge with user overrides
        if user_incar_overrides:
            calc_incar_settings.update(user_incar_overrides)
        
        # Create MPRelaxSet with overrides
        if calc_type == "gas":
            # For molecules, use gamma-only k-points
            vasp_input_set = MPRelaxSet(
                structure,
                user_incar_settings=calc_incar_settings,
                user_kpoints_settings={"reciprocal_density": 1}  # Gamma only
            )
        else:
            # For bulk/slab, use k_product density
            vasp_input_set = MPRelaxSet(
                structure,
                user_incar_settings=calc_incar_settings,
                user_kpoints_settings={"reciprocal_density": k_product}
            )
        
        # Write all input files
        vasp_input_set.write_input(str(output_dir))
        
        # Copy run template if provided
        if run_template and run_template.exists():
            import shutil
            shutil.copy(run_template, output_dir / "run.yaml")
    
    def _get_calc_type_settings(self, calc_type: str) -> Dict[str, Any]:
        """
        Get INCAR overrides for specific calculation types.
        These override MPRelaxSet defaults.
        """
        if calc_type == "gas":
            # Gas phase (molecule) specific overrides
            return {
                "ISIF": 2,      # Relax ions only (no cell)
                "ISMEAR": 0,    # Gaussian smearing for molecules
                "SIGMA": 0.05,  # Smaller smearing for molecules
                "ISYM": 0,      # No symmetry for molecules
            }
        elif calc_type == "slab":
            # Slab specific overrides
            return {
                "ISIF": 2,      # Relax ions only
                "LDIPOL": True, # Dipole correction
                "IDIPOL": 3,    # z-direction dipole
            }
        else:
            # Bulk uses MPRelaxSet defaults
            return {}