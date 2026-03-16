"""
VASP input writer using pymatgen input sets.
"""
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Kpoints

from catmaster.tools.base import resolve_workspace_path

class StructWriter:
    """
    VASP input writer using MPRelaxSet as base.
    """
    
    def write_vasp_inputs(
        self, 
        structure: Structure,
        output_dir: Path,
        calc_type: str = "bulk",
        relax_cell: bool = False,
        k_product: int = 35,
        use_d3: bool = True,
        user_incar_overrides: Optional[Dict[str, Any]] = None,  
        use_dft_plus_u: bool = False,
        single_point: bool = False,
        compute_dos: bool = False,
        enable_dipole: bool = False,
        run_template: Optional[Path] = None,
    ) -> None:
        """
        Write VASP input files for a structure using MPRelaxSet.
        
        Args:
            structure: Pymatgen Structure object
            output_dir: Directory to write inputs
            calc_type: Type of calculation (gas/bulk/slab)
            relax_cell: If True for bulk, set ISIF=3; otherwise ISIF=2
            k_product: K-point density product 
            user_incar_overrides: User INCAR overrides
            single_point: If True, prepare static single-point settings (NSW=0, IBRION=-1).
            compute_dos: If True, keep orbital projection output (LORBIT=11); else set LORBIT=0.
            enable_dipole: If True, default IDIPOL=3 and DIPOL to atomic center-of-mass fractional coordinates.
            run_template: Optional run.yaml template path
        """
        output_dir = resolve_workspace_path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        required_overrides = self._required_overrides(calc_type, relax_cell, user_incar_overrides or {})
        user_incar_settings = self._build_user_incar_settings(
            calc_type=calc_type,
            required_overrides=required_overrides,
            use_d3=use_d3,
            use_dft_plus_u=use_dft_plus_u,
            user_incar_overrides=user_incar_overrides or {},
            single_point=single_point,
            compute_dos=compute_dos,
        )
        if enable_dipole:
            user_incar_settings.setdefault("IDIPOL", 3)
            user_incar_settings.setdefault("DIPOL", self._compute_com_frac_dipol(structure))
        
        # Build VASP input set; k-points handled manually to match reference logic
        vasp_input_set = MPRelaxSet(
            structure,
            user_potcar_functional="PBE_54",
            user_incar_settings=user_incar_settings,
        )
        vasp_input_set.write_input(str(output_dir))

        # Remove keys explicitly set to None by user (unless task requires them)
        if user_incar_overrides:
            incar_path = output_dir / "INCAR"
            if incar_path.exists():
                incar_obj = Incar.from_file(incar_path)
                changed = False
                for key, val in user_incar_overrides.items():
                    if val is None and key in incar_obj and key not in required_overrides:
                        if enable_dipole and str(key).strip().upper() in {"DIPOL", "IDIPOL"}:
                            continue
                        del incar_obj[key]
                        changed = True
                if changed:
                    incar_obj.write_file(incar_path)

        # Write KPOINTS following k_product rule
        k_grid = self._generate_kgrid(calc_type, k_product, structure)
        kpt = Kpoints.gamma_automatic(kpts=k_grid)
        kpoints_path = output_dir / "KPOINTS"
        if kpoints_path.exists():
            kpoints_path.unlink()
        kpt.write_file(kpoints_path)

        # Copy run template if provided
        if run_template and run_template.exists():
            import shutil
            shutil.copy(run_template, output_dir / "run.yaml")

    def _compute_com_frac_dipol(self, structure: Structure) -> Tuple[float, float, float]:
        total_mass = 0.0
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0

        for site in structure.sites:
            mass = float(site.species.weight)
            x, y, z = site.coords
            total_mass += mass
            com_x += mass * float(x)
            com_y += mass * float(y)
            com_z += mass * float(z)

        if total_mass <= 0:
            raise ValueError("Cannot compute DIPOL center of mass: total atomic mass is non-positive.")

        com_cart = (com_x / total_mass, com_y / total_mass, com_z / total_mass)
        frac = structure.lattice.get_fractional_coords(com_cart)
        wrapped = tuple(float(v - math.floor(float(v))) for v in frac)
        return wrapped

    def _build_user_incar_settings(
        self,
        *,
        calc_type: str,
        required_overrides: Dict[str, Any],
        use_d3: bool,
        use_dft_plus_u: bool,
        user_incar_overrides: Dict[str, Any],
        single_point: bool,
        compute_dos: bool,
    ) -> Dict[str, Any]:
        def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in (d or {}).items() if v is not None}

        # User settings then task-required overrides (task wins per-key)
        user_incar_settings: Dict[str, Any] = {}
        user_incar_settings.update(_clean(user_incar_overrides or {}))
        user_incar_settings.update(required_overrides)

        # Global baseline
        user_incar_settings.setdefault("EDIFF", 1e-6)
        user_incar_settings.setdefault("NELM", 100)
        user_incar_settings.setdefault("LCHARG", False)
        user_incar_settings.setdefault("LWAVE", False)

        # Keep DOS/projection data optional; default off to reduce disk usage.
        if compute_dos:
            user_incar_settings.setdefault("LORBIT", 11)
        else:
            user_incar_settings.setdefault("LORBIT", 0)

        if single_point:
            user_incar_settings.setdefault("NSW", 0)
            user_incar_settings.setdefault("IBRION", -1)
        else:
            user_incar_settings.setdefault("NSW", 500)
            user_incar_settings.setdefault("EDIFFG", -0.02)

        if calc_type != "gas":
            user_incar_settings.setdefault("ISMEAR", 0)
            user_incar_settings.setdefault("SIGMA", 0.1)

        # DFT-D3(BJ) toggle
        if use_d3:
            user_incar_settings.setdefault("IVDW", 12)

        # DFT+U toggle
        if use_dft_plus_u:
            user_incar_settings.setdefault("LDAU", True)
        else:
            user_incar_settings.setdefault("LDAU", False)

        return user_incar_settings

    def _required_overrides(
        self,
        calc_type: str,
        relax_cell: bool,
        user_incar_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Task-specific INCAR overrides; wins over user values."""
        overrides: Dict[str, Any] = {}
        if relax_cell and calc_type in {"gas", "slab"}:
            raise ValueError("relax_cell=True is not allowed when calc_type is 'gas' or 'slab'.")

        if calc_type == "bulk":
            overrides["ISIF"] = 3 if relax_cell else 2
            if relax_cell and "ENCUT" in user_incar_overrides and user_incar_overrides["ENCUT"] is not None:
                try:
                    if float(user_incar_overrides["ENCUT"]) < 520:
                        import warnings
                        warnings.warn("Detected bulk cell relaxation with ENCUT < 520; may affect stress accuracy.")
                except Exception:
                    pass
        elif calc_type == "slab":
            overrides["ISIF"] = 2
        elif calc_type == "gas":
            overrides.update(
                {
                    "ISIF": 2,
                    "ISYM": 0,
                    "LREAL": False,
                    "ISMEAR": 0,
                    "SIGMA": 0.01,
                }
            )
        return overrides

    def _generate_kgrid(self, calc_type: str, k_product: int, structure: Structure) -> Tuple[int, int, int]:
        """Generate Gamma-centered k-grid so that k_i * L_i ~= k_product, min 1, force odd."""
        try:
            a_len, b_len, c_len = structure.lattice.abc
        except Exception:
            a_len = b_len = c_len = 1.0

        def _k_from_len(L: float) -> int:
            L = float(L) if L and L > 1e-8 else 1.0
            k = max(int(round(float(k_product) / L)), 1)
            if k % 2 == 0:
                k += 1
            return k

        if calc_type == "bulk":
            return (_k_from_len(a_len), _k_from_len(b_len), _k_from_len(c_len))
        if calc_type == "slab":
            return (_k_from_len(a_len), _k_from_len(b_len), 1)
        if calc_type == "gas":
            return (1, 1, 1)
        return (_k_from_len(a_len), _k_from_len(b_len), _k_from_len(c_len))
