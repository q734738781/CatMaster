"""Canonical VASP input planning/writing helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet

from catmaster.tools.base import resolve_workspace_path

VaspPreset = Literal["relax", "static", "freq", "dos", "md", "dimer"]
VaspRegime = Literal["bulk", "slab", "gas"]
PatchPolicy = Literal["safe", "force"]


@dataclass(frozen=True)
class VaspWritePlan:
    output_dir: Path
    preset: VaspPreset
    regime: VaspRegime
    relax_cell: bool
    patch_policy: PatchPolicy
    k_product: int
    k_grid: Tuple[int, int, int]
    user_incar_settings: Dict[str, Any]
    removal_keys: Tuple[str, ...]
    protected_keys: Tuple[str, ...]


class StructWriter:
    """Plan and write VASP inputs using MPRelaxSet as the canonical backend."""

    def plan_vasp_inputs(
        self,
        *,
        structure: Structure,
        output_dir: Path,
        preset: VaspPreset = "relax",
        regime: VaspRegime = "bulk",
        relax_cell: bool = False,
        k_product: int = 35,
        use_d3: bool = False,
        user_incar_patch: Optional[Dict[str, Any]] = None,
        use_dft_plus_u: bool = False,
        compute_dos: bool = False,
        enable_dipole: bool = False,
        dos_use_chgcar: bool = False,
        patch_policy: PatchPolicy = "safe",
    ) -> VaspWritePlan:
        output_dir = self._resolve_output_dir(output_dir)
        self._validate_scope(preset=preset, regime=regime, relax_cell=relax_cell)

        canonical_settings = self._canonical_incar_settings(
            structure=structure,
            preset=preset,
            regime=regime,
            relax_cell=relax_cell,
            use_d3=use_d3,
            use_dft_plus_u=use_dft_plus_u,
            compute_dos=compute_dos,
            enable_dipole=enable_dipole,
            dos_use_chgcar=dos_use_chgcar,
        )
        protected_keys = self._protected_keys_for_scope(
            preset=preset,
            regime=regime,
            relax_cell=relax_cell,
            dos_use_chgcar=dos_use_chgcar,
        )
        merged_settings, removal_keys = self._merge_user_patch(
            canonical_settings=canonical_settings,
            user_incar_patch=user_incar_patch or {},
            patch_policy=patch_policy,
            protected_keys=protected_keys,
        )
        k_grid = self._generate_kgrid(regime, k_product, structure)
        return VaspWritePlan(
            output_dir=output_dir,
            preset=preset,
            regime=regime,
            relax_cell=relax_cell,
            patch_policy=patch_policy,
            k_product=int(k_product),
            k_grid=k_grid,
            user_incar_settings=merged_settings,
            removal_keys=tuple(sorted(removal_keys)),
            protected_keys=tuple(sorted(protected_keys)),
        )

    def write_vasp_inputs(
        self,
        *,
        structure: Structure,
        output_dir: Path,
        preset: VaspPreset = "relax",
        regime: VaspRegime = "bulk",
        relax_cell: bool = False,
        k_product: int = 35,
        use_d3: bool = False,
        user_incar_patch: Optional[Dict[str, Any]] = None,
        use_dft_plus_u: bool = False,
        compute_dos: bool = False,
        enable_dipole: bool = False,
        dos_use_chgcar: bool = False,
        patch_policy: PatchPolicy = "safe",
        run_template: Optional[Path] = None,
    ) -> VaspWritePlan:
        plan = self.plan_vasp_inputs(
            structure=structure,
            output_dir=output_dir,
            preset=preset,
            regime=regime,
            relax_cell=relax_cell,
            k_product=k_product,
            use_d3=use_d3,
            user_incar_patch=user_incar_patch,
            use_dft_plus_u=use_dft_plus_u,
            compute_dos=compute_dos,
            enable_dipole=enable_dipole,
            dos_use_chgcar=dos_use_chgcar,
            patch_policy=patch_policy,
        )

        plan.output_dir.mkdir(parents=True, exist_ok=True)
        vasp_input_set = MPRelaxSet(
            structure,
            user_potcar_functional="PBE_54",
            user_incar_settings=plan.user_incar_settings,
            sort_structure=False,
        )
        vasp_input_set.write_input(str(plan.output_dir))

        if plan.removal_keys:
            incar_path = plan.output_dir / "INCAR"
            if incar_path.exists():
                incar_obj = Incar.from_file(incar_path)
                changed = False
                for key in plan.removal_keys:
                    if key in incar_obj:
                        del incar_obj[key]
                        changed = True
                if changed:
                    incar_obj.write_file(incar_path)

        kpt = Kpoints.gamma_automatic(kpts=plan.k_grid)
        kpoints_path = plan.output_dir / "KPOINTS"
        if kpoints_path.exists():
            kpoints_path.unlink()
        kpt.write_file(kpoints_path)

        if run_template and run_template.exists():
            import shutil

            shutil.copy(run_template, plan.output_dir / "run.yaml")
        return plan

    @staticmethod
    def _resolve_output_dir(output_dir: Path) -> Path:
        candidate = Path(output_dir).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return resolve_workspace_path(str(candidate))

    @staticmethod
    def _validate_scope(*, preset: VaspPreset, regime: VaspRegime, relax_cell: bool) -> None:
        if relax_cell and not (preset == "relax" and regime == "bulk"):
            raise ValueError("relax_cell=True is only allowed for preset='relax' and regime='bulk'.")

    def _canonical_incar_settings(
        self,
        *,
        structure: Structure,
        preset: VaspPreset,
        regime: VaspRegime,
        relax_cell: bool,
        use_d3: bool,
        use_dft_plus_u: bool,
        compute_dos: bool,
        enable_dipole: bool,
        dos_use_chgcar: bool,
    ) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "EDIFF": 1e-6,
            "NELM": 150,
            "LCHARG": False,
            "LWAVE": False,
            "LORBIT": 11 if (compute_dos or preset == "dos") else 0,
            "LDAU": bool(use_dft_plus_u),
        }

        if regime == "gas":
            settings.update(
                {
                    "ISIF": 2,
                    "ISYM": 0,
                    "ISMEAR": 0,
                    "SIGMA": 0.01,
                }
            )
        elif regime == "slab":
            settings.update(
                {
                    "ISIF": 2,
                    "ISMEAR": 0,
                    "SIGMA": 0.1,
                }
            )
        else:
            settings.update(
                {
                    "ISIF": 3 if relax_cell else 2,
                    "ISMEAR": 0,
                    "SIGMA": 0.1,
                }
            )

        if preset == "relax":
            settings.update(
                {
                    "IBRION": 2,
                    "NSW": 500,
                    "EDIFFG": -0.02,
                }
            )
        elif preset == "static":
            settings.update(
                {
                    "IBRION": -1,
                    "NSW": 1,
                }
            )
        elif preset == "freq":
            settings.update(
                {
                    "IBRION": 5,
                    "NSW": 1,
                    "POTIM": 0.015,
                    "NFREE": 2,
                    "ISYM": 0,
                }
            )
        elif preset == "dos":
            settings.update(
                {
                    "IBRION": -1,
                    "NSW": 0,
                    "ISMEAR": -5,
                    "NEDOS": 2001,
                }
            )
            if dos_use_chgcar:
                settings["ICHARG"] = 11
        elif preset == "md":
            settings.update(
                {
                    "IBRION": 0,
                    "NSW": 1000,
                    "POTIM": 1.0,
                    "MDALGO": 2,
                    "SMASS": 0.0,
                    "TEBEG": 300.0,
                    "TEEND": 300.0,
                    "ISYM": 0,
                }
            )
        elif preset == "dimer":
            settings.update(
                {
                    "IBRION": 44,
                    "NSW": 500,
                    "EDIFFG": -0.02,
                }
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported preset: {preset}")

        if use_d3:
            settings["IVDW"] = 12

        if enable_dipole:
            settings["IDIPOL"] = 3
            settings["DIPOL"] = self._compute_com_frac_dipol(structure)

        return settings

    @staticmethod
    def _merge_user_patch(
        *,
        canonical_settings: Dict[str, Any],
        user_incar_patch: Dict[str, Any],
        patch_policy: PatchPolicy,
        protected_keys: set[str],
    ) -> tuple[Dict[str, Any], Tuple[str, ...]]:
        merged = dict(canonical_settings)
        removal_keys: list[str] = []

        for raw_key, raw_value in (user_incar_patch or {}).items():
            key = str(raw_key).strip().upper()
            if not key:
                raise ValueError("INCAR key must be a non-empty string.")
            if patch_policy == "safe" and key in protected_keys:
                base_value = canonical_settings.get(key)
                if raw_value != base_value:
                    raise ValueError(
                        f"user_incar_patch attempts to override protected INCAR key {key} under patch_policy='safe'. "
                        "Use patch_policy='force' if you really need to replace preset/regime-bound defaults."
                    )
            if raw_value is None:
                merged.pop(key, None)
                removal_keys.append(key)
                continue
            merged[key] = raw_value

        return merged, tuple(removal_keys)

    @staticmethod
    def _protected_keys_for_scope(
        *,
        preset: VaspPreset,
        regime: VaspRegime,
        relax_cell: bool,
        dos_use_chgcar: bool,
    ) -> set[str]:
        protected = {"IBRION"}
        if preset in {"relax", "static", "freq", "dimer"}:
            protected.update({"ISIF", "NSW"})
        if regime == "gas" or preset == "freq":
            protected.add("ISYM")
        if preset == "freq":
            protected.update({"POTIM", "NFREE"})
        if preset == "dos" and dos_use_chgcar:
            protected.add("ICHARG")
        if relax_cell:
            protected.add("ISIF")
        return protected

    def _compute_com_frac_dipol(self, structure: Structure) -> list[float]:
        weights = [float(site.species.weight) for site in structure.sites]
        total_mass = float(sum(weights))
        if total_mass <= 0:
            raise ValueError("Cannot compute DIPOL center of mass: total atomic mass is non-positive.")
        center_of_mass = np.average(structure.frac_coords, weights=weights, axis=0).tolist()
        return [float(v - math.floor(float(v))) for v in center_of_mass]

    @staticmethod
    def _generate_kgrid(regime: VaspRegime, k_product: int, structure: Structure) -> Tuple[int, int, int]:
        try:
            a_len, b_len, c_len = structure.lattice.abc
        except Exception:
            a_len = b_len = c_len = 1.0

        def _k_from_len(length: float) -> int:
            safe_length = float(length) if length and length > 1e-8 else 1.0
            k_value = max(int(round(float(k_product) / safe_length)), 1)
            if k_value % 2 == 0:
                k_value += 1
            return k_value

        if regime == "gas":
            return (1, 1, 1)
        if regime == "slab":
            return (_k_from_len(a_len), _k_from_len(b_len), 1)
        return (_k_from_len(a_len), _k_from_len(b_len), _k_from_len(c_len))
