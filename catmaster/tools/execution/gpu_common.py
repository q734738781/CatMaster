"""Shared GPU-side helpers reused across algorithms and tools."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Dict, List, Optional

from ase import Atoms
from ase.constraints import Hookean
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import mace_mp


def make_mace_calc(
    *,
    model: str,
    use_d3: bool,
    device: str,
    amp_mode: str = "auto",
    amp_dtype: str = "fp16",
    tf32: bool = False,
) -> Calculator:
    """Instantiate the configured MACE calculator with optional AMP autocast."""
    # TODO: Remove AMP autocast as it do not accelerate the calculation but make the code more complex.
    base = mace_mp(model=model, dispersion=use_d3, device=device)
    if device.startswith("cuda"):
        try:
            import torch
            if tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            if amp_mode in ("on", "auto"):
                from ase.calculators.calculator import Calculator, all_changes
                import contextlib
                class _AutocastCalc(Calculator):
                    implemented_properties = getattr(base, "implemented_properties", ["energy", "forces", "stress"])
                    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            base.calculate(atoms=atoms, properties=properties, system_changes=system_changes)
                        self.results = dict(base.results)
                return _AutocastCalc()
        except Exception:
            pass
    return base


def load_meta(path_poscar: str) -> Optional[Dict]:
    meta_path = path_poscar.replace(".vasp", ".meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def hookeans_from_metadata(
    seed_poscar: str,
    *,
    constraint_entities: str = "molecule",
    bond_rt_scale: float = 0.10,
    k: float = 15.0,
) -> List[Hookean]:
    meta = load_meta(seed_poscar)
    hoops: List[Hookean] = []
    if not meta or constraint_entities == "none":
        return hoops
    want = {"molecule", "fragment"} if constraint_entities == "all" else {constraint_entities}
    for ent in meta.get("entities", []):
        role = ent.get("role", "molecule")
        if role not in want:
            continue
        for bond in ent.get("local_bonds", []):
            i_local = int(bond["i"])
            j_local = int(bond["j"])
            r0 = float(bond["r0"])
            idx_map = ent["global_indices"]
            hoops.append(
                Hookean(
                    a1=int(idx_map[i_local]),
                    a2=int(idx_map[j_local]),
                    rt=(1.0 + bond_rt_scale) * r0,
                    k=k,
                )
            )
    return hoops


def listify_constraints(constraints) -> List:
    if constraints is None:
        return []
    if isinstance(constraints, (list, tuple)):
        return list(constraints)
    return [constraints]


def apply_all_constraints(
    atoms: Atoms,
    *,
    seed_poscar: str,
    mode: str,
    constraint_entities: str,
    bond_rt_scale: float,
    hooke_k: float,
) -> None:
    """Merge Hookean constraints from metadata with any existing ASE constraints."""
    constraints = listify_constraints(atoms.constraints)
    effective = "none" if (mode == "dissociative" and constraint_entities == "auto") else constraint_entities
    if effective == "auto":
        effective = "molecule"
    if effective != "none":
        constraints.extend(
            hookeans_from_metadata(
                seed_poscar,
                constraint_entities=effective,
                bond_rt_scale=bond_rt_scale,
                k=hooke_k,
            )
        )
    if constraints:
        atoms.set_constraint(constraints)


__all__ = [
    "apply_all_constraints",
    "hookeans_from_metadata",
    "listify_constraints",
    "load_meta",
    "make_mace_calc",
]


