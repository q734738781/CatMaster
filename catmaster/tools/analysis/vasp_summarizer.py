"""
Utility to extract structured summaries from VASP runs.  recommend_device: CPU
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pymatgen.io.vasp.outputs import Vasprun


def summarize_vasp(work_dir: Path) -> Dict[str, Any]:
    """recommend_device: CPU"""
    vasprun_path = work_dir / "vasprun.xml"
    if not vasprun_path.exists():
        raise FileNotFoundError(f"{vasprun_path} is missing.")

    vasprun = Vasprun(
        filename=str(vasprun_path),
        parse_potcar_file=False,
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
    )

    outcar_path = work_dir / "OUTCAR"
    summary: Dict[str, Any] = {
        "formula": vasprun.final_structure.composition.reduced_formula,
        "nsites": len(vasprun.final_structure),
        "final_energy_eV": vasprun.final_energy,
        "final_energy_per_atom_eV": vasprun.final_energy / len(vasprun.final_structure),
        "efermi": vasprun.efermi,
        "is_spin": vasprun.is_spin,
        "ionic_steps": len(vasprun.ionic_steps),
        "electronic_steps_per_ionic": [len(step["electronic_steps"]) for step in vasprun.ionic_steps],
        "converged_electronic": vasprun.converged_electronic,
        "converged_ionic": vasprun.converged_ionic,
        "potcar_symbols": vasprun.potcar_symbols,
        "outcar_present": outcar_path.exists(),
    }
    if vasprun.converged_ionic:
        summary["final_structure"] = vasprun.final_structure.as_dict()
    return summary


def write_summary(work_dir: Path, output_path: Path) -> Dict[str, Any]:
    """recommend_device: CPU"""
    summary = summarize_vasp(work_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


