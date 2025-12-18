"""
Utility to extract structured summaries from VASP runs.  recommend_device: CPU
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp import Poscar


def summarize_vasp(work_dir: Path) -> Dict[str, Any]:
    """recommend_device: local"""
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
        "final_energy_eV": vasprun.final_energy,
        "ionic_steps": len(vasprun.ionic_steps),
        "converged_electronic": vasprun.converged_electronic,
        "converged_ionic": vasprun.converged_ionic,
    }
    if vasprun.converged_ionic:
        poscar_str = Poscar(vasprun.final_structure).get_string(significant_figures=6)
        summary["final_structure_poscar"] = poscar_str
    return summary


def write_summary(work_dir: Path, output_path: Path) -> Dict[str, Any]:
    """recommend_device: CPU"""
    summary = summarize_vasp(work_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary

