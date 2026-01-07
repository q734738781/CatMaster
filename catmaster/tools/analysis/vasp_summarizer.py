"""
Utility to extract structured summaries from VASP runs.  recommend_device: CPU
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp import Poscar
from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_workspace_path


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
        "final_structure_path": str((work_dir / "CONTCAR").resolve()),
    }
    return summary


def write_summary(work_dir: Path, output_path: Path) -> Dict[str, Any]:
    """recommend_device: CPU"""
    summary = summarize_vasp(work_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


class VaspSummarizeInput(BaseModel):
    """Summarize a VASP run by parsing vasprun.xml in the given work directory."""
    work_dir: str = Field(..., description="Directory containing VASP outputs (must include vasprun.xml).")


def vasp_summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured summary from a VASP run directory.

    Pydantic Args Schema: VaspSummarizeInput
    Returns: dict with energies, convergence flags, final structure (if converged), and metadata
    """
    params = VaspSummarizeInput(**payload)
    work_dir = str(resolve_workspace_path(params.work_dir))

    try:
        summary = summarize_vasp(Path(work_dir))
        return create_tool_output(
            tool_name="vasp_summarize",
            success=True,
            data=summary,
        )
    except Exception as e:
        return create_tool_output(
            tool_name="vasp_summarize",
            success=False,
            error=str(e),
        )


__all__ = ["VaspSummarizeInput", "vasp_summarize", "summarize_vasp", "write_summary"]
