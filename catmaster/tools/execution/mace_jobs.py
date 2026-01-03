from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def run_mace(
    structure_file: str = "POSCAR",
    fmax: float = 0.05,
    steps: int = 500,
    model: str = "medium-mpa-0",
    device: str = "auto",
) -> Dict[str, object]:
    """
    Run MACE force field relaxation in the current working directory.
    
    Args:
        structure_file: Input structure file (Must have lattice information, xyz files are NOT supported)
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        model: MACE model name
        device: Device to use (auto/cpu/cuda)
    
    Returns:
        dict with summary including final energy, convergence status, etc.
    """
    from ase.io import read, write
    from ase.optimize import BFGS
    import numpy as np
    import torch
    from mace.calculators import mace_mp
    
    # Read structure
    atoms = read(structure_file)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up calculator
    atoms.calc = mace_mp(model=model, device=device)
    
    # Run optimization
    opt = BFGS(atoms, trajectory="opt.traj", logfile="opt.log")
    opt.run(fmax=fmax, steps=steps)
    
    # Get final energy and forces
    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    max_force = float(np.max(np.abs(final_forces)))
    
    # Check convergence - optimization converged if max force < fmax
    converged = max_force < fmax
    
    # Save optimized structure using a format compatible with the input.
    # xyz files usually carry no lattice, so writing VASP would fail.
    input_suffix = Path(structure_file).suffix.lower()
    has_lattice = atoms.cell is not None and getattr(atoms.cell, "volume", 0) > 1e-6

    if has_lattice:
        write("opt.vasp", atoms, format="vasp")
        output_structure = "opt.vasp"
    else:
        write("opt.xyz", atoms, format="xyz")
        output_structure = "opt.xyz"
    
    # Create summary
    summary = {
        "device": device,
        "model": model,
        "final_energy_eV": final_energy,
        "fmax": fmax,
        "max_force": max_force,
        "steps": steps,
        "converged": converged,
        "nsteps": opt.nsteps,
        "output_structure": output_structure,
    }
    
    # Save summary to file
    try:
        Path("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    
    return {"summary": summary}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run a MACE relaxation in-place.")
    parser.add_argument("--structure", default="POSCAR", help="Structure file name")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold (eV/Å)")
    parser.add_argument("--steps", type=int, default=500, help="Maximum optimization steps")
    parser.add_argument("--model", default="medium-mpa-0", help="MACE model name")
    parser.add_argument("--device", default="auto", help="Device to use: auto|cpu|cuda|cuda:0")
    args = parser.parse_args()
    
    result = run_mace(
        structure_file=args.structure,
        fmax=args.fmax,
        steps=args.steps,
        model=args.model,
        device=args.device,
    )
    summary = result.get("summary", {})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
