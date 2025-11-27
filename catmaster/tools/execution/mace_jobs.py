from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from jobflow import job


@job
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
        structure_file: Input structure file name (default: POSCAR)
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
    
    # Save optimized structure
    write("CONTCAR", atoms, format="vasp")
    
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
    }
    
    # Save summary to file
    try:
        Path("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    
    return {"summary": summary}

