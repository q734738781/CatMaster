from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import os


def run_mace(
    structure_file: str = "POSCAR",
    fmax: float = 0.05,
    steps: int = 500,
    model: str = "medium-mpa-0",
    device: str = "auto",
    output_root: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run MACE force field relaxation in the current working directory.
    
    Args:
        structure_file: Input structure file (Must have lattice information, xyz files are NOT supported)
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        model: MACE model name
        device: Device to use (auto/cpu/cuda)
        output_root: Directory to write outputs; defaults to the structure file's directory.
    
    Returns:
        dict with summary including final energy, convergence status, etc.
    """
    return _run_mace_single(
        structure_path=Path(structure_file),
        output_dir=Path(output_root) if output_root else Path(structure_file).resolve().parent,
        fmax=fmax,
        steps=steps,
        model=model,
        device=device,
    )


def _collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    skip_prefixes = ("mace_batch_", "vasp_batch_")
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part.startswith(skip_prefixes) for part in path.parts):
            dirnames[:] = []
            continue
        if ".catmaster" in path.parts:
            dirnames[:] = []
            continue
        if "summary.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d != ".catmaster" and not d.startswith(skip_prefixes)
        ]
        for fname in filenames:
            p = path / fname
            if fname in {"POSCAR", "CONTCAR"}:
                files.append(p)
                continue
            if p.suffix.lower() in {".vasp", ".poscar", ".cif"}:
                files.append(p)
    return sorted(files, key=lambda p: str(p))


def _unique_output_dir(base_dir: Path, stem: str) -> Path:
    candidate = base_dir / stem
    idx = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_run{idx}"
        idx += 1
    return candidate


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _run_mace_single(
    structure_path: Path,
    output_dir: Path,
    fmax: float,
    steps: int,
    model: str,
    device: str,
    calc=None,
) -> Dict[str, object]:
    from ase.io import read, write
    from ase.optimize import BFGS
    import numpy as np
    import torch
    from mace.calculators import mace_mp

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(str(structure_path))
    if calc is None:
        calc = mace_mp(model=model, device=device)
    atoms.calc = calc

    traj_path = output_dir / "opt.traj"
    log_path = output_dir / "opt.log"
    opt = BFGS(atoms, trajectory=str(traj_path), logfile=str(log_path))
    opt.run(fmax=fmax, steps=steps)

    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    max_force = float(np.max(np.abs(final_forces)))
    converged = max_force < fmax

    has_lattice = atoms.cell is not None and getattr(atoms.cell, "volume", 0) > 1e-6
    if has_lattice:
        output_structure = output_dir / "opt.vasp"
        write(str(output_structure), atoms, format="vasp")
    else:
        output_structure = output_dir / "opt.xyz"
        write(str(output_structure), atoms, format="xyz")

    summary = {
        "device": device,
        "model": model,
        "final_energy_eV": final_energy,
        "fmax": fmax,
        "max_force": max_force,
        "steps": steps,
        "converged": converged,
        "nsteps": opt.nsteps,
        "output_structure": output_structure.name,
    }

    try:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"summary": summary, "output_dir": str(output_dir)}


def run_mace_path(
    input_path: str,
    fmax: float = 0.05,
    steps: int = 500,
    model: str = "medium-mpa-0",
    device: str = "auto",
    output_root: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run MACE relaxation for a file or a directory.

    - If input_path is a file, writes outputs to output_root if provided, otherwise to the file's directory.
    - If input_path is a directory, output_root is required and outputs are flattened under output_root using "__"
      to encode the relative path without file suffix (e.g. a/b/CO.vasp -> a__b__CO).
    """
    input_path = Path(input_path)
    output_root_path = Path(output_root) if output_root else None

    if input_path.is_dir():
        if output_root_path is None:
            raise ValueError("output_root is required when input_path is a directory.")
        input_resolved = input_path.resolve()
        output_resolved = output_root_path.resolve()
        if _is_within(output_resolved, input_resolved):
            raise ValueError("output_root must not be inside input_path.")
        structures = _collect_structure_files(input_path)
        if not structures:
            raise ValueError(f"No structure files found in directory: {input_path}")
        output_root_path.mkdir(parents=True, exist_ok=True)
        from mace.calculators import mace_mp
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_mp(model=model, device=device)

        results = []
        errors = []
        for struct in structures:
            rel_path = struct.relative_to(input_path).with_suffix("")
            slab_id = "__".join(rel_path.parts)
            base_dir = output_root_path
            out_dir = _unique_output_dir(base_dir, slab_id)
            try:
                res = _run_mace_single(
                    structure_path=struct,
                    output_dir=out_dir,
                    fmax=fmax,
                    steps=steps,
                    model=model,
                    device=device,
                    calc=calc,
                )
                results.append(
                    {
                        "input_rel": str(struct.relative_to(input_path)),
                        "slab_id": slab_id,
                        "output_rel": str(out_dir.relative_to(output_root_path)),
                        "summary": res.get("summary", {}),
                    }
                )
            except Exception as exc:
                errors.append({"input_rel": str(struct.relative_to(input_path)), "error": str(exc)})

        batch_summary = {
            "input_root": str(input_path),
            "output_root": str(output_root_path),
            "model": model,
            "device": device,
            "fmax": fmax,
            "steps": steps,
            "results": results,
            "errors": errors,
        }
        try:
            (output_root_path / "batch_summary.json").write_text(
                json.dumps(batch_summary, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        return batch_summary

    output_dir = output_root_path if output_root_path else input_path.resolve().parent
    return _run_mace_single(
        structure_path=input_path,
        output_dir=output_dir,
        fmax=fmax,
        steps=steps,
        model=model,
        device=device,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run a MACE relaxation in-place.")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--structure", default="POSCAR", help="Structure file name")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold (eV/Å)")
    parser.add_argument("--steps", type=int, default=500, help="Maximum optimization steps")
    parser.add_argument("--model", default="medium-mpa-0", help="MACE model name")
    parser.add_argument("--device", default="auto", help="Device to use: auto|cpu|cuda|cuda:0")
    parser.add_argument("--output_root", default=None, help="Output root directory")
    args = parser.parse_args()

    input_path = args.input or args.structure
    result = run_mace_path(
        input_path=input_path,
        fmax=args.fmax,
        steps=args.steps,
        model=args.model,
        device=args.device,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
