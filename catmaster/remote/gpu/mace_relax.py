from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import os


def _resolve_device(preference: str) -> str:
    import torch

    device = str(preference or "auto").strip().lower() or "auto"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested but torch.cuda.is_available() is false; "
            "check the remote driver/CUDA/PyTorch environment."
        )
    return device


def run_mace(
    structure_file: str = "POSCAR",
    fmax: float = 0.05,
    steps: int = 500,
    model: str = "mh-1",
    head: Optional[str] = "omat_pbe",
    dispersion: bool = False,
    relax_lattice: bool = False,
    device: str = "auto",
    default_dtype: str = "float64",
    enable_cueq: bool = False,
    output_root: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run MACE force field relaxation in the current working directory.
    
    Args:
        structure_file: Input structure file (Must have lattice information, xyz files are NOT supported)
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        model: MACE model name
        head: MACE model head for multi-head models; use None to disable.
        dispersion: Enable D3 dispersion correction in mace_mp.
        relax_lattice: Whether to relax cell vectors together with atomic positions.
        device: Device to use (auto/cpu/cuda)
        default_dtype: Floating-point precision for the MACE calculator.
        enable_cueq: Enable cuEquivariance acceleration on CUDA.
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
        head=head,
        dispersion=dispersion,
        relax_lattice=relax_lattice,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
    )


def _collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    skip_prefixes = ("mace_batch_", "vasp_batch_")
    internal_dirs = {"metadata", ".catmaster"}
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part.startswith(skip_prefixes) for part in path.parts):
            dirnames[:] = []
            continue
        if any(part in internal_dirs for part in path.parts):
            dirnames[:] = []
            continue
        if "summary.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in internal_dirs and not d.startswith(skip_prefixes)
        ]
        for fname in filenames:
            p = path / fname
            if fname in {"POSCAR", "CONTCAR"}:
                files.append(p)
                continue
            if p.suffix.lower() in {".vasp", ".poscar", ".cif"}:
                files.append(p)
    return sorted(files, key=lambda p: str(p))


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
    head: Optional[str],
    dispersion: bool,
    relax_lattice: bool,
    device: str,
    default_dtype: str,
    enable_cueq: bool,
    calc=None,
) -> Dict[str, object]:
    from ase.io import read, write
    from ase.filters import FrechetCellFilter
    from ase.io.trajectory import Trajectory
    from ase.optimize import FIRE
    import numpy as np
    device = _resolve_device(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(str(structure_path))
    if calc is None:
        calc, device = _make_calculator(
            model=model,
            head=head,
            dispersion=dispersion,
            device=device,
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
        )
    atoms.calc = calc

    traj_path = output_dir / "opt.traj"
    log_path = output_dir / "opt.log"
    has_lattice = atoms.cell is not None and getattr(atoms.cell, "volume", 0) > 1e-6
    if relax_lattice and not has_lattice:
        raise ValueError("relax_lattice=True requires periodic structure with a valid lattice.")
    target = FrechetCellFilter(atoms) if relax_lattice else atoms
    traj = Trajectory(str(traj_path), "w", atoms)
    opt = FIRE(target, logfile=str(log_path))
    opt.attach(traj)
    try:
        opt.run(fmax=fmax, steps=steps)
    finally:
        traj.close()

    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    max_force = float(np.max(np.abs(final_forces)))
    converged = max_force < fmax

    if has_lattice:
        output_structure = output_dir / "opt.vasp"
        write(str(output_structure), atoms, format="vasp")
    else:
        output_structure = output_dir / "opt.xyz"
        write(str(output_structure), atoms, format="xyz")

    summary = {
        "device": device,
        "model": model,
        "head": head,
        "dispersion": dispersion,
        "default_dtype": default_dtype,
        "enable_cueq": enable_cueq,
        "relax_lattice": relax_lattice,
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
    model: str = "mh-1",
    head: Optional[str] = "omat_pbe",
    dispersion: bool = False,
    relax_lattice: bool = False,
    device: str = "auto",
    default_dtype: str = "float64",
    enable_cueq: bool = False,
    output_root: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run MACE relaxation for a directory of structure files.

    - input_path must be a directory.
    - output_root is required.
    - Output mirrors the input directory structure and expands file names into folders
      (e.g. input_root/a/c.vasp -> output_root/a/c/opt.vasp).
    """
    input_path = Path(input_path)
    output_root_path = Path(output_root) if output_root else None
    if not input_path.is_dir():
        raise ValueError("input_path must be a directory for batch relaxation.")
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
    calc, device = _make_calculator(
        model=model,
        head=head,
        dispersion=dispersion,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
    )

    results = []
    errors = []
    for struct in structures:
        rel_path = struct.relative_to(input_path)
        rel_dir = rel_path.with_suffix("")
        out_dir = output_root_path / rel_dir
        try:
            res = _run_mace_single(
                structure_path=struct,
                output_dir=out_dir,
                fmax=fmax,
                steps=steps,
                model=model,
                head=head,
                dispersion=dispersion,
                relax_lattice=relax_lattice,
                device=device,
                default_dtype=default_dtype,
                enable_cueq=enable_cueq,
                calc=calc,
            )
            results.append(
                {
                    "input_rel": str(rel_path),
                    "output_rel": str(out_dir.relative_to(output_root_path)),
                    "summary": res.get("summary", {}),
                }
            )
        except Exception as exc:
            errors.append({"input_rel": str(rel_path), "error": str(exc)})

    batch_summary = {
        "input_root": str(input_path),
        "output_root": str(output_root_path),
        "model": model,
        "head": head,
        "dispersion": dispersion,
        "default_dtype": default_dtype,
        "enable_cueq": enable_cueq,
        "relax_lattice": relax_lattice,
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


def _make_calculator(
    *,
    model: str,
    head: Optional[str],
    dispersion: bool,
    device: str,
    default_dtype: str,
    enable_cueq: bool,
):
    from mace.calculators import mace_mp

    resolved_device = _resolve_device(device)
    kwargs = {
        "model": model,
        "dispersion": dispersion,
        "device": resolved_device,
        "default_dtype": default_dtype,
    }
    if head:
        kwargs["head"] = head
    if enable_cueq:
        if not resolved_device.startswith("cuda"):
            raise ValueError("enable_cueq requires a CUDA device.")
        kwargs["enable_cueq"] = True
    return mace_mp(**kwargs), resolved_device


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run MACE relaxations for a directory of structures.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold (eV/Å)")
    parser.add_argument("--steps", type=int, default=500, help="Maximum optimization steps")
    parser.add_argument("--model", default="mh-1", help="MACE model name")
    parser.add_argument(
        "--head",
        default="omat_pbe",
        help="Model head for multi-head models (e.g. 'omat_pbe'). Use '' for none.",
    )
    parser.add_argument(
        "--dispersion",
        type=_parse_bool,
        default=False,
        help="Enable dispersion correction in mace_mp (true|false). Default: false.",
    )
    parser.add_argument(
        "--relax_lattice",
        type=_parse_bool,
        default=False,
        help="Relax lattice/cell together with atomic positions (true|false). Default: false.",
    )
    parser.add_argument(
        "--default_dtype",
        default="float64",
        choices=("float32", "float64"),
        help="Floating-point precision for the MACE calculator.",
    )
    parser.add_argument(
        "--enable_cueq",
        type=_parse_bool,
        default=False,
        help="Enable cuEquivariance acceleration on CUDA (true|false). Default: false.",
    )
    parser.add_argument("--device", default="auto", help="Device to use: auto|cpu|cuda|cuda:0")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    args = parser.parse_args()

    input_path = args.input
    head = args.head.strip()
    if head == "":
        head = None
    result = run_mace_path(
        input_path=input_path,
        fmax=args.fmax,
        steps=args.steps,
        model=args.model,
        head=head,
        dispersion=args.dispersion,
        relax_lattice=args.relax_lattice,
        device=args.device,
        default_dtype=args.default_dtype,
        enable_cueq=args.enable_cueq,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
