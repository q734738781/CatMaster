from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import os


def _collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    skip_prefixes = ("mace_batch_", "mace_sp_batch_", "vasp_batch_")
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


def _run_mace_single_point(
    structure_path: Path,
    output_dir: Path,
    *,
    model: str,
    head: Optional[str],
    dispersion: bool,
    device: str,
    calc=None,
) -> Dict[str, object]:
    from ase.io import read, write
    import numpy as np
    import torch
    from mace.calculators import mace_mp

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(str(structure_path))

    if calc is None:
        kwargs = {"model": model, "dispersion": dispersion, "device": device}
        if head:
            kwargs["head"] = head
        calc = mace_mp(**kwargs)

    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    max_force_abs = float(np.max(np.abs(forces)))

    has_lattice = atoms.cell is not None and getattr(atoms.cell, "volume", 0) > 1e-6
    if has_lattice:
        output_structure = output_dir / "sp.vasp"
        write(str(output_structure), atoms, format="vasp")
    else:
        output_structure = output_dir / "sp.xyz"
        write(str(output_structure), atoms, format="xyz")

    summary = {
        "device": device,
        "model": model,
        "head": head,
        "dispersion": dispersion,
        "energy_eV": energy,
        "max_force_abs_eVA": max_force_abs,
        "output_structure": output_structure.name,
    }
    try:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {"summary": summary, "output_dir": str(output_dir)}


def run_mace_sp_batch(
    input_path: str,
    *,
    model: str = "mh-1",
    head: Optional[str] = "omat_pbe",
    dispersion: bool = False,
    device: str = "auto",
    output_root: Optional[str] = None,
) -> Dict[str, object]:
    input_root = Path(input_path)
    output_root_path = Path(output_root) if output_root else None
    if not input_root.is_dir():
        raise ValueError("input_path must be a directory for mace_sp_batch.")
    if output_root_path is None:
        raise ValueError("output_root is required for mace_sp_batch.")
    input_resolved = input_root.resolve()
    output_resolved = output_root_path.resolve()
    if _is_within(output_resolved, input_resolved):
        raise ValueError("output_root must not be inside input_path.")

    structures = _collect_structure_files(input_root)
    if not structures:
        raise ValueError(f"No structure files found in directory: {input_root}")
    output_root_path.mkdir(parents=True, exist_ok=True)

    from mace.calculators import mace_mp
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"model": model, "dispersion": dispersion, "device": device}
    if head:
        kwargs["head"] = head
    calc = mace_mp(**kwargs)

    results = []
    errors = []
    for struct in structures:
        rel_path = struct.relative_to(input_root)
        rel_dir = rel_path.with_suffix("")
        out_dir = output_root_path / rel_dir
        try:
            res = _run_mace_single_point(
                structure_path=struct,
                output_dir=out_dir,
                model=model,
                head=head,
                dispersion=dispersion,
                device=device,
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
        "input_root": str(input_root),
        "output_root": str(output_root_path),
        "model": model,
        "head": head,
        "dispersion": dispersion,
        "device": device,
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


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run MACE single-point batch for a directory of structures.")
    parser.add_argument("--input", required=True, help="Input root directory")
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
    parser.add_argument("--device", default="auto", help="Device to use: auto|cpu|cuda|cuda:0")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    args = parser.parse_args()

    head = args.head.strip()
    if head == "":
        head = None
    result = run_mace_sp_batch(
        input_path=args.input,
        model=args.model,
        head=head,
        dispersion=args.dispersion,
        device=args.device,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
