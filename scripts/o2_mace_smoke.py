from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catmaster.remote.gpu.mace_relax import run_mace_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal local O2 MACE geometry-optimization smoke test.")
    parser.add_argument(
        "--workspace",
        default="tmp_o2_mace_public",
        help="Workspace directory used for staged input and output.",
    )
    parser.add_argument(
        "--model",
        default="medium-omat-0",
        help="Public MACE foundation model to use for the local smoke test.",
    )
    parser.add_argument(
        "--head",
        default="",
        help="Optional model head. Use an empty string for none.",
    )
    parser.add_argument("--device", default="cpu", help="Device to use: auto|cpu|cuda|cuda:0")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force threshold in eV/Angstrom.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimization steps.")
    parser.add_argument(
        "--target-bond-length",
        type=float,
        default=1.23,
        help="Reference O-O bond length in Angstrom for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.workspace).expanduser().resolve()
    input_root = root / "input"
    output_root = root / "output"

    if root.exists():
        shutil.rmtree(root)
    input_root.mkdir(parents=True, exist_ok=True)

    source_poscar = REPO_ROOT / "tests" / "assets" / "O2_VASP_inputs" / "POSCAR"
    staged_input = input_root / "O2.vasp"
    shutil.copy2(source_poscar, staged_input)

    initial_atoms = read(staged_input)
    initial_distance = float(initial_atoms.get_distance(0, 1))

    result = run_mace_path(
        input_path=str(input_root),
        output_root=str(output_root),
        fmax=float(args.fmax),
        steps=int(args.steps),
        model=str(args.model),
        head=str(args.head or "") or None,
        dispersion=False,
        relax_lattice=False,
        device=str(args.device),
    )

    relaxed_path = output_root / "O2" / "opt.vasp"
    relaxed_atoms = read(relaxed_path)
    final_distance = float(relaxed_atoms.get_distance(0, 1))

    payload = {
        "workspace": str(root),
        "input_structure": str(staged_input),
        "relaxed_structure": str(relaxed_path),
        "model": str(args.model),
        "head": str(args.head or "") or None,
        "device": str(args.device),
        "fmax": float(args.fmax),
        "steps": int(args.steps),
        "initial_bond_length_A": initial_distance,
        "final_bond_length_A": final_distance,
        "target_bond_length_A": float(args.target_bond_length),
        "abs_error_A": abs(final_distance - float(args.target_bond_length)),
        "run_summary": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
