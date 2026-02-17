#!/usr/bin/env python3
"""
DPDispatcher demo: MACE relaxation of CO.
- Starts from tests/assets/CO_VASP_inputs/POSCAR
- Dry-run by default; add --run to actually submit
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from pprint import pprint

from catmaster.tools.base import resolve_workspace_path
from catmaster.tools.execution import mace_relax_batch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests" / "assets"

def _parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text}")


def stage_structure(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    input_root = workspace / "mace_inputs"
    if input_root.exists():
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "CO.vasp").write_bytes((ASSETS / "CO_VASP_inputs" / "POSCAR").read_bytes())
    return input_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CO MACE relax via DPDispatcher")
    parser.add_argument("--workspace", default="demo_mace_CO", help="Workspace under project files root")
    parser.add_argument("--fmax", type=float, default=0.05, help="Relaxation force threshold")
    parser.add_argument("--maxsteps", type=int, default=400, help="Max relaxation steps")
    parser.add_argument("--model", default="mh-1", help="MACE model name")
    parser.add_argument("--head", default="omat_pbe", help="MACE model head")
    parser.add_argument(
        "--relax-lattice",
        type=_parse_bool,
        default=False,
        help="Relax lattice/cell together with atomic positions (true|false).",
    )
    parser.add_argument("--check-interval", type=int, default=10, help="Polling interval seconds")
    parser.add_argument("--run", action="store_true", help="Actually submit; default prints payload")
    args = parser.parse_args()

    workspace = resolve_workspace_path(args.workspace)
    input_root = stage_structure(workspace)
    output_root = workspace / "mace_outputs"

    payload = {
        "input_dir": str(input_root),
        "output_root": str(output_root),
        "fmax": args.fmax,
        "maxsteps": args.maxsteps,
        "model": args.model,
        "head": args.head,
        "relax_lattice": args.relax_lattice,
        "check_interval": args.check_interval,
    }

    print("Planned payload (CO MACE batch):")
    pprint(payload)
    print("Resolved files-scope workspace:", workspace)

    if not args.run:
        print("Dry-run only. Use --run to submit via DPDispatcher.")
        return

    result = mace_relax_batch(payload)
    print("\nSubmission result:")
    pprint(result)
    print("\nOutput root:", output_root)


if __name__ == "__main__":
    main()
