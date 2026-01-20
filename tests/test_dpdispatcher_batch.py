#!/usr/bin/env python3
"""
Demonstrate batch tools for DPDispatcher:
- vasp_execute_batch on prepared VASP input subdirectories
- mace_relax_batch on a directory of structure files

Usage:
  python tests/test_dpdispatcher_batch.py --run   # actually submit
  python tests/test_dpdispatcher_batch.py         # dry-run, print payloads
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from pprint import pprint

from catmaster.tools.execution import vasp_execute_batch, mace_relax_batch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests" / "assets"


def stage_vasp_inputs(root: Path) -> Path:
    vasp_root = root / "vasp_inputs"
    if vasp_root.exists():
        shutil.rmtree(vasp_root)
    vasp_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS / "CO_VASP_inputs", vasp_root / "CO")
    shutil.copytree(ASSETS / "O2_VASP_inputs", vasp_root / "O2")
    return vasp_root


def stage_mace_structures(root: Path) -> Path:
    mace_root = root / "mace_inputs"
    if mace_root.exists():
        shutil.rmtree(mace_root)
    mace_root.mkdir(parents=True, exist_ok=True)
    (mace_root / "CO.vasp").write_bytes((ASSETS / "CO_VASP_inputs" / "POSCAR").read_bytes())
    (mace_root / "O2.vasp").write_bytes((ASSETS / "O2_VASP_inputs" / "POSCAR").read_bytes())
    return mace_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DPDispatcher demo for VASP + MACE on CO/O2")
    parser.add_argument("--workspace", default="workspace/test_dpdispatcher_batch", help="Workspace root")
    parser.add_argument("--run", action="store_true", help="Actually submit jobs; otherwise dry-run")
    parser.add_argument("--disable_vasp", action="store_true", help="Disable VASP batch test")
    parser.add_argument("--disable_mace", action="store_true", help="Disable MACE batch test")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    vasp_payload = None
    mace_payload = None

    if not args.disable_vasp:
        vasp_root = stage_vasp_inputs(workspace)
        vasp_output = workspace / "vasp_outputs"
        vasp_payload = {
            "input_dir": str(vasp_root),
            "output_dir": str(vasp_output),
            "check_interval": 60,
        }
        print("\nVASP Batch Payload:")
        pprint(vasp_payload)

    if not args.disable_mace:
        mace_root = stage_mace_structures(workspace)
        mace_output = workspace / "mace_outputs"
        mace_payload = {
            "input_dir": str(mace_root),
            "output_root": str(mace_output),
            "fmax": 0.05,
            "maxsteps": 300,
            "model": "medium-mpa-0",
            "check_interval": 10,
        }
        print("\nMACE Batch Payload:")
        pprint(mace_payload)

    if not args.run:
        print("\nDry-run only. Use --run to submit.")
        return

    if vasp_payload:
        print("\nSubmitting VASP batch...")
        res_vasp = vasp_execute_batch(vasp_payload)
        print("VASP batch result:")
        pprint(res_vasp)

    if mace_payload:
        print("\nSubmitting MACE batch...")
        res_mace = mace_relax_batch(mace_payload)
        print("MACE batch result:")
        pprint(res_mace)


if __name__ == "__main__":
    main()
