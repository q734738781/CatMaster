#!/usr/bin/env python3
"""
Demonstrate generic remote submission tools for DPDispatcher:
- remote_submission_batch with task_name=vasp_execute on prepared VASP stage subdirectories
- remote_submission_batch with task_name=mace_relax_dir on prepared MACE stage subdirectories

Usage:
  python tests/test_dpdispatcher_batch.py --run   # actually submit
  python tests/test_dpdispatcher_batch.py         # dry-run, print payloads
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from pprint import pprint

from catmaster.tools.base import resolve_workspace_path
from catmaster.tools.execution import remote_submission_batch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests" / "assets"


def _parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text}")


def stage_vasp_inputs(root: Path) -> Path:
    vasp_root = root / "vasp_inputs"
    if vasp_root.exists():
        shutil.rmtree(vasp_root)
    vasp_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS / "CO_VASP_inputs", vasp_root / "CO")
    shutil.copytree(ASSETS / "O2_VASP_inputs", vasp_root / "O2")
    return vasp_root


def stage_mace_structures(root: Path) -> Path:
    mace_root = root / "mace_stage"
    if mace_root.exists():
        shutil.rmtree(mace_root)
    mace_root.mkdir(parents=True, exist_ok=True)
    for name in ("CO", "O2"):
        task_dir = mace_root / name
        input_dir = task_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / f"{name}.vasp").write_bytes((ASSETS / f"{name}_VASP_inputs" / "POSCAR").read_bytes())
    return mace_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DPDispatcher demo for VASP + MACE on CO/O2")
    parser.add_argument("--workspace", default="test_dpdispatcher_batch", help="Workspace under project files root")
    parser.add_argument("--run", action="store_true", help="Actually submit jobs; otherwise dry-run")
    parser.add_argument("--disable_vasp", action="store_true", help="Disable VASP batch test")
    parser.add_argument("--disable_mace", action="store_true", help="Disable MACE batch test")
    parser.add_argument(
        "--mace-relax-lattice",
        type=_parse_bool,
        default=False,
        help="Relax lattice/cell in MACE batch path (true|false).",
    )
    args = parser.parse_args()

    workspace = resolve_workspace_path(args.workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    print("Resolved files-scope workspace:", workspace)

    vasp_payload = None
    mace_payload = None

    if not args.disable_vasp:
        vasp_root = stage_vasp_inputs(workspace)
        vasp_output = workspace / "vasp_outputs"
        vasp_payload = {
            "work_dir": str(vasp_root),
            "task_name": "vasp_execute",
            "submission_config": {"check_interval": 60},
        }
        print("\nVASP Batch Payload:")
        pprint(vasp_payload)

    if not args.disable_mace:
        mace_root = stage_mace_structures(workspace)
        mace_output = workspace / "mace_outputs"
        mace_payload = {
            "work_dir": str(mace_root),
            "task_name": "mace_relax_dir",
            "template_overrides": {
                "fmax": 0.05,
                "steps": 300,
                "model": "mh-1",
                "head": "omat_pbe",
                "relax_lattice": args.mace_relax_lattice,
            },
            "submission_config": {"check_interval": 10},
        }
        print("\nMACE Batch Payload:")
        pprint(mace_payload)

    if not args.run:
        print("\nDry-run only. Use --run to submit.")
        return

    if vasp_payload:
        print("\nSubmitting VASP batch...")
        res_vasp = remote_submission_batch(vasp_payload)
        print("VASP batch result:")
        pprint(res_vasp)

    if mace_payload:
        print("\nSubmitting MACE batch...")
        res_mace = remote_submission_batch(mace_payload)
        print("MACE batch result:")
        pprint(res_mace)


if __name__ == "__main__":
    main()
