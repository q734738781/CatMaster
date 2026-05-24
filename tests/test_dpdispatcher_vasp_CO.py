#!/usr/bin/env python3
"""
DPDispatcher demo: VASP relaxation of CO through generic remote_submission_batch.
- Uses prepared inputs from tests/assets/CO_VASP_inputs
- Dry-run by default; add --run to actually submit
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


def stage_inputs(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    vasp_root = workspace / "vasp_inputs"
    if vasp_root.exists():
        shutil.rmtree(vasp_root)
    vasp_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS / "CO_VASP_inputs", vasp_root / "CO")
    return vasp_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CO VASP via DPDispatcher")
    parser.add_argument("--workspace", default="demo_vasp_CO", help="Workspace under project files root")
    parser.add_argument("--check-interval", type=int, default=60, help="Polling interval seconds")
    parser.add_argument("--run", action="store_true", help="Actually submit; default prints payload")
    args = parser.parse_args()

    workspace = resolve_workspace_path(args.workspace)
    input_root = stage_inputs(workspace)
    output_root = workspace / "vasp_outputs"

    payload = {
        "work_dir": str(input_root),
        "task_name": "vasp_execute",
        "config": {"check_interval": args.check_interval},
    }

    print("Planned payload (CO VASP batch):")
    pprint(payload)
    print("Resolved files-scope workspace:", workspace)

    if not args.run:
        print("Dry-run only. Use --run to submit via DPDispatcher.")
        return

    result = remote_submission_batch(payload)
    print("\nSubmission result:")
    pprint(result)
    print("\nOutput root:", output_root)


if __name__ == "__main__":
    main()
