#!/usr/bin/env python3
"""
DPDispatcher demo: MACE relaxation of O2 on gpu_server.
- Starts from POSCAR in tests/assets/O2_in_the_box
- Dry-run by default; add --run to actually submit
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from pprint import pprint

from catmaster.tools.execution import mace_relax

ASSETS = Path(__file__).resolve().parents[1] / "tests" / "assets" / "O2_in_the_box"


def stage_poscar(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    dest_dir = workspace / "mace_o2"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ASSETS / "POSCAR", dest_dir / "POSCAR")
    return dest_dir / "POSCAR"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run O2 MACE relax via DPDispatcher")
    parser.add_argument("--workspace", default="workspace/demo_mace_o2", help="Local workspace root")
    parser.add_argument("--run", action="store_true", help="Actually submit; default prints payload")
    args = parser.parse_args()

    structure = stage_poscar(Path(args.workspace))

    payload = {
        "structure_file": str(structure),
        "work_dir": str(structure.parent),
        "fmax": 0.05,
        "maxsteps": 400,
        "model": None,  # use router default
        "check_interval": 30,
    }

    print("Planned payload (gpu_server MACE):")
    pprint(payload)

    if not args.run:
        print("Dry-run only. Use --run to submit via DPDispatcher.")
        return

    result = mace_relax(payload)
    print("\nSubmission result:")
    pprint(result)
    print("\nDownloaded results directory:", result.get("data", {}).get("download_path"))


if __name__ == "__main__":
    main()
