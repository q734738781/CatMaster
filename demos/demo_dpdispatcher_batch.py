#!/usr/bin/env python3
"""
Demonstrate submitting multiple DPDispatcher jobs in batch mode:
- One batch submission for two VASP relaxations (CO, O2) on cpu_hpc/vasp_cpu
- One batch submission for two MACE relaxations (CO, O2) on gpu_server/mace_gpu

Each batch uses task_work_path to separate tasks inside a single work_base.

Usage:
  python demos/demo_dpdispatcher_batch.py --run   # actually submit
  python demos/demo_dpdispatcher_batch.py         # dry-run, print payloads
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from pprint import pprint

from catmaster.tools.execution.dpdispatcher_runner import (
    TaskSpec,
    BatchDispatchRequest,
    dispatch_submission,
    make_work_base,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests" / "assets"


def stage_vasp_inputs(workspace: Path, name: str, src_dir: Path) -> Path:
    dest = workspace / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src_dir, dest)
    return dest


def stage_mace_structure(workspace: Path, name: str, src_file: Path) -> Path:
    dest_dir = workspace / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_file.name
    shutil.copy(src_file, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DPDispatcher demo for VASP + MACE on CO/O2")
    parser.add_argument("--workspace", default="workspace/demo_dpdispatcher_batch", help="Workspace root")
    parser.add_argument("--run", action="store_true", help="Actually submit jobs; otherwise dry-run")
    parser.add_argument("--disable_vasp", action="store_true", help="Disable VASP batch test")
    parser.add_argument("--disable_mace", action="store_true", help="Disable MACE batch test")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # ---------- VASP batch ----------
    vasp_base = make_work_base("vasp_batch")
    vasp_root = workspace / vasp_base
    (vasp_root).mkdir(parents=True, exist_ok=True)
    co_vasp = stage_vasp_inputs(vasp_root, "CO", ASSETS / "CO_VASP_inputs")
    o2_vasp = stage_vasp_inputs(vasp_root, "O2", ASSETS / "O2_VASP_inputs")

    vasp_tasks = []
    for name in ["CO", "O2"]:
        task_dir = vasp_root / name
        forwards = [p.name for p in task_dir.iterdir() if p.is_file()]
        vasp_tasks.append(
            TaskSpec(
                command="mpirun -n $SLURM_NTASKS vasp_std > vasp_stdout.txt 2>&1",
                task_work_path=name,
                forward_files=forwards,
                backward_files=["OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml"],
            )
        )
    vasp_batch = BatchDispatchRequest(
        machine="cpu_hpc",
        resources="vasp_cpu",
        work_base=vasp_base,
        local_root=str(workspace),
        tasks=vasp_tasks,
        forward_common_files=[],
        backward_common_files=[],
        clean_remote=False,
        check_interval=60,
    )

    # ---------- MACE batch ----------
    mace_base = make_work_base("mace_batch")
    mace_root = workspace / mace_base
    mace_root.mkdir(parents=True, exist_ok=True)
    co_struct = stage_mace_structure(mace_root, "CO", ASSETS / "CO_VASP_inputs" / "POSCAR")
    o2_struct = stage_mace_structure(mace_root, "O2", ASSETS / "O2_VASP_inputs" / "POSCAR")

    mace_tasks = []
    for struct in [co_struct, o2_struct]:
        task_dir = struct.parent
        forwards = [p.name for p in task_dir.iterdir() if p.is_file()]
        mace_tasks.append(
            TaskSpec(
                command=f"python -m catmaster.tools.execution.mace_jobs --structure {struct.name} --fmax 0.05 --steps 300 --model medium-mpa-0",
                task_work_path=task_dir.name,
                forward_files=forwards,
                backward_files=["opt.*", "summary.json", "opt.log", "opt.traj"],
            )
        )
    mace_batch = BatchDispatchRequest(
        machine="gpu_server",
        resources="mace_gpu",
        work_base=mace_base,
        local_root=str(workspace),
        tasks=mace_tasks,
        forward_common_files=[],
        backward_common_files=[],
        clean_remote=False,
        check_interval=10,
    )
    if not args.disable_vasp:
        print("\nVASP Batch Request:")
        pprint(vasp_batch.model_dump())
    if not args.disable_mace:
        print("\nMACE Batch Request:")
        pprint(mace_batch.model_dump())

    if not args.run:
        print("\nDry-run only. Use --run to submit.")
        return

    if not args.disable_vasp:
        print("\nSubmitting VASP batch...")
        res_vasp = dispatch_submission(vasp_batch)
        print("VASP batch result:")
        pprint(res_vasp.model_dump())

    if not args.disable_mace:
        print("\nSubmitting MACE batch...")
        res_mace = dispatch_submission(mace_batch)
        print("MACE batch result:")
        pprint(res_mace.model_dump())


if __name__ == "__main__":
    main()
