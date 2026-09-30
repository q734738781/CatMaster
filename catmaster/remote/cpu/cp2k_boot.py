from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from .node_local_scratch import ScratchStageOutError, node_local_workdir
except ImportError:  # Staged boot scripts run as task_script/cp2k_boot.py.
    from node_local_scratch import ScratchStageOutError, node_local_workdir


def _resolve_nprocs(raw: int | None) -> int:
    if raw is not None:
        if raw <= 0:
            raise ValueError("--nprocs must be positive")
        return raw
    env_val = os.environ.get("SLURM_NTASKS")
    if not env_val:
        raise ValueError("SLURM_NTASKS is not set; pass --nprocs explicitly")
    parsed = int(str(env_val).strip())
    if parsed <= 0:
        raise ValueError(f"Invalid SLURM_NTASKS: {env_val}")
    return parsed


def _collect_outputs(logical_root: Path | None = None) -> dict[str, str]:
    names: dict[str, str] = {}
    root = Path.cwd() if logical_root is None else logical_root
    suffixes = (".out", ".xyz", ".ener", ".restart", ".wfn", ".pdos", ".cube", ".bs")
    for path in sorted(Path.cwd().iterdir()):
        if not path.is_file():
            continue
        if path.name in {"cp2k_summary.json", "cp2k_stdout.out", "job.out", "job.inp"} or path.suffix.lower() in suffixes:
            names[path.name] = str((root / path.name).resolve())
    return names


def _normal_completion(output_path: Path) -> bool:
    if not output_path.is_file():
        return False
    text = output_path.read_text(encoding="utf-8", errors="replace")
    return "PROGRAM ENDED AT" in text and "ABORT" not in text


def _run(args: argparse.Namespace, *, nprocs: int, logical_root: Path) -> int:
    input_path = Path(args.input)
    cp2k_bin = shutil.which(args.cp2k_bin) or args.cp2k_bin
    launcher = shutil.which(args.launcher) or args.launcher

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    command = [launcher, "-n", str(nprocs), cp2k_bin, "-i", input_path.name, "-o", args.output]
    started = time.time()
    with open(args.log, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"[cp2k_boot] cwd={Path.cwd()}\n")
        for key in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NNODES",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "OMP_NUM_THREADS",
        ):
            value = "1" if key == "OMP_NUM_THREADS" else os.environ.get(key, "")
            log_handle.write(f"[cp2k_boot] env {key}={value}\n")
        log_handle.write(f"[cp2k_boot] command={' '.join(command)}\n")
        log_handle.flush()
        proc = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=False)
        log_handle.write(f"[cp2k_boot] returncode={proc.returncode}\n")

    output_path = Path(args.output)
    normal = _normal_completion(output_path)
    payload = {
        "completed": proc.returncode == 0 and normal,
        "returncode": int(proc.returncode),
        "normal_completion": normal,
        "command": command,
        "input": input_path.name,
        "output": args.output,
        "started_at": started,
        "finished_at": time.time(),
        "outputs": _collect_outputs(logical_root),
        "log_file": args.log,
        "omp_num_threads": 1,
        "mpi_ranks": nprocs,
    }
    Path("cp2k_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="CP2K boot wrapper for DPDispatcher tasks")
    parser.add_argument("--input", default="job.inp", help="CP2K input filename")
    parser.add_argument("--output", default="job.out", help="CP2K output filename")
    parser.add_argument("--cp2k_bin", default="cp2k.psmp", help="CP2K executable")
    parser.add_argument("--launcher", default="mpirun", help="MPI launcher")
    parser.add_argument("--nprocs", type=int, default=None, help="MPI ranks")
    parser.add_argument("--log", default="cp2k_stdout.out", help="Wrapper log file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.stderr.write(f"[cp2k_boot] input file missing: {input_path}\n")
        return 2
    try:
        nprocs = _resolve_nprocs(args.nprocs)
    except Exception as exc:
        sys.stderr.write(f"[cp2k_boot] {exc}\n")
        return 2
    try:
        with node_local_workdir("cp2k") as workdir:
            return _run(args, nprocs=nprocs, logical_root=workdir.shared_dir)
    except ScratchStageOutError as exc:
        sys.stderr.write(f"[cp2k_boot] {exc}\n")
        return getattr(os, "EX_IOERR", 74)


if __name__ == "__main__":
    raise SystemExit(main())
