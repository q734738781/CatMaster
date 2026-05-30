from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def _run_help(lmp_bin: str) -> str:
    try:
        proc = subprocess.run([lmp_bin, "-help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except Exception:
        return ""
    return proc.stdout or ""


def _detect_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        return len([item for item in visible.split(",") if item.strip()])
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0
    try:
        proc = subprocess.run([nvidia_smi, "-L"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    except Exception:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU "))


def _installed_packages(help_text: str) -> set[str]:
    packages: set[str] = set()
    for line in help_text.splitlines():
        if "Installed packages" not in line and not re.match(r"^\s*(ASPHERE|BODY|GPU|KOKKOS)\b", line):
            continue
        for token in re.split(r"[\s,]+", line.strip()):
            upper = token.strip().upper()
            if upper:
                packages.add(upper)
    return packages


def _gpu_command_prefix(lmp_bin: str, mode: str, gpu_count: int, help_text: str) -> tuple[list[str], str]:
    mode = str(mode or "auto").lower()
    if mode == "off" or gpu_count <= 0:
        return [lmp_bin], "cpu"
    packages = _installed_packages(help_text)
    if mode in {"kokkos", "kk"}:
        return [lmp_bin, "-k", "on", "g", str(gpu_count), "-sf", "kk"], "kokkos"
    if mode == "gpu":
        return [lmp_bin, "-sf", "gpu", "-pk", "gpu", str(gpu_count)], "gpu"
    if "KOKKOS" in packages:
        return [lmp_bin, "-k", "on", "g", str(gpu_count), "-sf", "kk"], "kokkos"
    if "GPU" in packages:
        return [lmp_bin, "-sf", "gpu", "-pk", "gpu", str(gpu_count)], "gpu"
    return [lmp_bin], "cpu"


def _collect_outputs() -> dict[str, str]:
    out: dict[str, str] = {}
    names = {
        "lammps_summary.json",
        "lammps_stdout.out",
        "log.lammps",
        "trajectory.lammpstrj",
        "final.data",
        "restart.final",
        "rdf.dat",
        "msd.dat",
    }
    for path in sorted(Path.cwd().iterdir()):
        if not path.is_file():
            continue
        if path.name in names or path.suffix in {".data", ".restart", ".lammpstrj"}:
            out[path.name] = str(path.resolve())
    return out


def _run_lammps(command: list[str], output_log: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    with output_log.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[lammps_boot] command={' '.join(command)}\n")
        log_handle.flush()
        return subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="LAMMPS boot wrapper for DPDispatcher tasks")
    parser.add_argument("--input", default="in.lammps", help="LAMMPS input script")
    parser.add_argument("--lammps-bin", default="lmp", help="LAMMPS executable")
    parser.add_argument("--gpu", default="auto", choices=["auto", "off", "gpu", "kokkos"], help="GPU acceleration selection")
    parser.add_argument("--allow-cpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log", default="lammps_stdout.out", help="Wrapper log path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.stderr.write(f"[lammps_boot] input file missing: {input_path}\n")
        return 2
    lmp_bin = shutil.which(args.lammps_bin) or args.lammps_bin
    help_text = _run_help(lmp_bin)
    gpu_count = _detect_gpu_count()
    prefix, acceleration = _gpu_command_prefix(lmp_bin, args.gpu, gpu_count, help_text)
    command = [*prefix, "-in", input_path.name]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    started = time.time()
    log_path = Path(args.log)
    log_path.write_text(f"[lammps_boot] cwd={Path.cwd()}\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log_handle:
        for key in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NNODES",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
        ):
            log_handle.write(f"[lammps_boot] env {key}={env.get(key, '')}\n")
        log_handle.write(f"[lammps_boot] gpu_count={gpu_count} acceleration={acceleration}\n")
    proc = _run_lammps(command, log_path, env)
    fallback_used = False
    if proc.returncode != 0 and acceleration != "cpu" and args.allow_cpu_fallback:
        fallback_used = True
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("[lammps_boot] accelerated run failed; retrying CPU path\n")
        command = [lmp_bin, "-in", input_path.name]
        acceleration = "cpu_fallback"
        proc = _run_lammps(command, log_path, env)

    payload = {
        "completed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "command": command,
        "input": input_path.name,
        "started_at": started,
        "finished_at": time.time(),
        "outputs": _collect_outputs(),
        "log_file": args.log,
        "gpu_count": gpu_count,
        "acceleration": acceleration,
        "cpu_fallback_used": fallback_used,
    }
    Path("lammps_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
