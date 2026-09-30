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


_CPU_LAMMPS_BIN_CANDIDATES = (
    "lmp_mpi",
    "lmp",
    "lammps",
    "lmp_openmpi",
    "lmp_intel_cpu_intelmpi",
    "lmp_intel_cpu",
    "lmp_serial",
)

_GPU_LAMMPS_BIN_CANDIDATES = (
    "lmp_kokkos_cuda_mpi",
    "lmp_kokkos_cuda",
    "lmp_kokkos",
    "lmp_gpu",
    "lmp_cuda",
    "lmp_mpi",
    "lmp",
    "lammps",
)

_MPI_LAUNCHER_CANDIDATES = (
    "mpirun",
    "mpiexec",
    "srun",
)


def _split_candidate_env(raw: str) -> list[str]:
    out: list[str] = []
    for item in re.split(r"[:,]", str(raw or "")):
        text = item.strip()
        if text:
            out.append(text)
    return out


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _binary_exists(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_lammps_binary(requested: str, *, mode: str, gpu_count: int) -> tuple[str, list[str]]:
    raw = str(requested or "auto").strip()
    candidates: list[str] = []
    if raw and raw.lower() not in {"auto", "__auto__"}:
        candidates.append(raw)
    else:
        candidates.extend(_split_candidate_env(os.environ.get("CATMASTER_LAMMPS_BIN", "")))
        candidates.extend(_split_candidate_env(os.environ.get("CATMASTER_LAMMPS_BIN_CANDIDATES", "")))
        if str(mode or "auto").lower() != "off" and gpu_count > 0:
            candidates.extend(_GPU_LAMMPS_BIN_CANDIDATES)
            candidates.extend(_CPU_LAMMPS_BIN_CANDIDATES)
        else:
            candidates.extend(_CPU_LAMMPS_BIN_CANDIDATES)
            candidates.extend(_GPU_LAMMPS_BIN_CANDIDATES)
    tried = _dedupe(candidates)
    for candidate in tried:
        if "/" in candidate:
            path = Path(candidate).expanduser()
            if _binary_exists(path):
                return str(path), tried
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved, tried
    raise FileNotFoundError(
        "Unable to resolve LAMMPS executable. "
        f"requested={raw!r}; tried={', '.join(tried) if tried else '(none)'}; "
        "set CATMASTER_LAMMPS_BIN or pass --lammps_bin to override."
    )


def _run_help(lmp_bin: str) -> str:
    try:
        proc = subprocess.run([lmp_bin, "-help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except Exception:
        return ""
    return proc.stdout or ""


def _resolve_nprocs(requested: int | None) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--nprocs must be positive")
        return requested
    env_value = os.environ.get("SLURM_NTASKS", "").strip()
    if not env_value:
        return 1
    try:
        parsed = int(env_value)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_NTASKS: {env_value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Invalid SLURM_NTASKS (<=0): {env_value}")
    return parsed


def _lammps_mpi_build(help_text: str) -> tuple[bool, str]:
    for line in help_text.splitlines():
        stripped = line.strip()
        if not re.match(r"^MPI\s+v", stripped, flags=re.IGNORECASE):
            continue
        upper = stripped.upper()
        if "STUB" in upper:
            return False, stripped
        return True, stripped
    return False, "MPI build metadata not reported by `lmp -help`"


def _resolve_mpi_launcher(requested: str, *, nprocs: int) -> tuple[list[str], str]:
    if nprocs <= 1:
        return [], "direct"
    raw = str(requested or "auto").strip()
    candidates: list[str] = []
    if raw.lower() not in {"auto", "__auto__"}:
        candidates.append(raw)
    else:
        candidates.extend(_split_candidate_env(os.environ.get("CATMASTER_LAMMPS_MPI_LAUNCHER", "")))
        candidates.extend(_MPI_LAUNCHER_CANDIDATES)
    tried = _dedupe(candidates)
    for candidate in tried:
        if "/" in candidate:
            path = Path(candidate).expanduser()
            if _binary_exists(path):
                resolved = str(path)
                return [resolved, "-n", str(nprocs)], resolved
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved, "-n", str(nprocs)], resolved
    raise FileNotFoundError(
        "Unable to resolve an MPI launcher for multi-rank LAMMPS execution. "
        f"tried={', '.join(tried) if tried else '(none)'}; "
        "set CATMASTER_LAMMPS_MPI_LAUNCHER or pass --mpi_launcher."
    )


def _configure_mpi_environment(
    env: dict[str, str],
    *,
    mpi_report: str,
    nprocs: int,
) -> tuple[dict[str, str], dict[str, object]]:
    configured = dict(env)
    intel_mpi = "INTEL" in str(mpi_report or "").upper()
    slurm_nodes_raw = (
        configured.get("SLURM_NNODES", "").strip()
        or configured.get("SLURM_JOB_NUM_NODES", "").strip()
    )
    try:
        slurm_nodes = int(slurm_nodes_raw) if slurm_nodes_raw else 0
    except ValueError:
        slurm_nodes = 0
    existing_bootstrap = configured.get("I_MPI_HYDRA_BOOTSTRAP", "").strip()
    srun_available = bool(shutil.which("srun", path=configured.get("PATH")))
    needs_local_fork = (
        intel_mpi
        and nprocs > 1
        and slurm_nodes == 1
        and not srun_available
        and existing_bootstrap.lower() in {"", "slurm"}
    )
    source = "unchanged"
    if existing_bootstrap and not needs_local_fork:
        source = "environment"
    elif needs_local_fork:
        # Intel MPI sees SLURM_JOB_ID and otherwise chooses Slurm bootstrap.
        # Some single-node Slurm installations do not expose srun on compute
        # nodes, while their login shell may still export a stale `slurm`
        # bootstrap. Hydra fork preserves the allocation without using srun.
        configured["I_MPI_HYDRA_BOOTSTRAP"] = "fork"
        existing_bootstrap = "fork"
        source = "auto_single_node_without_srun"
    return configured, {
        "intel_mpi": intel_mpi,
        "slurm_nodes": slurm_nodes,
        "srun_available": srun_available,
        "hydra_bootstrap": existing_bootstrap,
        "hydra_bootstrap_source": source,
    }


def _probe_mpi_launcher(
    launcher_prefix: list[str],
    *,
    expected_ranks: int,
    env: dict[str, str],
) -> dict[str, object]:
    if expected_ranks <= 1:
        return {
            "status": "skipped_single_rank",
            "expected_ranks": expected_ranks,
            "observed_processes": 1,
            "command": [],
        }
    token = "CATMASTER_LAMMPS_MPI_PROBE"
    command = [*launcher_prefix, "sh", "-c", f"printf '{token}\\n'"]
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MPI launcher probe timed out after 60 s: {' '.join(command)}") from exc
    output = proc.stdout or ""
    observed = output.count(token)
    if proc.returncode != 0 or observed != expected_ranks:
        tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(
            "MPI launcher probe did not match the allocated slots: "
            f"expected={expected_ranks}, observed={observed}, returncode={proc.returncode}; "
            f"output_tail={tail!r}"
        )
    return {
        "status": "passed",
        "expected_ranks": expected_ranks,
        "observed_processes": observed,
        "command": command,
    }


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
    if mode == "off":
        return [lmp_bin], "cpu"
    if mode in {"kokkos", "kk", "gpu"} and gpu_count <= 0:
        raise RuntimeError(f"Explicit LAMMPS GPU mode '{mode}' requires at least one visible GPU")
    if gpu_count <= 0:
        return [lmp_bin], "cpu"
    packages = _installed_packages(help_text)
    if mode in {"kokkos", "kk"}:
        if "KOKKOS" not in packages:
            raise RuntimeError("Explicit LAMMPS KOKKOS mode requested, but the selected executable lacks KOKKOS")
        return [lmp_bin, "-k", "on", "g", str(gpu_count), "-sf", "kk"], "kokkos"
    if mode == "gpu":
        if "GPU" not in packages:
            raise RuntimeError("Explicit LAMMPS GPU-package mode requested, but the selected executable lacks GPU")
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
    parser.add_argument("--lammps_bin", default="auto", help="LAMMPS executable or `auto`")
    parser.add_argument("--gpu", default="auto", choices=["auto", "off", "gpu", "kokkos"], help="GPU acceleration selection")
    parser.add_argument("--allow_cpu_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpi_launcher", default="auto", help="MPI launcher or `auto`")
    parser.add_argument("--nprocs", type=int, default=None, help="MPI ranks; defaults to SLURM_NTASKS or one")
    parser.add_argument("--mpi_probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log", default="lammps_stdout.out", help="Wrapper log path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.stderr.write(f"[lammps_boot] input file missing: {input_path}\n")
        return 2
    gpu_count = _detect_gpu_count()
    log_path = Path(args.log)
    started = time.time()
    try:
        lmp_bin, lmp_candidates = _resolve_lammps_binary(args.lammps_bin, mode=args.gpu, gpu_count=gpu_count)
    except FileNotFoundError as exc:
        log_path.write_text(
            "\n".join(
                [
                    f"[lammps_boot] cwd={Path.cwd()}",
                    f"[lammps_boot] requested_lammps_bin={args.lammps_bin}",
                    f"[lammps_boot] gpu_count={gpu_count}",
                    f"[lammps_boot] error={exc}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        Path("lammps_summary.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "returncode": 2,
                    "command": [],
                    "input": input_path.name,
                    "started_at": started,
                    "finished_at": time.time(),
                    "outputs": _collect_outputs(),
                    "log_file": args.log,
                    "gpu_count": gpu_count,
                    "acceleration": "unresolved",
                    "cpu_fallback_used": False,
                    "requested_lammps_bin": args.lammps_bin,
                    "resolved_lammps_bin": "",
                    "lammps_bin_candidates": [],
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        sys.stderr.write(f"[lammps_boot] {exc}\n")
        return 2
    log_path.write_text(f"[lammps_boot] cwd={Path.cwd()}\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[lammps_boot] requested_lammps_bin={args.lammps_bin}\n")
        log_handle.write(f"[lammps_boot] resolved_lammps_bin={lmp_bin}\n")
        log_handle.write(f"[lammps_boot] lammps_bin_candidates={', '.join(lmp_candidates)}\n")
    help_text = _run_help(lmp_bin)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    nprocs = 0
    mpi_build = False
    mpi_report = ""
    mpi_launcher = "unresolved"
    mpi_environment: dict[str, object] = {
        "intel_mpi": False,
        "slurm_nodes": 0,
        "srun_available": False,
        "hydra_bootstrap": "",
        "hydra_bootstrap_source": "not_configured",
    }
    mpi_probe: dict[str, object] = {
        "status": "not_run",
        "expected_ranks": 0,
        "observed_processes": 0,
        "command": [],
    }
    try:
        nprocs = _resolve_nprocs(args.nprocs)
        mpi_build, mpi_report = _lammps_mpi_build(help_text)
        if nprocs > 1 and not mpi_build:
            raise RuntimeError(
                f"Allocated {nprocs} MPI slots, but the selected LAMMPS executable is not a real MPI build: "
                f"{mpi_report}"
            )
        launcher_prefix, mpi_launcher = _resolve_mpi_launcher(args.mpi_launcher, nprocs=nprocs)
        env, mpi_environment = _configure_mpi_environment(
            env,
            mpi_report=mpi_report,
            nprocs=nprocs,
        )
        if args.mpi_probe:
            mpi_probe = _probe_mpi_launcher(launcher_prefix, expected_ranks=nprocs, env=env)
        else:
            mpi_probe = {
                "status": "disabled",
                "expected_ranks": nprocs,
                "observed_processes": 0,
                "command": [],
            }
        prefix, acceleration = _gpu_command_prefix(lmp_bin, args.gpu, gpu_count, help_text)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"[lammps_boot] error={exc}\n")
        Path("lammps_summary.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "returncode": 2,
                    "command": [],
                    "input": input_path.name,
                    "started_at": started,
                    "finished_at": time.time(),
                    "outputs": _collect_outputs(),
                    "log_file": args.log,
                    "gpu_count": gpu_count,
                    "acceleration": "unavailable",
                    "cpu_fallback_used": False,
                    "requested_lammps_bin": args.lammps_bin,
                    "resolved_lammps_bin": lmp_bin,
                    "lammps_bin_candidates": lmp_candidates,
                    "mpi_ranks": nprocs,
                    "mpi_launcher": mpi_launcher,
                    "mpi_build": mpi_build,
                    "mpi_report": mpi_report,
                    "mpi_environment": mpi_environment,
                    "mpi_probe": mpi_probe,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        sys.stderr.write(f"[lammps_boot] {exc}\n")
        return 2
    command = [*launcher_prefix, *prefix, "-in", input_path.name]
    with log_path.open("a", encoding="utf-8") as log_handle:
        for key in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NNODES",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "I_MPI_HYDRA_BOOTSTRAP",
        ):
            log_handle.write(f"[lammps_boot] env {key}={env.get(key, '')}\n")
        log_handle.write(
            f"[lammps_boot] mpi_ranks={nprocs} mpi_launcher={mpi_launcher} "
            f"mpi_build={mpi_build} mpi_report={mpi_report}\n"
        )
        log_handle.write(f"[lammps_boot] mpi_probe={json.dumps(mpi_probe, sort_keys=True)}\n")
        log_handle.write(f"[lammps_boot] gpu_count={gpu_count} acceleration={acceleration}\n")
    proc = _run_lammps(command, log_path, env)
    fallback_used = False
    if proc.returncode != 0 and acceleration != "cpu" and args.allow_cpu_fallback:
        fallback_used = True
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("[lammps_boot] accelerated run failed; retrying CPU path\n")
        command = [*launcher_prefix, lmp_bin, "-in", input_path.name]
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
        "requested_lammps_bin": args.lammps_bin,
        "resolved_lammps_bin": lmp_bin,
        "lammps_bin_candidates": lmp_candidates,
        "mpi_ranks": nprocs,
        "mpi_launcher": mpi_launcher,
        "mpi_build": mpi_build,
        "mpi_report": mpi_report,
        "mpi_environment": mpi_environment,
        "mpi_probe": mpi_probe,
    }
    Path("lammps_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
