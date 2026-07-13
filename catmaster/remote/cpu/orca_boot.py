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


def _resolve_orca_binary(requested: str) -> str:
    if requested and requested != "auto":
        return shutil.which(requested) or requested
    candidates = (
        "orca",
        "orca5",
        "orca_5_0_3",
        "orca-5.0.3",
        "orca-6.1.1",
    )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "orca"


_NPROCS_RE = re.compile(r"^\s*nprocs\s+([0-9]+)\s*$", re.IGNORECASE)
_SLURM_CPU_COUNT_RE = re.compile(r"^\s*([0-9]+)")


def _parse_cpu_count(raw: str) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    match = _SLURM_CPU_COUNT_RE.match(text)
    if not match:
        return None
    try:
        parsed = int(match.group(1))
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _resolve_nprocs() -> int:
    for key in ("SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "OMP_NUM_THREADS"):
        parsed = _parse_cpu_count(os.environ.get(key, ""))
        if parsed:
            return parsed
    return 1


def _upsert_pal_nprocs(input_path: Path, nprocs: int) -> str:
    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    filtered: list[str] = []
    inside_pal = False
    had_pal_block = False
    for raw in lines:
        stripped = raw.strip()
        lower = stripped.lower()
        if not inside_pal and lower == "%pal":
            inside_pal = True
            had_pal_block = True
            continue
        if inside_pal:
            if lower == "end":
                inside_pal = False
            continue
        filtered.append(raw)

    pal_block = ["%pal", f"  nprocs {int(nprocs)}", "end"]
    insert_at = 0
    if filtered and filtered[0].strip().startswith("!"):
        insert_at = 1
    updated = [*filtered[:insert_at], *pal_block, *filtered[insert_at:]]
    input_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return "replaced" if had_pal_block else "inserted"


def _collect_outputs() -> dict[str, str]:
    suffixes = (".out", ".gbw", ".xyz", ".trj", ".property.txt", ".property.json", ".engrad", ".hess")
    found: dict[str, str] = {}
    for path in sorted(Path.cwd().glob("job*")):
        if not path.is_file():
            continue
        if path.suffix not in suffixes:
            continue
        found[path.name] = str(path.resolve())
    return found


def _try_orca_2json() -> None:
    converter = shutil.which("orca_2json")
    if not converter:
        return
    candidates = (
        [converter, "job"],
        [converter, "job.out"],
        [converter, "job", "-property"],
        [converter, "job.out", "-property"],
    )
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            continue
        if proc.returncode == 0 and Path("job.property.json").exists():
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="ORCA boot wrapper for DPDispatcher tasks")
    parser.add_argument("--input", default="job.inp", help="ORCA input filename")
    parser.add_argument("--orca_bin", default="auto", help="ORCA executable or `auto`")
    parser.add_argument("--log", default="orca_stdout.out", help="Wrapper log path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.stderr.write(f"[orca_boot] input file missing: {input_path}\n")
        return 2

    orca_bin = _resolve_orca_binary(args.orca_bin)
    nprocs = _resolve_nprocs()
    pal_status = _upsert_pal_nprocs(input_path, nprocs)
    if nprocs > 1 and shutil.which("mpirun") is None:
        sys.stderr.write(
            "[orca_boot] inferred parallel execution "
            f"(nprocs={nprocs}) but `mpirun` is not available in the current environment. "
            "Provide an OpenMPI/MPI-enabled ORCA environment or reduce the allocated ranks to 1.\n"
        )
        return 2
    output_path = Path("job.out")
    started = time.time()
    with open(args.log, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"[orca_boot] cwd={Path.cwd()}\n")
        for key in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NNODES",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "SLURM_JOB_CPUS_PER_NODE",
            "OMP_NUM_THREADS",
        ):
            log_handle.write(f"[orca_boot] env {key}={os.environ.get(key, '')}\n")
        log_handle.write(f"[orca_boot] orca_bin={orca_bin}\n")
        log_handle.write(f"[orca_boot] parallel: nprocs={nprocs} pal_block={pal_status}\n")
        log_handle.flush()
        with output_path.open("w", encoding="utf-8") as out_handle:
            proc = subprocess.run([orca_bin, input_path.name], stdout=out_handle, stderr=subprocess.STDOUT, check=False)
        log_handle.write(f"[orca_boot] returncode={proc.returncode}\n")
        log_handle.flush()

    if proc.returncode == 0:
        _try_orca_2json()

    out_text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    payload = {
        "completed": proc.returncode == 0 and "ORCA TERMINATED NORMALLY" in out_text,
        "returncode": int(proc.returncode),
        "command": [orca_bin, input_path.name],
        "input": input_path.name,
        "started_at": started,
        "finished_at": time.time(),
        "outputs": _collect_outputs(),
        "normal_termination": "ORCA TERMINATED NORMALLY" in out_text,
        "log_file": args.log,
    }
    Path("orca_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
