from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import TextIO


def _is_gamma_only(kpoints_path: Path) -> bool:
    if not kpoints_path.is_file():
        return False
    try:
        raw_lines = kpoints_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", "!")):
                continue
            lines.append(stripped)
        if len(lines) < 4:
            return False
        try:
            npoints = int(lines[1].split()[0])
        except ValueError:
            return False
        if npoints != 0:
            # Only trust automatic Gamma with 1x1x1 for vasp_gam safety.
            return False
        scheme = lines[2].lower()
        if "gamma" not in scheme:
            return False
        grid_parts = lines[3].split()
        if len(grid_parts) < 3:
            return False
        try:
            grid = [int(grid_parts[0]), int(grid_parts[1]), int(grid_parts[2])]
        except ValueError:
            return False
        return grid == [1, 1, 1]
    except Exception:
        return False
    return False


_INCAR_KEY_RE = re.compile(r"(?i)^\s*(NCORE|NPAR)\b")
_INCAR_IMAGES_RE = re.compile(r"(?i)^\s*IMAGES\s*=\s*([0-9]+)\b")


def _strip_incar_comment(line: str) -> str:
    return line.split("!")[0].split("#")[0].strip()


def _read_incar_lines(incar_path: Path) -> list[str]:
    return incar_path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _read_neb_images(incar_path: Path) -> int | None:
    if not incar_path.is_file():
        return None
    try:
        for raw_line in _read_incar_lines(incar_path):
            line = _strip_incar_comment(raw_line)
            if not line:
                continue
            match = _INCAR_IMAGES_RE.match(line)
            if not match:
                continue
            parsed = int(match.group(1))
            return parsed if parsed > 0 else None
    except Exception:
        return None
    return None


def _infer_cpu_per_node(args: argparse.Namespace, log_file) -> int | None:
    if args.cpu_per_node is not None and args.cpu_per_node > 0:
        log_file.write(f"[vasp_boot] auto-ncore: cpu_per_node={args.cpu_per_node}\n")
        log_file.flush()
        return args.cpu_per_node
    env_val = os.environ.get("SLURM_CPUS_ON_NODE", "").strip()
    if env_val:
        try:
            parsed = int(env_val)
        except ValueError:
            return None
        if parsed > 0:
            return parsed
    return None


def _choose_ncore(total_ranks: int) -> int:
    target = total_ranks ** 0.5
    factors = [f for f in range(1, total_ranks + 1) if total_ranks % f == 0]
    return min(factors, key=lambda factor: (abs(factor - target), factor))


def _choose_effective_neb_ranks(requested: int, neb_images: int) -> tuple[int, int, str]:
    divisible = [candidate for candidate in range(requested, neb_images - 1, -1) if candidate % neb_images == 0]
    if not divisible:
        raise ValueError(
            f"After NEB divisibility adjustment, no valid ranks remain: requested={requested}, IMAGES={neb_images}"
        )
    for candidate in divisible:
        ranks_per_image = candidate // neb_images
        chosen_ncore = _choose_ncore(ranks_per_image)
        if chosen_ncore > 1:
            return candidate, ranks_per_image, "prefer_ncore_gt_1"
    fallback = divisible[0]
    return fallback, fallback // neb_images, "fallback_max_divisible"


def _resolve_effective_nprocs(
    args: argparse.Namespace,
    log_file: TextIO,
) -> tuple[int, int | None]:
    requested = _resolve_nprocs(args.nprocs)
    neb_images = _read_neb_images(Path(args.incar))
    if not neb_images:
        log_file.write(f"[vasp_boot] parallel: ranks={requested} (non-NEB or IMAGES absent)\n")
        log_file.flush()
        return requested, None
    if requested < neb_images:
        raise ValueError(
            f"Requested MPI ranks ({requested}) are fewer than IMAGES ({neb_images}); "
            "NEB requires at least one rank per image."
        )
    adjusted, ranks_per_image, reason = _choose_effective_neb_ranks(requested, neb_images)
    if adjusted == requested:
        log_file.write(
            f"[vasp_boot] parallel: NEB detected IMAGES={neb_images}, ranks={requested}, ranks/image={ranks_per_image}\n"
        )
        log_file.flush()
        return requested, neb_images
    reason_msg = (
        "preferring a divisible layout with NCORE>1"
        if reason == "prefer_ncore_gt_1"
        else "so ranks are divisible by IMAGES"
    )
    log_file.write(
        f"[vasp_boot] parallel: NEB detected IMAGES={neb_images}, reducing ranks from {requested} to {adjusted} "
        f"{reason_msg}; ranks/image={ranks_per_image}\n"
    )
    log_file.flush()
    return adjusted, neb_images


def _maybe_set_ncore(args: argparse.Namespace, log_file: TextIO, *, nprocs: int, neb_images: int | None) -> None:
    if not args.auto_ncore:
        return
    incar_path = Path(args.incar)
    if not incar_path.is_file():
        log_file.write("[vasp_boot] auto-ncore: INCAR missing, skip\n")
        log_file.flush()
        return
    cpu_per_node = _infer_cpu_per_node(args, log_file)
    if not cpu_per_node:
        log_file.write("[vasp_boot] auto-ncore: cpu_per_node unavailable, skip\n")
        log_file.flush()
        return
    lines = _read_incar_lines(incar_path)
    for ln in lines:
        head = _strip_incar_comment(ln)
        if not head:
            continue
        if _INCAR_KEY_RE.match(head):
            log_file.write("[vasp_boot] auto-ncore: NCORE/NPAR already set, skip\n")
            log_file.flush()
            return
    ncore_basis = cpu_per_node
    if neb_images:
        ncore_basis = max(1, nprocs // neb_images)
    ncore = _choose_ncore(ncore_basis)
    lines.append(f"NCORE = {ncore}")
    incar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if neb_images:
        log_file.write(
            f"[vasp_boot] auto-ncore: cpu_per_node={cpu_per_node}, IMAGES={neb_images}, ranks={nprocs}, "
            f"ranks/image={ncore_basis}, ncore={ncore}\n"
        )
    else:
        log_file.write(f"[vasp_boot] auto-ncore: cpu_per_node={cpu_per_node}, ncore={ncore}\n")
    log_file.flush()


def _resolve_nprocs(nprocs: int | None) -> int:
    if nprocs is not None:
        if nprocs <= 0:
            raise ValueError("nprocs must be positive")
        return nprocs
    env_val = os.environ.get("SLURM_NTASKS")
    if not env_val:
        raise ValueError("SLURM_NTASKS is not set; pass --nprocs explicitly")
    try:
        parsed = int(env_val)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_NTASKS: {env_val}") from exc
    if parsed <= 0:
        raise ValueError(f"Invalid SLURM_NTASKS (<=0): {env_val}")
    return parsed


def _build_command(args: argparse.Namespace, *, nprocs: int) -> tuple[list[str], str]:
    vasp_bin = args.vasp
    if args.auto_gamma and _is_gamma_only(Path(args.kpoints)):
        vasp_bin = args.gamma_vasp
    return [args.launcher, "-n", str(nprocs), vasp_bin], vasp_bin


def main() -> int:
    parser = argparse.ArgumentParser(description="VASP boot wrapper for DPDispatcher tasks")
    parser.add_argument("--vasp", default="vasp_std", help="VASP binary to run")
    parser.add_argument("--launcher", default="mpirun", help="MPI launcher command")
    parser.add_argument("--nprocs", type=int, default=None, help="MPI process count")
    parser.add_argument("--log", default="vasp_stdout.out", help="Log file path")
    parser.add_argument("--sync-delay", type=int, default=10, help="Sleep seconds after sync")
    parser.add_argument("--auto-gamma", action="store_true", help="Use gamma VASP when KPOINTS is Gamma")
    parser.add_argument("--gamma-vasp", default="vasp_gam", help="Gamma-point VASP binary")
    parser.add_argument("--kpoints", default="KPOINTS", help="KPOINTS file path")
    parser.add_argument("--auto-ncore", action="store_true", help="Auto-set NCORE when absent")
    parser.add_argument("--cpu-per-node", type=int, default=None, help="CPU per node hint for NCORE")
    parser.add_argument("--incar", default="INCAR", help="INCAR file path")
    args = parser.parse_args()

    with open(args.log, "w", encoding="utf-8") as log_file:
        log_file.write(f"[vasp_boot] cwd={Path.cwd()}\n")
        for key in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NNODES",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "SLURM_JOB_CPUS_PER_NODE",
            "OMP_NUM_THREADS",
            "I_MPI_HYDRA_BOOTSTRAP",
        ):
            log_file.write(f"[vasp_boot] env {key}={os.environ.get(key, '')}\n")
        log_file.flush()
        try:
            nprocs, neb_images = _resolve_effective_nprocs(args, log_file)
            _maybe_set_ncore(args, log_file, nprocs=nprocs, neb_images=neb_images)
            cmd, vasp_bin = _build_command(args, nprocs=nprocs)
        except Exception as exc:
            log_file.write(f"[vasp_boot] setup failed: {exc}\n")
            log_file.flush()
            sys.stderr.write(f"[vasp_boot] {exc}\n")
            return 2
        log_file.write(f"[vasp_boot] vasp_bin={vasp_bin}\n")
        log_file.write(f"[vasp_boot] command: {' '.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    try:
        os.sync()
    except AttributeError:
        pass

    if args.sync_delay > 0:
        time.sleep(args.sync_delay)

    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
