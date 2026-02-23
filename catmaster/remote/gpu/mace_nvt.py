#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run NVT MD sampling (ASE) with a MACE force field for a batch of structures.

What it does
------------
For each input structure file:
  1) Create 3 (or user-specified) isotropically scaled cells, e.g. 0.98, 1.00, 1.02.
     - Only lattice vectors are scaled.
     - Fractional coordinates are kept fixed (atoms positions are scaled with the cell).
  2) Run NVT MD for ~50 ps with 1 fs timestep at 800 K (defaults; configurable).
  3) Save all trajectories + MD logs into an output folder per structure, with
     subfolders per scale factor.

Parallelization
---------------
- One process per GPU.
- Each process pins itself to a single GPU via CUDA_VISIBLE_DEVICES and loads the
  MACE model once, then runs its assigned structures sequentially.

Thermostats (choose one)
------------------------
- Bussi stochastic velocity rescaling (CSVR): ase.md.bussi.Bussi
- Nose–Hoover chain NVT: ase.md.nose_hoover_chain.NoseHooverChainNVT

Example
-------
python run_mace_nvt_sampling.py \
  --input /path/to/selected_structures.zip \
  --output ./md_out \
  --model /path/to/mace-mh-1.model \
  --head omat_pbe \
  --thermostat bussi \
  --temperature_K 800 \
  --timestep_fs 1.0 \
  --total_time_ps 50.0 \
  --scales 0.98,1.0,1.02 \
  --gpu_ids 0,1,2,3 \
  --traj_interval 1 \
  --log_interval 10

Notes
-----
- This script assumes you have installed:
    pip install ase mace-torch
- For multi-head models (e.g. MACE-MH-1), set --head (e.g. omat_pbe).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

# -----------------------------
# Utilities
# -----------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_int_hash(text: str, modulo: int = 2**31 - 1) -> int:
    """Stable (cross-run) hash -> int, suitable for RNG seeds."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % modulo


def _format_scale_tag(scale: float) -> str:
    """
    Make a filesystem-friendly tag for a scale factor.
    Prefer +/-pct style when close to 1.
    """
    pct = (scale - 1.0) * 100.0
    # e.g. 0.98 -> -2.0
    if abs(pct - round(pct)) < 1e-6:
        ipct = int(round(pct))
        if ipct < 0:
            return f"scale_m{abs(ipct)}pct"
        elif ipct > 0:
            return f"scale_p{ipct}pct"
        else:
            return "scale_0pct"
    # fallback
    return "scale_" + f"{scale:.6f}".replace(".", "p")


def _is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def _extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def _collect_structure_files(
    input_path: Path,
    exts: Sequence[str],
) -> List[Path]:
    """
    Recursively collect files with selected extensions.
    exts should be like ["vasp", "cif", "xyz"] (no leading dot).
    """
    exts_norm = tuple("." + e.lower().lstrip(".") for e in exts)
    files: List[Path] = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_norm:
            files.append(p)
    files.sort()
    return files


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Core MD routine
# -----------------------------


@dataclasses.dataclass(frozen=True)
class RunConfig:
    model: str
    head: Optional[str]
    default_dtype: str
    device: str  # usually "cuda" (after CUDA_VISIBLE_DEVICES is set per worker)
    enable_cueq: bool
    dispersion: bool
    compute_atomic_stresses: bool

    thermostat: str  # "bussi" or "nhc"
    temperature_K: float
    timestep_fs: float
    total_time_ps: float
    steps: Optional[int]

    tau_fs: float  # thermostat time constant / damping
    tchain: int
    tloop: int

    scales: Tuple[float, ...]
    traj_interval: int
    log_interval: int
    log_stress: bool
    zero_rotation: bool
    force_temp: bool

    overwrite: bool
    seed: int


def _make_calculator(cfg: RunConfig):
    """
    Create MACE ASE calculator.
    Import is done here so the worker can set CUDA_VISIBLE_DEVICES first.
    """
    try:
        from mace.calculators import mace_mp
    except Exception as e:
        raise RuntimeError(
            "Failed to import MACE. Make sure you installed mace-torch, e.g.\n"
            "  pip install mace-torch\n"
            f"Original import error: {e}"
        ) from e

    kwargs = dict(
        model=cfg.model,
        device=cfg.device,
        default_dtype=cfg.default_dtype,
    )
    if cfg.head:
        kwargs["head"] = cfg.head
    # optional accelerations / additions
    if cfg.enable_cueq:
        kwargs["enable_cueq"] = True
    if cfg.dispersion:
        kwargs["dispersion"] = True
    if cfg.compute_atomic_stresses:
        kwargs["compute_atomic_stresses"] = True


    return mace_mp(**kwargs)


def _make_dynamics(cfg: RunConfig, atoms, rng: np.random.Generator):
    dt = cfg.timestep_fs * units.fs

    if cfg.thermostat.lower() in ("bussi", "csvr"):
        try:
            from ase.md.bussi import Bussi
        except Exception as e:
            raise RuntimeError(
                "Your ASE installation does not provide ase.md.bussi.Bussi.\n"
                "Please update ASE (newer versions include Bussi/CSVR).\n"
                f"Import error: {e}"
            ) from e
        dyn = Bussi(
            atoms,
            timestep=dt,
            temperature_K=cfg.temperature_K,
            taut=cfg.tau_fs * units.fs,
            rng=rng,
        )
        return dyn

    if cfg.thermostat.lower() in ("nhc", "nosehoover", "nose-hoover", "nose_hoover"):
        try:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
        except Exception as e:
            raise RuntimeError(
                "Your ASE installation does not provide ase.md.nose_hoover_chain.NoseHooverChainNVT.\n"
                "Please update ASE.\n"
                f"Import error: {e}"
            ) from e
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=dt,
            temperature_K=cfg.temperature_K,
            tdamp=cfg.tau_fs * units.fs,
            tchain=cfg.tchain,
            tloop=cfg.tloop,
        )
        return dyn

    raise ValueError(f"Unknown thermostat '{cfg.thermostat}'. Use 'bussi' or 'nhc'.")


def _compute_nsteps(cfg: RunConfig) -> int:
    if cfg.steps is not None and cfg.steps > 0:
        return int(cfg.steps)
    # total_time_ps -> fs = ps * 1000
    total_fs = cfg.total_time_ps * 1000.0
    steps = int(round(total_fs / cfg.timestep_fs))
    if steps <= 0:
        raise ValueError("Computed MD steps <= 0; check total_time_ps and timestep_fs.")
    return steps


def _run_one_scale(
    *,
    cfg: RunConfig,
    base_atoms,
    calc,
    out_dir: Path,
    scale: float,
    gpu_id: str,
    structure_id: str,
) -> None:
    tag = _format_scale_tag(scale)
    run_dir = out_dir / tag
    _safe_mkdir(run_dir)

    info_path = run_dir / "md_info.json"
    if info_path.exists() and not cfg.overwrite:
        try:
            info = _read_json(info_path)
            if info.get("completed", False):
                print(f"[SKIP] {structure_id} {tag}: already completed", flush=True)
                return
        except Exception:
            # If corrupt, re-run unless overwrite==False but we'll continue and overwrite md_info.
            pass

    # Prepare atoms: isotropic cell scaling + keep fractional coords fixed
    atoms = base_atoms.copy()
    # scale_atoms=True keeps fractional coordinates fixed (scales cartesian positions with the cell)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)

    # Attach calculator
    atoms.calc = calc

    # RNG / initial velocities
    # Make seed deterministic from global seed + structure_id + scale
    seed_i = (cfg.seed + _stable_int_hash(f"{structure_id}|{scale}")) % (2**32 - 1)
    rng = np.random.default_rng(seed_i)

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=cfg.temperature_K,
        force_temp=cfg.force_temp,
        rng=rng,
    )
    Stationary(atoms)  # remove COM drift (preserve temperature by default)
    if cfg.zero_rotation:
        # For periodic solids this is often unnecessary, but provided as an option.
        ZeroRotation(atoms)

    # Save starting configuration
    with contextlib.suppress(Exception):
        write(run_dir / "start.vasp", atoms, format="vasp")

    # Build dynamics
    dyn = _make_dynamics(cfg, atoms, rng=rng)

    # Output: trajectory + logger
    traj_path = run_dir / "md.traj"
    log_path = run_dir / "md.log"

    # If overwrite, reset outputs
    if cfg.overwrite:
        for p in (traj_path, log_path):
            if p.exists():
                p.unlink()

    traj = Trajectory(str(traj_path), mode="w", atoms=atoms)
    logger = MDLogger(
        dyn,
        atoms,
        str(log_path),
        header=True,
        stress=cfg.log_stress,
        peratom=False,
        mode="w",
    )

    nsteps = _compute_nsteps(cfg)

    # Metadata
    md_info = dict(
        structure_id=structure_id,
        gpu_id=gpu_id,
        scale=scale,
        scale_tag=tag,
        thermostat=cfg.thermostat,
        temperature_K=cfg.temperature_K,
        timestep_fs=cfg.timestep_fs,
        total_time_ps=cfg.total_time_ps,
        steps=nsteps,
        tau_fs=cfg.tau_fs,
        tchain=cfg.tchain,
        tloop=cfg.tloop,
        model=cfg.model,
        head=cfg.head,
        default_dtype=cfg.default_dtype,
        device=cfg.device,
        enable_cueq=cfg.enable_cueq,
        dispersion=cfg.dispersion,
        compute_atomic_stresses=cfg.compute_atomic_stresses,
        traj_interval=cfg.traj_interval,
        log_interval=cfg.log_interval,
        log_stress=cfg.log_stress,
        zero_rotation=cfg.zero_rotation,
        force_temp=cfg.force_temp,
        started_at=_now_iso(),
        completed=False,
        error=None,
        traceback=None,
    )
    _write_json(info_path, md_info)

    try:
        dyn.attach(traj.write, interval=cfg.traj_interval)
        dyn.attach(logger, interval=cfg.log_interval)

        dyn.run(nsteps)

        # Final snapshot
        with contextlib.suppress(Exception):
            write(run_dir / "final.vasp", atoms, format="vasp")

        md_info["completed"] = True
        md_info["finished_at"] = _now_iso()
        _write_json(info_path, md_info)

        print(f"[DONE] {structure_id} {tag} on GPU {gpu_id}", flush=True)

    except Exception as e:
        md_info["completed"] = False
        md_info["finished_at"] = _now_iso()
        md_info["error"] = repr(e)
        md_info["traceback"] = traceback.format_exc()
        _write_json(info_path, md_info)

        # Also save a plain text traceback
        with open(run_dir / "error.txt", "w", encoding="utf-8") as f:
            f.write(md_info["traceback"] or "")
        print(f"[FAIL] {structure_id} {tag} on GPU {gpu_id}: {e}", flush=True)

    finally:
        with contextlib.suppress(Exception):
            traj.close()
        with contextlib.suppress(Exception):
            logger.close()


def run_structure_file(
    *,
    cfg: RunConfig,
    structure_file: Path,
    calc,
    out_root: Path,
    gpu_id: str,
) -> None:
    structure_id = structure_file.stem
    struct_out = out_root / structure_id
    _safe_mkdir(struct_out)

    # Copy original input into folder for provenance
    with contextlib.suppress(Exception):
        shutil.copy2(structure_file, struct_out / structure_file.name)

    # Read atoms (first frame only)
    atoms = read(structure_file, index=0)

    for scale in cfg.scales:
        _run_one_scale(
            cfg=cfg,
            base_atoms=atoms,
            calc=calc,
            out_dir=struct_out,
            scale=float(scale),
            gpu_id=gpu_id,
            structure_id=structure_id,
        )


# -----------------------------
# Multiprocessing orchestration
# -----------------------------


def _worker_main(
    gpu_id: str,
    files: List[str],
    cfg_dict: dict,
    out_root: str,
) -> int:
    """
    Worker process:
      - Pin to 1 GPU with CUDA_VISIBLE_DEVICES
      - Create calculator
      - Run assigned structures
    """
    # IMPORTANT: set env before importing torch/mace
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Avoid CPU oversubscription if multiple GPU workers share the node
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = RunConfig(**cfg_dict)
    out_root_p = Path(out_root)
    _safe_mkdir(out_root_p)

    try:
        # Import torch here (after CUDA_VISIBLE_DEVICES is set)
        import torch  # noqa: F401

        # Optional: limit CPU threads used by PyTorch in each process
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        calc = _make_calculator(cfg)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to initialize calculator: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 2

    for fp in files:
        try:
            run_structure_file(
                cfg=cfg,
                structure_file=Path(fp),
                calc=calc,
                out_root=out_root_p,
                gpu_id=str(gpu_id),
            )
        except Exception as e:
            # Structure-level failure: continue with next
            print(f"[GPU {gpu_id}] Structure failed: {fp}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue

    return 0



def _worker_entry(gpu_id: str, files: List[str], cfg_dict: dict, out_root: str) -> None:
    """Multiprocessing entrypoint (must be top-level for spawn pickling)."""
    rc = _worker_main(gpu_id=gpu_id, files=files, cfg_dict=cfg_dict, out_root=out_root)
    raise SystemExit(rc)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel NVT sampling with ASE + MACE on multiple GPUs."
    )

    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory or .zip containing structure files.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory (per-structure folders will be created here).",
    )

    # MACE
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="MACE model identifier or path (passed to mace.calculators.mace_mp).",
    )
    p.add_argument(
        "--head",
        type=str,
        default="omat_pbe",
        help="Model head for multi-head models (e.g. 'omat_pbe'). Use '' for none.",
    )
    p.add_argument(
        "--default_dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Torch default dtype used by the calculator.",
    )
    p.add_argument(
        "--enable_cueq",
        action="store_true",
        help="Enable cuEquivariance acceleration if available.",
    )
    p.add_argument(
        "--dispersion",
        action="store_true",
        help="Enable D3 dispersion in mace_mp if supported by the chosen model.",
    )
    p.add_argument(
        "--compute_atomic_stresses",
        action="store_true",
        help="Enable MACE atomic stresses/virials computation (if supported). Useful when you also want stress logging.",
    )


    # MD settings
    p.add_argument(
        "--thermostat",
        type=str,
        default="bussi",
        choices=["bussi", "nhc"],
        help="Thermostat: 'bussi' (CSVR) or 'nhc' (Nose–Hoover chain NVT).",
    )
    p.add_argument("--temperature_K", type=float, default=800.0)
    p.add_argument("--timestep_fs", type=float, default=1.0)
    p.add_argument("--total_time_ps", type=float, default=50.0)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total steps. If provided, total_time_ps is ignored.",
    )

    p.add_argument(
        "--tau_fs",
        type=float,
        default=100.0,
        help="Thermostat time constant (Bussi taut / NHC tdamp) in fs.",
    )
    p.add_argument("--tchain", type=int, default=3, help="NHC chain length (nhc only).")
    p.add_argument("--tloop", type=int, default=1, help="NHC substeps (nhc only).")

    # Scaling
    p.add_argument(
        "--scales",
        type=str,
        default="0.98,1.0,1.02",
        help="Comma-separated isotropic cell scale factors, e.g. '0.98,1.0,1.02'.",
    )

    # Output frequency
    p.add_argument(
        "--traj_interval",
        type=int,
        default=1,
        help="Write trajectory every N steps (1 = write every step).",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Write md.log every N steps.",
    )
    p.add_argument(
        "--log_stress",
        action="store_true",
        help="Include stress tensor in md.log (requires calculator stress support).",
    )

    # Velocity init options
    p.add_argument(
        "--zero_rotation",
        action="store_true",
        help="Remove total angular momentum after velocity init.",
    )
    p.add_argument(
        "--force_temp",
        action="store_true",
        help="Force exact initial temperature (slight deviation from exact MB distribution).",
    )

    # Input filtering
    p.add_argument(
        "--extensions",
        type=str,
        default="vasp,cif,xyz,poscar",
        help="Comma-separated extensions to search for (no dots). Default: vasp,cif,xyz,poscar",
    )

    # Parallel/GPU
    p.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs to use (as in nvidia-smi), e.g. '0,1,2,3'.",
    )

    # Misc
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--seed", type=int, default=2026, help="Global RNG seed base.")
    p.add_argument("--dry_run", action="store_true", help="Only list tasks, do not run.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    _safe_mkdir(out_root)

    # Handle zip extraction
    tmpdir: Optional[tempfile.TemporaryDirectory] = None
    if _is_zip(input_path):
        tmpdir = tempfile.TemporaryDirectory(prefix="mace_md_inputs_")
        extract_dir = Path(tmpdir.name)
        _extract_zip(input_path, extract_dir)
        # If the zip contains a single top-level folder, use it
        children = [p for p in extract_dir.iterdir() if p.name not in ("__MACOSX",)]
        if len(children) == 1 and children[0].is_dir():
            input_dir = children[0]
        else:
            input_dir = extract_dir
    else:
        if not input_path.exists():
            print(f"ERROR: input path does not exist: {input_path}", file=sys.stderr)
            return 2
        input_dir = input_path

    exts = [e.strip() for e in args.extensions.split(",") if e.strip()]
    files = _collect_structure_files(input_dir, exts=exts)
    if not files:
        print(f"ERROR: No structure files found in {input_dir} with extensions {exts}", file=sys.stderr)
        return 2

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip() != ""]
    if not gpu_ids:
        print("ERROR: --gpu_ids is empty.", file=sys.stderr)
        return 2

    # Parse scales
    scales = tuple(float(x) for x in args.scales.split(",") if x.strip() != "")

    # Prepare run config
    head = args.head.strip()
    if head == "":
        head = None

    cfg = RunConfig(
        model=args.model,
        head=head,
        default_dtype=args.default_dtype,
        device="cuda",  # each worker sets CUDA_VISIBLE_DEVICES then uses "cuda"
        enable_cueq=bool(args.enable_cueq),
        dispersion=bool(args.dispersion),
        compute_atomic_stresses=bool(args.compute_atomic_stresses),
        thermostat=args.thermostat,
        temperature_K=float(args.temperature_K),
        timestep_fs=float(args.timestep_fs),
        total_time_ps=float(args.total_time_ps),
        steps=args.steps if args.steps is not None else None,
        tau_fs=float(args.tau_fs),
        tchain=int(args.tchain),
        tloop=int(args.tloop),
        scales=scales,
        traj_interval=int(args.traj_interval),
        log_interval=int(args.log_interval),
        log_stress=bool(args.log_stress),
        zero_rotation=bool(args.zero_rotation),
        force_temp=bool(args.force_temp),
        overwrite=bool(args.overwrite),
        seed=int(args.seed),
    )

    print(f"Found {len(files)} structure files.", flush=True)
    print(f"GPUs: {gpu_ids}", flush=True)
    print(f"Scales: {cfg.scales}", flush=True)
    print(
        f"MD: T={cfg.temperature_K} K, dt={cfg.timestep_fs} fs, total={cfg.total_time_ps} ps, "
        f"steps={_compute_nsteps(cfg)}",
        flush=True,
    )
    print(f"Thermostat: {cfg.thermostat} (tau={cfg.tau_fs} fs)", flush=True)
    print(f"MACE model: {cfg.model}, head={cfg.head}, dtype={cfg.default_dtype}", flush=True)
    print(f"Output: {out_root}", flush=True)

    if args.dry_run:
        for i, fp in enumerate(files):
            print(f"[TASK] {i:04d} -> GPU {gpu_ids[i % len(gpu_ids)]}: {fp}", flush=True)
        return 0

    # Partition tasks (round-robin)
    assignments: List[List[str]] = [[] for _ in gpu_ids]
    for i, fp in enumerate(files):
        assignments[i % len(gpu_ids)].append(str(fp))

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []
    exit_codes: List[int] = []

    print("Launching workers...", flush=True)
    for gi, gpu in enumerate(gpu_ids):
        flist = assignments[gi]
        if not flist:
            continue
        cfg_dict = dataclasses.asdict(cfg)
        proc = ctx.Process(
            target=_worker_entry,
            args=(str(gpu), flist, cfg_dict, str(out_root)),
            name=f"gpu_worker_{gpu}",
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
        exit_codes.append(proc.exitcode or 0)

    # Cleanup temp extraction dir
    if tmpdir is not None:
        tmpdir.cleanup()

    if any(code != 0 for code in exit_codes):
        print(f"Some workers exited with non-zero codes: {exit_codes}", file=sys.stderr, flush=True)
        return 1

    print("All workers finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
