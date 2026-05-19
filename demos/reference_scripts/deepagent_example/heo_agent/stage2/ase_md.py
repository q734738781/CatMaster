from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from scipy.stats import linregress

from ..core.config import DEFAULT_GPU_IDS, DEFAULT_MACE_DTYPE, DEFAULT_MACE_HEAD, DEFAULT_MACE_MODEL, DEFAULT_MD_TEMPERATURES_K

KB_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class MDSettings:
    model_path: str
    head: str | None
    default_dtype: str
    timestep_fs: float
    steps: int
    friction: float
    sample_interval: int
    temperatures_K: tuple[float, ...]
    gpu_ids: tuple[str, ...]
    seed: int = 2026


def _make_calc(model_path: str, head: str | None, default_dtype: str, device: str):
    from mace.calculators import mace_mp

    kwargs: dict[str, Any] = {
        "model": model_path,
        "device": device,
        "default_dtype": default_dtype,
    }
    if head:
        kwargs["head"] = head
    return mace_mp(**kwargs)


def _ensure_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _unwrap_fractional(fractional: np.ndarray, cell: np.ndarray) -> np.ndarray:
    unwrapped = np.empty_like(fractional)
    unwrapped[0] = fractional[0]
    delta = fractional[1:] - fractional[:-1]
    delta -= np.round(delta)
    for index in range(1, len(fractional)):
        unwrapped[index] = unwrapped[index - 1] + delta[index - 1]
    return np.einsum("tni,ij->tnj", unwrapped, cell)


def _compute_diffusion_summary(
    scaled_positions: list[np.ndarray],
    cell: np.ndarray,
    timestep_fs: float,
    sample_interval: int,
    temperature_K: float,
) -> dict[str, Any]:
    scaled = np.asarray(scaled_positions, dtype=float)
    cart = _unwrap_fractional(scaled, cell)
    displacements = cart - cart[0]
    msd_ang2 = np.mean(np.sum(displacements**2, axis=2), axis=1)
    times_ps = np.arange(len(msd_ang2), dtype=float) * timestep_fs * sample_interval / 1000.0
    fit_start = max(1, int(len(times_ps) * 0.2))
    fit = linregress(times_ps[fit_start:], msd_ang2[fit_start:])
    slope_ang2_per_ps = max(float(fit.slope), 0.0)
    diffusion_m2_s = slope_ang2_per_ps / 6.0 * 1e-8
    return {
        "temperature_K": temperature_K,
        "diffusion_m2_s": diffusion_m2_s,
        "slope_ang2_per_ps": slope_ang2_per_ps,
        "intercept_ang2": float(fit.intercept),
        "rvalue": float(fit.rvalue),
        "msd_times_ps": times_ps.tolist(),
        "msd_ang2": msd_ang2.tolist(),
    }


def _md_worker(
    gpu_id: str,
    structure_path: str,
    out_dir: str,
    temperature_K: float,
    settings_dict: dict[str, Any],
    queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = MDSettings(**settings_dict)
    traj_path = run_dir / "md.traj"
    log_path = run_dir / "md.log"
    summary_path = run_dir / "diffusion_summary.json"
    msd_path = run_dir / "msd.csv"

    try:
        atoms = read(structure_path)
        atoms.calc = _make_calc(
            model_path=settings.model_path,
            head=settings.head,
            default_dtype=settings.default_dtype,
            device="cuda",
        )

        rng = np.random.default_rng(settings.seed + int(round(temperature_K)))
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)

        na_indices = [index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Na"]
        if not na_indices:
            raise ValueError(f"No Na atoms found in structure: {structure_path}")

        dyn = Langevin(
            atoms,
            timestep=settings.timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=settings.friction,
        )
        traj = Trajectory(str(traj_path), mode="w", atoms=atoms)
        logger = MDLogger(dyn, atoms, str(log_path), header=True, stress=False, peratom=False, mode="w")

        scaled_positions: list[np.ndarray] = []

        def sample_frame() -> None:
            scaled_positions.append(atoms.get_scaled_positions()[na_indices].copy())
            traj.write(atoms)

        dyn.attach(sample_frame, interval=settings.sample_interval)
        dyn.attach(logger, interval=settings.sample_interval)
        sample_frame()
        dyn.run(settings.steps)
        if len(scaled_positions) < 3:
            sample_frame()

        summary = _compute_diffusion_summary(
            scaled_positions=scaled_positions,
            cell=np.asarray(atoms.cell.array, dtype=float),
            timestep_fs=settings.timestep_fs,
            sample_interval=settings.sample_interval,
            temperature_K=temperature_K,
        )
        summary.update(
            {
                "gpu_id": gpu_id,
                "structure_path": structure_path,
                "traj_path": str(traj_path),
                "log_path": str(log_path),
            }
        )
        _ensure_json(summary_path, summary)
        with msd_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_ps", "msd_ang2"])
            writer.writerows(zip(summary["msd_times_ps"], summary["msd_ang2"], strict=False))
        queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        error = {
            "ok": False,
            "temperature_K": temperature_K,
            "gpu_id": gpu_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _ensure_json(run_dir / "error.json", error)
        queue.put(error)


def _fit_arrhenius(temperature_summaries: list[dict[str, Any]], reference_temperature_K: float) -> dict[str, Any]:
    valid = [
        summary
        for summary in temperature_summaries
        if summary.get("diffusion_m2_s", 0.0) and float(summary["diffusion_m2_s"]) > 0.0
    ]
    if len(valid) < 2:
        return {
            "activation_barrier_ev": math.nan,
            "prefactor_m2_s": math.nan,
            "reference_temperature_K": reference_temperature_K,
            "diffusion_at_reference_m2_s": math.nan,
            "arrhenius_rvalue": math.nan,
        }

    inv_t = np.array([1.0 / float(summary["temperature_K"]) for summary in valid], dtype=float)
    ln_d = np.log(np.array([float(summary["diffusion_m2_s"]) for summary in valid], dtype=float))
    fit = linregress(inv_t, ln_d)
    activation_barrier_ev = -float(fit.slope) * KB_EV_PER_K
    prefactor_m2_s = math.exp(float(fit.intercept))
    diffusion_at_reference_m2_s = prefactor_m2_s * math.exp(-activation_barrier_ev / (KB_EV_PER_K * reference_temperature_K))
    return {
        "activation_barrier_ev": activation_barrier_ev,
        "prefactor_m2_s": prefactor_m2_s,
        "reference_temperature_K": reference_temperature_K,
        "diffusion_at_reference_m2_s": diffusion_at_reference_m2_s,
        "arrhenius_rvalue": float(fit.rvalue),
    }


def run_multitemperature_md(
    *,
    structure_path: str,
    output_dir: str,
    model_path: str | None = None,
    head: str | None = DEFAULT_MACE_HEAD,
    default_dtype: str = DEFAULT_MACE_DTYPE,
    temperatures_K: list[float] | None = None,
    gpu_ids: list[str] | None = None,
    timestep_fs: float = 1.0,
    steps: int = 4000,
    sample_interval: int = 20,
    friction: float = 0.02,
    seed: int = 2026,
    reference_temperature_K: float = 800.0,
) -> dict[str, Any]:
    structure_path = str(Path(structure_path).expanduser().resolve())
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    temps = tuple(temperatures_K or DEFAULT_MD_TEMPERATURES_K)
    gpu_pool = tuple(gpu_ids or DEFAULT_GPU_IDS)
    if not gpu_pool:
        raise ValueError("At least one GPU id is required for multitemperature MD.")

    settings = MDSettings(
        model_path=str(Path(model_path or DEFAULT_MACE_MODEL).expanduser().resolve()),
        head=head,
        default_dtype=default_dtype,
        timestep_fs=timestep_fs,
        steps=steps,
        friction=friction,
        sample_interval=sample_interval,
        temperatures_K=temps,
        gpu_ids=gpu_pool,
        seed=seed,
    )

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []
    for index, temperature_K in enumerate(temps):
        gpu_id = gpu_pool[index % len(gpu_pool)]
        run_dir = output_root / f"T{int(round(temperature_K))}K"
        proc = ctx.Process(
            target=_md_worker,
            args=(
                str(gpu_id),
                structure_path,
                str(run_dir),
                float(temperature_K),
                asdict(settings),
                queue,
            ),
            name=f"md_T{int(round(temperature_K))}_gpu{gpu_id}",
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    results = []
    deadline = time.time() + 5.0
    while len(results) < len(temps) and time.time() < deadline:
        try:
            results.append(queue.get(timeout=0.5))
        except Empty:
            if all(not proc.is_alive() for proc in procs):
                break

    if len(results) != len(temps):
        exit_codes = [proc.exitcode for proc in procs]
        raise RuntimeError(f"Missing MD worker results. Expected {len(temps)}, got {len(results)}. Exit codes: {exit_codes}")
    errors = [result for result in results if not result.get("ok")]
    if errors:
        raise RuntimeError(f"Multi-temperature MD failed: {errors[0]['error']}")

    summaries = sorted((result["summary"] for result in results), key=lambda item: item["temperature_K"])
    arrhenius = _fit_arrhenius(summaries, reference_temperature_K=reference_temperature_K)
    payload = {
        "structure_path": structure_path,
        "output_dir": str(output_root),
        "temperature_summaries": summaries,
        **arrhenius,
    }
    _ensure_json(output_root / "arrhenius_summary.json", payload)
    return payload


def evaluate_volume_change(
    *,
    sodiated_structure_path: str,
    desodiated_structure_path: str,
    output_dir: str,
    model_path: str | None = None,
    head: str | None = DEFAULT_MACE_HEAD,
    default_dtype: str = DEFAULT_MACE_DTYPE,
    device: str = "cuda:0",
    fmax: float = 0.05,
    max_steps: int = 200,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    calc = _make_calc(
        model_path=str(Path(model_path or DEFAULT_MACE_MODEL).expanduser().resolve()),
        head=head,
        default_dtype=default_dtype,
        device=device,
    )

    def relax_one(structure_path: str, tag: str) -> dict[str, Any]:
        atoms = read(structure_path)
        atoms.calc = calc
        from ase.filters import FrechetCellFilter

        filter_atoms = FrechetCellFilter(atoms)
        optimizer = BFGS(
            filter_atoms,
            logfile=str(output_root / f"{tag}_relax.log"),
            trajectory=str(output_root / f"{tag}_relax.traj"),
        )
        optimizer.run(fmax=fmax, steps=max_steps)
        final_path = output_root / f"{tag}_relaxed.vasp"
        write(final_path, atoms, format="vasp")
        return {
            "initial_volume": float(read(structure_path).get_volume()),
            "relaxed_volume": float(atoms.get_volume()),
            "relaxed_path": str(final_path),
            "potential_energy": float(atoms.get_potential_energy()),
        }

    sodiated = relax_one(sodiated_structure_path, "sodiated")
    desodiated = relax_one(desodiated_structure_path, "desodiated")
    delta_v_over_v = (desodiated["relaxed_volume"] - sodiated["relaxed_volume"]) / sodiated["relaxed_volume"]
    payload = {
        "sodiated": sodiated,
        "desodiated": desodiated,
        "volume_deformation": float(delta_v_over_v),
    }
    _ensure_json(output_root / "volume_change_summary.json", payload)
    return payload
