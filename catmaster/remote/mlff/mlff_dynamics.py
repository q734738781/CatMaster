# Code writing date: 2026-07-17
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: keep molecular-dynamics integration independent of
# the calculator provider and reuse one adapter-created calculator per stage.
# Purpose: shared ASE MD execution for all managed MLFF backends.
from __future__ import annotations

"""Calculator-independent ASE molecular-dynamics execution."""

import contextlib
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.velocitydistribution import Stationary, ZeroRotation, thermalize_momenta


_RESTART_INFO_KEY = "_catmaster_md_restart"
_RESTART_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _max_force_eva(forces: Any) -> float:
    array = np.asarray(forces, dtype=float)
    if array.size == 0:
        return 0.0
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Force arrays must have shape (n_atoms, 3).")
    return float(np.max(np.linalg.norm(array, axis=1)))


def _require_finite_md_state(atoms: Any) -> tuple[float, np.ndarray]:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if not np.all(np.isfinite(positions)):
        raise FloatingPointError("MLFF MD produced non-finite atomic positions.")
    velocities = atoms.get_velocities()
    if velocities is not None and not np.all(np.isfinite(np.asarray(velocities, dtype=float))):
        raise FloatingPointError("MLFF MD produced non-finite atomic velocities.")
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        raise FloatingPointError("MLFF MD calculator returned a non-finite potential energy.")
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.ndim != 2 or forces.shape != (len(atoms), 3):
        raise ValueError("MLFF MD calculator forces must have shape (n_atoms, 3).")
    if not np.all(np.isfinite(forces)):
        raise FloatingPointError("MLFF MD calculator returned non-finite forces.")
    return energy, forces


def _timing_statistics(values: list[float], *, warmup_steps: int = 10) -> dict[str, Any]:
    def stats(samples: list[float]) -> dict[str, float | int | None]:
        if not samples:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p05": None,
                "p95": None,
                "min": None,
                "max": None,
            }
        array = np.asarray(samples, dtype=float)
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p05": float(np.percentile(array, 5)),
            "p95": float(np.percentile(array, 95)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    skipped = min(max(int(warmup_steps), 0), len(values))
    return {
        "all_steps": stats(values),
        "steady_state": stats(values[skipped:]),
        "warmup_steps_excluded": skipped,
        "first_step_s": float(values[0]) if values else None,
    }


def _write_step_timings(path: Path, rows: list[tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "elapsed_s", "steps_per_s"])
        for step, elapsed in rows:
            writer.writerow([step, f"{elapsed:.12g}", f"{1.0 / max(elapsed, 1e-12):.12g}"])


def _require_npt_cell(atoms: Any) -> None:
    if atoms.cell is None or float(getattr(atoms.cell, "volume", 0.0)) <= 1e-6:
        raise ValueError("NPT dynamics requires a nonzero periodic cell.")
    if not bool(np.all(atoms.pbc)):
        raise ValueError("NPT dynamics requires periodic boundary conditions in all three directions.")


def _make_dynamics(atoms: Any, task: dict[str, Any], *, rng: np.random.Generator) -> Any:
    dynamics = dict(task["dynamics"])
    thermostat = dict(task["thermostat"])
    barostat = dict(task["barostat"])
    ensemble = str(dynamics["ensemble"])
    timestep = float(dynamics["timestep_fs"]) * units.fs
    temperature_k = float(dynamics["temperature_K"])

    if ensemble == "nve":
        from ase.md.verlet import VelocityVerlet

        return VelocityVerlet(atoms, timestep=timestep)

    if ensemble == "nvt":
        thermostat_type = str(thermostat["type"])
        if thermostat_type == "bussi":
            from ase.md.bussi import Bussi

            return Bussi(
                atoms,
                timestep=timestep,
                temperature_K=temperature_k,
                taut=float(thermostat["tau_fs"]) * units.fs,
                rng=rng,
            )
        if thermostat_type == "nhc":
            from ase.md.nose_hoover_chain import NoseHooverChainNVT

            return NoseHooverChainNVT(
                atoms,
                timestep=timestep,
                temperature_K=temperature_k,
                tdamp=float(thermostat["tau_fs"]) * units.fs,
                tchain=int(thermostat["tchain"]),
                tloop=int(thermostat["tloop"]),
            )
        if thermostat_type == "langevin":
            from ase.md.langevin import Langevin

            friction = float(thermostat.get("friction_per_fs") or 0.01)
            return Langevin(
                atoms,
                timestep=timestep,
                temperature_K=temperature_k,
                friction=friction / units.fs,
                rng=rng,
            )
        if thermostat_type == "berendsen":
            from ase.md.nvtberendsen import NVTBerendsen

            return NVTBerendsen(
                atoms,
                timestep=timestep,
                temperature_K=temperature_k,
                taut=float(thermostat["tau_fs"]) * units.fs,
            )
        raise ValueError(f"Unsupported thermostat: {thermostat_type}")

    _require_npt_cell(atoms)
    pressure_au = float(barostat["pressure_bar"]) * units.bar
    barostat_type = str(barostat["type"])
    if barostat_type == "berendsen":
        from ase.md.nptberendsen import NPTBerendsen

        return NPTBerendsen(
            atoms,
            timestep=timestep,
            temperature_K=temperature_k,
            taut=float(thermostat["tau_fs"]) * units.fs,
            pressure_au=pressure_au,
            taup=float(barostat["taup_fs"]) * units.fs,
            compressibility_au=float(barostat["compressibility_bar_inv"]) / units.bar,
        )
    if barostat_type == "isotropic_mtk":
        from ase.md.nose_hoover_chain import IsotropicMTKNPT

        return IsotropicMTKNPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature_k,
            pressure_au=pressure_au,
            tdamp=float(thermostat["tau_fs"]) * units.fs,
            pdamp=float(barostat["pdamp_fs"]) * units.fs,
            tchain=int(thermostat["tchain"]),
            pchain=int(barostat["pchain"]),
            tloop=int(thermostat["tloop"]),
            ploop=int(barostat["ploop"]),
        )
    if barostat_type == "inhomogeneous_mtk":
        from ase.md.nose_hoover_chain import MTKNPT

        return MTKNPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature_k,
            pressure_au=pressure_au,
            tdamp=float(thermostat["tau_fs"]) * units.fs,
            pdamp=float(barostat["pdamp_fs"]) * units.fs,
            tchain=int(thermostat["tchain"]),
            pchain=int(barostat["pchain"]),
            tloop=int(thermostat["tloop"]),
            ploop=int(barostat["ploop"]),
        )
    raise ValueError(f"Unsupported barostat: {barostat_type}")


def _temperature_schedule(dynamics_config: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved target-temperature schedule for one MD segment."""

    start_k = float(dynamics_config["temperature_K"])
    configured_end_k = float(dynamics_config.get("temperature_end_K") or 0.0)
    variable = configured_end_k > 0 and not math.isclose(configured_end_k, start_k)
    return {
        "mode": "linear" if variable else "constant",
        "start_K": start_k,
        "end_K": configured_end_k if variable else start_k,
        "steps": int(dynamics_config["steps"]),
        "update_interval_steps": 1 if variable else 0,
        "temperature_api": "set_temperature" if variable else "constructor",
    }


def _run_dynamics(dynamics: Any, dynamics_config: dict[str, Any]) -> dict[str, Any]:
    """Run constant-temperature MD or a public-ASE-API linear schedule."""

    schedule = _temperature_schedule(dynamics_config)
    steps = int(schedule["steps"])
    if schedule["mode"] == "constant":
        dynamics.run(steps)
        return schedule

    setter = getattr(dynamics, "set_temperature", None)
    if not callable(setter):
        raise ValueError(
            f"{type(dynamics).__name__} does not expose ASE set_temperature(); "
            "choose a supported thermostat/barostat for variable-temperature MD."
        )
    if steps < 2:
        raise ValueError("A variable-temperature schedule requires at least two MD steps.")

    # Dynamics.irun() yields the initial state before advancing the first step.
    # Set each target immediately before its corresponding integration step,
    # using only ASE's public generator and temperature setter interfaces.
    iterator = iter(dynamics.irun(steps))
    try:
        next(iterator)
    except StopIteration as exc:  # pragma: no cover - defensive API guard
        raise RuntimeError("ASE dynamics stopped before the first scheduled MD step.") from exc

    start_k = float(schedule["start_K"])
    end_k = float(schedule["end_K"])
    initial_nsteps = int(getattr(dynamics, "nsteps", 0))
    for index in range(steps):
        fraction = index / (steps - 1)
        setter(temperature_K=start_k + (end_k - start_k) * fraction)
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"ASE dynamics stopped after {getattr(dynamics, 'nsteps', 0)} steps "
                f"during a {steps}-step temperature schedule."
            ) from exc

    completed_steps = int(getattr(dynamics, "nsteps", 0)) - initial_nsteps
    if completed_steps != steps:  # pragma: no cover - defensive API guard
        raise RuntimeError(f"ASE dynamics completed {completed_steps} of {steps} scheduled MD steps.")
    return schedule


def _prepare_initial_velocities(
    atoms: Any,
    *,
    dynamics: dict[str, Any],
    rng: np.random.Generator,
) -> str:
    if atoms.has("momenta") and not bool(dynamics["reinitialize_velocities"]):
        velocities = atoms.get_velocities()
        if velocities is None or not np.all(np.isfinite(velocities)):
            raise ValueError("Input momenta must produce finite velocities.")
        return "input_last_frame"

    initial_temperature = float(dynamics.get("initial_temperature_K") or dynamics["temperature_K"])
    thermalize_momenta(
        atoms,
        initial_temperature,
        exact_temperature=bool(dynamics["force_temp"]),
        rng=rng,
    )
    Stationary(atoms)
    if bool(dynamics["zero_rotation"]):
        ZeroRotation(atoms)
    if bool(dynamics["reinitialize_velocities"]):
        return "generated_explicit_reinitialization"
    return "generated_missing_input_velocities"


def _integrator_id(task: dict[str, Any]) -> str:
    dynamics = dict(task["dynamics"])
    if dynamics["ensemble"] == "nvt":
        return f"nvt:{task['thermostat']['type']}"
    if dynamics["ensemble"] == "npt":
        return f"npt:{task['barostat']['type']}"
    return "nve:verlet"


def _prepare_rng(
    atoms: Any,
    *,
    task: dict[str, Any],
    seed: int,
) -> tuple[np.random.Generator, str, dict[str, Any] | None]:
    checkpoint = atoms.info.pop(_RESTART_INFO_KEY, None)
    rng = np.random.default_rng(seed)
    if checkpoint is None:
        return rng, "configured_seed", None
    if not isinstance(checkpoint, dict):
        raise ValueError("CatMaster MD restart metadata must be an object.")
    if bool(task["dynamics"]["reinitialize_velocities"]):
        return rng, "configured_seed", checkpoint
    if checkpoint.get("schema_version") != _RESTART_SCHEMA_VERSION:
        raise ValueError("Unsupported CatMaster MD restart schema version.")
    if checkpoint.get("integrator") != _integrator_id(task):
        return rng, "configured_seed", checkpoint
    state = checkpoint.get("rng_state")
    if not isinstance(state, dict):
        raise ValueError("CatMaster MD restart metadata is missing rng_state.")
    if state.get("bit_generator") != type(rng.bit_generator).__name__:
        raise ValueError("CatMaster MD restart RNG is incompatible with the current NumPy generator.")
    rng.bit_generator.state = state
    return rng, "restart_checkpoint", checkpoint


def _restore_integrator_state(dynamics: Any, checkpoint: dict[str, Any] | None, *, rng_source: str) -> str:
    if rng_source != "restart_checkpoint" or checkpoint is None:
        return "not_restored"
    state = checkpoint.get("integrator_state")
    if not isinstance(state, dict):
        return "not_present"
    if type(dynamics).__name__ == "Bussi" and "transferred_energy" in state:
        dynamics.transferred_energy = float(state["transferred_energy"])
        return "restored"
    if type(dynamics).__name__ == "Langevin":
        return "not_required"
    return "not_supported"


def _checkpoint_payload(task: dict[str, Any], rng: np.random.Generator, seed: int, dynamics: Any) -> dict[str, Any]:
    integrator_state: dict[str, Any] = {}
    if type(dynamics).__name__ == "Bussi":
        integrator_state["transferred_energy"] = float(dynamics.transferred_energy)
    return {
        "schema_version": _RESTART_SCHEMA_VERSION,
        "integrator": _integrator_id(task),
        "rng_seed": seed,
        "rng_state": rng.bit_generator.state,
        "integrator_state": integrator_state,
    }


def _write_checkpoint_frame(
    trajectory: Trajectory,
    *,
    atoms: Any,
    task: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    dynamics: Any,
) -> None:
    previous = atoms.info.get(_RESTART_INFO_KEY)
    atoms.info[_RESTART_INFO_KEY] = _checkpoint_payload(task, rng, seed, dynamics)
    try:
        trajectory.write(atoms)
    finally:
        if previous is None:
            atoms.info.pop(_RESTART_INFO_KEY, None)
        else:
            atoms.info[_RESTART_INFO_KEY] = previous


def _write_restart(path: Path, *, atoms: Any, task: dict[str, Any], rng: np.random.Generator, seed: int, dynamics: Any) -> None:
    restart_atoms = atoms.copy()
    restart_atoms.info[_RESTART_INFO_KEY] = _checkpoint_payload(task, rng, seed, dynamics)
    write(path, restart_atoms)


def run_single(
    *,
    source: Path,
    output_dir: Path,
    config: dict[str, Any],
    item_config: dict[str, Any],
    adapter: Any,
) -> dict[str, Any]:
    task = dict(config["task_config"])
    dynamics_config = dict(task["dynamics"])
    output_config = dict(task["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not bool(output_config["overwrite"]):
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("completed") is True:
            return existing
        raise FileExistsError(f"Incomplete output already exists: {output_dir}")

    atoms = read(str(source), index=-1)
    calculator = adapter.calculator_for(atoms, item_config)
    atoms.calc = calculator
    seed = int(dynamics_config["seed"])
    rng, rng_source, restart_checkpoint = _prepare_rng(atoms, task=task, seed=seed)
    velocity_source = _prepare_initial_velocities(atoms, dynamics=dynamics_config, rng=rng)

    with contextlib.suppress(Exception):
        write(output_dir / "start.vasp", atoms, format="vasp")
    write(output_dir / "start.traj", atoms)
    trajectory_path = output_dir / "md.traj"
    log_path = output_dir / "md.log"
    if bool(output_config["overwrite"]):
        for path in (trajectory_path, log_path):
            if path.exists():
                path.unlink()

    dynamics = _make_dynamics(atoms, task, rng=rng)
    temperature_schedule = _temperature_schedule(dynamics_config)
    integrator_state_source = _restore_integrator_state(
        dynamics,
        restart_checkpoint,
        rng_source=rng_source,
    )
    trajectory = Trajectory(str(trajectory_path), "w", atoms=atoms)
    logger = MDLogger(
        dynamics,
        atoms,
        str(log_path),
        header=True,
        stress=bool(output_config["log_stress"]),
        peratom=False,
        mode="w",
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": "md",
        "backend": str(config["backend"]),
        "model": str(item_config.get("model") or item_config.get("checkpoint_artifact") or ""),
        "provider_version": adapter.provider_version,
        "device": str(item_config.get("device") or "auto"),
        "input": str(source),
        "dynamics": dynamics_config,
        "temperature_schedule": temperature_schedule,
        "thermostat": dict(task["thermostat"]),
        "barostat": dict(task["barostat"]) if dynamics_config["ensemble"] == "npt" else None,
        "output": output_config,
        "input_frame": -1,
        "velocity_source": velocity_source,
        "rng_seed": seed,
        "rng_source": rng_source,
        "integrator_state_source": integrator_state_source,
        "start_trajectory": "start.traj",
        "total_time_ps": float(dynamics_config["steps"]) * float(dynamics_config["timestep_fs"]) / 1000.0,
        "started_at": _now_iso(),
        "completed": False,
        "error": None,
    }
    _write_json(summary_path, summary)
    started = time.perf_counter()
    step_timings: list[tuple[int, float]] = []
    timing_previous = [started]

    def record_step_timing() -> None:
        _require_finite_md_state(atoms)
        now = time.perf_counter()
        step = int(getattr(dynamics, "nsteps", 0))
        if step > 0:
            step_timings.append((step, now - timing_previous[0]))
        timing_previous[0] = now

    try:
        initial_evaluation_started = time.perf_counter()
        initial_energy, initial_forces = _require_finite_md_state(atoms)
        summary.update(
            {
                "initial_energy_eV": initial_energy,
                "initial_max_force_eVA": _max_force_eva(initial_forces),
                "initial_evaluation_s": time.perf_counter() - initial_evaluation_started,
            }
        )
        _write_json(summary_path, summary)
        dynamics.attach(
            _write_checkpoint_frame,
            interval=int(output_config["traj_interval"]),
            trajectory=trajectory,
            atoms=atoms,
            task=task,
            rng=rng,
            seed=seed,
            dynamics=dynamics,
        )
        dynamics.attach(logger, interval=int(output_config["log_interval"]))
        dynamics.attach(record_step_timing, interval=1)
        _run_dynamics(dynamics, dynamics_config)

        periodic = bool(any(bool(value) for value in atoms.pbc)) and float(atoms.cell.volume) > 1e-6
        output_name = "final.vasp" if periodic else "final.xyz"
        write(output_dir / output_name, atoms, format="vasp" if periodic else "xyz")
        restart_path = output_dir / "restart.traj"
        _write_restart(restart_path, atoms=atoms, task=task, rng=rng, seed=seed, dynamics=dynamics)
        energy, final_forces = _require_finite_md_state(atoms)
        max_force = _max_force_eva(final_forces)
        elapsed = time.perf_counter() - started
        timings_path = output_dir / "step_timings.csv"
        _write_step_timings(timings_path, step_timings)
        timing_values = [value for _, value in step_timings]
        timing_stats = _timing_statistics(timing_values)
        steady_mean = timing_stats["steady_state"]["mean"]
        summary.update(
            {
                "completed": True,
                "finished_at": _now_iso(),
                "final_energy_eV": energy,
                "max_force_eVA": max_force,
                "provider_metadata": adapter.provider_metadata(atoms, item_config, calculator),
                "trajectory": trajectory_path.name,
                "log": log_path.name,
                "output_structure": output_name,
                "restart_trajectory": restart_path.name,
                "elapsed_s": elapsed,
                "steps_per_s": float(dynamics_config["steps"]) / max(elapsed, 1e-12),
                "step_timings": timings_path.name,
                "step_timing_statistics_s": timing_stats,
                "timed_steps_elapsed_s": float(sum(timing_values)),
                "pre_step_overhead_s": max(elapsed - float(sum(timing_values)), 0.0),
                "steady_state_steps_per_s": (
                    1.0 / float(steady_mean) if steady_mean not in (None, 0.0) else None
                ),
            }
        )
        _write_json(summary_path, summary)
        return summary
    except Exception as exc:
        if step_timings:
            _write_step_timings(output_dir / "step_timings.csv", step_timings)
        summary.update(
            {
                "completed": False,
                "finished_at": _now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.perf_counter() - started,
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(summary_path, summary)
        (output_dir / "error.txt").write_text(summary["traceback"], encoding="utf-8")
        raise
    finally:
        with contextlib.suppress(Exception):
            trajectory.close()
        with contextlib.suppress(Exception):
            logger.close()


__all__ = ["_max_force_eva", "_require_finite_md_state", "_timing_statistics", "run_single"]
