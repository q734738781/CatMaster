from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


def _resolve_device(preference: str) -> str:
    import torch

    device = str(preference or "auto").strip().lower() or "auto"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested but torch.cuda.is_available() is false; "
            "check the remote driver/CUDA/PyTorch environment."
        )
    return device


def _collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    skip_prefixes = ("mace_batch_", "mace_sp_batch_", "mace_md_batch_", "vasp_batch_")
    internal_dirs = {"metadata", ".catmaster"}
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part.startswith(skip_prefixes) for part in path.parts):
            dirnames[:] = []
            continue
        if any(part in internal_dirs for part in path.parts):
            dirnames[:] = []
            continue
        if "summary.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in internal_dirs and not d.startswith(skip_prefixes)
        ]
        for fname in filenames:
            p = path / fname
            if fname in {"POSCAR", "CONTCAR"}:
                files.append(p)
                continue
            if p.suffix.lower() in {".vasp", ".poscar", ".cif", ".xyz"}:
                files.append(p)
    return sorted(files, key=lambda p: str(p))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _positive_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _positive_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _timing_statistics(values: list[float], *, warmup_steps: int = 10) -> dict[str, Any]:
    def _stats(samples: list[float]) -> dict[str, float | int | None]:
        if not samples:
            return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
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
        "all_steps": _stats(values),
        "steady_state": _stats(values[skipped:]),
        "warmup_steps_excluded": skipped,
        "first_step_s": float(values[0]) if values else None,
    }


def _write_step_timings(path: Path, rows: list[tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "elapsed_s", "steps_per_s"])
        for step, elapsed in rows:
            writer.writerow([step, f"{elapsed:.12g}", f"{1.0 / max(elapsed, 1e-12):.12g}"])


def _default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "calculator": {
            "model": "mh-1",
            "head": "omat_pbe",
            "dispersion": False,
            "default_dtype": "float32",
            "enable_cueq": False,
            "compile_mode": None,
            "compute_atomic_stresses": False,
        },
        "dynamics": {
            "ensemble": "nvt",
            "temperature_K": 300.0,
            "initial_temperature_K": None,
            "timestep_fs": 1.0,
            "steps": 1000,
            "seed": 2026,
            "zero_rotation": False,
            "force_temp": False,
        },
        "thermostat": {
            "type": "bussi",
            "tau_fs": 100.0,
            "friction_per_fs": None,
            "tchain": 3,
            "tloop": 1,
        },
        "barostat": None,
        "output": {
            "traj_interval": 10,
            "log_interval": 10,
            "log_stress": False,
            "overwrite": False,
        },
    }


def _default_barostat_config() -> dict[str, Any]:
    return {
        "type": "isotropic_mtk",
        "pressure_bar": 1.01325,
        "taup_fs": 1000.0,
        "pdamp_fs": 1000.0,
        "compressibility_bar_inv": None,
        "pchain": 3,
        "ploop": 1,
    }


def _merge_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_from_compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = _default_config()
    md_config = payload.get("md_config") or {}
    if not isinstance(md_config, dict):
        raise ValueError("md_config must be an object.")
    if isinstance(md_config.get("barostat"), dict):
        md_config = dict(md_config)
        md_config["barostat"] = _merge_dict(_default_barostat_config(), md_config["barostat"])
    cfg = _merge_dict(cfg, md_config)
    calc = cfg["calculator"]
    for key in ("model", "head", "dispersion", "default_dtype", "enable_cueq", "compile_mode"):
        if key in payload:
            calc[key] = payload[key]
    return cfg


def _load_config(path: str | None) -> dict[str, Any]:
    cfg = _default_config()
    if not path:
        return cfg
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("params JSON must contain an object.")
    if "md_config" in payload:
        return _config_from_compact_payload(payload)
    return _merge_dict(cfg, payload)


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    calc = config["calculator"]
    dyn = config["dynamics"]
    thermo = config["thermostat"]
    output = config["output"]
    barostat = config.get("barostat")

    ensemble = str(dyn.get("ensemble", "")).lower()
    if ensemble not in {"nve", "nvt", "npt"}:
        raise ValueError("dynamics.ensemble must be one of: nve, nvt, npt.")
    dyn["ensemble"] = ensemble
    dyn["temperature_K"] = _positive_float(dyn["temperature_K"], name="dynamics.temperature_K")
    if dyn.get("initial_temperature_K") is not None:
        dyn["initial_temperature_K"] = _positive_float(
            dyn["initial_temperature_K"],
            name="dynamics.initial_temperature_K",
        )
    dyn["timestep_fs"] = _positive_float(dyn["timestep_fs"], name="dynamics.timestep_fs")
    dyn["steps"] = _positive_int(dyn["steps"], name="dynamics.steps")
    dyn["seed"] = int(dyn.get("seed", 2026))
    dyn["zero_rotation"] = bool(dyn.get("zero_rotation", False))
    dyn["force_temp"] = bool(dyn.get("force_temp", False))

    thermostat_type = str(thermo.get("type", "")).lower()
    if thermostat_type not in {"bussi", "nhc", "langevin", "berendsen"}:
        raise ValueError("thermostat.type must be one of: bussi, nhc, langevin, berendsen.")
    thermo["type"] = thermostat_type
    thermo["tau_fs"] = _positive_float(thermo["tau_fs"], name="thermostat.tau_fs")
    if thermo.get("friction_per_fs") is not None:
        thermo["friction_per_fs"] = _positive_float(
            thermo["friction_per_fs"],
            name="thermostat.friction_per_fs",
        )
    thermo["tchain"] = _positive_int(thermo["tchain"], name="thermostat.tchain")
    thermo["tloop"] = _positive_int(thermo["tloop"], name="thermostat.tloop")

    if ensemble == "npt":
        if barostat is None:
            barostat = _default_barostat_config()
            config["barostat"] = barostat
        else:
            barostat = _merge_dict(_default_barostat_config(), barostat)
            config["barostat"] = barostat
        barostat_type = str(barostat.get("type", "")).lower()
        if barostat_type not in {"isotropic_mtk", "full_mtk", "berendsen"}:
            raise ValueError("barostat.type must be one of: isotropic_mtk, full_mtk, berendsen.")
        barostat["type"] = barostat_type
        barostat["pressure_bar"] = _positive_float(barostat["pressure_bar"], name="barostat.pressure_bar")
        barostat["taup_fs"] = _positive_float(barostat["taup_fs"], name="barostat.taup_fs")
        barostat["pdamp_fs"] = _positive_float(barostat["pdamp_fs"], name="barostat.pdamp_fs")
        barostat["pchain"] = _positive_int(barostat["pchain"], name="barostat.pchain")
        barostat["ploop"] = _positive_int(barostat["ploop"], name="barostat.ploop")
        if barostat_type == "berendsen":
            if barostat.get("compressibility_bar_inv") is None:
                raise ValueError("barostat.compressibility_bar_inv is required for Berendsen NPT.")
            barostat["compressibility_bar_inv"] = _positive_float(
                barostat["compressibility_bar_inv"],
                name="barostat.compressibility_bar_inv",
            )
    elif barostat is not None:
        raise ValueError("barostat is only valid when dynamics.ensemble is npt.")

    output["traj_interval"] = _positive_int(output["traj_interval"], name="output.traj_interval")
    output["log_interval"] = _positive_int(output["log_interval"], name="output.log_interval")
    output["log_stress"] = bool(output.get("log_stress", False))
    output["overwrite"] = bool(output.get("overwrite", False))

    dtype = str(calc.get("default_dtype", "float32"))
    if dtype not in {"float32", "float64"}:
        raise ValueError("calculator.default_dtype must be float32 or float64.")
    calc["default_dtype"] = dtype
    calc["head"] = str(calc.get("head") or "").strip() or None
    calc["dispersion"] = bool(calc.get("dispersion", False))
    calc["enable_cueq"] = bool(calc.get("enable_cueq", False))
    compile_mode = calc.get("compile_mode")
    if compile_mode in (None, "", "none", "None"):
        calc["compile_mode"] = None
    else:
        compile_mode = str(compile_mode).strip()
        if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(
                "calculator.compile_mode must be one of: none, default, reduce-overhead, max-autotune."
            )
        calc["compile_mode"] = compile_mode
    calc["compute_atomic_stresses"] = bool(calc.get("compute_atomic_stresses", False))
    return config


def _make_calculator(config: dict[str, Any], *, device: str):
    from mace.calculators import mace_mp

    calc = config["calculator"]
    device = _resolve_device(device)

    kwargs: dict[str, Any] = {
        "model": calc["model"],
        "device": device,
        "default_dtype": calc["default_dtype"],
        "dispersion": bool(calc["dispersion"]),
    }
    if calc.get("head"):
        kwargs["head"] = calc["head"]
    if calc.get("enable_cueq"):
        if not device.startswith("cuda"):
            raise ValueError("calculator.enable_cueq requires a CUDA device.")
        kwargs["enable_cueq"] = True
    if calc.get("compile_mode"):
        if not device.startswith("cuda"):
            raise ValueError("calculator.compile_mode requires a CUDA device for this managed MD path.")
        kwargs["compile_mode"] = calc["compile_mode"]
    if calc.get("compute_atomic_stresses"):
        kwargs["compute_atomic_stresses"] = True
    return mace_mp(**kwargs), device


def _require_npt_cell(atoms) -> None:
    if atoms.cell is None or getattr(atoms.cell, "volume", 0.0) <= 1e-6:
        raise ValueError("NPT dynamics requires a nonzero periodic cell.")
    if not bool(np.all(atoms.pbc)):
        raise ValueError("NPT dynamics requires periodic boundary conditions in all three directions.")


def _make_dynamics(atoms, config: dict[str, Any], *, rng: np.random.Generator):
    dyn = config["dynamics"]
    thermo = config["thermostat"]
    barostat = config.get("barostat")
    ensemble = dyn["ensemble"]
    timestep = float(dyn["timestep_fs"]) * units.fs
    temperature_K = float(dyn["temperature_K"])

    if ensemble == "nve":
        from ase.md.verlet import VelocityVerlet

        return VelocityVerlet(atoms, timestep=timestep)

    if ensemble == "nvt":
        thermostat = thermo["type"]
        if thermostat == "bussi":
            from ase.md.bussi import Bussi

            return Bussi(
                atoms,
                timestep=timestep,
                temperature_K=temperature_K,
                taut=float(thermo["tau_fs"]) * units.fs,
                rng=rng,
            )
        if thermostat == "nhc":
            from ase.md.nose_hoover_chain import NoseHooverChainNVT

            return NoseHooverChainNVT(
                atoms,
                timestep=timestep,
                temperature_K=temperature_K,
                tdamp=float(thermo["tau_fs"]) * units.fs,
                tchain=int(thermo["tchain"]),
                tloop=int(thermo["tloop"]),
            )
        if thermostat == "langevin":
            from ase.md.langevin import Langevin

            friction_per_fs = float(thermo["friction_per_fs"] or 0.01)
            return Langevin(
                atoms,
                timestep=timestep,
                temperature_K=temperature_K,
                friction=friction_per_fs / units.fs,
                rng=rng,
            )
        if thermostat == "berendsen":
            from ase.md.nvtberendsen import NVTBerendsen

            return NVTBerendsen(
                atoms,
                timestep=timestep,
                temperature_K=temperature_K,
                taut=float(thermo["tau_fs"]) * units.fs,
            )
        raise ValueError(f"Unsupported thermostat: {thermostat}")

    _require_npt_cell(atoms)
    assert barostat is not None
    pressure_au = float(barostat["pressure_bar"]) * units.bar
    if barostat["type"] == "berendsen":
        from ase.md.nptberendsen import NPTBerendsen

        return NPTBerendsen(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            taut=float(thermo["tau_fs"]) * units.fs,
            pressure_au=pressure_au,
            taup=float(barostat["taup_fs"]) * units.fs,
            compressibility_au=float(barostat["compressibility_bar_inv"]) / units.bar,
        )
    if barostat["type"] == "isotropic_mtk":
        from ase.md.nose_hoover_chain import IsotropicMTKNPT

        return IsotropicMTKNPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            pressure_au=pressure_au,
            tdamp=float(thermo["tau_fs"]) * units.fs,
            pdamp=float(barostat["pdamp_fs"]) * units.fs,
            tchain=int(thermo["tchain"]),
            pchain=int(barostat["pchain"]),
            tloop=int(thermo["tloop"]),
            ploop=int(barostat["ploop"]),
        )
    if barostat["type"] == "full_mtk":
        from ase.md.nose_hoover_chain import MTKNPT

        return MTKNPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            pressure_au=pressure_au,
            tdamp=float(thermo["tau_fs"]) * units.fs,
            pdamp=float(barostat["pdamp_fs"]) * units.fs,
            tchain=int(thermo["tchain"]),
            pchain=int(barostat["pchain"]),
            tloop=int(thermo["tloop"]),
            ploop=int(barostat["ploop"]),
        )
    raise ValueError(f"Unsupported barostat: {barostat['type']}")


def _run_md_single(
    *,
    structure_path: Path,
    output_dir: Path,
    calc,
    config: dict[str, Any],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    dyn_cfg = config["dynamics"]
    out_cfg = config["output"]
    calc_cfg = config["calculator"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not out_cfg["overwrite"]:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("completed") is True:
                return existing
        except Exception:
            pass

    atoms = read(str(structure_path), index=0)
    atoms.calc = calc

    rng = np.random.default_rng(int(seed))
    initial_temperature_K = dyn_cfg.get("initial_temperature_K") or dyn_cfg["temperature_K"]
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=float(initial_temperature_K),
        force_temp=bool(dyn_cfg["force_temp"]),
        rng=rng,
    )
    Stationary(atoms)
    if dyn_cfg["zero_rotation"]:
        ZeroRotation(atoms)

    with contextlib.suppress(Exception):
        write(output_dir / "start.vasp", atoms, format="vasp")

    traj_path = output_dir / "md.traj"
    log_path = output_dir / "md.log"
    if out_cfg["overwrite"]:
        for path in (traj_path, log_path):
            if path.exists():
                path.unlink()

    dyn = _make_dynamics(atoms, config, rng=rng)
    traj = Trajectory(str(traj_path), mode="w", atoms=atoms)
    logger = MDLogger(
        dyn,
        atoms,
        str(log_path),
        header=True,
        stress=bool(out_cfg["log_stress"]),
        peratom=False,
        mode="w",
    )
    summary: dict[str, Any] = {
        "input": str(structure_path),
        "device": device,
        "calculator": calc_cfg,
        "dynamics": dyn_cfg,
        "thermostat": config["thermostat"] if dyn_cfg["ensemble"] == "nvt" else config["thermostat"],
        "barostat": config.get("barostat") if dyn_cfg["ensemble"] == "npt" else None,
        "output": out_cfg,
        "total_time_ps": float(dyn_cfg["steps"]) * float(dyn_cfg["timestep_fs"]) / 1000.0,
        "started_at": _now_iso(),
        "completed": False,
        "error": None,
    }
    run_started = time.perf_counter()
    step_timings: list[tuple[int, float]] = []
    timing_previous = [run_started]

    def _record_step_timing() -> None:
        now = time.perf_counter()
        step = int(getattr(dyn, "nsteps", 0))
        if step <= 0:
            timing_previous[0] = now
            return
        step_timings.append((step, now - timing_previous[0]))
        timing_previous[0] = now
    _write_json(summary_path, summary)

    try:
        dyn.attach(traj.write, interval=int(out_cfg["traj_interval"]))
        dyn.attach(logger, interval=int(out_cfg["log_interval"]))
        dyn.attach(_record_step_timing, interval=1)
        dyn.run(int(dyn_cfg["steps"]))

        has_lattice = atoms.cell is not None and getattr(atoms.cell, "volume", 0) > 1e-6
        final_name = "final.vasp" if has_lattice else "final.xyz"
        write(output_dir / final_name, atoms, format="vasp" if has_lattice else "xyz")
        final_energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        elapsed_s = time.perf_counter() - run_started
        timing_path = output_dir / "step_timings.csv"
        _write_step_timings(timing_path, step_timings)
        timing_values = [elapsed for _, elapsed in step_timings]
        timed_steps_elapsed_s = float(sum(timing_values))
        timing_statistics = _timing_statistics(timing_values)
        steady_mean = timing_statistics["steady_state"]["mean"]
        summary.update(
            {
                "completed": True,
                "finished_at": _now_iso(),
                "final_energy_eV": final_energy,
                "max_force_abs_eVA": float(np.max(np.abs(forces))),
                "trajectory": traj_path.name,
                "log": log_path.name,
                "output_structure": final_name,
                "elapsed_s": elapsed_s,
                "steps_per_s": float(dyn_cfg["steps"]) / max(elapsed_s, 1e-12),
                "step_timings": timing_path.name,
                "step_timing_statistics_s": timing_statistics,
                "timed_steps_elapsed_s": timed_steps_elapsed_s,
                "pre_step_overhead_s": max(elapsed_s - timed_steps_elapsed_s, 0.0),
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
                "error": repr(exc),
                "elapsed_s": time.perf_counter() - run_started,
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(summary_path, summary)
        (output_dir / "error.txt").write_text(summary.get("traceback") or "", encoding="utf-8")
        raise
    finally:
        with contextlib.suppress(Exception):
            traj.close()
        with contextlib.suppress(Exception):
            logger.close()


def run_mace_md_batch(
    input_path: str,
    *,
    output_root: str,
    params_path: str | None = None,
    config: dict[str, Any] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    input_root = Path(input_path)
    output_root_path = Path(output_root)
    if not input_root.is_dir():
        raise ValueError("input_path must be a directory for mace_md_batch.")
    if _is_within(output_root_path, input_root):
        raise ValueError("output_root must not be inside input_path.")

    md_config = _validate_config(_merge_dict(_load_config(params_path), config or {}))
    structures = _collect_structure_files(input_root)
    if not structures:
        raise ValueError(f"No structure files found in directory: {input_root}")
    output_root_path.mkdir(parents=True, exist_ok=True)

    calc, resolved_device = _make_calculator(md_config, device=device)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    base_seed = int(md_config["dynamics"]["seed"])
    for idx, struct in enumerate(structures):
        rel_path = struct.relative_to(input_root)
        out_dir = output_root_path / rel_path.with_suffix("")
        try:
            summary = _run_md_single(
                structure_path=struct,
                output_dir=out_dir,
                calc=calc,
                config=md_config,
                device=resolved_device,
                seed=base_seed + idx,
            )
            results.append(
                {
                    "input_rel": rel_path.as_posix(),
                    "output_rel": out_dir.relative_to(output_root_path).as_posix(),
                    "summary": summary,
                }
            )
        except Exception as exc:
            errors.append({"input_rel": rel_path.as_posix(), "error": str(exc)})

    batch_summary = {
        "input_root": str(input_root),
        "output_root": str(output_root_path),
        "device": resolved_device,
        "calculator": md_config["calculator"],
        "dynamics": md_config["dynamics"],
        "thermostat": md_config["thermostat"],
        "barostat": md_config.get("barostat"),
        "output": md_config["output"],
        "total_time_ps": float(md_config["dynamics"]["steps"]) * float(md_config["dynamics"]["timestep_fs"]) / 1000.0,
        "results": results,
        "errors": errors,
    }
    _write_json(output_root_path / "batch_summary.json", batch_summary)
    return batch_summary


def _parse_json_object(text: str | None, *, name: str) -> dict[str, Any]:
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return payload


def _cli_compact_config(args: argparse.Namespace) -> dict[str, Any]:
    return _config_from_compact_payload(
        {
            "model": args.model,
            "head": args.head.strip() or None,
            "dispersion": bool(args.dispersion),
            "default_dtype": args.default_dtype,
            "md_config": _parse_json_object(args.md_config, name="--md_config"),
        }
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run an ASE-backed MACE molecular-dynamics batch.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    parser.add_argument("--params", default=None, help="JSON parameter file produced by mace_md_batch")
    parser.add_argument("--model", default="mh-1", help="MACE model identifier or path for direct CLI use.")
    parser.add_argument("--head", default="omat_pbe", help="Model head for direct CLI use. Use '' for none.")
    parser.add_argument("--dispersion", type=_parse_bool, default=False, help="Enable dispersion for direct CLI use.")
    parser.add_argument("--default_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="auto", help="Device to use: auto|cpu|cuda|cuda:0")
    parser.add_argument(
        "--md_config",
        default="{}",
        help="Compact JSON object with optional dynamics/thermostat/barostat/output/calculator groups.",
    )
    args = parser.parse_args()

    run_mace_md_batch(
        input_path=args.input,
        output_root=args.output_root,
        params_path=args.params,
        config=None if args.params else _cli_compact_config(args),
        device=args.device,
    )


if __name__ == "__main__":
    _cli()
