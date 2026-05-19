from __future__ import annotations

import argparse
import copy
import csv
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

_IMAGE_SUFFIXES = {".vasp", ".poscar", ".cif"}
_AUTONEB_TARGET_TOTAL_IMAGES = 7


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_image_file(path: Path) -> int | None:
    if not path.is_file():
        return None
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    stem = path.stem
    if not stem.isdigit():
        return None
    return int(stem)


def _list_task_image_files(task_dir: Path) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in sorted(task_dir.iterdir()):
        if path.is_dir():
            raise ValueError(f"Nested directories are forbidden inside MACE NEB task directories: {task_dir}")
        idx = _parse_image_file(path)
        if idx is None:
            continue
        indexed.append((idx, path))
    if len(indexed) < 2:
        raise ValueError(f"MACE NEB task directory must contain numbered image files like 00.vasp, 01.vasp, ...: {task_dir}")
    expected = list(range(len(indexed)))
    found = [idx for idx, _ in indexed]
    if found != expected:
        raise ValueError(f"MACE NEB image numbering must be contiguous from 00. Found {found} in {task_dir}")
    return [path for _, path in indexed]


def _discover_task_dirs(input_root: Path) -> list[Path]:
    child_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    root_indexed_images = [path.name for path in sorted(input_root.iterdir()) if _parse_image_file(path) is not None]
    if child_dirs and root_indexed_images:
        raise ValueError(
            "input root mixes batch task subdirectories with root-level numbered image files. "
            "Choose exactly one layout: either a single flat image tree or a batch root containing only task directories."
        )
    if not child_dirs:
        raise ValueError("input root contains no task directories")
    for child in child_dirs:
        _list_task_image_files(child)
    return child_dirs


def _constraint_signature(atoms) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for constraint in atoms.constraints or []:
        entry: dict[str, Any] = {"type": type(constraint).__name__}
        for attr in ("index", "indices", "mask", "a"):
            if not hasattr(constraint, attr):
                continue
            value = getattr(constraint, attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (tuple, list)):
                value = [_json_ready(v) for v in value]
            else:
                value = _json_ready(value)
            entry[attr] = value
        signatures.append(entry)
    return signatures


def _validate_image_set(images: list[Any]) -> None:
    if len(images) < 2:
        raise ValueError("Each MACE NEB task requires at least two images.")
    first = images[0]
    first_numbers = first.get_atomic_numbers()
    first_cell = first.cell.array
    first_pbc = first.pbc
    first_constraints = _constraint_signature(first)
    for idx, atoms in enumerate(images[1:], start=1):
        if len(atoms) != len(first):
            raise ValueError(f"Image {idx:02d} has a different atom count.")
        if not np.array_equal(atoms.get_atomic_numbers(), first_numbers):
            raise ValueError(f"Image {idx:02d} has a different element ordering.")
        if not np.allclose(atoms.cell.array, first_cell, atol=1e-6, rtol=0):
            raise ValueError(f"Image {idx:02d} has a different cell.")
        if not np.array_equal(atoms.pbc, first_pbc):
            raise ValueError(f"Image {idx:02d} has different PBC settings.")
        if _constraint_signature(atoms) != first_constraints:
            raise ValueError(f"Image {idx:02d} has different constraints.")


def _resolve_device(preference: str) -> str:
    import torch

    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_calculator(*, model: str, head: Optional[str], dispersion: bool, device: str, default_dtype: str):
    from mace.calculators import MACECalculator, mace_mp

    model_text = str(model or "").strip()
    if not model_text:
        raise ValueError("model is required")
    model_path = Path(model_text)
    if model_path.exists():
        if dispersion:
            raise NotImplementedError("dispersion=True with local/staged model files is not supported in remote MACE NEB.")
        kwargs = {"model_paths": str(model_path.resolve()), "device": device, "default_dtype": default_dtype}
        if head:
            kwargs["head"] = head
        return MACECalculator(**kwargs), "local_file", str(model_path)
    kwargs = {"model": model_text, "dispersion": dispersion, "device": device, "default_dtype": default_dtype}
    if head:
        kwargs["head"] = head
    return mace_mp(**kwargs), "pretrained", model_text


def _freeze_single_point_results(atoms, *, energy: float, forces: np.ndarray) -> None:
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)


def _downgrade_runtime_calculator(atoms) -> None:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return
    if type(calc).__name__ == "SinglePointCalculator":
        return
    results = getattr(calc, "results", {}) or {}
    energy = results.get("energy")
    forces = results.get("forces")
    if energy is not None and forces is not None:
        _freeze_single_point_results(atoms, energy=float(energy), forces=np.asarray(forces, dtype=float))
        return
    atoms.calc = None


def _evaluate_endpoint_single_points(images: list[Any], *, calc_factory) -> None:
    for idx in (0, len(images) - 1):
        atoms = images[idx]
        calc = calc_factory()
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        _freeze_single_point_results(atoms, energy=energy, forces=forces)


def _write_final_image_files(images: list[Any], *, output_dir: Path) -> list[str]:
    from ase.io import write

    final_image_files: list[str] = []
    for idx, atoms in enumerate(images):
        final_path = output_dir / f"{idx:02d}.vasp"
        write(str(final_path), atoms, format="vasp", direct=False, vasp5=True)
        final_image_files.append(final_path.name)
    return final_image_files


def _collect_profile_rows(images: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
    from ase.utils.forcecurve import fit_images

    forcefit = fit_images(images)
    rows: list[dict[str, Any]] = []
    for idx, atoms in enumerate(images):
        forces = np.asarray(atoms.get_forces(), dtype=float)
        norms = np.linalg.norm(forces, axis=1)
        rows.append(
            {
                "image_index": idx,
                "path_A": float(forcefit.path[idx]),
                "energy_eV": float(atoms.get_potential_energy()),
                "max_force_eV_per_A": float(norms.max()) if norms.size else 0.0,
                "rms_force_eV_per_A": float(np.sqrt(np.mean(norms**2))) if norms.size else 0.0,
            }
        )
    reference = rows[0]["energy_eV"]
    for row in rows:
        row["relative_energy_eV"] = row["energy_eV"] - reference
    return forcefit, rows


def _write_profile_outputs(*, rows: list[dict[str, Any]], forcefit: Any, energies_csv: Path, profile_png: Path) -> None:
    import matplotlib.pyplot as plt

    with energies_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_index",
                "path_A",
                "energy_eV",
                "relative_energy_eV",
                "max_force_eV_per_A",
                "rms_force_eV_per_A",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    path = [float(row["path_A"]) for row in rows]
    relative = [float(row["relative_energy_eV"]) for row in rows]
    ax.plot(path, relative, "-o", color="C0", linewidth=1.6, markersize=4.5)
    ax.set_xlabel("Reaction coordinate (A)")
    ax.set_ylabel("Relative energy (eV)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(profile_png)
    plt.close(fig)


def _compute_projected_neb_force(images: list[Any], *, climb: bool) -> float | None:
    try:
        from ase.mep import NEB

        neb = NEB(images, climb=climb, allow_shared_calculator=True)
        neb_forces = neb.get_forces()
        if neb_forces.size:
            return float(np.linalg.norm(neb_forces, axis=1).max())
    except Exception:
        return None
    return 0.0


def _summarize_run(
    *,
    images: list[Any],
    output_dir: Path,
    summary: dict[str, Any],
    climb: bool,
    artifact_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ase.mep import NEBTools

    energies_csv = output_dir / "image_energies.csv"
    profile_png = output_dir / "profile.png"
    final_image_files = _write_final_image_files(images, output_dir=output_dir)
    forcefit, rows = _collect_profile_rows(images)
    _write_profile_outputs(rows=rows, forcefit=forcefit, energies_csv=energies_csv, profile_png=profile_png)

    tools = NEBTools(images)
    fit_barrier, endpoint_delta = tools.get_barrier(fit=True, raw=False)
    raw_barrier, _ = tools.get_barrier(fit=False, raw=False)
    max_image = max(rows, key=lambda row: row["energy_eV"])
    max_neb_force = _compute_projected_neb_force(images, climb=climb)
    if max_neb_force is None:
        max_neb_force = max((row["max_force_eV_per_A"] for row in rows), default=0.0)

    artifacts = {
        "summary_rel": "summary.json",
        "energies_csv_rel": "image_energies.csv",
        "profile_png_rel": "profile.png",
        "final_image_files": final_image_files,
    }
    if artifact_extras:
        artifacts.update(artifact_extras)

    summary.update(
        {
            "status": "completed",
            "results": {
                "barrier_eV": float(fit_barrier),
                "barrier_raw_image_eV": float(raw_barrier),
                "endpoint_energy_difference_eV": float(endpoint_delta),
                "highest_energy_image_index": int(max_image["image_index"]),
                "highest_energy_eV": float(max_image["energy_eV"]),
                "number_of_images_total": len(images),
                "number_of_intermediate_images": max(0, len(images) - 2),
                "converged": bool(summary.get("results", {}).get("converged", True)),
                "max_neb_force_eV_per_A": float(max_neb_force),
            },
            "artifacts": artifacts,
        }
    )
    _write_json(output_dir / "summary.json", summary)
    return summary


def _run_plain_neb(
    *,
    images: list[Any],
    summary: dict[str, Any],
    optimizer_log: Path,
    optimizer_traj: Path,
    model: str,
    head: Optional[str],
    dispersion: bool,
    device_preference: str,
    default_dtype: str,
    fmax: float,
    steps: int,
    climb: bool,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    import torch
    from ase.mep import NEB
    from ase.optimize import FIRE

    device = _resolve_device(device_preference)
    calc, model_source_kind, model_source_ref = _make_calculator(
        model=model,
        head=head,
        dispersion=dispersion,
        device=device,
        default_dtype=default_dtype,
    )
    summary["environment"] = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary["method"]["model_source_kind"] = model_source_kind
    summary["method"]["model_source_ref"] = model_source_ref

    neb = NEB(images, climb=climb, allow_shared_calculator=True)
    for atoms in images:
        atoms.set_constraint(copy.deepcopy(images[0].constraints))
        atoms.calc = calc

    optimizer = FIRE(
        neb,
        trajectory=str(optimizer_traj),
        logfile=str(optimizer_log),
        dt=0.1,
        maxstep=0.2,
    )
    converged = optimizer.run(fmax=fmax, steps=steps)
    summary["results"] = {"converged": bool(converged)}
    artifact_extras = {
        "optimizer_log_rel": optimizer_log.name,
        "optimizer_traj_rel": optimizer_traj.name,
    }
    return images, summary, artifact_extras


def _autoneb_defaults(initial_total_images: int) -> tuple[int, int]:
    target_total = max(int(initial_total_images), _AUTONEB_TARGET_TOTAL_IMAGES)
    initial_window = max(1, initial_total_images - 2)
    n_simul = max(1, min(4, target_total - 2, initial_window))
    return target_total, n_simul


def _resolve_autoneb_settings(
    *,
    initial_total_images: int,
    target_total_images: int,
    n_simul: int,
    space_energy_ratio: float,
) -> tuple[int, int, float]:
    auto_target, auto_n_simul = _autoneb_defaults(initial_total_images)
    resolved_target = auto_target if target_total_images <= 0 else int(target_total_images)
    if resolved_target < initial_total_images:
        raise ValueError(
            f"AutoNEB target_images ({resolved_target}) cannot be smaller than the supplied image count ({initial_total_images})."
        )
    resolved_n_simul = auto_n_simul if n_simul <= 0 else int(n_simul)
    max_window = max(1, resolved_target - 2)
    if resolved_n_simul > max_window:
        raise ValueError(
            f"AutoNEB n_simul ({resolved_n_simul}) cannot exceed target_images-2 ({max_window})."
        )
    resolved_ratio = float(space_energy_ratio)
    if not 0.0 <= resolved_ratio <= 1.0:
        raise ValueError(f"AutoNEB space_energy_ratio must be within [0, 1]. Got {resolved_ratio}.")
    return resolved_target, resolved_n_simul, resolved_ratio


def _run_autoneb(
    *,
    images: list[Any],
    output_dir: Path,
    summary: dict[str, Any],
    model: str,
    head: Optional[str],
    dispersion: bool,
    device_preference: str,
    default_dtype: str,
    fmax: float,
    steps: int,
    climb: bool,
    target_total_images: int,
    n_simul: int,
    space_energy_ratio: float,
    interpolate_method: Literal["idpp", "linear"],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    import torch
    from ase.io import write
    from ase.mep.autoneb import AutoNEB
    from ase.optimize import FIRE

    device = _resolve_device(device_preference)
    probe_calc, model_source_kind, model_source_ref = _make_calculator(
        model=model,
        head=head,
        dispersion=dispersion,
        device=device,
        default_dtype=default_dtype,
    )
    _ = probe_calc
    summary["environment"] = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary["method"]["model_source_kind"] = model_source_kind
    summary["method"]["model_source_ref"] = model_source_ref

    calc_pool: list[Any] = []
    controller_ref: dict[str, Any] = {}

    def _calc_factory():
        calc, _kind, _ref = _make_calculator(
            model=model,
            head=head,
            dispersion=dispersion,
            device=device,
            default_dtype=default_dtype,
        )
        return calc

    def _attach_calculators(target_images: list[Any]) -> None:
        controller = controller_ref.get("controller")
        if controller is not None:
            for atoms in controller.all_images:
                _downgrade_runtime_calculator(atoms)
        while len(calc_pool) < len(target_images):
            calc_pool.append(_calc_factory())
        for atoms, calc in zip(target_images, calc_pool):
            atoms.calc = calc

    _evaluate_endpoint_single_points(images, calc_factory=_calc_factory)

    autoneb_root = output_dir / "autoneb"
    autoneb_root.mkdir(parents=True, exist_ok=True)
    prefix = autoneb_root / "image-"
    for idx, atoms in enumerate(images):
        write(str(autoneb_root / f"image-{idx:03d}.traj"), atoms)

    target_total_images, n_simul, space_energy_ratio = _resolve_autoneb_settings(
        initial_total_images=len(images),
        target_total_images=target_total_images,
        n_simul=n_simul,
        space_energy_ratio=space_energy_ratio,
    )
    summary["method"]["autoneb_n_simul"] = n_simul
    summary["method"]["autoneb_target_total_images"] = target_total_images
    summary["method"]["autoneb_space_energy_ratio"] = float(space_energy_ratio)
    summary["method"]["autoneb_interpolate_method"] = interpolate_method

    autoneb = AutoNEB(
        attach_calculators=_attach_calculators,
        prefix=str(prefix),
        n_simul=n_simul,
        n_max=target_total_images,
        fmax=fmax,
        maxsteps=steps,
        climb=climb,
        optimizer=FIRE,
        parallel=False,
        space_energy_ratio=space_energy_ratio,
        interpolate_method=interpolate_method,
    )
    controller_ref["controller"] = autoneb
    final_images = autoneb.run()
    summary["results"] = {"converged": True}
    artifact_extras = {
        "autoneb_prefix_rel": str(prefix.relative_to(output_dir)),
        "autoneb_iter_dir_rel": str((autoneb_root / "AutoNEB_iter").relative_to(output_dir)),
    }
    return final_images, summary, artifact_extras


def _run_task(
    *,
    task_dir: Path,
    output_dir: Path,
    mode: Literal["plain", "autoneb"],
    model: str,
    head: Optional[str],
    dispersion: bool,
    device_preference: str,
    default_dtype: str,
    fmax: float,
    steps: int,
    climb: bool,
    autoneb_target_images: int,
    autoneb_n_simul: int,
    autoneb_space_energy_ratio: float,
    autoneb_interpolate_method: Literal["idpp", "linear"],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")

    from ase.io import read

    image_paths = _list_task_image_files(task_dir)
    images = [read(str(path)) for path in image_paths]
    _validate_image_set(images)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    optimizer_log = output_dir / "optimizer.log"
    optimizer_traj = output_dir / "neb.traj"
    summary: dict[str, Any] = {
        "status": "started",
        "input_task": str(task_dir),
        "method": {
            "mode": mode,
            "optimizer": "FIRE",
            "controller": ("AutoNEB" if mode == "autoneb" else "NEB"),
            "climbing_image": climb,
            "optimizer_fmax_eV_per_A": fmax,
            "optimizer_steps": steps,
            "dispersion": dispersion,
            "head": head,
            "model_request": model,
            "default_dtype": default_dtype,
            "number_of_input_images_total": len(images),
            "number_of_input_intermediate_images": max(0, len(images) - 2),
        },
    }
    _write_json(summary_path, summary)

    try:
        if mode == "autoneb":
            final_images, summary, artifact_extras = _run_autoneb(
                images=images,
                output_dir=output_dir,
                summary=summary,
                model=model,
                head=head,
                dispersion=dispersion,
                device_preference=device_preference,
                default_dtype=default_dtype,
                fmax=fmax,
                steps=steps,
                climb=climb,
                target_total_images=autoneb_target_images,
                n_simul=autoneb_n_simul,
                space_energy_ratio=autoneb_space_energy_ratio,
                interpolate_method=autoneb_interpolate_method,
            )
        else:
            final_images, summary, artifact_extras = _run_plain_neb(
                images=images,
                summary=summary,
                optimizer_log=optimizer_log,
                optimizer_traj=optimizer_traj,
                model=model,
                head=head,
                dispersion=dispersion,
                device_preference=device_preference,
                default_dtype=default_dtype,
                fmax=fmax,
                steps=steps,
                climb=climb,
            )
        return _summarize_run(
            images=final_images,
            output_dir=output_dir,
            summary=summary,
            climb=climb,
            artifact_extras=artifact_extras,
        )
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "failure": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        _write_json(summary_path, summary)
        raise


def run_mace_neb_batch(
    *,
    input_path: str,
    output_root: str,
    fmax: float,
    steps: int,
    mode: Literal["plain", "autoneb"],
    autoneb_target_images: int,
    autoneb_n_simul: int,
    autoneb_space_energy_ratio: float,
    autoneb_interpolate_method: Literal["idpp", "linear"],
    climb: bool,
    model: str,
    head: Optional[str],
    dispersion: bool,
    device: str = "auto",
    default_dtype: str = "float64",
) -> dict[str, Any]:
    input_root = Path(input_path)
    output_root_path = Path(output_root)
    if not input_root.is_dir():
        raise ValueError(f"input_path is not a directory: {input_root}")
    output_root_path.mkdir(parents=True, exist_ok=True)
    task_dirs = _discover_task_dirs(input_root)

    task_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for task_dir in task_dirs:
        task_output = output_root_path / task_dir.name
        try:
            summary = _run_task(
                task_dir=task_dir,
                output_dir=task_output,
                mode=mode,
                model=model,
                head=head,
                dispersion=dispersion,
                device_preference=device,
                default_dtype=default_dtype,
                fmax=fmax,
                steps=steps,
                autoneb_target_images=autoneb_target_images,
                autoneb_n_simul=autoneb_n_simul,
                autoneb_space_energy_ratio=autoneb_space_energy_ratio,
                autoneb_interpolate_method=autoneb_interpolate_method,
                climb=climb,
            )
            task_results.append(
                {
                    "task_rel": str(task_dir.relative_to(input_root)),
                    "output_rel": str(task_output.relative_to(output_root_path)),
                    "summary_rel": str((task_output / "summary.json").relative_to(output_root_path)),
                    "status": summary.get("status"),
                    "barrier_eV": ((summary.get("results") or {}).get("barrier_eV")),
                    "converged": ((summary.get("results") or {}).get("converged")),
                }
            )
        except Exception as exc:
            warnings.append(f"{task_dir.name}: {type(exc).__name__}: {exc}")
            task_results.append(
                {
                    "task_rel": str(task_dir.relative_to(input_root)),
                    "output_rel": str(task_output.relative_to(output_root_path)),
                    "summary_rel": str((task_output / "summary.json").relative_to(output_root_path)),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    batch_summary = {
        "input_root": str(input_root),
        "output_root": str(output_root_path),
        "task_count": len(task_dirs),
        "optimizer": "FIRE",
        "mode": mode,
        "climb": climb,
        "autoneb_target_images": autoneb_target_images,
        "autoneb_n_simul": autoneb_n_simul,
        "autoneb_space_energy_ratio": autoneb_space_energy_ratio,
        "autoneb_interpolate_method": autoneb_interpolate_method,
        "model": model,
        "head": head,
        "dispersion": dispersion,
        "device": device,
        "default_dtype": default_dtype,
        "tasks": task_results,
        "warnings": warnings,
    }
    _write_json(output_root_path / "batch_summary.json", batch_summary)
    return batch_summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run MACE NEB batch remotely inside a staged DPDispatcher task.")
    parser.add_argument("--input", required=True, help="Input root containing task directories")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--mode", choices=("plain", "autoneb"), default="plain")
    parser.add_argument("--autoneb_target_images", type=int, default=0)
    parser.add_argument("--autoneb_n_simul", type=int, default=0)
    parser.add_argument("--autoneb_space_energy_ratio", type=float, default=0.5)
    parser.add_argument("--autoneb_interpolate_method", choices=("idpp", "linear"), default="idpp")
    parser.add_argument("--climb", type=_parse_bool, default=False)
    parser.add_argument("--model", default="mh-1")
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--dispersion", type=_parse_bool, default=False)
    parser.add_argument("--default_dtype", default="float64", choices=("float32", "float64"))
    args = parser.parse_args()
    head = args.head.strip() or None
    summary = run_mace_neb_batch(
        input_path=args.input,
        output_root=args.output_root,
        fmax=args.fmax,
        steps=args.steps,
        mode=args.mode,
        autoneb_target_images=args.autoneb_target_images,
        autoneb_n_simul=args.autoneb_n_simul,
        autoneb_space_energy_ratio=args.autoneb_space_energy_ratio,
        autoneb_interpolate_method=args.autoneb_interpolate_method,
        climb=args.climb,
        model=args.model,
        head=head,
        dispersion=args.dispersion,
        default_dtype=args.default_dtype,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
