# Code writing date: 2026-07-17
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: optimize a locally prepared fixed-image path with
# ASE while obtaining every calculator exclusively from the provider adapter.
# Purpose: shared managed MLFF NEB execution for every enabled backend.
from __future__ import annotations

import argparse
import copy
import csv
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package import in tests; flat import after DPDispatcher staging.
    from .mlff_common import _ADAPTER_TYPES, _load_run_config
except ImportError:  # pragma: no cover - exercised by remote staged scripts
    from mlff_common import _ADAPTER_TYPES, _load_run_config


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


def _write_final_images(images: list[Any], output_dir: Path) -> list[str]:
    from ase.io import write

    width = max(2, len(str(len(images) - 1)))
    names: list[str] = []
    for index, atoms in enumerate(images):
        name = f"{index:0{width}d}.vasp"
        write(str(output_dir / name), atoms, format="vasp")
        names.append(name)
    return names


def _profile_rows(images: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
    from ase.utils.forcecurve import fit_images

    fit = fit_images(images)
    rows: list[dict[str, Any]] = []
    for index, atoms in enumerate(images):
        rows.append(
            {
                "image_index": index,
                "path_A": float(fit.path[index]),
                "energy_eV": float(atoms.get_potential_energy()),
                "max_force_eVA": _max_force_eva(atoms.get_forces()),
            }
        )
    reference = float(rows[0]["energy_eV"])
    for row in rows:
        row["relative_energy_eV"] = float(row["energy_eV"]) - reference
    return fit, rows


def _write_profile(rows: list[dict[str, Any]], output_dir: Path) -> tuple[str, str]:
    csv_path = output_dir / "image_energies.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_index",
                "path_A",
                "energy_eV",
                "relative_energy_eV",
                "max_force_eVA",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    png_name = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 4), dpi=150)
        axis.plot(
            [float(row["path_A"]) for row in rows],
            [float(row["relative_energy_eV"]) for row in rows],
            "-o",
            linewidth=1.6,
            markersize=4.5,
        )
        axis.set_xlabel("Reaction coordinate (A)")
        axis.set_ylabel("Relative energy (eV)")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        png_path = output_dir / "profile.png"
        figure.savefig(png_path)
        plt.close(figure)
        png_name = png_path.name
    except Exception:
        png_name = ""
    return csv_path.name, png_name


def run(run_config: str) -> dict[str, Any]:
    from ase.io import read
    from ase.mep import NEB, NEBTools
    from ase.optimize import FIRE

    config = _load_run_config(run_config)
    if config["operation"] != "neb":
        raise ValueError(f"Managed MLFF NEB received operation={config['operation']!r}.")
    task = dict(config["task_config"])
    if task["mode"] != "plain":
        raise ValueError("Managed MLFF NEB accepts only locally prepared fixed-image plain mode.")

    items = dict(config["items"])
    ordered = sorted(items.items(), key=lambda item: int(Path(item[0]).stem))
    if len(ordered) < 3:
        raise ValueError("MLFF NEB requires endpoints plus at least one intermediate image.")
    image_paths = [Path("input") / rel for rel, _ in ordered]
    images = [read(str(path), index=-1) for path in image_paths]
    backend = str(config["backend"])
    adapter = _ADAPTER_TYPES[backend]()
    calculators: list[Any] = []
    for atoms, (_, item_config) in zip(images, ordered):
        atoms.set_constraint(copy.deepcopy(images[0].constraints))
        calculator = adapter.calculator_for(atoms, dict(item_config))
        atoms.calc = calculator
        calculators.append(calculator)

    output_root = Path("output")
    output_dir = output_root / "path"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": "neb",
        "backend": backend,
        "provider_version": adapter.provider_version,
        "status": "started",
        "method": {
            "controller": "ASE NEB",
            "optimizer": str(task["optimizer"]),
            "mode": str(task["mode"]),
            "climbing_image": bool(task["climb"]),
            "fmax_eVA": float(task["fmax"]),
            "steps": int(task["steps"]),
            "allow_shared_calculator": True,
            "number_of_images_total": len(images),
        },
    }
    _write_json(summary_path, summary)
    try:
        neb = NEB(
            images,
            climb=bool(task["climb"]),
            method="improvedtangent",
            parallel=False,
            allow_shared_calculator=True,
        )
        optimizer = FIRE(
            neb,
            trajectory=str(output_dir / "neb.traj"),
            logfile=str(output_dir / "optimizer.log"),
            dt=0.1,
            maxstep=0.2,
        )
        converged = bool(optimizer.run(fmax=float(task["fmax"]), steps=int(task["steps"])))
        projected = np.asarray(neb.get_forces(), dtype=float)
        max_neb_force = _max_force_eva(projected)
        final_files = _write_final_images(images, output_dir)
        _fit, rows = _profile_rows(images)
        csv_name, png_name = _write_profile(rows, output_dir)
        tools = NEBTools(images)
        fit_barrier, endpoint_delta = tools.get_barrier(fit=True, raw=False)
        raw_barrier, _ = tools.get_barrier(fit=False, raw=False)
        highest = max(rows, key=lambda row: float(row["energy_eV"]))
        provider_metadata = [
            adapter.provider_metadata(atoms, dict(item_config), calculator)
            for atoms, (_, item_config), calculator in zip(images, ordered, calculators)
        ]
        artifacts = {
            "summary": summary_path.name,
            "trajectory": "neb.traj",
            "optimizer_log": "optimizer.log",
            "image_energies": csv_name,
            "final_image_files": final_files,
        }
        if png_name:
            artifacts["profile_plot"] = png_name
        summary.update(
            {
                "status": "completed",
                "results": {
                    "converged": converged,
                    "nsteps": int(optimizer.nsteps),
                    "barrier_eV": float(fit_barrier),
                    "barrier_raw_image_eV": float(raw_barrier),
                    "endpoint_energy_difference_eV": float(endpoint_delta),
                    "highest_energy_image_index": int(highest["image_index"]),
                    "highest_energy_eV": float(highest["energy_eV"]),
                    "max_force_eVA": max_neb_force,
                    "number_of_images_total": len(images),
                    "number_of_intermediate_images": len(images) - 2,
                },
                "image_profile": rows,
                "provider_metadata": provider_metadata,
                "artifacts": artifacts,
            }
        )
        _write_json(summary_path, summary)
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

    batch = {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": "neb",
        "backend": backend,
        "provider_version": adapter.provider_version,
        "tasks": [
            {
                "task_rel": "path",
                "output_rel": "path",
                "summary_rel": "path/summary.json",
                "status": summary["status"],
                "barrier_eV": (summary.get("results") or {}).get("barrier_eV"),
                "converged": (summary.get("results") or {}).get("converged"),
            }
        ],
        "warnings": [],
    }
    _write_json(output_root / "batch_summary.json", batch)
    if summary["status"] != "completed":
        raise RuntimeError("Managed MLFF NEB failed; see output/path/summary.json.")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run managed calculator-independent MLFF NEB.")
    parser.add_argument("--run_config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
