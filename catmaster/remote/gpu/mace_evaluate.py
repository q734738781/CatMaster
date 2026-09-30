from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read as ase_read


_EVALUATION_PARAM_KEYS = {"dataset_file", "model", "head", "default_dtype", "device"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_calculator(model: str, *, head: Optional[str], default_dtype: str, device: str):
    from mace.calculators import MACECalculator, mace_mp

    model_path = Path(model)
    if model_path.exists():
        try:
            return MACECalculator(model_paths=[str(model_path)], default_dtype=default_dtype, device=device)
        except Exception:
            return MACECalculator(model_paths=str(model_path), default_dtype=default_dtype, device=device)
    kwargs = {"model": model, "device": device, "default_dtype": default_dtype}
    if head:
        kwargs["head"] = head
    return mace_mp(**kwargs)


def _reference_energy(atoms: Atoms) -> float | None:
    if atoms.calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    value = atoms.info.get("REF_energy")
    return float(value) if value is not None else None


def _reference_forces(atoms: Atoms) -> np.ndarray | None:
    if "REF_forces" in atoms.arrays:
        return np.asarray(atoms.arrays["REF_forces"], dtype=float)
    if atoms.calc is not None:
        try:
            return np.asarray(atoms.get_forces(), dtype=float)
        except Exception:
            pass
    return None


def _reference_stress(atoms: Atoms) -> np.ndarray | None:
    value = atoms.info.get("REF_stress")
    if value is not None:
        try:
            arr = np.asarray(value, dtype=float)
            if arr.shape == (6,):
                return arr
        except Exception:
            pass
    if atoms.calc is not None:
        try:
            return np.asarray(atoms.get_stress(voigt=True), dtype=float)
        except Exception:
            pass
    return None


def run_evaluation(dataset_root: Path, output_root: Path, params_path: Path) -> dict[str, Any]:
    params = _read_json(params_path)
    if not isinstance(params, dict):
        raise ValueError("Evaluation params JSON must contain an object.")
    unknown = sorted(str(key) for key in params if key not in _EVALUATION_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown evaluation parameter key(s): {', '.join(unknown)}")
    dataset_file = dataset_root / str(params["dataset_file"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    calc = _load_calculator(
        str(params["model"]),
        head=params.get("head"),
        default_dtype=str(params.get("default_dtype") or "float32"),
        device=str(params.get("device") or "cuda"),
    )
    frames = list(ase_read(str(dataset_file), index=":"))
    energy_errors: list[float] = []
    force_errors: list[np.ndarray] = []
    stress_errors: list[np.ndarray] = []
    for idx, atoms in enumerate(frames):
        ref_energy = _reference_energy(atoms)
        ref_forces = _reference_forces(atoms)
        ref_stress = _reference_stress(atoms)
        sample = atoms.copy()
        sample.calc = calc
        pred_energy = float(sample.get_potential_energy())
        pred_forces = np.asarray(sample.get_forces(), dtype=float)
        try:
            pred_stress = np.asarray(sample.get_stress(voigt=True), dtype=float)
        except Exception:
            pred_stress = None
        row = {
            "frame_index": idx,
            "natoms": len(sample),
            "pred_energy_eV": pred_energy,
            "pred_energy_eV_per_atom": pred_energy / max(len(sample), 1),
        }
        if ref_energy is not None:
            row["ref_energy_eV"] = float(ref_energy)
            row["ref_energy_eV_per_atom"] = float(ref_energy) / max(len(sample), 1)
            row["energy_error_meV_per_atom"] = 1000.0 * (
                row["pred_energy_eV_per_atom"] - row["ref_energy_eV_per_atom"]
            )
            energy_errors.append(row["energy_error_meV_per_atom"])
        if ref_forces is not None:
            delta = pred_forces - ref_forces
            force_errors.append(delta.reshape(-1))
            row["force_rmse_eVA"] = float(np.sqrt(np.mean(delta**2)))
        if ref_stress is not None and pred_stress is not None:
            delta = pred_stress - ref_stress
            stress_errors.append(delta.reshape(-1))
            row["stress_rmse_eVA3"] = float(np.sqrt(np.mean(delta**2)))
        rows.append(row)

    per_config_csv = output_root / "per_config.csv"
    metrics_json = output_root / "metrics.json"
    scatter_png = output_root / "energy_scatter.png"
    _write_csv(per_config_csv, rows)

    metrics = {
        "dataset_file": str(dataset_file.relative_to(dataset_root)),
        "frames_evaluated": len(rows),
        "energy_rmse_meV_per_atom": float(np.sqrt(np.mean(np.square(energy_errors)))) if energy_errors else None,
        "energy_mae_meV_per_atom": float(np.mean(np.abs(energy_errors))) if energy_errors else None,
        "force_rmse_eVA": float(np.sqrt(np.mean(np.concatenate(force_errors) ** 2))) if force_errors else None,
        "force_mae_eVA": float(np.mean(np.abs(np.concatenate(force_errors)))) if force_errors else None,
        "stress_rmse_eVA3": float(np.sqrt(np.mean(np.concatenate(stress_errors) ** 2))) if stress_errors else None,
        "stress_mae_eVA3": float(np.mean(np.abs(np.concatenate(stress_errors)))) if stress_errors else None,
        "per_config_csv": str(per_config_csv.relative_to(output_root)),
    }
    if energy_errors:
        ref = [row["ref_energy_eV_per_atom"] for row in rows if "ref_energy_eV_per_atom" in row]
        pred = [row["pred_energy_eV_per_atom"] for row in rows if "ref_energy_eV_per_atom" in row]
        plt.figure(figsize=(5.0, 5.0))
        plt.scatter(ref, pred, s=18)
        bounds = [min(ref + pred), max(ref + pred)]
        plt.plot(bounds, bounds, linestyle="--", color="black", linewidth=1)
        plt.xlabel("Reference energy / atom (eV)")
        plt.ylabel("Predicted energy / atom (eV)")
        plt.tight_layout()
        plt.savefig(scatter_png, dpi=180)
        plt.close()
        metrics["energy_scatter_png"] = str(scatter_png.relative_to(output_root))
    _write_json(metrics_json, metrics)
    summary = {
        "metrics": metrics,
        "batch_summary_version": 1,
    }
    _write_json(output_root / "batch_summary.json", summary)
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MACE model on an extxyz dataset.")
    parser.add_argument("--dataset_root", required=True, help="Staged dataset directory")
    parser.add_argument("--output_root", required=True, help="Output directory for metrics")
    parser.add_argument("--params", required=True, help="Evaluation params JSON path")
    args = parser.parse_args()

    summary = run_evaluation(Path(args.dataset_root), Path(args.output_root), Path(args.params))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
