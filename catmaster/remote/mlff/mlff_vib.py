# Code writing date: 2026-07-24
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: treat vibrational analysis as a general constrained
# stationary-point property task, independent of transition-state optimization.
# Purpose: managed MLFF normal-mode analysis without ASE displacement caches.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package import in tests; flat import after DPDispatcher staging.
    from .mlff_common import _ADAPTER_TYPES, _base_summary, _load_run_config
    from .mlff_vibrations import (
        _constraint_projection,
        _finite_difference_hessian_reduced,
        _frequency_analysis,
        _max_projected_force,
        _resolve_full_hessian_strategy,
        _stationary_point_class,
        _write_vibration_artifacts,
    )
except ImportError:  # pragma: no cover - exercised by remote staged scripts
    from mlff_common import _ADAPTER_TYPES, _base_summary, _load_run_config
    from mlff_vibrations import (
        _constraint_projection,
        _finite_difference_hessian_reduced,
        _frequency_analysis,
        _max_projected_force,
        _resolve_full_hessian_strategy,
        _stationary_point_class,
        _write_vibration_artifacts,
    )


def run_single(
    *,
    source: Path,
    output_dir: Path,
    config: dict[str, Any],
    item_config: dict[str, Any],
    adapter: Any,
) -> dict[str, Any]:
    from ase.io import read

    task = dict(config["task_config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(str(source), index=-1)
    constraints = list(atoms.constraints or [])
    basis, _ = _constraint_projection(atoms, constraints)
    calculator = adapter.calculator_for(atoms, item_config)
    atoms.calc = calculator

    requested_method = str(task["hessian_method"])
    resolved_method, analytic_hessian, warnings = _resolve_full_hessian_strategy(
        requested=requested_method,
        atoms=atoms,
        calculator=calculator,
    )
    hessian_force_calls = 0
    finite_difference_asymmetry = 0.0
    if resolved_method == "analytic":
        if analytic_hessian is None:
            raise RuntimeError("Resolved analytic Hessian was not materialized.")
        hessian = analytic_hessian
    else:
        reduced_hessian, hessian_force_calls, finite_difference_asymmetry = (
            _finite_difference_hessian_reduced(
                atoms,
                basis,
                delta=float(task["hessian_delta"]),
                nfree=int(task["nfree"]),
            )
        )
        hessian = basis @ reduced_hessian @ basis.T
        hessian = 0.5 * (hessian + hessian.T)

    rows, modes, significant_imaginary_count = _frequency_analysis(
        atoms=atoms,
        hessian=hessian,
        basis=basis,
        threshold_cm1=float(task["imaginary_threshold_cm1"]),
    )
    artifacts = _write_vibration_artifacts(
        output_dir,
        atoms=atoms,
        hessian=hessian,
        basis=basis,
        rows=rows,
        modes=modes,
    )

    max_projected_force = _max_projected_force(
        atoms=atoms,
        basis=basis,
    )
    stationary_threshold = float(task["stationary_force_threshold"])
    stationary = bool(max_projected_force <= stationary_threshold)
    if not stationary:
        warnings.append(
            "projected_force_above_stationary_threshold="
            f"{max_projected_force:.8g}>{stationary_threshold:.8g} eV/Angstrom"
        )

    summary = _base_summary(
        config=config,
        item_config=item_config,
        adapter=adapter,
        atoms=atoms,
        calculator=calculator,
    )
    summary.update(
        {
            "energy_eV": float(atoms.get_potential_energy()),
            "free_dof": int(basis.shape[1]),
            "constrained_dof_count": int(3 * len(atoms) - basis.shape[1]),
            "max_projected_force_eVA": max_projected_force,
            "stationary_force_threshold_eVA": stationary_threshold,
            "stationary_point": stationary,
            "stationary_point_class": _stationary_point_class(
                stationary=stationary,
                significant_imaginary_count=significant_imaginary_count,
            ),
            "hessian_method_requested": requested_method,
            "hessian_method_resolved": resolved_method,
            "hessian_delta_A": float(task["hessian_delta"]),
            "finite_difference_nfree": int(task["nfree"]),
            "hessian_force_evaluations": hessian_force_calls,
            "finite_difference_hessian_asymmetry": finite_difference_asymmetry,
            "mode_count": len(rows),
            "significant_imaginary_mode_count": significant_imaginary_count,
            "imaginary_threshold_cm1": float(task["imaginary_threshold_cm1"]),
            "lowest_frequency_cm1": float(rows[0]["frequency_cm1"]),
            "highest_frequency_cm1": float(rows[-1]["frequency_cm1"]),
            **artifacts,
            "warnings": [*summary.get("warnings", []), *warnings],
        }
    )
    return summary


def run(run_config: str) -> dict[str, Any]:
    config = _load_run_config(run_config)
    if config["operation"] != "vib":
        raise ValueError(
            f"Managed MLFF VIB received operation={config['operation']!r}."
        )
    backend = str(config["backend"])
    adapter = _ADAPTER_TYPES[backend]()
    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for rel, item_config in sorted(dict(config["items"]).items()):
        output_dir = output_root / Path(rel).with_suffix("")
        try:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"Output already exists: {output_dir}")
            summary = run_single(
                source=Path("input") / rel,
                output_dir=output_dir,
                config=config,
                item_config=dict(item_config),
                adapter=adapter,
            )
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "input_rel": rel,
                    "output_rel": output_dir.relative_to(output_root).as_posix(),
                    "summary": summary,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "input_rel": rel,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    batch = {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": "vib",
        "backend": backend,
        "provider_version": adapter.provider_version,
        "results": results,
        "errors": errors,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(
            f"{len(errors)} of {len(results) + len(errors)} MLFF vibration inputs "
            "failed; see output/batch_summary.json."
        )
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run general constrained managed MLFF normal-mode analysis."
    )
    parser.add_argument("--run_config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["run", "run_single"]
