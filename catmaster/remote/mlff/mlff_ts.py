# Code writing date: 2026-07-24
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: keep one provider-independent constrained RS-pRFO
# workflow while resolving Hessian acquisition from calculator capabilities.
# Purpose: robust local refinement and validation of one TS-like structure.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:  # Package import in tests; flat import after DPDispatcher staging.
    from .mlff_common import (
        _ADAPTER_TYPES,
        _base_summary,
        _load_run_config,
        _output_structure,
    )
    from .mlff_vibrations import (
        _analytic_hessian,
        _constraint_projection,
        _finite_difference_hessian,
        _frequency_analysis,
        _normalize_hessian,
        _write_vibration_artifacts,
    )
except ImportError:  # pragma: no cover - exercised by remote staged scripts
    from mlff_common import _ADAPTER_TYPES, _base_summary, _load_run_config, _output_structure
    from mlff_vibrations import (
        _analytic_hessian,
        _constraint_projection,
        _finite_difference_hessian,
        _frequency_analysis,
        _normalize_hessian,
        _write_vibration_artifacts,
    )


_AUTO_FULL_HESSIAN_DOF_LIMIT = 60


def _build_sella_constraints(atoms: Any, constraints: list[Any]) -> Any:
    """Translate ASE file constraints without relying on Sella's FixCartesian mask parser."""

    from ase.constraints import FixAtoms, FixCartesian, FixScaled
    from sella import Constraints
    from sella.internal import Coordinate

    class LinearCoordinate(Coordinate):
        nindices = 1

        def __init__(self, atom_index: int, direction: np.ndarray) -> None:
            super().__init__((int(atom_index),))
            vector = np.asarray(direction, dtype=float).reshape(3)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-14:
                raise ValueError("Linear constraint direction has zero norm.")
            self.kwargs = {"direction": vector / norm}

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, LinearCoordinate)
                and np.array_equal(self.indices, other.indices)
                and np.allclose(self.kwargs["direction"], other.kwargs["direction"])
            )

        @staticmethod
        def _eval0(pos: np.ndarray, *, direction: np.ndarray) -> float:
            return pos[0] @ direction

        @staticmethod
        def _eval1(pos: np.ndarray, *, direction: np.ndarray) -> np.ndarray:
            return np.asarray(direction, dtype=float).reshape(1, 3)

        @staticmethod
        def _eval2(pos: np.ndarray, *, direction: np.ndarray) -> np.ndarray:
            return np.zeros((1, 3, 1, 3), dtype=float)

    translated = Constraints(atoms)
    inverse_cell = None
    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            for atom_index in np.asarray(constraint.index, dtype=int):
                translated.fix_translation(int(atom_index))
        elif isinstance(constraint, FixCartesian):
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis, fixed in enumerate(np.asarray(constraint.mask, dtype=bool)):
                    if fixed:
                        translated.fix_translation(int(atom_index), dim=axis)
        elif isinstance(constraint, FixScaled):
            if inverse_cell is None:
                inverse_cell = np.linalg.inv(np.asarray(atoms.cell.complete(), dtype=float))
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis, fixed in enumerate(np.asarray(constraint.mask, dtype=bool)):
                    if fixed:
                        translated.fix_other(
                            LinearCoordinate(int(atom_index), inverse_cell[:, axis])
                        )
        else:
            raise ValueError(f"Unsupported MLFF TS constraint: {type(constraint).__name__}.")
    return translated


def _resolve_hessian_strategy(
    *,
    requested: str,
    atoms: Any,
    calculator: Any,
    free_dof: int,
) -> tuple[str, np.ndarray | None, list[str]]:
    warnings: list[str] = []
    initial_analytic = None
    getter_available = callable(getattr(calculator, "get_hessian", None))
    if requested in {"auto", "analytic"} and getter_available:
        try:
            initial_analytic = _analytic_hessian(atoms, calculator)
        except Exception as exc:
            if requested == "analytic":
                raise
            warnings.append(f"analytic_hessian_unavailable={type(exc).__name__}: {exc}")
    elif requested == "analytic":
        raise ValueError("hessian_method=analytic requires a calculator that exposes get_hessian().")

    if requested == "auto":
        if initial_analytic is not None:
            return "analytic", initial_analytic, warnings
        if free_dof <= _AUTO_FULL_HESSIAN_DOF_LIMIT:
            return "finite_difference", None, warnings
        return "iterative", None, warnings
    return requested, initial_analytic, warnings


def _constraint_drift(
    atoms: Any,
    initial_positions: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    displacement = np.asarray(atoms.positions, dtype=float) - initial_positions
    cartesian: list[float] = []
    scaled: list[float] = []
    for record in records:
        atom_index = int(record["atom_index"])
        direction = np.asarray(record["direction"], dtype=float)
        value = abs(float(displacement[atom_index] @ direction))
        if record["kind"] == "scaled":
            scaled.append(value)
        else:
            cartesian.append(value)
    return {
        "max_cartesian_constraint_drift_A": max(cartesian, default=0.0),
        "max_scaled_constraint_drift": max(scaled, default=0.0),
    }


def run_single(
    *,
    source: Path,
    output_dir: Path,
    config: dict[str, Any],
    item_config: dict[str, Any],
    adapter: Any,
) -> dict[str, Any]:
    from ase.io import read
    from sella import Sella

    task = dict(config["task_config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(str(source), index=-1)
    original_constraints = list(atoms.constraints or [])
    initial_positions = np.asarray(atoms.positions, dtype=float).copy()
    basis, constraint_records = _constraint_projection(atoms, original_constraints)

    # Sella must receive raw calculator forces. Its native ASE FixCartesian
    # translation reverses ASE's current True=fixed mask semantics, so all
    # supported file constraints are translated explicitly below.
    atoms.set_constraint()
    translated_constraints = _build_sella_constraints(atoms, original_constraints)
    calculator = adapter.calculator_for(atoms, item_config)
    atoms.calc = calculator
    requested_method = str(task["hessian_method"])
    resolved_method, initial_analytic, warnings = _resolve_hessian_strategy(
        requested=requested_method,
        atoms=atoms,
        calculator=calculator,
        free_dof=basis.shape[1],
    )

    hessian_calls = 0
    hessian_force_calls = 0
    cached_initial = initial_analytic

    def analytic_function(current_atoms: Any) -> np.ndarray:
        nonlocal cached_initial, hessian_calls
        hessian_calls += 1
        if cached_initial is not None:
            value = cached_initial
            cached_initial = None
            return value
        return _analytic_hessian(current_atoms, calculator)

    def finite_difference_function(current_atoms: Any) -> np.ndarray:
        nonlocal hessian_calls, hessian_force_calls
        hessian_calls += 1
        value, calls = _finite_difference_hessian(
            current_atoms,
            basis,
            delta=float(task["hessian_delta"]),
        )
        hessian_force_calls += calls
        return value

    hessian_function: Callable[[Any], np.ndarray] | None
    if resolved_method == "analytic":
        hessian_function = analytic_function
    elif resolved_method == "finite_difference":
        hessian_function = finite_difference_function
    else:
        hessian_function = None

    optimizer = Sella(
        atoms,
        order=1,
        method="prfo",
        rs="ras",
        internal=False,
        constraints=translated_constraints,
        hessian_function=hessian_function,
        nsteps_per_diag=3,
        logfile=str(output_dir / "optimizer.log"),
        trajectory=str(output_dir / "ts.traj"),
    )
    try:
        optimizer.run(fmax=float(task["fmax"]), steps=int(task["steps"]))
        converged = bool(optimizer.converged())
        nsteps = int(optimizer.nsteps)
    finally:
        optimizer.close()

    validation_method = "analytic" if resolved_method == "analytic" else "finite_difference"
    validation_force_calls = 0
    if validation_method == "analytic":
        try:
            final_hessian = _analytic_hessian(atoms, calculator)
        except Exception as exc:
            if requested_method != "auto":
                raise
            warnings.append(f"final_analytic_hessian_failed={type(exc).__name__}: {exc}")
            validation_method = "finite_difference"
            final_hessian, validation_force_calls = _finite_difference_hessian(
                atoms,
                basis,
                delta=float(task["hessian_delta"]),
            )
    else:
        final_hessian, validation_force_calls = _finite_difference_hessian(
            atoms,
            basis,
            delta=float(task["hessian_delta"]),
        )

    frequency_rows, modes, imaginary_count = _frequency_analysis(
        atoms=atoms,
        hessian=final_hessian,
        basis=basis,
        threshold_cm1=float(task["imaginary_threshold_cm1"]),
    )
    vibration_artifacts = _write_vibration_artifacts(
        output_dir,
        atoms=atoms,
        hessian=final_hessian,
        basis=basis,
        rows=frequency_rows,
        modes=modes,
    )
    if len(modes):
        np.savetxt(output_dir / "reaction_mode.txt", modes[0], fmt="% .16e")

    projected_forces = np.asarray(optimizer.pes.get_projected_forces(), dtype=float)
    projected_atom_norms = np.linalg.norm(projected_forces.reshape(len(atoms), 3), axis=1)
    final_energy = float(atoms.get_potential_energy())
    atoms.set_constraint(original_constraints)

    summary = _base_summary(
        config=config,
        item_config=item_config,
        adapter=adapter,
        atoms=atoms,
        calculator=calculator,
    )
    validated = bool(converged and imaginary_count == 1)
    summary.update(
        {
            "final_energy_eV": final_energy,
            "converged": converged,
            "validated_first_order_saddle": validated,
            "nsteps": nsteps,
            "free_dof": int(basis.shape[1]),
            "constrained_dof_count": len(constraint_records),
            **_constraint_drift(atoms, initial_positions, constraint_records),
            "max_projected_force_eVA": (
                float(np.max(projected_atom_norms)) if projected_atom_norms.size else 0.0
            ),
            "hessian_method_requested": requested_method,
            "hessian_method_resolved": resolved_method,
            "optimization_hessian_evaluations": hessian_calls,
            "optimization_hessian_force_evaluations": hessian_force_calls,
            "validation_hessian_method": validation_method,
            "validation_hessian_force_evaluations": validation_force_calls,
            "significant_imaginary_mode_count": imaginary_count,
            "imaginary_threshold_cm1": float(task["imaginary_threshold_cm1"]),
            "lowest_frequency_cm1": float(frequency_rows[0]["frequency_cm1"]),
            **vibration_artifacts,
            "reaction_mode_file": "reaction_mode.txt",
            "output_structure": _output_structure(
                output_dir,
                atoms,
                stem="ts",
                source_name=source.name,
            ),
            "warnings": [*summary.get("warnings", []), *warnings],
        }
    )
    return summary


def run(run_config: str) -> dict[str, Any]:
    config = _load_run_config(run_config)
    if config["operation"] != "ts":
        raise ValueError(f"Managed MLFF TS received operation={config['operation']!r}.")
    items = dict(config["items"])
    if len(items) != 1:
        raise ValueError("MLFF TS requires exactly one TS-like structure per stage.")
    backend = str(config["backend"])
    adapter = _ADAPTER_TYPES[backend]()
    rel, item_config = next(iter(sorted(items.items())))
    output_root = Path("output")
    output_dir = output_root / Path(rel).with_suffix("")
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
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
        errors.append({"input_rel": rel, "error": f"{type(exc).__name__}: {exc}"})
    batch = {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": "ts",
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
        raise RuntimeError("Managed MLFF TS failed; see output/batch_summary.json.")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained managed MLFF RS-pRFO TS refinement.")
    parser.add_argument("--run_config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "_constraint_projection",
    "_finite_difference_hessian",
    "_frequency_analysis",
    "_normalize_hessian",
    "_resolve_hessian_strategy",
    "run",
    "run_single",
]
