# Code writing date: 2026-07-24
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: compute and persist constrained normal modes without
# ASE Vibrations' per-displacement JSON cache or atom-only active-index model.
# Purpose: shared Hessian, constraint, and mode analysis for MLFF VIB and TS.
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value, dtype=float)


def _normalize_hessian(raw_hessian: Any, atom_count: int) -> np.ndarray:
    hessian = _as_numpy(raw_hessian)
    expected = atom_count * 3
    if hessian.shape == (atom_count, 3, atom_count, 3):
        hessian = hessian.reshape(expected, expected)
    elif hessian.shape != (expected, expected):
        raise ValueError(
            f"Unsupported Hessian shape {hessian.shape}; expected {(expected, expected)} "
            f"or {(atom_count, 3, atom_count, 3)}."
        )
    if not np.isfinite(hessian).all():
        raise ValueError("Calculator Hessian contains non-finite values.")
    return 0.5 * (hessian + hessian.T)


def _analytic_hessian(atoms: Any, calculator: Any) -> np.ndarray:
    getter = getattr(calculator, "get_hessian", None)
    if not callable(getter):
        raise AttributeError("The selected calculator does not expose get_hessian().")
    try:
        raw = getter(atoms=atoms)
    except TypeError:
        try:
            raw = getter(atoms)
        except TypeError:
            raw = getter()
    return _normalize_hessian(raw, len(atoms))


def _constraint_projection(
    atoms: Any,
    constraints: list[Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build an orthonormal Cartesian basis satisfying structure constraints."""

    from ase.constraints import FixAtoms, FixCartesian, FixScaled
    from scipy.linalg import null_space

    ndof = 3 * len(atoms)
    rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    def add_row(*, atom_index: int, axis: int, direction: np.ndarray, kind: str) -> None:
        vector = np.asarray(direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-14:
            raise ValueError(
                f"Degenerate {kind} constraint direction for atom {atom_index}, axis {axis}."
            )
        row = np.zeros(ndof, dtype=float)
        row[3 * atom_index : 3 * atom_index + 3] = vector / norm
        rows.append(row)
        records.append(
            {
                "kind": kind,
                "atom_index": int(atom_index),
                "axis": int(axis),
                "direction": vector.tolist(),
            }
        )

    inverse_cell = None
    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis in range(3):
                    add_row(
                        atom_index=int(atom_index),
                        axis=axis,
                        direction=np.eye(3)[axis],
                        kind="cartesian",
                    )
        elif isinstance(constraint, FixCartesian):
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis, fixed in enumerate(np.asarray(constraint.mask, dtype=bool)):
                    if fixed:
                        add_row(
                            atom_index=int(atom_index),
                            axis=axis,
                            direction=np.eye(3)[axis],
                            kind="cartesian",
                        )
        elif isinstance(constraint, FixScaled):
            if inverse_cell is None:
                cell = np.asarray(atoms.cell.complete(), dtype=float)
                if abs(float(np.linalg.det(cell))) <= 1e-14:
                    raise ValueError("FixScaled constraints require an invertible cell.")
                inverse_cell = np.linalg.inv(cell)
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis, fixed in enumerate(np.asarray(constraint.mask, dtype=bool)):
                    if fixed:
                        add_row(
                            atom_index=int(atom_index),
                            axis=axis,
                            direction=inverse_cell[:, axis],
                            kind="scaled",
                        )
        else:
            raise ValueError(
                "Managed MLFF vibration analysis supports only FixAtoms, "
                f"FixCartesian, and FixScaled; got {type(constraint).__name__}."
            )

    if not rows:
        return np.eye(ndof, dtype=float), records
    basis = null_space(np.vstack(rows), rcond=1e-12)
    if basis.shape[1] == 0:
        raise ValueError("The structure has no unconstrained Cartesian degrees of freedom.")
    return np.asarray(basis, dtype=float), records


def _finite_difference_hessian_reduced(
    atoms: Any,
    basis: np.ndarray,
    *,
    delta: float,
    nfree: int = 2,
) -> tuple[np.ndarray, int, float]:
    """Differentiate raw gradients along the exact unconstrained subspace."""

    if nfree not in {2, 4}:
        raise ValueError("nfree must be 2 or 4.")
    positions = np.asarray(atoms.positions, dtype=float).copy()
    ndof, free_dof = basis.shape
    if ndof != 3 * len(atoms):
        raise ValueError("Constraint projection has the wrong Cartesian dimension.")
    reduced = np.empty((free_dof, free_dof), dtype=float)
    force_calls = 0

    def gradient_at(scale: float, column: int) -> np.ndarray:
        nonlocal force_calls
        displacement = scale * delta * basis[:, column].reshape(len(atoms), 3)
        # Direct array assignment intentionally bypasses ASE constraint
        # adjustment. The displacement direction already lies in the exact
        # constraint null space, including Cartesian-component and FixScaled
        # constraints.
        atoms.positions[:] = positions + displacement
        gradient = -np.asarray(
            atoms.get_forces(apply_constraint=False),
            dtype=float,
        ).reshape(-1)
        force_calls += 1
        return basis.T @ gradient

    try:
        for column in range(free_dof):
            gradient_minus = gradient_at(-1.0, column)
            gradient_plus = gradient_at(1.0, column)
            if nfree == 2:
                reduced[:, column] = (
                    gradient_plus - gradient_minus
                ) / (2.0 * delta)
            else:
                gradient_minus_2 = gradient_at(-2.0, column)
                gradient_plus_2 = gradient_at(2.0, column)
                reduced[:, column] = (
                    gradient_minus_2
                    - 8.0 * gradient_minus
                    + 8.0 * gradient_plus
                    - gradient_plus_2
                ) / (12.0 * delta)
    finally:
        atoms.positions[:] = positions

    if not np.isfinite(reduced).all():
        raise ValueError("Finite-difference Hessian contains non-finite values.")
    scale = max(float(np.linalg.norm(reduced)), np.finfo(float).eps)
    asymmetry = float(np.linalg.norm(reduced - reduced.T) / scale)
    return 0.5 * (reduced + reduced.T), force_calls, asymmetry


def _finite_difference_hessian(
    atoms: Any,
    basis: np.ndarray,
    *,
    delta: float,
    nfree: int = 2,
) -> tuple[np.ndarray, int]:
    reduced, force_calls, _ = _finite_difference_hessian_reduced(
        atoms,
        basis,
        delta=delta,
        nfree=nfree,
    )
    full = basis @ reduced @ basis.T
    return 0.5 * (full + full.T), force_calls


def _resolve_full_hessian_strategy(
    *,
    requested: str,
    atoms: Any,
    calculator: Any,
) -> tuple[str, np.ndarray | None, list[str]]:
    """Resolve a complete Hessian method suitable for a full mode spectrum."""

    if requested not in {"auto", "analytic", "finite_difference"}:
        raise ValueError(
            "Full vibrational analysis supports hessian_method=auto, analytic, "
            "or finite_difference."
        )
    warnings: list[str] = []
    if requested in {"auto", "analytic"}:
        if callable(getattr(calculator, "get_hessian", None)):
            try:
                return "analytic", _analytic_hessian(atoms, calculator), warnings
            except Exception as exc:
                if requested == "analytic":
                    raise
                warnings.append(
                    f"analytic_hessian_unavailable={type(exc).__name__}: {exc}"
                )
        elif requested == "analytic":
            raise ValueError(
                "hessian_method=analytic requires a calculator exposing get_hessian()."
            )
    return "finite_difference", None, warnings


def _frequency_analysis(
    *,
    atoms: Any,
    hessian: np.ndarray,
    basis: np.ndarray,
    threshold_cm1: float,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Solve the constrained mass-weighted generalized eigenproblem."""

    from ase import units
    from scipy.linalg import eigh

    reduced_hessian = basis.T @ hessian @ basis
    masses = np.repeat(np.asarray(atoms.get_masses(), dtype=float), 3)
    if np.any(masses <= 0):
        raise ValueError("All atomic masses must be positive for vibration analysis.")
    reduced_mass = basis.T @ (masses[:, None] * basis)
    omega2, reduced_modes = eigh(
        reduced_hessian,
        reduced_mass,
        check_finite=True,
    )
    factor = units._hbar * units.m / np.sqrt(units._e * units._amu) / units.invcm
    frequencies = np.sign(omega2) * factor * np.sqrt(np.abs(omega2))
    full_modes = (basis @ reduced_modes).T.reshape(len(omega2), len(atoms), 3)
    rows: list[dict[str, Any]] = []
    significant_imaginary = 0
    for index, (eigenvalue, frequency) in enumerate(zip(omega2, frequencies)):
        significant = bool(frequency < -threshold_cm1)
        significant_imaginary += int(significant)
        rows.append(
            {
                "mode_index": int(index),
                "eigenvalue_eVA2_amu": float(eigenvalue),
                "frequency_cm1": float(frequency),
                "imaginary": bool(frequency < 0),
                "significant_imaginary": significant,
            }
        )
    return rows, full_modes, significant_imaginary


def _write_vibration_artifacts(
    output_dir: Path,
    *,
    atoms: Any,
    hessian: np.ndarray,
    basis: np.ndarray,
    rows: list[dict[str, Any]],
    modes: np.ndarray,
) -> dict[str, str]:
    """Write one canonical array bundle plus compact tabular/viewer outputs."""

    from ase.io import write

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "vibrations.npz"
    archive_tmp = output_dir / ".vibrations.npz.tmp"
    reduced_hessian = basis.T @ hessian @ basis
    with archive_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(1, dtype=np.int64),
            atomic_numbers=np.asarray(atoms.numbers, dtype=np.int32),
            positions_A=np.asarray(atoms.positions, dtype=float),
            cell_A=np.asarray(atoms.cell.array, dtype=float),
            pbc=np.asarray(atoms.pbc, dtype=bool),
            masses_amu=np.asarray(atoms.get_masses(), dtype=float),
            constraint_basis=np.asarray(basis, dtype=float),
            hessian_reduced_eVA2=np.asarray(reduced_hessian, dtype=float),
            eigenvalues_eVA2_amu=np.asarray(
                [row["eigenvalue_eVA2_amu"] for row in rows],
                dtype=float,
            ),
            frequencies_cm1=np.asarray(
                [row["frequency_cm1"] for row in rows],
                dtype=float,
            ),
            modes_mass_normalized=np.asarray(modes, dtype=float),
        )
    archive_tmp.replace(archive_path)

    frequencies_path = output_dir / "frequencies.csv"
    fieldnames = [
        "mode_index",
        "eigenvalue_eVA2_amu",
        "frequency_cm1",
        "imaginary",
        "significant_imaginary",
    ]
    with frequencies_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mode_frames = []
    for row, mode in zip(rows, modes):
        frame = atoms.copy()
        # modes.extxyz is a viewer/interchange artifact. The exact constraint
        # subspace is stored once in vibrations.npz and must not be degraded to
        # extxyz's Cartesian-only move_mask representation.
        frame.set_constraint()
        frame.info = {
            "mode_index": int(row["mode_index"]),
            "frequency_cm1": float(row["frequency_cm1"]),
            "imaginary": bool(row["imaginary"]),
            "significant_imaginary": bool(row["significant_imaginary"]),
            "mode_normalization": "mass_normalized",
        }
        frame.set_array("mode", np.asarray(mode, dtype=float))
        mode_frames.append(frame)
    modes_path = output_dir / "modes.extxyz"
    write(str(modes_path), mode_frames, format="extxyz")

    return {
        "vibrations_file": archive_path.name,
        "frequencies_csv": frequencies_path.name,
        "modes_file": modes_path.name,
    }


def _max_projected_force(
    *,
    atoms: Any,
    basis: np.ndarray,
) -> float:
    raw = np.asarray(
        atoms.get_forces(apply_constraint=False),
        dtype=float,
    ).reshape(-1)
    projected = basis @ (basis.T @ raw)
    atom_norms = np.linalg.norm(projected.reshape(len(atoms), 3), axis=1)
    return float(np.max(atom_norms)) if atom_norms.size else 0.0


def _stationary_point_class(
    *,
    stationary: bool,
    significant_imaginary_count: int,
) -> str:
    if not stationary:
        return "nonstationary"
    if significant_imaginary_count == 0:
        return "minimum"
    if significant_imaginary_count == 1:
        return "first_order_saddle"
    return "higher_order_saddle"


__all__ = [
    "_analytic_hessian",
    "_constraint_projection",
    "_finite_difference_hessian",
    "_finite_difference_hessian_reduced",
    "_frequency_analysis",
    "_normalize_hessian",
    "_max_projected_force",
    "_resolve_full_hessian_strategy",
    "_stationary_point_class",
    "_write_vibration_artifacts",
]
