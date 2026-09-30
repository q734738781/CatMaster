#!/usr/bin/env python3
# Code writing date: 2026-08-26
# Responsible/related agent: ExperimentSpecialist, materials_worker, dynamics_worker, orca_xtb_worker
# Implementation principle: use PBC-aware deterministic geometry evidence before visual or physical relaxation checks.
# Purpose: report absolute and covalent-radius-normalized short contacts, expected-contact errors, and basic cell/PBC issues.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, neighbor_list


def _load_atoms(path: Path, index: int) -> Atoms:
    loaded = ase_read(str(path), index=index)
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"No structure frames found in {path}")
        loaded = loaded[-1]
    if not isinstance(loaded, Atoms):
        raise TypeError(f"ASE did not return an Atoms object for {path}")
    if len(loaded) == 0:
        raise ValueError("Structure contains no atoms")
    return loaded


def _covalent_radius(atomic_number: int) -> float:
    radius = float(covalent_radii[int(atomic_number)])
    if not math.isfinite(radius) or radius <= 0.0:
        return 0.70
    return radius


def _load_context(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Validation context must be a JSON object")
    return raw


def _body_names(atoms: Atoms, context: dict[str, Any]) -> list[str]:
    names = ["structure"] * len(atoms)
    groups = context.get("body_ranges", [])
    applied_context = False
    if isinstance(groups, list):
        for row in groups:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "body")
            start = int(row.get("start", -1))
            stop = int(row.get("stop", -1))
            if 0 <= start <= stop <= len(atoms):
                names[start:stop] = [name] * (stop - start)
                applied_context = True
    if not applied_context and "assembly_group" in atoms.arrays:
        names = [f"group_{int(value)}" for value in atoms.arrays["assembly_group"]]
    return names


def _cell_report(atoms: Atoms) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)

    if positions.shape != (len(atoms), 3) or not np.all(np.isfinite(positions)):
        issues.append({"severity": "FAIL", "code": "nonfinite_positions"})
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        issues.append({"severity": "FAIL", "code": "nonfinite_cell"})
        return issues, {"pbc": pbc.tolist(), "cell_rank": 0, "outside_primary_cell": []}

    lengths = np.linalg.norm(cell, axis=1)
    for axis in np.flatnonzero(pbc):
        if float(lengths[axis]) <= 1.0e-10:
            issues.append(
                {
                    "severity": "FAIL",
                    "code": "periodic_axis_has_zero_cell_vector",
                    "axis": int(axis),
                }
            )

    rank = int(np.linalg.matrix_rank(cell, tol=1.0e-10))
    periodic_dimensions = int(np.count_nonzero(pbc))
    periodic_rank = (
        int(np.linalg.matrix_rank(cell[pbc], tol=1.0e-10))
        if periodic_dimensions
        else 0
    )
    if periodic_rank < periodic_dimensions:
        issues.append(
            {
                "severity": "FAIL",
                "code": "cell_rank_incompatible_with_pbc",
                "cell_rank": rank,
                "periodic_cell_rank": periodic_rank,
                "periodic_dimensions": periodic_dimensions,
            }
        )
    volume = float(abs(np.linalg.det(cell)))
    if bool(np.all(pbc)) and volume <= 1.0e-10:
        issues.append({"severity": "FAIL", "code": "fully_periodic_cell_has_zero_volume"})

    outside: list[int] = []
    if rank == 3 and np.all(np.isfinite(positions)):
        scaled = np.linalg.solve(cell.T, positions.T).T
        periodic_scaled = scaled[:, pbc]
        if periodic_scaled.size:
            mask = np.any((periodic_scaled < -1.0e-7) | (periodic_scaled >= 1.0 + 1.0e-7), axis=1)
            outside = np.flatnonzero(mask).astype(int).tolist()

    return issues, {
        "pbc": pbc.tolist(),
        "cell_rank": rank,
        "periodic_cell_rank": periodic_rank,
        "cell_lengths_A": [float(value) for value in lengths],
        "cell_volume_A3": volume,
        "outside_primary_cell": outside,
        "outside_primary_cell_count": len(outside),
    }


def _canonical_pair(i: int, j: int, shift: np.ndarray) -> tuple[tuple[int, int, int, int, int], int, int, np.ndarray]:
    shift = np.asarray(shift, dtype=int).reshape(3)
    if i < j:
        return (i, j, *shift.tolist()), i, j, shift
    if i > j:
        normalized = -shift
        return (j, i, *normalized.tolist()), j, i, normalized
    forward = tuple(int(value) for value in shift)
    reverse = tuple(-int(value) for value in shift)
    normalized_tuple = min(forward, reverse)
    normalized = np.asarray(normalized_tuple, dtype=int)
    return (i, i, *normalized.tolist()), i, i, normalized


def _severity(
    distance: float,
    ratio: float,
    *,
    absolute_fail: float,
    fail_ratio: float,
    warn_ratio: float,
    review_ratio: float,
) -> str:
    if distance < absolute_fail or ratio < fail_ratio:
        return "FAIL"
    if ratio < warn_ratio:
        return "WARN"
    if ratio < review_ratio:
        return "REVIEW"
    return "OK"


def _pair_records(
    atoms: Atoms,
    *,
    body_names: list[str],
    absolute_fail: float,
    fail_ratio: float,
    warn_ratio: float,
    review_ratio: float,
    pair_scan_scale: float,
) -> list[dict[str, Any]]:
    radii = np.asarray([_covalent_radius(number) for number in atoms.numbers], dtype=float)
    scan_radii = radii * max(float(pair_scan_scale), float(review_ratio), 1.0)
    i_values, j_values, distances, shifts = neighbor_list(
        "ijdS",
        atoms,
        scan_radii,
        self_interaction=False,
    )
    unique: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    symbols = atoms.get_chemical_symbols()

    for raw_i, raw_j, raw_distance, raw_shift in zip(i_values, j_values, distances, shifts):
        key, i, j, shift = _canonical_pair(int(raw_i), int(raw_j), np.asarray(raw_shift))
        distance = float(raw_distance)
        radius_sum = float(radii[i] + radii[j])
        ratio = distance / radius_sum if radius_sum > 0.0 else math.inf
        severity = _severity(
            distance,
            ratio,
            absolute_fail=absolute_fail,
            fail_ratio=fail_ratio,
            warn_ratio=warn_ratio,
            review_ratio=review_ratio,
        )
        body_i = body_names[i]
        body_j = body_names[j]
        record = {
            "i": i,
            "j": j,
            "i_1based": i + 1,
            "j_1based": j + 1,
            "elements": [symbols[i], symbols[j]],
            "distance_A": distance,
            "covalent_radius_sum_A": radius_sum,
            "distance_ratio": float(ratio),
            "severity": severity,
            "image_shift": [int(value) for value in shift],
            "periodic_self_image": bool(i == j and np.any(shift != 0)),
            "bodies": [body_i, body_j],
            "scope": "intra_body" if body_i == body_j else "inter_body",
        }
        previous = unique.get(key)
        if previous is None or distance < float(previous["distance_A"]):
            unique[key] = record

    return sorted(
        unique.values(),
        key=lambda row: (
            float(row["distance_A"]),
            float(row["distance_ratio"]),
            int(row["i"]),
            int(row["j"]),
            tuple(row["image_shift"]),
        ),
    )


def _expected_contact_report(atoms: Atoms, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = context.get("expected_contacts", [])
    if not isinstance(rows, list):
        raise ValueError("expected_contacts in the validation context must be a list")
    report: list[dict[str, Any]] = []
    use_mic = bool(np.any(atoms.pbc))
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"expected_contacts[{index}] must be an object")
        i = int(row["i"])
        j = int(row["j"])
        if not (0 <= i < len(atoms) and 0 <= j < len(atoms)):
            raise ValueError(f"expected contact index out of range: {i}, {j}")
        target = float(row["target"])
        tolerance = float(row.get("tolerance", 0.0))
        if not all(math.isfinite(value) for value in (target, tolerance)):
            raise ValueError("Expected-contact target and tolerance must be finite")
        if target < 0.0 or tolerance < 0.0:
            raise ValueError("Expected-contact target and tolerance must be non-negative")
        distance = float(atoms.get_distance(i, j, mic=use_mic))
        error = abs(distance - target)
        report.append(
            {
                "label": str(row.get("label") or f"contact_{index + 1}"),
                "i": i,
                "j": j,
                "i_1based": i + 1,
                "j_1based": j + 1,
                "target_A": target,
                "tolerance_A": tolerance,
                "distance_A": distance,
                "absolute_error_A": error,
                "status": "PASS" if error <= tolerance + 1.0e-12 else "FAIL",
            }
        )
    return report


def _coordination_report(atoms: Atoms, scale: float) -> dict[str, Any]:
    cutoffs = natural_cutoffs(atoms, mult=float(scale))
    i_values = neighbor_list("i", atoms, cutoffs, self_interaction=False)
    counts = np.bincount(np.asarray(i_values, dtype=int), minlength=len(atoms)).astype(int)
    return {
        "heuristic": "ASE natural_cutoffs based on covalent radii",
        "scale": float(scale),
        "counts": counts.tolist(),
        "isolated_indices": np.flatnonzero(counts == 0).astype(int).tolist(),
        "isolated_indices_1based": (np.flatnonzero(counts == 0) + 1).astype(int).tolist(),
    }


def check_structure(
    *,
    structure_path: Path,
    context_path: Path | None,
    frame_index: int,
    top_pairs: int,
    absolute_fail: float,
    fail_ratio: float,
    warn_ratio: float,
    review_ratio: float,
    pair_scan_scale: float,
    coordination_scale: float,
) -> dict[str, Any]:
    numeric_controls = (
        absolute_fail,
        fail_ratio,
        warn_ratio,
        review_ratio,
        pair_scan_scale,
        coordination_scale,
    )
    if not all(math.isfinite(value) for value in numeric_controls):
        raise ValueError("Distance and scale controls must be finite")
    if not (0.0 < fail_ratio < warn_ratio < review_ratio):
        raise ValueError("Require 0 < fail_ratio < warn_ratio < review_ratio")
    if absolute_fail <= 0.0 or pair_scan_scale <= 0.0 or coordination_scale <= 0.0:
        raise ValueError("Distance and scale controls must be positive")
    if top_pairs < 1:
        raise ValueError("top_pairs must be positive")

    atoms = _load_atoms(structure_path, frame_index)
    context = _load_context(context_path)
    body_names = _body_names(atoms, context)
    cell_issues, cell = _cell_report(atoms)

    positions_finite = bool(np.all(np.isfinite(np.asarray(atoms.positions, dtype=float))))
    cell_finite = bool(np.all(np.isfinite(np.asarray(atoms.cell.array, dtype=float))))
    pairs: list[dict[str, Any]] = []
    expected_contacts: list[dict[str, Any]] = []
    coordination: dict[str, Any] = {}
    if positions_finite and cell_finite and not any(row["severity"] == "FAIL" for row in cell_issues):
        pairs = _pair_records(
            atoms,
            body_names=body_names,
            absolute_fail=absolute_fail,
            fail_ratio=fail_ratio,
            warn_ratio=warn_ratio,
            review_ratio=review_ratio,
            pair_scan_scale=pair_scan_scale,
        )
        expected_contacts = _expected_contact_report(atoms, context)
        coordination = _coordination_report(atoms, coordination_scale)

    flagged = [row for row in pairs if row["severity"] != "OK"]
    severity_counts = {
        level: sum(row["severity"] == level for row in pairs)
        for level in ("FAIL", "WARN", "REVIEW", "OK")
    }
    hard_fail = (
        any(row["severity"] == "FAIL" for row in cell_issues)
        or severity_counts["FAIL"] > 0
        or any(row["status"] == "FAIL" for row in expected_contacts)
    )
    has_warning = severity_counts["WARN"] > 0 or severity_counts["REVIEW"] > 0
    status = "FAIL" if hard_fail else ("WARN" if has_warning else "PASS")

    return {
        "status": status,
        "structure_path": str(structure_path),
        "context_path": str(context_path) if context_path else "",
        "frame_index": int(frame_index),
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "thresholds": {
            "absolute_fail_A": float(absolute_fail),
            "fail_ratio": float(fail_ratio),
            "warn_ratio": float(warn_ratio),
            "review_ratio": float(review_ratio),
            "pair_scan_scale": float(pair_scan_scale),
        },
        "cell": cell,
        "cell_issues": cell_issues,
        "severity_counts": severity_counts,
        "flagged_pairs": flagged,
        "shortest_pairs": pairs[: max(1, int(top_pairs))],
        "expected_contacts": expected_contacts,
        "coordination": coordination,
        "interpretation": (
            "Distance bands are geometry-screening heuristics, not a bond-order or energetic-stability model. "
            "Review WARN/REVIEW contacts in their chemical context."
        ),
    }


def _default_output_path(structure_path: Path) -> Path:
    return structure_path.with_name(f"{structure_path.stem}_geometry_check.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PBC-aware atomic overlap and short-contact checker with covalent-radius normalization."
    )
    parser.add_argument("structure", type=Path, help="Input structure or trajectory readable by ASE.")
    parser.add_argument("--context", type=Path, help="Optional assembly validation context JSON.")
    parser.add_argument("--output", type=Path, help="Output JSON; defaults beside the structure.")
    parser.add_argument("--index", type=int, default=-1, help="ASE frame index, default last frame.")
    parser.add_argument("--top-pairs", type=int, default=20, help="Number of shortest scanned pairs to summarize.")
    parser.add_argument("--absolute-fail", type=float, default=0.50, help="Unconditional failure distance in angstrom.")
    parser.add_argument("--fail-ratio", type=float, default=0.60)
    parser.add_argument("--warn-ratio", type=float, default=0.75)
    parser.add_argument("--review-ratio", type=float, default=0.85)
    parser.add_argument(
        "--pair-scan-scale",
        type=float,
        default=1.25,
        help="Covalent-radius multiplier used only to collect nearby pairs for reporting.",
    )
    parser.add_argument(
        "--coordination-scale",
        type=float,
        default=1.15,
        help="ASE natural-cutoff multiplier for diagnostic neighbor counts.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("fail", "warn"),
        default="fail",
        help="Exit nonzero on FAIL only, or on either FAIL/WARN.",
    )
    parser.add_argument("--verbose", action="store_true", help="Also print the complete JSON report; it is always saved to the report file")
    return parser.parse_args()


def format_summary(report: dict[str, Any], report_path: Path | str) -> str:
    """Console triage; complete pair/contact/coordination data stays in JSON."""
    lines = [
        f"status={report['status']} atoms={report['natoms']}",
        f"report={report_path}",
        f"severity_counts={json.dumps(report['severity_counts'], sort_keys=True)}",
    ]
    if report["cell_issues"]:
        lines.append("cell_issues=" + json.dumps(report["cell_issues"], sort_keys=True))
    flagged = report["flagged_pairs"]
    if flagged:
        # Absolute distance and radius-normalized distance are independent gates.
        # Report both extrema so a short normal bond cannot mask a metal clash.
        worst = [min(flagged, key=lambda pair: pair["distance_A"])]
        lowest_ratio = min(flagged, key=lambda pair: pair["distance_ratio"])
        if lowest_ratio != worst[0]:
            worst.append(lowest_ratio)
        for pair in worst:
            lines.append(
                "flagged_extreme="
                f"{pair['i_1based']}-{pair['j_1based']} {'-'.join(pair['elements'])} "
                f"d={pair['distance_A']:.6f}A q={pair['distance_ratio']:.6f} "
                f"{pair['severity']} scope={pair['scope']} image={pair['image_shift']}"
            )
    failed_contacts = [row for row in report["expected_contacts"] if row["status"] == "FAIL"]
    if failed_contacts:
        worst_contact = max(
            failed_contacts, key=lambda row: row["absolute_error_A"] - row["tolerance_A"]
        )
        lines.append(
            f"failed_expected_contacts={len(failed_contacts)} worst="
            f"{worst_contact['i_1based']}-{worst_contact['j_1based']} "
            f"d={worst_contact['distance_A']:.6f}A "
            f"target={worst_contact['target_A']:.6f}A "
            f"tolerance={worst_contact['tolerance_A']:.6f}A"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_path = args.output or _default_output_path(args.structure)
    report = check_structure(
        structure_path=args.structure,
        context_path=args.context,
        frame_index=args.index,
        top_pairs=args.top_pairs,
        absolute_fail=args.absolute_fail,
        fail_ratio=args.fail_ratio,
        warn_ratio=args.warn_ratio,
        review_ratio=args.review_ratio,
        pair_scan_scale=args.pair_scan_scale,
        coordination_scale=args.coordination_scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(format_summary(report, output_path))
    if args.verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    should_fail = report["status"] == "FAIL" or (args.fail_on == "warn" and report["status"] == "WARN")
    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
