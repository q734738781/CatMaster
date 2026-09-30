#!/usr/bin/env python3
# Code writing date: 2026-08-26
# Responsible/related agent: ExperimentSpecialist, materials_worker, orca_xtb_worker
# Implementation principle: let the agent define chemical contacts while a deterministic optimizer solves rigid 3D poses.
# Purpose: assemble a fixed host and rigid fragments with multi-start soft-repulsion and explicit geometric constraints.
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.io import write as ase_write
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


@dataclass
class RigidBody:
    name: str
    source_path: Path
    atoms: Atoms
    seed_positions: np.ndarray
    seed_pivot: np.ndarray
    radii: np.ndarray
    translation_jitter: float
    rotation_jitter_rad: float
    translation_limit: float


def _read_atoms(path: Path) -> Atoms:
    loaded = ase_read(str(path), index=-1)
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"No structure frames found in {path}")
        loaded = loaded[-1]
    if not isinstance(loaded, Atoms) or len(loaded) == 0:
        raise ValueError(f"Expected a nonempty atomic structure at {path}")
    if not np.all(np.isfinite(np.asarray(loaded.positions, dtype=float))):
        raise ValueError(f"Non-finite coordinates in {path}")
    return loaded


def _vec3(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return array


def _unit(value: Any, *, name: str) -> np.ndarray:
    vector = _vec3(value, name=name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must not be a zero vector")
    return vector / norm


def _radius(number: int) -> float:
    value = float(covalent_radii[int(number)])
    if not math.isfinite(value) or value <= 0.0:
        return 0.70
    return value


def _body_atom(positions: dict[str, np.ndarray], ref: dict[str, Any], *, name: str) -> np.ndarray:
    body = str(ref.get("body") or "")
    if body not in positions:
        raise ValueError(f"{name}.body refers to unknown or not-yet-seeded body {body!r}")
    atom = int(ref.get("atom", -1))
    if not 0 <= atom < len(positions[body]):
        raise ValueError(f"{name}.atom out of range for {body}: {atom}")
    return positions[body][atom]


def _target_point(positions: dict[str, np.ndarray], target: Any, *, name: str) -> np.ndarray:
    if not isinstance(target, dict):
        raise ValueError(f"{name} must be an object")
    if "point" in target:
        return _vec3(target["point"], name=f"{name}.point")
    return _body_atom(positions, target, name=name)


def _seed_fragment(
    row: dict[str, Any],
    *,
    default_translation_jitter: float,
    default_rotation_jitter_deg: float,
    default_translation_limit: float,
    seeded_positions: dict[str, np.ndarray],
) -> RigidBody:
    name = str(row.get("name") or "").strip()
    if not name or name == "host":
        raise ValueError("Every fragment needs a unique nonempty name other than 'host'")
    path = Path(str(row.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"Fragment file not found: {path}")
    atoms = _read_atoms(path)
    positions = np.asarray(atoms.positions, dtype=float).copy()

    pivot_atom = row.get("pivot_atom")
    if pivot_atom is None:
        pivot = positions.mean(axis=0)
    else:
        pivot_index = int(pivot_atom)
        if not 0 <= pivot_index < len(atoms):
            raise ValueError(f"pivot_atom out of range for {name}: {pivot_index}")
        pivot = positions[pivot_index].copy()

    initial_rotation = _vec3(row.get("initial_rotation_deg", [0.0, 0.0, 0.0]), name=f"{name}.initial_rotation_deg")
    if np.any(np.abs(initial_rotation) > 1.0e-12):
        rotation = Rotation.from_euler("xyz", initial_rotation, degrees=True)
        positions = rotation.apply(positions - pivot) + pivot

    align = row.get("align")
    if align is not None:
        if not isinstance(align, dict):
            raise ValueError(f"{name}.align must be an object")
        axis_atoms = [int(value) for value in align.get("axis_atoms", [])]
        if len(axis_atoms) != 2 or any(index < 0 or index >= len(atoms) for index in axis_atoms):
            raise ValueError(f"{name}.align.axis_atoms must contain two valid atom indices")
        source_axis = _unit(
            positions[axis_atoms[1]] - positions[axis_atoms[0]],
            name=f"{name}.align.source_axis",
        )
        target_axis = _unit(align.get("target_vector"), name=f"{name}.align.target_vector")
        rotation, _ = Rotation.align_vectors([target_axis], [source_axis])
        positions = rotation.apply(positions - pivot) + pivot

    translation = _vec3(row.get("initial_translation", [0.0, 0.0, 0.0]), name=f"{name}.initial_translation")
    positions = positions + translation

    placement = row.get("placement")
    if placement is not None:
        if not isinstance(placement, dict):
            raise ValueError(f"{name}.placement must be an object")
        fragment_atom = int(placement.get("fragment_atom", -1))
        if not 0 <= fragment_atom < len(atoms):
            raise ValueError(f"{name}.placement.fragment_atom out of range: {fragment_atom}")
        target = _target_point(seeded_positions, placement.get("target"), name=f"{name}.placement.target")
        direction = _unit(placement.get("direction"), name=f"{name}.placement.direction")
        distance = float(placement.get("distance", 0.0))
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"{name}.placement.distance must be finite and non-negative")
        desired = target + distance * direction
        positions = positions + (desired - positions[fragment_atom])

    seed_pivot = positions[int(pivot_atom)].copy() if pivot_atom is not None else positions.mean(axis=0)
    translation_jitter = float(row.get("translation_jitter", default_translation_jitter))
    rotation_jitter_deg = float(row.get("rotation_jitter_deg", default_rotation_jitter_deg))
    translation_limit = float(row.get("translation_limit", default_translation_limit))
    for label, value in (
        ("translation_jitter", translation_jitter),
        ("rotation_jitter_deg", rotation_jitter_deg),
        ("translation_limit", translation_limit),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}.{label} must be finite and non-negative")

    return RigidBody(
        name=name,
        source_path=path,
        atoms=atoms,
        seed_positions=positions,
        seed_pivot=seed_pivot,
        radii=np.asarray([_radius(number) for number in atoms.numbers], dtype=float),
        translation_jitter=translation_jitter,
        rotation_jitter_rad=math.radians(rotation_jitter_deg),
        translation_limit=translation_limit,
    )


class AssemblyProblem:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        settings = spec.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("settings must be a JSON object")
        self.settings = settings

        host_spec = spec.get("host")
        if isinstance(host_spec, str):
            host_path = Path(host_spec)
        elif isinstance(host_spec, dict):
            host_path = Path(str(host_spec.get("path") or ""))
        else:
            raise ValueError("host must be a path string or an object with path")
        if not host_path.is_file():
            raise FileNotFoundError(f"Host file not found: {host_path}")
        self.host_path = host_path
        self.host = _read_atoms(host_path)
        self.host_positions = np.asarray(self.host.positions, dtype=float).copy()
        self.host_radii = np.asarray([_radius(number) for number in self.host.numbers], dtype=float)

        self.cell = np.asarray(self.host.cell.array, dtype=float)
        self.pbc = np.asarray(self.host.pbc, dtype=bool)
        if np.any(self.pbc):
            if not np.all(np.isfinite(self.cell)):
                raise ValueError("Periodic host has a non-finite cell")
            for axis in np.flatnonzero(self.pbc):
                if np.linalg.norm(self.cell[axis]) <= 1.0e-10:
                    raise ValueError(f"Periodic host axis {axis} has a zero cell vector")

        default_translation_jitter = float(settings.get("translation_jitter", 0.35))
        default_rotation_jitter_deg = float(settings.get("rotation_jitter_deg", 35.0))
        default_translation_limit = float(settings.get("translation_limit", 3.0))
        for label, value in (
            ("translation_jitter", default_translation_jitter),
            ("rotation_jitter_deg", default_rotation_jitter_deg),
            ("translation_limit", default_translation_limit),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"settings.{label} must be finite and non-negative")

        fragment_rows = spec.get("fragments")
        if not isinstance(fragment_rows, list) or not fragment_rows:
            raise ValueError("fragments must be a nonempty list")
        seeded_positions: dict[str, np.ndarray] = {"host": self.host_positions}
        self.fragments: list[RigidBody] = []
        for index, row in enumerate(fragment_rows):
            if not isinstance(row, dict):
                raise ValueError(f"fragments[{index}] must be an object")
            body = _seed_fragment(
                row,
                default_translation_jitter=default_translation_jitter,
                default_rotation_jitter_deg=default_rotation_jitter_deg,
                default_translation_limit=default_translation_limit,
                seeded_positions=seeded_positions,
            )
            if body.name in seeded_positions:
                raise ValueError(f"Duplicate fragment name: {body.name}")
            self.fragments.append(body)
            seeded_positions[body.name] = body.seed_positions

        self.body_names = ["host", *(body.name for body in self.fragments)]
        self.anchors = self._validated_object_list("anchors")
        self.orientations = self._validated_object_list("orientations")
        self.regions = self._validated_object_list("regions")
        self._validate_constraints()

        self.clash_scale = float(settings.get("clash_scale", 0.75))
        self.absolute_clash = float(settings.get("absolute_clash", 0.50))
        self.clash_weight = float(settings.get("clash_weight", 100.0))
        self.seed_tether_weight = float(settings.get("seed_tether_weight", 0.0))
        self.seed_tether_tolerance = float(settings.get("seed_tether_tolerance", 0.0))
        self.feasibility_tolerance = float(settings.get("feasibility_tolerance", 0.01))
        geometry_controls = (
            self.clash_scale,
            self.absolute_clash,
            self.clash_weight,
            self.seed_tether_weight,
            self.seed_tether_tolerance,
            self.feasibility_tolerance,
        )
        if not all(math.isfinite(value) for value in geometry_controls):
            raise ValueError("Assembly geometry controls must be finite")
        if self.clash_scale <= 0.0 or self.absolute_clash <= 0.0 or self.clash_weight <= 0.0:
            raise ValueError("clash_scale, absolute_clash, and clash_weight must be positive")
        if self.seed_tether_weight < 0.0 or self.seed_tether_tolerance < 0.0:
            raise ValueError("Seed-tether controls must be non-negative")
        if self.feasibility_tolerance < 0.0:
            raise ValueError("feasibility_tolerance must be non-negative")

        self.excluded_clash_pairs: set[tuple[str, int, str, int]] = set()
        for anchor in self.anchors:
            a = self._validate_atom_ref(anchor.get("a"), name="anchor.a")
            b = self._validate_atom_ref(anchor.get("b"), name="anchor.b")
            self.excluded_clash_pairs.add((a[0], a[1], b[0], b[1]))
            self.excluded_clash_pairs.add((b[0], b[1], a[0], a[1]))

    def _validated_object_list(self, key: str) -> list[dict[str, Any]]:
        rows = self.spec.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"{key} must be a list")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Every {key} entry must be an object")
        return rows

    def _body_size(self, body: str) -> int:
        if body == "host":
            return len(self.host)
        for fragment in self.fragments:
            if fragment.name == body:
                return len(fragment.atoms)
        raise ValueError(f"Unknown body: {body}")

    def _validate_atom_ref(self, raw: Any, *, name: str) -> tuple[str, int]:
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be an object")
        body = str(raw.get("body") or "")
        if body not in self.body_names:
            raise ValueError(f"{name}.body is unknown: {body!r}")
        atom = int(raw.get("atom", -1))
        if not 0 <= atom < self._body_size(body):
            raise ValueError(f"{name}.atom out of range for {body}: {atom}")
        return body, atom

    def _validate_constraints(self) -> None:
        position_constraints = {name: 0 for name in self.body_names if name != "host"}
        for index, anchor in enumerate(self.anchors):
            a = self._validate_atom_ref(anchor.get("a"), name=f"anchors[{index}].a")
            b = self._validate_atom_ref(anchor.get("b"), name=f"anchors[{index}].b")
            if a[0] == b[0]:
                raise ValueError("Rigid assembly anchors must join different bodies")
            target = float(anchor.get("target", -1.0))
            tolerance = float(anchor.get("tolerance", 0.0))
            weight = float(anchor.get("weight", 100.0))
            if not all(math.isfinite(value) for value in (target, tolerance, weight)):
                raise ValueError("Anchor target, tolerance, and weight must be finite")
            if target < 0.0 or tolerance < 0.0 or weight <= 0.0:
                raise ValueError("Anchor target/tolerance must be non-negative and weight positive")
            if a[0] != "host":
                position_constraints[a[0]] += 1
            if b[0] != "host":
                position_constraints[b[0]] += 1

        for index, row in enumerate(self.orientations):
            body = str(row.get("body") or "")
            if body == "host" or body not in self.body_names:
                raise ValueError(f"orientations[{index}].body must name a fragment")
            axis_atoms = [int(value) for value in row.get("axis_atoms", [])]
            if len(axis_atoms) != 2 or any(atom < 0 or atom >= self._body_size(body) for atom in axis_atoms):
                raise ValueError(f"orientations[{index}].axis_atoms must contain two valid indices")
            _unit(row.get("target_vector"), name=f"orientations[{index}].target_vector")
            target_angle = float(row.get("target_angle_deg", 0.0))
            tolerance = float(row.get("tolerance_deg", 0.0))
            weight = float(row.get("weight", 10.0))
            if not all(math.isfinite(value) for value in (target_angle, tolerance, weight)):
                raise ValueError("Orientation target, tolerance, and weight must be finite")
            if not 0.0 <= target_angle <= 180.0 or tolerance < 0.0 or weight <= 0.0:
                raise ValueError("Orientation target must be 0..180 degrees, tolerance non-negative, weight positive")

        for index, row in enumerate(self.regions):
            body = str(row.get("body") or "")
            if body == "host" or body not in self.body_names:
                raise ValueError(f"regions[{index}].body must name a fragment")
            lower = _vec3(row.get("lower"), name=f"regions[{index}].lower")
            upper = _vec3(row.get("upper"), name=f"regions[{index}].upper")
            if np.any(lower > upper):
                raise ValueError(f"regions[{index}] lower bounds exceed upper bounds")
            coordinate_type = str(row.get("coordinate_type", "cartesian"))
            if coordinate_type not in {"cartesian", "fractional"}:
                raise ValueError("Region coordinate_type must be cartesian or fractional")
            if coordinate_type == "fractional" and np.linalg.matrix_rank(self.cell) < 3:
                raise ValueError("Fractional regions require a full-rank host cell")
            if "reference_atom" in row:
                atom = int(row["reference_atom"])
                if not 0 <= atom < self._body_size(body):
                    raise ValueError(f"regions[{index}].reference_atom out of range")
            weight = float(row.get("weight", 50.0))
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("Region weight must be positive")
            position_constraints[body] += 1

        if float(self.settings.get("seed_tether_weight", 0.0)) <= 0.0:
            unconstrained = [name for name, count in position_constraints.items() if count == 0]
            if unconstrained:
                raise ValueError(
                    "Every fragment needs an anchor or region unless settings.seed_tether_weight is positive: "
                    + ", ".join(unconstrained)
                )

    def positions(self, variables: np.ndarray) -> dict[str, np.ndarray]:
        variables = np.asarray(variables, dtype=float)
        expected = 6 * len(self.fragments)
        if variables.shape != (expected,):
            raise ValueError(f"Expected {expected} pose variables, got {variables.shape}")
        result = {"host": self.host_positions}
        for index, body in enumerate(self.fragments):
            translation = variables[6 * index : 6 * index + 3]
            rotvec = variables[6 * index + 3 : 6 * index + 6]
            rotation = Rotation.from_rotvec(rotvec)
            result[body.name] = rotation.apply(body.seed_positions - body.seed_pivot) + body.seed_pivot + translation
        return result

    def _mic_distances(self, vectors: np.ndarray) -> np.ndarray:
        flat = np.asarray(vectors, dtype=float).reshape(-1, 3)
        if np.any(self.pbc):
            _, distances = find_mic(flat, self.cell, pbc=self.pbc)
            return np.asarray(distances, dtype=float).reshape(vectors.shape[:-1])
        return np.linalg.norm(flat, axis=1).reshape(vectors.shape[:-1])

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self._mic_distances(np.asarray(b - a, dtype=float).reshape(1, 3))[0])

    def evaluate(self, variables: np.ndarray, *, details: bool = False) -> tuple[float, dict[str, Any]]:
        positions = self.positions(variables)
        radii = {"host": self.host_radii, **{body.name: body.radii for body in self.fragments}}
        clash_loss = 0.0
        clash_rows: list[dict[str, Any]] = []
        max_penetration = 0.0

        for left_index, left_name in enumerate(self.body_names):
            for right_name in self.body_names[left_index + 1 :]:
                left = positions[left_name]
                right = positions[right_name]
                vectors = right[None, :, :] - left[:, None, :]
                distances = self._mic_distances(vectors)
                cutoff = np.maximum(
                    self.clash_scale * (radii[left_name][:, None] + radii[right_name][None, :]),
                    self.absolute_clash,
                )
                penetration = np.maximum(0.0, cutoff - distances)
                for i, j in np.argwhere(penetration > 0.0):
                    if (left_name, int(i), right_name, int(j)) in self.excluded_clash_pairs:
                        penetration[int(i), int(j)] = 0.0
                normalized = np.divide(
                    penetration,
                    cutoff,
                    out=np.zeros_like(penetration),
                    where=cutoff > 0.0,
                )
                clash_loss += self.clash_weight * float(np.sum(normalized * normalized))
                if penetration.size:
                    max_penetration = max(max_penetration, float(np.max(penetration)))
                if details:
                    for i, j in np.argwhere(penetration > self.feasibility_tolerance):
                        clash_rows.append(
                            {
                                "a": {"body": left_name, "atom": int(i)},
                                "b": {"body": right_name, "atom": int(j)},
                                "distance_A": float(distances[i, j]),
                                "exclusion_A": float(cutoff[i, j]),
                                "penetration_A": float(penetration[i, j]),
                            }
                        )

        anchor_loss = 0.0
        anchor_rows: list[dict[str, Any]] = []
        max_anchor_violation = 0.0
        for index, row in enumerate(self.anchors):
            a_body, a_atom = self._validate_atom_ref(row["a"], name=f"anchors[{index}].a")
            b_body, b_atom = self._validate_atom_ref(row["b"], name=f"anchors[{index}].b")
            target = float(row["target"])
            tolerance = float(row.get("tolerance", 0.0))
            weight = float(row.get("weight", 100.0))
            distance = self._distance(positions[a_body][a_atom], positions[b_body][b_atom])
            violation = max(0.0, abs(distance - target) - tolerance)
            anchor_loss += weight * violation * violation
            max_anchor_violation = max(max_anchor_violation, violation)
            if details:
                anchor_rows.append(
                    {
                        "label": str(row.get("label") or f"anchor_{index + 1}"),
                        "a": {"body": a_body, "atom": a_atom},
                        "b": {"body": b_body, "atom": b_atom},
                        "target_A": target,
                        "tolerance_A": tolerance,
                        "distance_A": distance,
                        "violation_A": violation,
                    }
                )

        orientation_loss = 0.0
        orientation_rows: list[dict[str, Any]] = []
        max_orientation_violation_deg = 0.0
        for index, row in enumerate(self.orientations):
            body = str(row["body"])
            a, b = (int(value) for value in row["axis_atoms"])
            axis = _unit(positions[body][b] - positions[body][a], name=f"orientation[{index}].axis")
            target_vector = _unit(row["target_vector"], name=f"orientation[{index}].target_vector")
            angle = math.degrees(math.acos(float(np.clip(np.dot(axis, target_vector), -1.0, 1.0))))
            target_angle = float(row.get("target_angle_deg", 0.0))
            tolerance = float(row.get("tolerance_deg", 0.0))
            violation_deg = max(0.0, abs(angle - target_angle) - tolerance)
            violation_rad = math.radians(violation_deg)
            orientation_loss += float(row.get("weight", 10.0)) * violation_rad * violation_rad
            max_orientation_violation_deg = max(max_orientation_violation_deg, violation_deg)
            if details:
                orientation_rows.append(
                    {
                        "body": body,
                        "axis_atoms": [a, b],
                        "angle_deg": angle,
                        "target_angle_deg": target_angle,
                        "tolerance_deg": tolerance,
                        "violation_deg": violation_deg,
                    }
                )

        region_loss = 0.0
        region_rows: list[dict[str, Any]] = []
        max_region_violation = 0.0
        for index, row in enumerate(self.regions):
            body = str(row["body"])
            body_positions = positions[body]
            point = (
                body_positions[int(row["reference_atom"])]
                if "reference_atom" in row
                else body_positions.mean(axis=0)
            )
            coordinate_type = str(row.get("coordinate_type", "cartesian"))
            coordinates = np.linalg.solve(self.cell.T, point) if coordinate_type == "fractional" else point
            lower = _vec3(row["lower"], name=f"regions[{index}].lower")
            upper = _vec3(row["upper"], name=f"regions[{index}].upper")
            below = np.maximum(0.0, lower - coordinates)
            above = np.maximum(0.0, coordinates - upper)
            violation = below + above
            region_loss += float(row.get("weight", 50.0)) * float(np.dot(violation, violation))
            max_region_violation = max(max_region_violation, float(np.max(violation)))
            if details:
                region_rows.append(
                    {
                        "body": body,
                        "coordinate_type": coordinate_type,
                        "coordinates": [float(value) for value in coordinates],
                        "lower": [float(value) for value in lower],
                        "upper": [float(value) for value in upper],
                        "max_violation": float(np.max(violation)),
                    }
                )

        tether_loss = 0.0
        if self.seed_tether_weight > 0.0:
            for index in range(len(self.fragments)):
                translation = np.asarray(variables[6 * index : 6 * index + 3], dtype=float)
                excess = max(0.0, float(np.linalg.norm(translation)) - self.seed_tether_tolerance)
                tether_loss += self.seed_tether_weight * excess * excess

        total = clash_loss + anchor_loss + orientation_loss + region_loss + tether_loss
        metrics = {
            "objective": float(total),
            "losses": {
                "clash": float(clash_loss),
                "anchor": float(anchor_loss),
                "orientation": float(orientation_loss),
                "region": float(region_loss),
                "seed_tether": float(tether_loss),
            },
            "max_penetration_A": float(max_penetration),
            "max_anchor_violation_A": float(max_anchor_violation),
            "max_orientation_violation_deg": float(max_orientation_violation_deg),
            "max_region_violation": float(max_region_violation),
        }
        if details:
            metrics.update(
                {
                    "clashes": clash_rows,
                    "anchors": anchor_rows,
                    "orientations": orientation_rows,
                    "regions": region_rows,
                }
            )
        return float(total), metrics

    def feasible(self, metrics: dict[str, Any]) -> bool:
        tol = self.feasibility_tolerance
        return bool(
            float(metrics["max_penetration_A"]) <= tol
            and float(metrics["max_anchor_violation_A"]) <= tol
            and float(metrics["max_orientation_violation_deg"]) <= 0.5
            and float(metrics["max_region_violation"]) <= tol
        )

    def bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for body in self.fragments:
            limit = float(body.translation_limit)
            bounds.extend([(-limit, limit)] * 3)
            bounds.extend([(-math.pi, math.pi)] * 3)
        return bounds

    def initial_variables(self, rng: np.random.Generator, *, first: bool) -> np.ndarray:
        variables = np.zeros(6 * len(self.fragments), dtype=float)
        if first:
            return variables
        for index, body in enumerate(self.fragments):
            variables[6 * index : 6 * index + 3] = rng.uniform(
                -body.translation_jitter,
                body.translation_jitter,
                size=3,
            )
            axis = rng.normal(size=3)
            norm = float(np.linalg.norm(axis))
            if norm <= 1.0e-12:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / norm
            angle = rng.uniform(-body.rotation_jitter_rad, body.rotation_jitter_rad)
            variables[6 * index + 3 : 6 * index + 6] = axis * angle
        return variables

    def combined_atoms(self, variables: np.ndarray, *, score: float, feasible: bool) -> Atoms:
        positions = self.positions(variables)
        combined = self.host.copy()
        host_constraints = copy.deepcopy(self.host.constraints)
        groups = [0] * len(self.host)
        for group, body in enumerate(self.fragments, start=1):
            fragment = body.atoms.copy()
            fragment.positions = positions[body.name]
            fragment.set_constraint()
            combined.extend(fragment)
            groups.extend([group] * len(fragment))
        combined.set_cell(self.host.cell)
        combined.set_pbc(self.host.pbc)
        combined.set_constraint(host_constraints)
        combined.set_array("assembly_group", np.asarray(groups, dtype=int))
        combined.info["assembly_score"] = float(score)
        combined.info["assembly_feasible"] = bool(feasible)
        return combined

    def body_ranges(self) -> list[dict[str, Any]]:
        rows = [{"name": "host", "group": 0, "start": 0, "stop": len(self.host)}]
        offset = len(self.host)
        for group, body in enumerate(self.fragments, start=1):
            rows.append(
                {
                    "name": body.name,
                    "group": group,
                    "start": offset,
                    "stop": offset + len(body.atoms),
                }
            )
            offset += len(body.atoms)
        return rows

    def validation_context(self) -> dict[str, Any]:
        ranges = self.body_ranges()
        offsets = {row["name"]: int(row["start"]) for row in ranges}
        contacts: list[dict[str, Any]] = []
        for index, row in enumerate(self.anchors):
            a_body, a_atom = self._validate_atom_ref(row["a"], name=f"anchors[{index}].a")
            b_body, b_atom = self._validate_atom_ref(row["b"], name=f"anchors[{index}].b")
            contacts.append(
                {
                    "label": str(row.get("label") or f"anchor_{index + 1}"),
                    "i": offsets[a_body] + a_atom,
                    "j": offsets[b_body] + b_atom,
                    "target": float(row["target"]),
                    "tolerance": float(row.get("tolerance", 0.0)),
                }
            )
        return {"body_ranges": ranges, "expected_contacts": contacts}

    def mobile_positions(self, variables: np.ndarray) -> np.ndarray:
        positions = self.positions(variables)
        return np.concatenate([positions[body.name] for body in self.fragments], axis=0)


def _pose_rmsd(problem: AssemblyProblem, left: np.ndarray, right: np.ndarray) -> float:
    difference = problem.mobile_positions(left) - problem.mobile_positions(right)
    if np.any(problem.pbc):
        difference, _ = find_mic(difference, problem.cell, pbc=problem.pbc)
    return float(np.sqrt(np.mean(np.sum(np.asarray(difference) ** 2, axis=1))))


def _output_suffix(output_format: str) -> str:
    return {"extxyz": ".extxyz", "vasp": ".vasp", "xyz": ".xyz"}[output_format]


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {path}")
    report_path = path / "assembly_report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Assembly output already exists: {report_path}; pass --overwrite intentionally")
    path.mkdir(parents=True, exist_ok=True)


def run(spec_path: Path, output_dir: Path, *, overwrite: bool, verbose: bool = False) -> tuple[dict[str, Any], int]:
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Assembly specification must be a JSON object")
    problem = AssemblyProblem(raw)
    settings = problem.settings
    starts = int(settings.get("starts", 32))
    keep = int(settings.get("keep", 3))
    seed = int(settings.get("seed", 20260826))
    maxiter = int(settings.get("maxiter", 500))
    diversity_rmsd = float(settings.get("diversity_rmsd", 0.35))
    output_format = str(settings.get("output_format", "extxyz")).lower()
    if not math.isfinite(diversity_rmsd):
        raise ValueError("diversity_rmsd must be finite")
    if starts < 1 or keep < 1 or maxiter < 1 or diversity_rmsd < 0.0:
        raise ValueError("starts, keep, and maxiter must be positive; diversity_rmsd must be non-negative")
    if output_format not in {"extxyz", "vasp", "xyz"}:
        raise ValueError("settings.output_format must be extxyz, vasp, or xyz")
    _prepare_output_dir(output_dir, overwrite=overwrite)

    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    for start_index in range(starts):
        initial = problem.initial_variables(rng, first=start_index == 0)
        result = minimize(
            lambda variables: problem.evaluate(np.asarray(variables, dtype=float))[0],
            initial,
            method="L-BFGS-B",
            bounds=problem.bounds(),
            options={"maxiter": maxiter, "ftol": 1.0e-12, "gtol": 1.0e-8},
        )
        variables = np.asarray(result.x, dtype=float)
        score, metrics = problem.evaluate(variables, details=True)
        trials.append(
            {
                "start_index": start_index,
                "variables": variables,
                "score": score,
                "metrics": metrics,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "optimizer_iterations": int(getattr(result, "nit", 0) or 0),
                "feasible": problem.feasible(metrics),
            }
        )

    trials.sort(key=lambda row: (not bool(row["feasible"]), float(row["score"]), int(row["start_index"])))
    feasible_trials = [row for row in trials if row["feasible"]]
    selected: list[dict[str, Any]] = []
    for trial in feasible_trials:
        if all(
            _pose_rmsd(problem, trial["variables"], kept["variables"]) >= diversity_rmsd
            for kept in selected
        ):
            selected.append(trial)
        if len(selected) >= keep:
            break
    if not selected and feasible_trials:
        selected = [feasible_trials[0]]

    suffix = _output_suffix(output_format)
    context = problem.validation_context()
    candidate_rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(selected, start=1):
        structure_path = output_dir / f"candidate_{rank:03d}{suffix}"
        context_path = output_dir / f"candidate_{rank:03d}.context.json"
        atoms = problem.combined_atoms(trial["variables"], score=trial["score"], feasible=True)
        ase_write(str(structure_path), atoms, format=output_format)
        context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        candidate_rows.append(
            {
                "rank": rank,
                "structure_path": str(structure_path),
                "validation_context_path": str(context_path),
                "start_index": int(trial["start_index"]),
                "score": float(trial["score"]),
                "optimizer_success": bool(trial["optimizer_success"]),
                "optimizer_message": trial["optimizer_message"],
                "metrics": trial["metrics"],
            }
        )

    best_infeasible_path = ""
    if not selected:
        best = trials[0]
        path = output_dir / f"best_infeasible{suffix}"
        atoms = problem.combined_atoms(best["variables"], score=best["score"], feasible=False)
        ase_write(str(path), atoms, format=output_format)
        context_path = output_dir / "best_infeasible.context.json"
        context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        best_infeasible_path = str(path)

    report = {
        "status": "PASS" if selected else "FAIL",
        "spec_path": str(spec_path),
        "host_path": str(problem.host_path),
        "fragment_paths": {body.name: str(body.source_path) for body in problem.fragments},
        "starts": starts,
        "feasible_trial_count": len(feasible_trials),
        "selected_candidate_count": len(selected),
        "body_ranges": problem.body_ranges(),
        "expected_contacts": context["expected_contacts"],
        "candidates": candidate_rows,
        "best_infeasible_path": best_infeasible_path,
        "best_trial": {
            "start_index": int(trials[0]["start_index"]),
            "score": float(trials[0]["score"]),
            "feasible": bool(trials[0]["feasible"]),
            "optimizer_success": bool(trials[0]["optimizer_success"]),
            "optimizer_message": trials[0]["optimizer_message"],
            "metrics": trials[0]["metrics"],
        },
        "interpretation": (
            "PASS means the explicit rigid geometric constraints were met. "
            "Run the independent atomic-structure checker and visual review before physical relaxation."
        ),
    }
    report_path = output_dir / "assembly_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    print(f"report={report_path}")
    print(f"feasible_trials={len(feasible_trials)} selected_candidates={len(selected)}")
    if not selected:
        print(format_metrics_summary(report["best_trial"]["metrics"]))
    if best_infeasible_path:
        print(f"best_infeasible={best_infeasible_path}")
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report, 0 if selected else 2


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Describe violated terms without embedding full constraint/pair tables."""
    return " ".join(
        f"{key}={metrics[key]:.6g}"
        for key in (
            "max_penetration_A", "max_anchor_violation_A",
            "max_orientation_violation_deg", "max_region_violation",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-start rigid-fragment assembly with soft repulsion and explicit geometric constraints."
    )
    parser.add_argument("spec", type=Path, help="Assembly JSON specification.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for candidates and report.")
    parser.add_argument("--overwrite", action="store_true", help="Intentionally replace deterministic output names.")
    parser.add_argument("--verbose", action="store_true", help="Also print the complete assembly report; it is always saved in the output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, exit_code = run(args.spec, args.output_dir, overwrite=bool(args.overwrite), verbose=args.verbose)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
