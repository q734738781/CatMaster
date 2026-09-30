from __future__ import annotations

import csv
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read as ase_read, iread as ase_iread
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Oszicar, Outcar, Vasprun

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_RUN_MARKERS = ("vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR")
# Matplotlib maintains process-global state and is not safe when LangGraph runs
# several analysis tool calls concurrently. Keep numerical work parallel, but
# serialize the short figure-rendering sections.
_MATPLOTLIB_LOCK = threading.RLock()


@dataclass(frozen=True)
class TrajectoryFrames:
    frames: list[Atoms]
    positions_are_wrapped: bool
    coordinate_mode: str
    species_labels: list[str]
    frame_steps: list[int]


def _error(tool_name: str, message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    lines = [str(message).strip()]
    if data:
        for key in (
            "result_root_rel",
            "result_dir_rel",
            "output_dir_rel",
            "summary_json_rel",
            "csv_rel",
            "png_rel",
        ):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            lines.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(lines),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
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


def _is_vasp_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _RUN_MARKERS)


def _discover_vasp_runs(root: Path) -> list[Path]:
    if _is_vasp_run_dir(root):
        return [root]
    runs: list[Path] = []
    for path in root.rglob("*"):
        if _is_vasp_run_dir(path):
            runs.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(runs):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _oszicar_last_energy(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        return float(Oszicar(str(path)).final_energy)
    except Exception:
        return None


def _outcar_total_mag(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        return float(Outcar(str(path)).total_mag)
    except Exception:
        return None


_OUTCAR_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")


def _outcar_last_toten(path: Path) -> float | None:
    if not path.is_file():
        return None
    energy: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _OUTCAR_TOTEN_RE.search(line)
        if match is None:
            continue
        try:
            energy = float(match.group(1))
        except Exception:
            continue
    return energy


def _load_selective_dynamics_mask(run_dir: Path, natoms: int) -> np.ndarray | None:
    for candidate in (run_dir / "CONTCAR", run_dir / "POSCAR"):
        if not candidate.is_file():
            continue
        try:
            poscar = Poscar.from_file(str(candidate))
        except Exception:
            continue
        selective_dynamics = getattr(poscar, "selective_dynamics", None)
        if selective_dynamics is None:
            continue
        mask = np.asarray(selective_dynamics, dtype=bool)
        if mask.shape == (natoms, 3):
            return mask
    return None


def _parse_vasp_run(run_dir: Path) -> dict[str, Any]:
    vasprun_path = run_dir / "vasprun.xml"
    oszicar_path = run_dir / "OSZICAR"
    outcar_path = run_dir / "OUTCAR"
    contcar_path = run_dir / "CONTCAR"
    issues: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "result_dir_rel": workspace_relpath(run_dir),
        "state": "unknown",
        "electronic_converged": None,
        "ionic_converged": None,
        "final_energy_ev": None,
        "max_force_ev_per_a": None,
        "total_magnetization": None,
        "bandgap_ev": None,
        "final_structure_path": workspace_relpath(contcar_path) if contcar_path.exists() else None,
        "issues": issues,
    }

    vasprun = None
    if vasprun_path.is_file():
        try:
            vasprun = Vasprun(str(vasprun_path), parse_projected_eigen=False)
            summary["electronic_converged"] = bool(getattr(vasprun, "converged_electronic", False))
            summary["ionic_converged"] = bool(getattr(vasprun, "converged_ionic", False))
            summary["final_energy_ev"] = float(getattr(vasprun, "final_energy", np.nan))
            eig = getattr(vasprun, "eigenvalue_band_properties", None)
            if eig and len(eig) >= 1 and eig[0] is not None:
                summary["bandgap_ev"] = float(eig[0])
            if vasprun.ionic_steps:
                forces = vasprun.ionic_steps[-1].get("forces")
                if forces is not None:
                    forces_arr = np.asarray(forces, dtype=float)
                    selective_mask = _load_selective_dynamics_mask(run_dir, natoms=int(forces_arr.shape[0]))
                    if selective_mask is not None:
                        forces_arr = np.where(selective_mask, forces_arr, 0.0)
                    norms = np.linalg.norm(forces_arr, axis=1)
                    summary["max_force_ev_per_a"] = float(np.max(norms))
            if summary["electronic_converged"] and summary["ionic_converged"]:
                summary["state"] = "completed"
            else:
                summary["state"] = "incomplete"
        except Exception as exc:
            issues.append(
                {
                    "kind": "vasprun_parse_failed",
                    "evidence": str(exc),
                    "next_action_hint": "Inspect vasprun.xml and stdout/stderr for a truncated or failed run.",
                }
            )

    if summary["final_energy_ev"] is None:
        summary["final_energy_ev"] = _oszicar_last_energy(oszicar_path)
    if summary["total_magnetization"] is None:
        summary["total_magnetization"] = _outcar_total_mag(outcar_path)

    if summary["electronic_converged"] is False:
        issues.append(
            {
                "kind": "electronic_not_converged",
                "evidence": "Vasprun reports unconverged electronic steps.",
                "next_action_hint": "Consider ALGO/NELM/smearing changes before trusting energies or forces.",
            }
        )
    if summary["ionic_converged"] is False:
        issues.append(
            {
                "kind": "ionic_not_converged",
                "evidence": "Vasprun reports unconverged ionic relaxation.",
                "next_action_hint": "Increase NSW or relax in stages before analysis.",
            }
        )
    if summary["final_energy_ev"] is None:
        issues.append(
            {
                "kind": "missing_final_energy",
                "evidence": "Could not parse final energy from vasprun.xml or OSZICAR.",
                "next_action_hint": "Check whether the run finished and whether result files were downloaded.",
            }
        )
    if summary["state"] == "unknown" and not issues:
        summary["state"] = "partial"
    return summary


def _parse_neb_energies(result_dir: Path) -> list[dict[str, Any]]:
    image_dirs = [path for path in sorted(result_dir.iterdir()) if path.is_dir() and path.name.isdigit()]
    if not image_dirs:
        return []
    records: list[dict[str, Any]] = []
    for path in image_dirs:
        energy = _outcar_last_toten(path / "OUTCAR")
        if energy is None:
            continue
        records.append(
            {
                "image": int(path.name),
                "image_dir_rel": workspace_relpath(path),
                "energy_ev": float(energy),
            }
        )
    return records


def _scan_neb_images(result_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    image_dirs = sorted(
        [path for path in result_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if not image_dirs:
        return [], ["No numbered NEB image directories found."]

    image_numbers = [int(path.name) for path in image_dirs]
    issues: list[str] = []
    if image_numbers[0] != 0:
        issues.append(f"image numbering does not start at 00 (first={image_dirs[0].name})")
    expected = list(range(image_numbers[0], image_numbers[-1] + 1))
    missing = sorted(set(expected) - set(image_numbers))
    if missing:
        issues.append("missing image directories: " + ", ".join(f"{idx:02d}" for idx in missing))

    records: list[dict[str, Any]] = []
    for path in image_dirs:
        energy = _outcar_last_toten(path / "OUTCAR")
        if energy is None:
            issues.append(f"image {path.name}: no energy parsed from OUTCAR")
            continue
        text = (path / "OUTCAR").read_text(errors="replace")
        ionic_converged = "reached required accuracy" in text.lower()
        process_completed = "General timing and accounting informations" in text
        records.append(
            {
                "image": int(path.name),
                "image_dir_rel": workspace_relpath(path),
                "energy_ev": float(energy),
                "ionic_converged": ionic_converged,
                "process_completed": process_completed,
                "electronic_converged": None,
            }
        )
    return records, issues


def _neb_missing_endpoint_outcar_hint(issues: list[str]) -> str | None:
    missing_endpoint_images: list[str] = []
    for issue in issues:
        if issue.startswith("image 00: no energy parsed from OUTCAR"):
            missing_endpoint_images.append("00")
        elif issue.startswith("image 0: no energy parsed from OUTCAR"):
            missing_endpoint_images.append("00")
        elif issue.startswith("image "):
            image_label = issue.split(":", 1)[0].replace("image", "").strip()
            if image_label.isdigit() and issue.endswith("no energy parsed from OUTCAR"):
                missing_endpoint_images.append(image_label.zfill(2))
    if len(missing_endpoint_images) < 2:
        return None
    normalized = sorted(set(missing_endpoint_images), key=int)
    first = normalized[0]
    last = normalized[-1]
    if first != "00":
        return None
    return (
        f"hint=VASP NEB endpoint images do not produce their own OUTCAR energies. Copy the original relax OUTCAR files into "
        f"{first}/OUTCAR and {last}/OUTCAR under result_dir, "
        "then rerun analyze_vasp_neb_results."
    )


def _validate_requested_coordinate_semantics(
    trajectory: TrajectoryFrames,
    requested: Literal["auto", "wrapped", "unwrapped"],
) -> TrajectoryFrames:
    if requested == "auto":
        return trajectory
    expected_wrapped = requested == "wrapped"
    if trajectory.positions_are_wrapped != expected_wrapped:
        raise ValueError(
            f"coordinate_semantics={requested!r} conflicts with native trajectory columns "
            f"({trajectory.coordinate_mode})"
        )
    return trajectory


def _read_trajectory_frames(
    path: Path,
    *,
    coordinate_semantics: Literal["auto", "wrapped", "unwrapped"],
) -> TrajectoryFrames:
    if path.suffix.lower() == ".lammpstrj":
        return _validate_requested_coordinate_semantics(_read_lammps_dump(path), coordinate_semantics)
    if path.name == "XDATCAR":
        frames = list(ase_read(str(path), index=":", format="vasp-xdatcar"))
        return _validate_requested_coordinate_semantics(
            TrajectoryFrames(
                frames=frames,
                positions_are_wrapped=True,
                coordinate_mode="wrapped_fractional",
                species_labels=frames[0].get_chemical_symbols() if frames else [],
                frame_steps=list(range(len(frames))),
            ),
            coordinate_semantics,
        )
    frames = list(ase_read(str(path), index=":"))
    labels = frames[0].get_chemical_symbols() if frames else []
    if any(frame.get_chemical_symbols() != labels for frame in frames[1:]):
        raise ValueError("Trajectory atom ordering/species changed between frames")
    periodic = any(any(bool(value) for value in frame.get_pbc()) for frame in frames)
    if periodic and coordinate_semantics == "auto":
        raise ValueError(
            "Periodic generic trajectories do not encode whether coordinates are wrapped; "
            "pass coordinate_semantics='wrapped' or 'unwrapped' explicitly"
        )
    if not periodic and coordinate_semantics == "wrapped":
        raise ValueError("coordinate_semantics='wrapped' is not meaningful for a nonperiodic trajectory")
    wrapped = periodic and coordinate_semantics == "wrapped"
    return TrajectoryFrames(
        frames=frames,
        positions_are_wrapped=wrapped,
        coordinate_mode="wrapped_cartesian" if wrapped else "unwrapped_cartesian",
        species_labels=labels,
        frame_steps=list(range(len(frames))),
    )


def _iter_lammps_dump(path: Path):
    previous_step = None
    step_interval = None
    labels_reference: list[str] | None = None
    ids_reference: list[int] | None = None
    pbc_reference: list[bool] | None = None
    coordinate_mode = ""
    positions_are_wrapped = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            lines = [first] + [handle.readline() for _ in range(8)]
            if any(not line for line in lines):
                raise ValueError("Truncated LAMMPS frame header")
            natoms = int(lines[3].strip())
            lines += [handle.readline() for _ in range(natoms)]
            if any(not line for line in lines):
                raise ValueError("Truncated LAMMPS atom rows")
            cursor = 0
            if lines[cursor].strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Malformed LAMMPS dump near line {cursor + 1}: expected ITEM: TIMESTEP")
            step = int(lines[cursor + 1].strip())
            if lines[cursor + 2].strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Malformed LAMMPS dump: missing NUMBER OF ATOMS")
            natoms = int(lines[cursor + 3].strip())
            bounds_header = lines[cursor + 4].split()
            if bounds_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Malformed LAMMPS dump: missing BOX BOUNDS")
            bounds = [[float(value) for value in lines[cursor + offset].split()] for offset in (5, 6, 7)]
            tilt = len(bounds[0]) >= 3 or any(token in {"xy", "xz", "yz"} for token in bounds_header)
            if tilt:
                xlo_bound, xhi_bound, xy = bounds[0][:3]
                ylo_bound, yhi_bound, xz = bounds[1][:3]
                zlo_bound, zhi_bound, yz = bounds[2][:3]
                xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
                xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
                ylo = ylo_bound - min(0.0, yz)
                yhi = yhi_bound - max(0.0, yz)
            else:
                xlo, xhi = bounds[0][:2]
                ylo, yhi = bounds[1][:2]
                zlo_bound, zhi_bound = bounds[2][:2]
                xy = xz = yz = 0.0
            cell = np.asarray(
                [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi_bound - zlo_bound]],
                dtype=float,
            )
            origin = np.asarray([xlo, ylo, zlo_bound], dtype=float)
            boundary_tokens = bounds_header[-3:]
            pbc = [token.lower().startswith("p") for token in boundary_tokens] if len(boundary_tokens) == 3 else [True] * 3
            if pbc_reference is None:
                pbc_reference = pbc
            elif pbc != pbc_reference:
                raise ValueError("LAMMPS dump boundary conditions changed between frames")
            atom_header = lines[cursor + 8].split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Malformed LAMMPS dump: missing ATOMS header")
            columns = atom_header[2:]
            column_index = {name: index for index, name in enumerate(columns)}
            if "id" not in column_index:
                raise ValueError("LAMMPS dump analysis requires an id column")
            rows = [lines[cursor + 9 + index].split() for index in range(natoms)]
            rows.sort(key=lambda row: int(row[column_index["id"]]))
            ids = [int(row[column_index["id"]]) for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("LAMMPS dump contains duplicate atom IDs")
            if all(name in column_index for name in ("xu", "yu", "zu")):
                positions = np.asarray([[float(row[column_index[name]]) for name in ("xu", "yu", "zu")] for row in rows]) - origin
                mode = "unwrapped_cartesian"
                wrapped = False
            elif all(name in column_index for name in ("xsu", "ysu", "zsu")):
                fractions = np.asarray([[float(row[column_index[name]]) for name in ("xsu", "ysu", "zsu")] for row in rows])
                positions = fractions @ cell
                mode = "unwrapped_scaled"
                wrapped = False
            elif all(name in column_index for name in ("x", "y", "z")):
                positions = np.asarray([[float(row[column_index[name]]) for name in ("x", "y", "z")] for row in rows]) - origin
                mode = "wrapped_cartesian"
                wrapped = True
            elif all(name in column_index for name in ("xs", "ys", "zs")):
                fractions = np.asarray([[float(row[column_index[name]]) for name in ("xs", "ys", "zs")] for row in rows])
                positions = fractions @ cell
                mode = "wrapped_scaled"
                wrapped = True
            else:
                raise ValueError("LAMMPS dump requires x/y/z, xs/ys/zs, xu/yu/zu, or xsu/ysu/zsu coordinates")
            if wrapped and all(name in column_index for name in ("ix", "iy", "iz")):
                images = np.asarray([[int(row[column_index[name]]) for name in ("ix", "iy", "iz")] for row in rows])
                positions = positions + images @ cell
                mode += "+image_flags"
                wrapped = False
            if "element" in column_index:
                labels = [row[column_index["element"]] for row in rows]
                symbols = labels
            else:
                labels = [f"type:{row[column_index['type']]}" if "type" in column_index else "atom" for row in rows]
                symbols = ["X"] * natoms
            if labels_reference is None:
                labels_reference = labels
                ids_reference = ids
            elif ids != ids_reference:
                raise ValueError("LAMMPS dump atom IDs are not stable across frames")
            elif labels != labels_reference:
                raise ValueError("LAMMPS dump atom IDs/types are not stable across frames")
            if coordinate_mode and (mode != coordinate_mode or wrapped != positions_are_wrapped):
                raise ValueError("LAMMPS dump coordinate columns changed between frames")
            if previous_step is not None:
                delta = step - previous_step
                if step_interval is not None and delta != step_interval:
                    raise ValueError("LAMMPS dump has nonuniform frame-step spacing")
                step_interval = delta
            previous_step = step
            coordinate_mode = mode
            positions_are_wrapped = wrapped
            yield TrajectoryFrames(frames=[Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)],
                                   positions_are_wrapped=wrapped, coordinate_mode=mode,
                                   species_labels=labels, frame_steps=[step])


def _read_lammps_dump(path: Path) -> TrajectoryFrames:
    frames, steps = [], []
    last = TrajectoryFrames([], False, "", [], [])
    for item in _iter_lammps_dump(path):
        frames.extend(item.frames)
        steps.extend(item.frame_steps)
        last = item
    return TrajectoryFrames(frames, last.positions_are_wrapped, last.coordinate_mode, last.species_labels, steps)


def _selected_trajectory(path: Path, params) -> TrajectoryFrames:
    if path.suffix.lower() == ".lammpstrj":
        source = _iter_lammps_dump(path)
    else:
        def generic():
            native = path.name == "XDATCAR"
            for index, atoms in enumerate(ase_iread(str(path), index=":", format="vasp-xdatcar" if native else None)):
                periodic = bool(np.any(atoms.pbc))
                if periodic and params.coordinate_semantics == "auto" and not native:
                    raise ValueError("Periodic generic trajectories require wrapped or unwrapped coordinate_semantics explicitly")
                wrapped = periodic and (native or params.coordinate_semantics == "wrapped")
                if not periodic and params.coordinate_semantics == "wrapped":
                    raise ValueError("wrapped coordinates require periodic boundaries")
                yield TrajectoryFrames([atoms], wrapped, "wrapped_fractional" if native else "wrapped_cartesian" if wrapped else "unwrapped_cartesian", atoms.get_chemical_symbols(), [index])
        source = generic()
    frames, steps = [], []
    previous_fraction = cumulative = None
    labels = None
    mode = ""
    unwrap = False
    for index, item in enumerate(source):
        if params.frame_stop and index >= params.frame_stop:
            break
        item = _validate_requested_coordinate_semantics(item, params.coordinate_semantics)
        atoms = item.frames[0]
        if labels is not None and labels != item.species_labels:
            raise ValueError("Trajectory atom ordering/species changed between frames")
        labels = item.species_labels
        mode = item.coordinate_mode
        unwrap = item.positions_are_wrapped
        if unwrap and params.compute_msd:
            fraction = atoms.positions @ np.linalg.inv(atoms.cell.array)
            if previous_fraction is None:
                cumulative = fraction.copy()
            else:
                delta = fraction - previous_fraction
                cumulative += delta - np.round(delta) * np.asarray(atoms.pbc)[None, :]
            previous_fraction = fraction
            atoms = atoms.copy()
            atoms.positions = cumulative @ atoms.cell.array
        if index >= params.frame_start and (index - params.frame_start) % params.frame_stride == 0:
            frames.append(atoms)
            steps.append(item.frame_steps[0])
    return TrajectoryFrames(frames, False if params.compute_msd and unwrap else unwrap,
                            mode + ("_unwrapped_before_sampling" if params.compute_msd and unwrap else ""), labels or [], steps)


def _unwrap_positions(frames: list[Atoms]) -> np.ndarray:
    positions = [atoms.get_positions() for atoms in frames]
    cells = [np.asarray(atoms.cell.array, dtype=float) for atoms in frames]
    frac = []
    for pos, cell in zip(positions, cells):
        inv = np.linalg.inv(cell)
        frac.append(np.dot(pos, inv))
    frac = np.asarray(frac, dtype=float)
    unwrapped = np.zeros_like(frac)
    unwrapped[0] = frac[0]
    for idx in range(1, len(frac)):
        delta = frac[idx] - frac[idx - 1]
        periodic_axes = np.asarray(frames[idx].get_pbc(), dtype=float)
        delta -= np.round(delta) * periodic_axes[None, :]
        unwrapped[idx] = unwrapped[idx - 1] + delta
    cart = np.zeros_like(unwrapped)
    for idx, cell in enumerate(cells):
        cart[idx] = np.dot(unwrapped[idx], cell)
    return cart


def _axes_for_dimension(dimension: str) -> tuple[int, ...]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    return tuple(axis_map[label] for label in dimension)


def _mean_square_displacement(
    frames: list[Atoms],
    species_labels: list[str],
    species: str,
    *,
    axes: tuple[int, ...],
    positions_are_wrapped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        return np.array([]), np.array([])
    symbols = np.array(species_labels)
    if species:
        mask = symbols == species
        if not np.any(mask):
            raise ValueError(f"No atoms with species={species!r} found in trajectory.")
    else:
        mask = np.ones(len(symbols), dtype=bool)
    positions = _unwrap_positions(frames) if positions_are_wrapped else np.asarray(
        [atoms.get_positions() for atoms in frames],
        dtype=float,
    )
    reference = positions[0, mask, :]
    delta = positions[:, mask, :] - reference[None, :, :]
    msd = np.mean(np.sum(delta[:, :, axes] ** 2, axis=2), axis=1)
    return np.arange(len(frames), dtype=float), msd


def _rdf(
    frames: list[Atoms],
    *,
    species_labels: list[str],
    species: str = "",
    bins: int = 150,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not frames:
        return np.array([]), np.array([]), "not_calculated"
    if any(not all(bool(value) for value in atoms.get_pbc()) for atoms in frames):
        return np.array([]), np.array([]), "not_applicable"
    if r_max is None:
        heights: list[float] = []
        for atoms in frames:
            inverse = np.linalg.inv(np.asarray(atoms.cell.array, dtype=float))
            heights.extend(float(1.0 / norm) for norm in np.linalg.norm(inverse, axis=0) if norm > 0)
        r_max = 0.5 * min(heights)
    edges = np.linspace(0.0, float(r_max), bins + 1)
    observed = np.zeros(bins, dtype=float)
    expected = np.zeros(bins, dtype=float)
    labels = np.asarray(species_labels)
    mask = labels == species if species else np.ones(len(labels), dtype=bool)
    for atoms in frames:
        if species:
            if not np.any(mask):
                continue
            selected = atoms[mask]
        else:
            selected = atoms
        if len(selected) < 2:
            continue
        for atom_index in range(len(selected) - 1):
            distances = selected.get_distances(atom_index, range(atom_index + 1, len(selected)), mic=True)
            observed += np.histogram(distances[(distances > 1e-8) & (distances <= r_max)], bins=edges)[0]
        shell_volume = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
        pair_count = len(selected) * (len(selected) - 1) / 2.0
        expected += pair_count * shell_volume / float(atoms.get_volume())
    if not np.any(expected):
        return np.array([]), np.array([]), "not_calculated"
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.divide(observed, expected, out=np.zeros_like(observed), where=expected > 0), "calculated"


def _parse_oszicar_series(path: Path) -> tuple[list[float], list[float]]:
    temps: list[float] = []
    energies: list[float] = []
    if not path.is_file():
        return temps, energies
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "T=" not in line:
            continue
        tokens = line.replace("=", " = ").split()
        for idx, token in enumerate(tokens):
            if token == "T" and idx + 2 < len(tokens):
                try:
                    temps.append(float(tokens[idx + 2]))
                except Exception:
                    pass
            if token in {"E0", "F"} and idx + 2 < len(tokens):
                try:
                    energies.append(float(tokens[idx + 2]))
                except Exception:
                    pass
    return temps, energies


class AnalyzeVaspResultsInput(BaseModel):
    """[vasp/analysis] Summarize one VASP result directory or a batch of result directories into JSON and CSV artifacts. final_energy_ev uses VASP e_0_energy."""

    result_root: str = Field(..., description="Single VASP result directory or batch root directory.")
    output_dir: Optional[str] = Field(
        None,
        description="Directory to write summary artifacts. Defaults to <result_root>_analysis next to the input.",
    )


class AnalyzeVaspNebResultsInput(BaseModel):
    """[vasp/analysis] Summarize current VASP NEB image energies and separately report convergence; current barriers may be provisional. Read a result directory into barrier, profile CSV, and profile plot artifacts."""

    result_dir: str = Field(..., description="VASP NEB result directory containing image folders like 00/01/02/...")
    output_dir: Optional[str] = Field(
        None,
        description="Directory to write summary artifacts. Defaults to <result_dir>_analysis next to the input.",
    )


class AnalyzeTrajectoryInput(BaseModel):
    """[md/analysis] Analyze an MD trajectory with MSD, diffusion-fit, RDF, and temperature/energy summary outputs."""

    model_config = ConfigDict(extra="forbid")

    compute_msd: bool = Field(True, description="Compute MSD and a diffusion fit; false skips that analysis entirely.")
    compute_rdf: bool = Field(True, description="Compute the normalized 3D RDF; false skips it.")
    make_plots: bool = Field(True, description="Write plots alongside numeric outputs; false writes only JSON/CSV.")
    frame_start: int = Field(0, ge=0, description="First source frame included in analysis, zero-based.")
    frame_stop: int = Field(0, ge=0, description="Exclusive source frame end; 0 reads to the end.")
    frame_stride: int = Field(1, ge=1, description="Keep every Nth frame after frame_start. Wrapped trajectories are unwrapped through every source frame before thinning for MSD. Selected frames remain in memory.")
    path: str = Field(..., description="Trajectory file path or a result directory containing XDATCAR/trajectory outputs.")
    output_dir: str = Field(
        "",
        description="Directory to write summary artifacts; leave empty to create <path>_analysis next to the input.",
    )
    frame_interval_fs: float = Field(
        ...,
        gt=0.0,
        description="Physical time between stored trajectory frames in femtoseconds, including the engine dump stride.",
    )
    coordinate_semantics: Literal["auto", "wrapped", "unwrapped"] = Field(
        "auto",
        description=(
            "Coordinate wrapping for generic trajectory formats. auto reads native LAMMPS dump columns and XDATCAR; "
            "periodic XYZ/EXTXYZ/ASE trajectories require wrapped or unwrapped explicitly."
        ),
    )
    species: str = Field(
        "",
        description="Optional species filter for MSD and diffusion, e.g. 'Li'. Defaults to all atoms.",
    )
    rdf_species: str = Field(
        "",
        description="Optional species filter for RDF. Defaults to species if species is set, otherwise all atoms.",
    )
    diffusion_dimension: Literal["x", "y", "z", "xy", "xz", "yz", "xyz"] = Field(
        "xyz",
        description="Axis set used for the Einstein diffusion fit. Use xy for surface diffusion, x/y/z for 1D projections, or xyz for isotropic 3D diffusion.",
    )
    fit_start_frame: int = Field(
        -1,
        ge=-1,
        description="Required when compute_msd=true: explicitly choose the first selected frame for fitting. Omit only when compute_msd=false; -1 means no window was supplied.",
    )
    fit_end_frame: int = Field(
        0,
        ge=0,
        description="Exclusive end frame for the MSD fit; pass 0 to use the final stored frame.",
    )
    rdf_bins: int = Field(150, ge=2, description="Number of radial bins for the normalized three-dimensional g(r).")
    rdf_max_angstrom: float = Field(
        0.0,
        ge=0.0,
        description="Maximum RDF radius in angstrom; pass 0 to use half the shortest periodic cell height.",
    )

    @model_validator(mode="after")
    def _require_fit_window(self):
        if self.compute_msd and self.fit_start_frame < 0:
            raise ValueError("Set fit_start_frame explicitly when compute_msd=true")
        return self


def analyze_vasp_results(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[vasp/analysis] Analyze one VASP result directory or a batch of VASP result directories. final_energy_ev uses VASP e_0_energy."""
    try:
        params = AnalyzeVaspResultsInput(**payload)
        result_root = resolve_workspace_path(params.result_root, must_exist=True)
        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
            result_root.parent / f"{result_root.name}_analysis"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        runs = _discover_vasp_runs(result_root)
        if not runs:
            _error(
                "analyze_vasp_results",
                "No VASP result directories found under result_root.",
                data={"result_root_rel": workspace_relpath(result_root)},
                error_code="no_vasp_runs",
            )

        records = [_parse_vasp_run(run) for run in runs]
        summary_json = output_dir / "vasp_results_summary.json"
        csv_path = output_dir / "vasp_results_summary.csv"
        payload_json = {
            "result_root_rel": workspace_relpath(result_root),
            "records": records,
            "runs_analyzed": len(records),
        }
        _write_json(summary_json, payload_json)
        _write_csv(csv_path, records)
        failed = sum(1 for record in records if record.get("state") != "completed")
        data = {
            "result_root_rel": workspace_relpath(result_root),
            "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_json),
            "csv_rel": workspace_relpath(csv_path),
            "runs_analyzed": len(records),
            "failed_or_incomplete": failed,
        }
        content = (
            "analyze_vasp_results completed.\n"
            f"result_root_rel={data['result_root_rel']} runs_analyzed={data['runs_analyzed']} "
            f"failed_or_incomplete={failed} summary_json_rel={data['summary_json_rel']}"
        )
        return content, {"tool_name": "analyze_vasp_results", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error("analyze_vasp_results", f"analyze_vasp_results failed: {exc}", error_code="analyze_vasp_results_failed")


def analyze_vasp_neb_results(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[vasp/analysis] Analyze a completed VASP NEB result directory and export barrier/profile artifacts."""
    try:
        params = AnalyzeVaspNebResultsInput(**payload)
        result_dir = resolve_workspace_path(params.result_dir, must_exist=True)
        if not result_dir.is_dir():
            _error(
                "analyze_vasp_neb_results",
                f"result_dir is not a directory: {result_dir}",
                error_code="invalid_result_dir",
            )
        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
            result_dir.parent / f"{result_dir.name}_analysis"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        records, issues = _scan_neb_images(result_dir)
        if issues:
            hint = _neb_missing_endpoint_outcar_hint(issues)
            message = "Incomplete NEB image energies; refusing to report a barrier.\nissues=" + "; ".join(issues)
            if hint:
                message += "\n" + hint
            _error(
                "analyze_vasp_neb_results",
                message,
                data={"result_dir_rel": workspace_relpath(result_dir)},
                error_code="incomplete_neb_profile",
            )
        if len(records) < 2:
            _error(
                "analyze_vasp_neb_results",
                "Could not parse enough NEB image energies.",
                data={"result_dir_rel": workspace_relpath(result_dir)},
                error_code="missing_neb_energies",
            )
        energies = np.array([record["energy_ev"] for record in records], dtype=float)
        relative = energies - energies.min()
        initial_rel = float(energies[0] - energies.min())
        final_rel = float(energies[-1] - energies.min())
        ts_index = int(np.argmax(energies))
        forward_barrier = float(energies[ts_index] - energies[0])
        reverse_barrier = float(energies[ts_index] - energies[-1])
        for record, rel in zip(records, relative):
            record["relative_energy_ev"] = float(rel)

        csv_path = output_dir / "neb_profile.csv"
        png_path = output_dir / "neb_profile.png"
        summary_json = output_dir / "neb_summary.json"
        _write_csv(csv_path, records)
        with _MATPLOTLIB_LOCK:
            figure = plt.figure(figsize=(6.5, 4.0))
            try:
                plt.plot([record["image"] for record in records], relative, marker="o")
                plt.xlabel("Image")
                plt.ylabel("Relative energy (eV)")
                plt.tight_layout()
                plt.savefig(png_path, dpi=180)
            finally:
                plt.close(figure)
        convergence_confirmed = bool(records[1:-1]) and all(record.get("ionic_converged") for record in records[1:-1])
        summary = {
            "profile_state": "ionic_convergence_confirmed" if convergence_confirmed else "provisional",
            "result_dir_rel": workspace_relpath(result_dir),
            "images_analyzed": len(records),
            "initial_relative_energy_ev": initial_rel,
            "final_relative_energy_ev": final_rel,
            "forward_barrier_ev": forward_barrier,
            "reverse_barrier_ev": reverse_barrier,
            "ts_image": int(records[ts_index]["image"]),
            "csv_rel": workspace_relpath(csv_path),
            "png_rel": workspace_relpath(png_path),
            "records": records,
        }
        _write_json(summary_json, summary)
        data = {
            "result_dir_rel": workspace_relpath(result_dir),
            "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_json),
            "csv_rel": workspace_relpath(csv_path),
            "png_rel": workspace_relpath(png_path),
            "profile_state": summary["profile_state"],
            "ts_image": summary["ts_image"],
            "forward_barrier_ev": forward_barrier,
            "reverse_barrier_ev": reverse_barrier,
        }
        content = (
            f"analyze_vasp_neb_results parsed; profile_state={summary['profile_state']}.\n"
            f"result_dir_rel={data['result_dir_rel']} ts_image={data['ts_image']} "
            f"forward_barrier_ev={forward_barrier:.6f}\n"
            f"summary_json_rel={data['summary_json_rel']} csv_rel={data['csv_rel']} png_rel={data['png_rel']}"
        )
        return content, {"tool_name": "analyze_vasp_neb_results", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(
            "analyze_vasp_neb_results",
            f"analyze_vasp_neb_results failed: {exc}",
            error_code="analyze_vasp_neb_results_failed",
        )


def analyze_trajectory(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[md/analysis] Analyze a trajectory or MD result directory with MSD, diffusion-fit, RDF, and time-series artifacts."""
    try:
        params = AnalyzeTrajectoryInput(**payload)
        source_path = resolve_workspace_path(params.path, must_exist=True)
        if source_path.is_dir():
            candidates = [
                source_path / "md.traj",
                source_path / "XDATCAR",
                source_path / "trajectory.traj",
                source_path / "opt.traj",
                source_path / "trajectory.lammpstrj",
            ]
            discovered = [path for path in candidates if path.exists()]
            discovered.extend(
                path
                for path in sorted(source_path.iterdir())
                if path.is_file()
                and path.suffix.lower() in {".traj", ".xyz", ".extxyz", ".lammpstrj"}
                and path not in discovered
            )
            if not discovered:
                _error(
                    "analyze_trajectory",
                    "No recognizable trajectory file found in the provided directory.",
                    data={"result_dir_rel": workspace_relpath(source_path)},
                    error_code="missing_trajectory",
                )
            if len(discovered) > 1:
                _error(
                    "analyze_trajectory",
                    "Multiple trajectory files found; pass the intended file path explicitly.",
                    data={"result_dir_rel": workspace_relpath(source_path), "candidates": [workspace_relpath(path) for path in discovered]},
                    error_code="ambiguous_trajectory",
                )
            trajectory_path = discovered[0]
            result_dir = source_path
        else:
            trajectory_path = source_path
            result_dir = source_path.parent
        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir.strip() else (
            result_dir.parent / f"{result_dir.name}_analysis"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if params.frame_stop and params.frame_stop <= params.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        if not (params.compute_msd or params.compute_rdf):
            raise ValueError("Enable at least one of compute_msd/compute_rdf")
        trajectory = _selected_trajectory(trajectory_path, params)
        frames = trajectory.frames
        if len(frames) < (2 if params.compute_msd else 1):
            _error(
                "analyze_trajectory",
                "Trajectory must contain at least two frames.",
                data={"result_dir_rel": workspace_relpath(result_dir)},
                error_code="too_few_frames",
            )
        indices, msd, time_ps = np.array([]), np.array([]), np.array([])
        slope = intercept = diffusion = None
        fit_end = 0
        if params.compute_msd:
            axes = _axes_for_dimension(params.diffusion_dimension)
            indices, msd = _mean_square_displacement(
                frames,
                trajectory.species_labels,
                params.species,
                axes=axes,
                positions_are_wrapped=trajectory.positions_are_wrapped,
            )
            time_ps = indices * params.frame_interval_fs * params.frame_stride / 1000.0
            fit_end = params.fit_end_frame or len(msd)
            if params.fit_start_frame >= fit_end or fit_end > len(msd) or fit_end - params.fit_start_frame < 2:
                raise ValueError("The explicit MSD fit interval must contain at least two stored frames within the trajectory")
            fit_x = time_ps[params.fit_start_frame:fit_end]
            fit_y = msd[params.fit_start_frame:fit_end]
            slope = None
            intercept = float(msd[0])
            diffusion = None
            dim = len(axes)
            if len(fit_x) >= 2 and np.ptp(fit_x) > 0:
                slope, intercept = np.polyfit(fit_x, fit_y, deg=1)
                diffusion = float(slope / (2.0 * dim))
        rdf_species = params.rdf_species or params.species
        rdf_r, rdf_g, rdf_state = np.array([]), np.array([]), "not_requested"
        if params.compute_rdf:
            rdf_r, rdf_g, rdf_state = _rdf(
                frames,
                species_labels=trajectory.species_labels,
                species=rdf_species,
                bins=params.rdf_bins,
                r_max=params.rdf_max_angstrom or None,
            )
        temps, energies = _parse_oszicar_series(result_dir / "OSZICAR")

        msd_csv = output_dir / "trajectory_msd.csv"
        rdf_csv = output_dir / "trajectory_rdf.csv"
        msd_png = output_dir / "trajectory_msd.png"
        thermo_png = output_dir / "trajectory_thermo.png"
        summary_json = output_dir / "trajectory_summary.json"
        if params.compute_msd:
            _write_csv(
                msd_csv,
                [
                    {
                        "frame": int(frame),
                        "time_ps": float(t),
                        "msd_a2": float(val),
                    }
                    for frame, t, val in zip(indices, time_ps, msd)
                ],
            )
        if params.compute_rdf:
            _write_csv(
                rdf_csv,
                [
                    {
                        "r_a": float(r),
                        "g_r": float(g),
                    }
                    for r, g in zip(rdf_r, rdf_g)
                ],
            )

        with _MATPLOTLIB_LOCK:
            if params.make_plots and params.compute_msd:
                figure = plt.figure(figsize=(6.5, 4.0))
                try:
                    plt.plot(time_ps, msd, label="MSD")
                    if diffusion is not None and slope is not None:
                        plt.plot(time_ps, slope * time_ps + intercept, linestyle="--", label="linear fit")
                    plt.xlabel("Time (ps)")
                    plt.ylabel("MSD ($\\AA^2$)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(msd_png, dpi=180)
                finally:
                    plt.close(figure)

            if params.make_plots and (temps or energies):
                figure = plt.figure(figsize=(6.5, 4.0))
                try:
                    if temps:
                        plt.plot(np.arange(len(temps)), temps, label="Temperature (K)")
                    if energies:
                        plt.plot(np.arange(len(energies)), energies, label="Energy")
                    plt.xlabel("Step")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(thermo_png, dpi=180)
                finally:
                    plt.close(figure)

        summary = {
            "trajectory_rel": workspace_relpath(trajectory_path),
            "coordinate_mode": trajectory.coordinate_mode,
            "positions_are_wrapped": trajectory.positions_are_wrapped,
            "result_dir_rel": workspace_relpath(result_dir),
            "nframes": len(frames),
            "natoms": len(frames[0]),
            "species_filter": params.species,
            "rdf_species": rdf_species,
            "diffusion_dimension": params.diffusion_dimension,
            "frame_interval_fs": params.frame_interval_fs,
            "fit_start_frame": params.fit_start_frame,
            "fit_end_frame": fit_end,
            "source_step_start": trajectory.frame_steps[0] if trajectory.frame_steps else None,
            "source_step_end": trajectory.frame_steps[-1] if trajectory.frame_steps else None,
            "source_step_interval": (
                trajectory.frame_steps[1] - trajectory.frame_steps[0]
                if len(trajectory.frame_steps) >= 2
                else None
            ),
            "msd_fit_slope_a2_per_ps": float(slope) if slope is not None else None,
            "final_msd_a2": float(msd[-1]) if len(msd) else None,
            "frame_start": params.frame_start, "frame_stop": params.frame_stop, "frame_stride": params.frame_stride,
            "diffusion_coefficient_a2_per_ps": diffusion,
            "rdf_state": rdf_state,
            "rdf_bins": params.rdf_bins,
            "rdf_max_angstrom": float(rdf_r[-1]) if len(rdf_r) else None,
            "msd_csv_rel": (workspace_relpath(msd_csv) if params.compute_msd else ""),
            "rdf_csv_rel": (workspace_relpath(rdf_csv) if params.compute_rdf else ""),
            "msd_png_rel": (workspace_relpath(msd_png) if params.make_plots and params.compute_msd else ""),
            "thermo_png_rel": workspace_relpath(thermo_png) if params.make_plots and (temps or energies) else None,
            "temperature_mean_K": float(np.mean(temps)) if temps else None,
            "energy_mean": float(np.mean(energies)) if energies else None,
        }
        _write_json(summary_json, summary)
        data = {
            "result_dir_rel": workspace_relpath(result_dir),
            "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_json),
            "csv_rel": (workspace_relpath(msd_csv) if params.compute_msd else ""),
            "png_rel": (workspace_relpath(msd_png) if params.make_plots and params.compute_msd else ""),
            "nframes": len(frames),
            "coordinate_mode": trajectory.coordinate_mode,
            "positions_are_wrapped": trajectory.positions_are_wrapped,
            "diffusion_coefficient_a2_per_ps": diffusion,
        }
        content = (
            "analyze_trajectory completed.\n"
            f"result_dir_rel={data['result_dir_rel']} nframes={data['nframes']}\n"
            f"summary_json_rel={data['summary_json_rel']} csv_rel={data['csv_rel']} png_rel={data['png_rel']}"
        )
        return content, {"tool_name": "analyze_trajectory", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error("analyze_trajectory", f"analyze_trajectory failed: {exc}", error_code="analyze_trajectory_failed")


__all__ = [
    "AnalyzeVaspNebResultsInput",
    "AnalyzeTrajectoryInput",
    "analyze_vasp_neb_results",
    "analyze_trajectory",
]
