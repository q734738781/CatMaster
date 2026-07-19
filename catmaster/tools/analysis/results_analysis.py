from __future__ import annotations

import csv
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from pydantic import BaseModel, Field
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
    coordinate_mode: Literal["wrapped_fractional", "unwrapped_cartesian"]


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
        records.append(
            {
                "image": int(path.name),
                "image_dir_rel": workspace_relpath(path),
                "energy_ev": float(energy),
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


def _read_trajectory_frames(path: Path) -> TrajectoryFrames:
    if path.name == "XDATCAR":
        return TrajectoryFrames(
            frames=list(ase_read(str(path), index=":", format="vasp-xdatcar")),
            positions_are_wrapped=True,
            coordinate_mode="wrapped_fractional",
        )
    return TrajectoryFrames(
        frames=list(ase_read(str(path), index=":")),
        positions_are_wrapped=False,
        coordinate_mode="unwrapped_cartesian",
    )


def _unwrap_positions(frames: list[Atoms]) -> np.ndarray:
    positions = [atoms.get_positions() for atoms in frames]
    cells = [np.asarray(atoms.cell.array, dtype=float) for atoms in frames]
    frac = []
    for pos, cell in zip(positions, cells):
        inv = np.linalg.inv(cell.T)
        frac.append(np.dot(pos, inv))
    frac = np.asarray(frac, dtype=float)
    unwrapped = np.zeros_like(frac)
    unwrapped[0] = frac[0]
    for idx in range(1, len(frac)):
        delta = frac[idx] - frac[idx - 1]
        delta -= np.round(delta)
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
    species: Optional[str],
    *,
    axes: tuple[int, ...],
    positions_are_wrapped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        return np.array([]), np.array([])
    symbols = np.array(frames[0].get_chemical_symbols())
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
    species: Optional[str] = None,
    bins: int = 150,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        return np.array([]), np.array([])
    sample_frames = frames[: min(len(frames), 25)]
    if r_max is None:
        cell_lengths = [min(atoms.cell.lengths()) for atoms in sample_frames if atoms.cell is not None]
        r_max = 0.5 * min(cell_lengths) if cell_lengths else 6.0
    all_distances: list[np.ndarray] = []
    for atoms in sample_frames:
        if species:
            mask = np.array(atoms.get_chemical_symbols()) == species
            if not np.any(mask):
                continue
            selected = atoms[mask]
        else:
            selected = atoms
        if len(selected) < 2:
            continue
        distances = selected.get_all_distances(mic=True)
        tri = distances[np.triu_indices(len(selected), k=1)]
        tri = tri[(tri > 1e-8) & (tri <= r_max)]
        if tri.size:
            all_distances.append(tri)
    if not all_distances:
        return np.array([]), np.array([])
    values = np.concatenate(all_distances)
    hist, edges = np.histogram(values, bins=bins, range=(0.0, r_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


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
    """[vasp/analysis] Summarize a completed VASP NEB result directory into barrier, profile CSV, and profile plot artifacts."""

    result_dir: str = Field(..., description="VASP NEB result directory containing image folders like 00/01/02/...")
    output_dir: Optional[str] = Field(
        None,
        description="Directory to write summary artifacts. Defaults to <result_dir>_analysis next to the input.",
    )


class AnalyzeTrajectoryInput(BaseModel):
    """[md/analysis] Analyze an MD trajectory with MSD, diffusion-fit, RDF, and temperature/energy summary outputs."""

    path: str = Field(..., description="Trajectory file path or a result directory containing XDATCAR/trajectory outputs.")
    output_dir: Optional[str] = Field(
        None,
        description="Directory to write summary artifacts. Defaults to <path>_analysis next to the input.",
    )
    timestep_fs: float = Field(1.0, gt=0.0, description="Trajectory timestep in femtoseconds used for MSD and diffusion fits.")
    species: Optional[str] = Field(
        None,
        description="Optional species filter for MSD and diffusion, e.g. 'Li'. Defaults to all atoms.",
    )
    rdf_species: Optional[str] = Field(
        None,
        description="Optional species filter for RDF. Defaults to species if species is set, otherwise all atoms.",
    )
    diffusion_dimension: Literal["x", "y", "z", "xy", "xz", "yz", "xyz"] = Field(
        "xyz",
        description="Axis set used for the Einstein diffusion fit. Use xy for surface diffusion, x/y/z for 1D projections, or xyz for isotropic 3D diffusion.",
    )
    fit_fraction_start: float = Field(
        0.2,
        ge=0.0,
        lt=1.0,
        description="Starting fraction of the trajectory used for linear MSD fitting.",
    )


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
        summary = {
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
            "ts_image": summary["ts_image"],
            "forward_barrier_ev": forward_barrier,
            "reverse_barrier_ev": reverse_barrier,
        }
        content = (
            "analyze_vasp_neb_results completed.\n"
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
            ]
            trajectory_path = next((path for path in candidates if path.exists()), None)
            if trajectory_path is None:
                ext_candidates = sorted(
                    path for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in {".traj", ".xyz", ".extxyz"}
                )
                trajectory_path = ext_candidates[0] if ext_candidates else None
            if trajectory_path is None:
                _error(
                    "analyze_trajectory",
                    "No recognizable trajectory file found in the provided directory.",
                    data={"result_dir_rel": workspace_relpath(source_path)},
                    error_code="missing_trajectory",
                )
            result_dir = source_path
        else:
            trajectory_path = source_path
            result_dir = source_path.parent
        output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
            result_dir.parent / f"{result_dir.name}_analysis"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectory = _read_trajectory_frames(trajectory_path)
        frames = trajectory.frames
        if len(frames) < 2:
            _error(
                "analyze_trajectory",
                "Trajectory must contain at least two frames.",
                data={"result_dir_rel": workspace_relpath(result_dir)},
                error_code="too_few_frames",
            )
        axes = _axes_for_dimension(params.diffusion_dimension)
        indices, msd = _mean_square_displacement(
            frames,
            params.species,
            axes=axes,
            positions_are_wrapped=trajectory.positions_are_wrapped,
        )
        time_ps = indices * params.timestep_fs / 1000.0
        fit_start = max(1, int(math.floor(len(msd) * params.fit_fraction_start)))
        fit_x = time_ps[fit_start:]
        fit_y = msd[fit_start:]
        slope = 0.0
        intercept = float(msd[0])
        diffusion = None
        dim = len(axes)
        if len(fit_x) >= 2 and np.ptp(fit_x) > 0:
            slope, intercept = np.polyfit(fit_x, fit_y, deg=1)
            diffusion = float(max(slope, 0.0) / (2.0 * dim))
        rdf_species = params.rdf_species or params.species
        rdf_r, rdf_g = _rdf(frames, species=rdf_species)
        temps, energies = _parse_oszicar_series(result_dir / "OSZICAR")

        msd_csv = output_dir / "trajectory_msd.csv"
        rdf_csv = output_dir / "trajectory_rdf.csv"
        msd_png = output_dir / "trajectory_msd.png"
        thermo_png = output_dir / "trajectory_thermo.png"
        summary_json = output_dir / "trajectory_summary.json"
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
            figure = plt.figure(figsize=(6.5, 4.0))
            try:
                plt.plot(time_ps, msd, label="MSD")
                if diffusion is not None:
                    plt.plot(time_ps, slope * time_ps + intercept, linestyle="--", label="linear fit")
                plt.xlabel("Time (ps)")
                plt.ylabel("MSD ($\\AA^2$)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(msd_png, dpi=180)
            finally:
                plt.close(figure)

            if temps or energies:
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
            "timestep_fs": params.timestep_fs,
            "final_msd_a2": float(msd[-1]),
            "diffusion_coefficient_a2_per_ps": diffusion,
            "msd_csv_rel": workspace_relpath(msd_csv),
            "rdf_csv_rel": workspace_relpath(rdf_csv),
            "msd_png_rel": workspace_relpath(msd_png),
            "thermo_png_rel": workspace_relpath(thermo_png) if thermo_png.exists() else None,
            "temperature_mean_K": float(np.mean(temps)) if temps else None,
            "energy_mean": float(np.mean(energies)) if energies else None,
        }
        _write_json(summary_json, summary)
        data = {
            "result_dir_rel": workspace_relpath(result_dir),
            "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_json),
            "csv_rel": workspace_relpath(msd_csv),
            "png_rel": workspace_relpath(msd_png),
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
