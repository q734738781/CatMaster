from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import iread as ase_iread
from ase.io import write as ase_write
from pydantic import BaseModel, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from .native_stage import require_fresh_stage

_MOLECULE_EXTS = {".xyz", ".mol", ".sdf", ".mol2", ".pdb", ".vasp", ".cif"}
_XTB_OUTPUT_NAMES = ("xtbopt.xyz", "xtblast.xyz", "xtb.trj")
_ORCA_OUTPUT_NAMES = (
    "job.runtime.xyz",
    "job.xyz",
    "job.runtime_trj.xyz",
    "job_trj.xyz",
    "job_IRC_Full_trj.xyz",
    "job_IRC_F.xyz",
    "job_IRC_B.xyz",
)


def _tool_error(tool_name: str, message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
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
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _iter_structure_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _MOLECULE_EXTS and path.name not in {"POSCAR", "CONTCAR"}:
            continue
        files.append(path)
    return files


def _atoms_from_path(path: Path, *, index: int = 0) -> Atoms:
    try:
        return ase_read(str(path), index=index)
    except Exception as exc:
        raise ValueError(f"Failed to read structure {path}: {exc}") from exc


def _write_atoms_xyz(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms, format="xyz")


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Coordinate shapes do not match for RMSD calculation")
    if a.size == 0:
        return 0.0
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    cov = ac.T @ bc
    v, _, wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, d]) @ wt
    aligned = ac @ rot
    diff = aligned - bc
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _load_energy_records(summary_path: Path) -> dict[str, dict[str, Any]]:
    suffix = summary_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("Energy summary JSON must contain a list under `records` or be a list itself")
        out: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            rel = str(record.get("structure_rel") or record.get("output_rel") or "").strip()
            if not rel:
                continue
            key = f"{rel}#{record['frame_index']}" if record.get("frame_index") is not None else rel
            if key in out:
                raise ValueError(f"Duplicate energy record: {key}")
            out[key] = record
        return out
    if suffix == ".csv":
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        out = {}
        for row in rows:
            rel = str(row.get("structure_rel") or row.get("output_rel") or "").strip()
            if rel:
                key = f"{rel}#{row['frame_index']}" if row.get("frame_index") not in (None, "") else rel
                if key in out:
                    raise ValueError(f"Duplicate energy record: {key}")
                out[key] = dict(row)
        return out
    raise ValueError(f"Unsupported energy summary format: {summary_path}")


class EnumerateMolecularConformersInput(BaseModel):
    """[molecule/modeling] Enumerate 3D molecular conformers from a SMILES string with RDKit ETKDGv3."""

    smiles: str = Field(..., description="SMILES string for the target molecule.")
    output_dir: str = Field(..., description="Fresh empty output directory for conformer XYZ files and per-conformer optimization status.")
    max_conformers: int = Field(20, ge=1, le=500, description="Maximum number of RDKit conformers to embed.")
    rms_prune_threshold: float = Field(0.35, ge=0.0, description="RMSD pruning threshold in Angstrom applied during embedding.")
    optimize: str = Field("mmff", pattern="^(mmff|uff|none)$", description="Force-field cleanup applied to each embedded conformer.")
    random_seed: int = Field(42, description="Random seed used by ETKDGv3.")
    max_iterations: int = Field(500, ge=1, description="Maximum force-field optimization iterations per conformer.")


class FilterConformerEnsembleInput(BaseModel):
    """[molecule/modeling] Filter a conformer ensemble by energy window and geometry similarity."""

    input_dir: str = Field(..., description="Directory containing conformer structures or extracted optimized molecules.")
    output_dir: str = Field(..., description="Directory where the filtered ensemble and manifest will be written.")
    energy_summary_path: str = Field(
        "",
        description="Optional JSON/CSV summary with structure_rel/output_rel and energy information. If absent, the tool tries common summary files under input_dir.",
    )
    energy_window_kcal_mol: float = Field(5.0, ge=0.0, description="Keep only conformers within this relative-energy window.")
    rmsd_threshold_angstrom: float = Field(0.35, ge=0.0, description="Reject geometrically redundant conformers below this RMSD threshold.")
    frames: str = Field(":", description="ASE frame index or slice, e.g. ':' for every frame or '-1' for the final frame. Multi-frame energy records must include zero-based frame_index.")
    apply_energy_window: bool = Field(True, description="Set false for geometry-only filtering without energy data.")
    missing_energy: Literal["error", "exclude", "keep"] = Field("error", description="Policy for a frame missing energy when energy filtering is enabled; keep explicitly retains it with unknown energy.")

    @model_validator(mode="before")
    @classmethod
    def _legacy_null(cls, values):
        if isinstance(values, dict) and values.get("energy_summary_path") is None:
            return {**values, "energy_summary_path": ""}
        return values


class ExtractOptimizedMoleculesInput(BaseModel):
    """[molecule/modeling] Collect optimized molecular geometries from ORCA/xTB result folders into one reusable ensemble directory."""

    input_dir: str = Field("", description="Single run or recursive root directory; omit when supplying source_files.")
    source_files: list[str] = Field(default_factory=list, description="Explicit geometry/trajectory files, including custom basenames. Use instead of input_dir.")
    output_dir: str = Field(..., description="Fresh empty output directory for extracted XYZ files and per-run outcomes.")
    source: str = Field("auto", pattern="^(auto|xtb|orca)$", description="Result family to scan for optimized structures.")
    include_failed: bool = Field(False, description="Whether to include geometries from runs that do not look completed.")
    require_converged: bool = Field(True, description="Require an explicit optimization convergence signal. Set false to extract available geometry while reporting its actual convergence state; include_failed also bypasses this filter.")
    frame_index: int = Field(-1, description="Frame to extract; -1 selects the final frame for all supported trajectory filenames.")


def _optimize_conformer(mol, conf_id: int, method: str, max_iterations: int) -> tuple[float | None, str, str]:
    """Keep an embedded geometry even when its requested optimization fails."""
    from rdkit.Chem import AllChem
    if method == "none":
        return None, "not_requested", ""
    try:
        if method == "mmff":
            properties = AllChem.MMFFGetMoleculeProperties(mol)
            if properties is None:
                raise ValueError("MMFF parameters are unavailable for this molecule")
            ff = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=int(conf_id))
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
        if ff is None:
            raise ValueError(f"{method.upper()} force field is unavailable")
        status = ff.Minimize(maxIts=max_iterations)
        energy = float(ff.CalcEnergy())
        if not math.isfinite(energy):
            raise ValueError("Force field returned a non-finite energy")
        return energy, "converged" if status == 0 else "not_converged", "" if status == 0 else f"Minimize returned {status}"
    except Exception as exc:
        return None, "failed", str(exc)


def enumerate_molecular_conformers(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = EnumerateMolecularConformersInput(**payload)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:
        _tool_error(
            "enumerate_molecular_conformers",
            f"RDKit is required for conformer enumeration: {exc}",
            data={"smiles": params.smiles},
            error_code="missing_rdkit",
        )

    mol = Chem.MolFromSmiles(params.smiles)
    if mol is None:
        _tool_error(
            "enumerate_molecular_conformers",
            f"Invalid SMILES: {params.smiles}",
            data={"smiles": params.smiles},
            error_code="invalid_smiles",
        )
    mol = Chem.AddHs(mol)
    embed_params = AllChem.ETKDGv3()
    embed_params.randomSeed = int(params.random_seed)
    embed_params.pruneRmsThresh = float(params.rms_prune_threshold)
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(params.max_conformers), params=embed_params))
    if not conformer_ids:
        _tool_error(
            "enumerate_molecular_conformers",
            "RDKit failed to embed any conformers.",
            data={"smiles": params.smiles},
            error_code="embedding_failed",
        )

    records: list[dict[str, Any]] = []
    output_dir = resolve_workspace_path(params.output_dir)
    require_fresh_stage(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, conf_id in enumerate(conformer_ids):
        energy_kcal_mol, optimization_state, optimization_error = _optimize_conformer(
            mol, conf_id, params.optimize, params.max_iterations,
        )
        xyz_block = Chem.MolToXYZBlock(mol, confId=conf_id)
        xyz_path = output_dir / f"conf_{rank:03d}.xyz"
        xyz_path.write_text(xyz_block, encoding="utf-8")
        records.append(
            {
                "conformer_id": int(conf_id),
                "rank": rank,
                "structure_rel": workspace_relpath(xyz_path),
                "energy_kcal_mol": energy_kcal_mol,
                "optimization_state": optimization_state,
                "optimization_error": optimization_error,
            }
        )

    finite_energies = [record["energy_kcal_mol"] for record in records if record["energy_kcal_mol"] is not None]
    if finite_energies:
        emin = min(float(value) for value in finite_energies)
        for record in records:
            value = record.get("energy_kcal_mol")
            record["relative_energy_kcal_mol"] = None if value is None else float(value) - emin
    else:
        for record in records:
            record["relative_energy_kcal_mol"] = None

    failed_count = sum(row["optimization_state"] in {"failed", "not_converged"} for row in records)
    state = "partial" if failed_count else "completed"
    summary_json = output_dir / "conformers.json"
    summary_csv = output_dir / "conformers.csv"
    _write_json(summary_json, {"smiles": params.smiles, "records": records})
    _write_csv(summary_csv, records)
    data = {
        "smiles": params.smiles,
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_json),
        "summary_csv_rel": workspace_relpath(summary_csv),
        "count": len(records),
        "state": state,
        "optimization_failed_count": failed_count,
    }
    content = (
        f"enumerate_molecular_conformers {state}; optimization_failed_count={failed_count}.\n"
        f"count={len(records)} output_dir_rel={data['output_dir_rel']}\n"
        f"summary_json_rel={data['summary_json_rel']} summary_csv_rel={data['summary_csv_rel']}"
    )
    return content, {"tool_name": "enumerate_molecular_conformers", "data": data}


def filter_conformer_ensemble(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[molecule/modeling] Filter selected frames by relative energy and element-preserving RMSD."""
    params = FilterConformerEnsembleInput(**payload)
    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_dir.is_dir():
        raise ValueError("input_dir must be a directory")
    output_dir = resolve_workspace_path(params.output_dir)
    candidate_paths = _iter_structure_files(input_dir)
    if not candidate_paths:
        raise ValueError("No molecular structures found under input_dir")
    require_fresh_stage(output_dir, source_paths=candidate_paths)
    summary_path = resolve_workspace_path(params.energy_summary_path, must_exist=True) if params.energy_summary_path else None
    if summary_path is None:
        for name in ("conformers.json", "conformers.csv", "batch_summary.json", "summary.json", "crest_conformers.csv"):
            if (input_dir / name).is_file():
                summary_path = input_dir / name
                break
    energy_records = _load_energy_records(summary_path) if summary_path else {}
    candidates = []
    rejected = []
    for path in candidate_paths:
        try:
            frames = list(ase_iread(str(path), index=":"))
            selection = params.frames.strip()
            indices = list(range(len(frames)))
            if ":" in selection:
                parts = [int(value) if value else None for value in selection.split(":")]
                if len(parts) not in {2, 3}:
                    raise ValueError("frames must be an integer or start:stop:step slice")
                indices = indices[slice(*parts)]
            else:
                indices = [indices[int(selection)]]
            for frame_index in indices:
                rel = str(path.relative_to(input_dir))
                keys = [rel, workspace_relpath(path)]
                if len(frames) > 1:
                    keys = [f"{key}#{frame_index}" for key in keys]
                record = next((energy_records[key] for key in keys if key in energy_records), {})
                def number(key):
                    value = record.get(key)
                    if value in (None, ""):
                        return None
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite {key} for {rel}, frame {frame_index}")
                    return value
                candidates.append({"path": path, "frame_index": frame_index, "atoms": frames[frame_index],
                                   "energy": number("energy_kcal_mol"), "relative": number("relative_energy_kcal_mol")})
        except Exception as exc:
            rejected.append({"source_rel": workspace_relpath(path), "reason": "read_or_energy_error", "error": str(exc)})
    absolute = [item["energy"] for item in candidates if item["energy"] is not None]
    absolute_only = any(item["energy"] is not None and item["relative"] is None for item in candidates)
    relative_only = any(item["energy"] is None and item["relative"] is not None for item in candidates)
    if params.apply_energy_window and absolute_only and relative_only:
        raise ValueError("Cannot mix absolute-only and relative-only energies without a common reference; provide one consistent energy column")
    if absolute and not relative_only:
        minimum = min(absolute)
        for item in candidates:
            if item["energy"] is not None:
                item["relative"] = item["energy"] - minimum
    if params.apply_energy_window and params.missing_energy == "error":
        missing = [f"{workspace_relpath(item['path'])}#{item['frame_index']}" for item in candidates if item["relative"] is None]
        if missing:
            raise ValueError("Missing energy for selected frames: " + ", ".join(missing) + "; supply energy records or choose apply_energy_window=false/missing_energy explicitly")
    candidates.sort(key=lambda item: math.inf if item["relative"] is None else item["relative"])
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    accepted_atoms = []
    for item in candidates:
        source = {"source_rel": workspace_relpath(item["path"]), "frame_index": item["frame_index"]}
        energy = item["relative"]
        reason = ""
        if params.apply_energy_window:
            if energy is None and params.missing_energy == "exclude":
                reason = "missing_energy"
            elif energy is not None and energy > params.energy_window_kcal_mol:
                reason = "outside_energy_window"
        atoms = item["atoms"]
        if not reason and any(
            previous.get_chemical_symbols() == atoms.get_chemical_symbols()
            and _kabsch_rmsd(previous.positions, atoms.positions) < params.rmsd_threshold_angstrom
            for previous in accepted_atoms
        ):
            reason = "redundant_geometry"
        if reason:
            rejected.append({**source, "reason": reason})
            continue
        dest = output_dir / f"kept_{len(kept):03d}.xyz"
        _write_atoms_xyz(dest, atoms)
        kept.append({**source, "rank": len(kept), "structure_rel": workspace_relpath(dest),
                     "relative_energy_kcal_mol": energy, "energy_kcal_mol": item["energy"]})
        accepted_atoms.append(atoms)
    summary_json = output_dir / "filtered_conformers.json"
    summary_csv = output_dir / "filtered_conformers.csv"
    failed_count = sum(row["reason"] == "read_or_energy_error" for row in rejected)
    state = "partial" if failed_count else "completed"
    _write_json(summary_json, {"input_dir_rel": workspace_relpath(input_dir),
                             "energy_summary_rel": workspace_relpath(summary_path) if summary_path else None,
                             "state": state, "records": kept, "rejected": rejected})
    _write_csv(summary_csv, kept)
    data = {"input_dir_rel": workspace_relpath(input_dir), "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_json), "summary_csv_rel": workspace_relpath(summary_csv),
            "count": len(kept), "rejected_count": len(rejected), "failed_count": failed_count, "state": state}
    content = (f"filter_conformer_ensemble {state}.\ncount={len(kept)} rejected_count={len(rejected)} failed_count={failed_count}\n"
               f"summary_json_rel={data['summary_json_rel']} summary_csv_rel={data['summary_csv_rel']}")
    return content, {"tool_name": "filter_conformer_ensemble", "data": data}


def _looks_finished(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            return int(status.get("returncode", 1)) == 0
        except Exception:
            return False
    summary_candidates = ("xtb_summary.json", "orca_summary.json")
    for name in summary_candidates:
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if bool(payload.get("completed")):
                return True
        except Exception:
            continue
    return False


def _pick_optimized_geometry(run_dir: Path, *, source: str) -> Path | None:
    names = _XTB_OUTPUT_NAMES if source == "xtb" else (_ORCA_OUTPUT_NAMES if source == "orca" else _XTB_OUTPUT_NAMES + _ORCA_OUTPUT_NAMES)
    return next((run_dir / name for name in names if (run_dir / name).is_file()), None)


def _optimization_state(run_dir: Path) -> tuple[str, str]:
    converged = False
    failed = False
    completed = _looks_finished(run_dir)
    signals = ("THE OPTIMIZATION HAS CONVERGED", "OPTIMIZATION RUN DONE", "GEOMETRY OPTIMIZATION CONVERGED")
    for path in sorted({*run_dir.glob("*.out"), *run_dir.glob("*.log")}):
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                upper = line.upper()
                converged |= any(signal in upper for signal in signals)
                failed |= "NOT_CONVERGED" in upper or "OPTIMIZATION DID NOT CONVERGE" in upper
                completed |= "ORCA TERMINATED NORMALLY" in upper or "NORMAL TERMINATION OF XTB" in upper
    if (run_dir / "status.json").is_file():
        status = json.loads((run_dir / "status.json").read_text())
        failed |= status.get("returncode") not in (None, 0)
    return ("not_converged" if failed else "converged" if converged else "unknown",
            "failed" if failed else "completed" if completed else "unknown")


def extract_optimized_molecules(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[molecule/modeling] Extract a selected final frame, preserving each run's convergence and extraction outcome."""
    params = ExtractOptimizedMoleculesInput(**payload)
    if bool(params.input_dir) == bool(params.source_files):
        raise ValueError("Provide exactly one of input_dir or source_files")
    source_root = resolve_workspace_path(params.input_dir, must_exist=True) if params.input_dir else None
    if source_root and not source_root.is_dir():
        raise ValueError("input_dir must be a directory")
    if source_root:
        candidates = [(directory, _pick_optimized_geometry(directory, source=params.source))
                      for directory in [source_root, *sorted(p for p in source_root.rglob("*") if p.is_dir())]]
        candidates = [(directory, geometry) for directory, geometry in candidates
                      if geometry is not None or any((directory / marker).is_file() for marker in ("orca_summary.json", "xtb_summary.json", "job.out", "status.json"))]
    else:
        paths = [resolve_workspace_path(path, must_exist=True) for path in params.source_files]
        candidates = [(path.parent, path) for path in paths]
    output_dir = resolve_workspace_path(params.output_dir)
    require_fresh_stage(output_dir, source_paths=[path for _, path in candidates if path])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = []
    for run_dir, geometry in candidates:
        source = {"source_dir_rel": workspace_relpath(run_dir),
                  "source_file_rel": workspace_relpath(geometry) if geometry else ""}
        try:
            convergence, execution = _optimization_state(run_dir)
            source.update(optimization_state=convergence, execution_state=execution)
            if geometry is None:
                raise ValueError("No known optimized geometry found; supply source_files for custom basenames")
            if not params.include_failed and (execution == "failed" or (params.require_converged and convergence != "converged")):
                skipped.append({**source, "reason": "optimization_convergence_not_established"})
                continue
            atoms = ase_read(str(geometry), index=params.frame_index, format="xyz" if geometry.suffix == ".trj" else None)
            dest = output_dir / f"{run_dir.name}_{len(records):03d}.xyz"
            _write_atoms_xyz(dest, atoms)
            records.append({**source, "structure_rel": workspace_relpath(dest), "source": params.source, "frame_index": params.frame_index})
        except Exception as exc:
            skipped.append({**source, "reason": "extraction_error", "error": str(exc)})
    summary_json = output_dir / "optimized_molecules.json"
    summary_csv = output_dir / "optimized_molecules.csv"
    state = "partial" if skipped and records else "no_geometries" if not records else "completed"
    _write_json(summary_json, {"state": state, "records": records, "skipped": skipped})
    _write_csv(summary_csv, records)
    data = {"input_dir_rel": workspace_relpath(source_root) if source_root else "", "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_json), "summary_csv_rel": workspace_relpath(summary_csv),
            "count": len(records), "skipped_count": len(skipped), "state": state}
    content = (f"extract_optimized_molecules {state}.\ncount={len(records)} skipped_count={len(skipped)}\n"
               f"summary_json_rel={data['summary_json_rel']} summary_csv_rel={data['summary_csv_rel']}")
    return content, {"tool_name": "extract_optimized_molecules", "data": data}


__all__ = [
    "EnumerateMolecularConformersInput",
    "FilterConformerEnsembleInput",
    "ExtractOptimizedMoleculesInput",
    "enumerate_molecular_conformers",
    "filter_conformer_ensemble",
    "extract_optimized_molecules",
]
