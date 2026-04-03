from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_MOLECULE_EXTS = {".xyz", ".mol", ".sdf", ".mol2", ".pdb", ".vasp", ".cif"}
_XTB_OUTPUT_NAMES = ("xtbopt.xyz", "xtblast.xyz", "xtb.trj")
_ORCA_OUTPUT_NAMES = (
    "job.xyz",
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
            out[rel] = record
        return out
    if suffix == ".csv":
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        out = {}
        for row in rows:
            rel = str(row.get("structure_rel") or row.get("output_rel") or "").strip()
            if rel:
                out[rel] = dict(row)
        return out
    raise ValueError(f"Unsupported energy summary format: {summary_path}")


class EnumerateMolecularConformersInput(BaseModel):
    """[molecule/modeling] Enumerate 3D molecular conformers from a SMILES string with RDKit ETKDGv3."""

    smiles: str = Field(..., description="SMILES string for the target molecule.")
    output_dir: str = Field(..., description="Workspace-relative output directory for conformer XYZ files and summaries.")
    max_conformers: int = Field(20, ge=1, le=500, description="Maximum number of RDKit conformers to embed.")
    rms_prune_threshold: float = Field(0.35, ge=0.0, description="RMSD pruning threshold in Angstrom applied during embedding.")
    optimize: str = Field("mmff", pattern="^(mmff|uff|none)$", description="Force-field cleanup applied to each embedded conformer.")
    random_seed: int = Field(42, description="Random seed used by ETKDGv3.")


class FilterConformerEnsembleInput(BaseModel):
    """[molecule/modeling] Filter a conformer ensemble by energy window and geometry similarity."""

    input_dir: str = Field(..., description="Directory containing conformer structures or extracted optimized molecules.")
    output_dir: str = Field(..., description="Directory where the filtered ensemble and manifest will be written.")
    energy_summary_path: str | None = Field(
        None,
        description="Optional JSON/CSV summary with structure_rel/output_rel and energy information. If absent, the tool tries common summary files under input_dir.",
    )
    energy_window_kcal_mol: float = Field(5.0, ge=0.0, description="Keep only conformers within this relative-energy window.")
    rmsd_threshold_angstrom: float = Field(0.35, ge=0.0, description="Reject geometrically redundant conformers below this RMSD threshold.")


class ExtractOptimizedMoleculesInput(BaseModel):
    """[molecule/modeling] Collect optimized molecular geometries from ORCA/xTB result folders into one reusable ensemble directory."""

    input_dir: str = Field(..., description="Root directory containing ORCA/xTB result folders.")
    output_dir: str = Field(..., description="Directory where extracted optimized XYZ files and a manifest will be written.")
    source: str = Field("auto", pattern="^(auto|xtb|orca)$", description="Result family to scan for optimized structures.")
    include_failed: bool = Field(False, description="Whether to include geometries from runs that do not look completed.")


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
    output_dir.mkdir(parents=True, exist_ok=True)

    mmff_props = None
    if params.optimize == "mmff":
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        except Exception:
            mmff_props = None

    for rank, conf_id in enumerate(conformer_ids):
        energy_kcal_mol: float | None = None
        if params.optimize == "mmff" and mmff_props is not None:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                ff.Minimize(maxIts=500)
                energy_kcal_mol = float(ff.CalcEnergy())
            except Exception:
                energy_kcal_mol = None
        elif params.optimize == "uff":
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                ff.Minimize(maxIts=500)
                energy_kcal_mol = float(ff.CalcEnergy())
            except Exception:
                energy_kcal_mol = None
        xyz_block = Chem.MolToXYZBlock(mol, confId=conf_id)
        xyz_path = output_dir / f"conf_{rank:03d}.xyz"
        xyz_path.write_text(xyz_block, encoding="utf-8")
        records.append(
            {
                "conformer_id": int(conf_id),
                "rank": rank,
                "structure_rel": workspace_relpath(xyz_path),
                "energy_kcal_mol": energy_kcal_mol,
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
    }
    content = (
        "enumerate_molecular_conformers completed.\n"
        f"count={len(records)} output_dir_rel={data['output_dir_rel']}\n"
        f"summary_json_rel={data['summary_json_rel']}"
    )
    return content, {"tool_name": "enumerate_molecular_conformers", "data": data}


@dataclass(frozen=True)
class _CandidateConformer:
    path: Path
    structure_rel: str
    energy_kcal_mol: float | None
    relative_energy_kcal_mol: float | None


def filter_conformer_ensemble(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = FilterConformerEnsembleInput(**payload)
    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_dir.is_dir():
        _tool_error(
            "filter_conformer_ensemble",
            f"input_dir is not a directory: {input_dir}",
            error_code="invalid_input_dir",
        )

    summary_path: Path | None = None
    if params.energy_summary_path:
        summary_path = resolve_workspace_path(params.energy_summary_path, must_exist=True)
    else:
        for candidate_name in ("conformers.json", "conformers.csv", "batch_summary.json", "summary.json", "crest_conformers.csv"):
            candidate = input_dir / candidate_name
            if candidate.is_file():
                summary_path = candidate
                break

    energy_records: dict[str, dict[str, Any]] = {}
    if summary_path is not None:
        try:
            energy_records = _load_energy_records(summary_path)
        except Exception:
            energy_records = {}

    candidate_paths = _iter_structure_files(input_dir)
    if not candidate_paths:
        _tool_error(
            "filter_conformer_ensemble",
            f"No molecular structures found under {input_dir}",
            error_code="no_structures",
        )

    candidates: list[_CandidateConformer] = []
    for path in candidate_paths:
        rel = str(path.relative_to(input_dir))
        record = energy_records.get(rel) or energy_records.get(workspace_relpath(path)) or {}
        energy = record.get("energy_kcal_mol")
        if energy is None and record.get("relative_energy_kcal_mol") is not None:
            energy = float(record.get("relative_energy_kcal_mol"))
        rel_energy = record.get("relative_energy_kcal_mol")
        if rel_energy is None and energy is not None:
            rel_energy = float(energy)
        candidates.append(
            _CandidateConformer(
                path=path,
                structure_rel=rel,
                energy_kcal_mol=None if energy is None else float(energy),
                relative_energy_kcal_mol=None if rel_energy is None else float(rel_energy),
            )
        )

    candidates.sort(
        key=lambda item: (
            math.inf if item.relative_energy_kcal_mol is None else item.relative_energy_kcal_mol,
            item.structure_rel,
        )
    )

    kept: list[dict[str, Any]] = []
    kept_coords: list[np.ndarray] = []
    output_dir = resolve_workspace_path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        rel_energy = candidate.relative_energy_kcal_mol
        if rel_energy is not None and rel_energy > float(params.energy_window_kcal_mol):
            continue
        try:
            atoms = _atoms_from_path(candidate.path)
        except Exception:
            continue
        coords = np.asarray(atoms.get_positions(), dtype=float)
        redundant = False
        for accepted_coords in kept_coords:
            if accepted_coords.shape != coords.shape:
                continue
            if _kabsch_rmsd(coords, accepted_coords) < float(params.rmsd_threshold_angstrom):
                redundant = True
                break
        if redundant:
            continue
        dest = output_dir / f"kept_{len(kept):03d}.xyz"
        _write_atoms_xyz(dest, atoms)
        kept.append(
            {
                "rank": len(kept),
                "source_rel": workspace_relpath(candidate.path),
                "structure_rel": workspace_relpath(dest),
                "relative_energy_kcal_mol": rel_energy,
                "energy_kcal_mol": candidate.energy_kcal_mol,
            }
        )
        kept_coords.append(coords)

    summary_json = output_dir / "filtered_conformers.json"
    summary_csv = output_dir / "filtered_conformers.csv"
    _write_json(
        summary_json,
        {
            "input_dir_rel": workspace_relpath(input_dir),
            "energy_summary_rel": workspace_relpath(summary_path) if summary_path else None,
            "records": kept,
        },
    )
    _write_csv(summary_csv, kept)
    data = {
        "input_dir_rel": workspace_relpath(input_dir),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_json),
        "summary_csv_rel": workspace_relpath(summary_csv),
        "count": len(kept),
    }
    content = (
        "filter_conformer_ensemble completed.\n"
        f"count={len(kept)} output_dir_rel={data['output_dir_rel']}\n"
        f"summary_json_rel={data['summary_json_rel']}"
    )
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
    ordered = _XTB_OUTPUT_NAMES if source == "xtb" else _ORCA_OUTPUT_NAMES
    for name in ordered:
        path = run_dir / name
        if not path.exists():
            continue
        if name.endswith(".trj"):
            try:
                atoms = _atoms_from_path(path, index=-1)
            except Exception:
                continue
            dest = run_dir / (path.stem + "_last.xyz")
            _write_atoms_xyz(dest, atoms)
            return dest
        return path
    if source == "auto":
        for fallback in ("xtbopt.xyz", "job.xyz", "job_IRC_Full_trj.xyz"):
            path = run_dir / fallback
            if path.exists():
                return path
    return None


def extract_optimized_molecules(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = ExtractOptimizedMoleculesInput(**payload)
    input_dir = resolve_workspace_path(params.input_dir, must_exist=True)
    if not input_dir.is_dir():
        _tool_error(
            "extract_optimized_molecules",
            f"input_dir is not a directory: {input_dir}",
            error_code="invalid_input_dir",
        )
    output_dir = resolve_workspace_path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for path in sorted(input_dir.rglob("*")):
        if not path.is_dir():
            continue
        if path in seen_dirs:
            continue
        source = params.source
        if source == "auto":
            if (path / "xtb_summary.json").exists() or (path / "xtbopt.xyz").exists():
                source = "xtb"
            elif (path / "orca_summary.json").exists() or (path / "job.inp").exists() or (path / "job.out").exists():
                source = "orca"
            else:
                continue
        if not params.include_failed and not _looks_finished(path):
            continue
        geometry = _pick_optimized_geometry(path, source=source)
        if geometry is None or not geometry.exists():
            continue
        try:
            atoms = _atoms_from_path(geometry, index=-1 if geometry.name.endswith(".trj") else 0)
        except Exception:
            continue
        dest = output_dir / f"{path.name}_{len(records):03d}.xyz"
        _write_atoms_xyz(dest, atoms)
        records.append(
            {
                "source_dir_rel": workspace_relpath(path),
                "source_file_rel": workspace_relpath(geometry),
                "structure_rel": workspace_relpath(dest),
                "source": source,
            }
        )
        seen_dirs.add(path)

    summary_json = output_dir / "optimized_molecules.json"
    summary_csv = output_dir / "optimized_molecules.csv"
    _write_json(summary_json, {"records": records})
    _write_csv(summary_csv, records)
    data = {
        "input_dir_rel": workspace_relpath(input_dir),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_json),
        "summary_csv_rel": workspace_relpath(summary_csv),
        "count": len(records),
    }
    content = (
        "extract_optimized_molecules completed.\n"
        f"count={len(records)} output_dir_rel={data['output_dir_rel']}\n"
        f"summary_json_rel={data['summary_json_rel']}"
    )
    return content, {"tool_name": "extract_optimized_molecules", "data": data}


__all__ = [
    "EnumerateMolecularConformersInput",
    "FilterConformerEnsembleInput",
    "ExtractOptimizedMoleculesInput",
    "enumerate_molecular_conformers",
    "filter_conformer_ensemble",
    "extract_optimized_molecules",
]
