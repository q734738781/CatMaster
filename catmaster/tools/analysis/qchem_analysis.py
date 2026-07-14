from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_HARTREE_TO_KCAL = 627.509474
_XTB_ENERGY_PATTERNS = (
    re.compile(r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"total energy\s+(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+Eh", re.IGNORECASE),
    re.compile(r":: total energy\s+(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+Eh ::", re.IGNORECASE),
)
_XTB_FREE_ENERGY_PATTERNS = (
    re.compile(r"TOTAL FREE ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r":: total free energy\s+(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+Eh ::", re.IGNORECASE),
)
_XTB_FREQ_RE = re.compile(r"^\s*\d+\s+(-?\d+(?:\.\d+)?)\s+cm-1", re.IGNORECASE)
_ORCA_FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
_ORCA_FREQ_RE = re.compile(r"^\s*\d+:\s+(-?\d+(?:\.\d+)?)\s+cm\*\*-1", re.IGNORECASE)
_ORCA_GIBBS_RE = re.compile(r"Final Gibbs free energy\s*\.\.\.\s*(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.IGNORECASE)
_ORCA_ENTHALPY_RE = re.compile(r"Total Enthalpy\s*\.\.\.\s*(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.IGNORECASE)
_ORCA_ZPE_RE = re.compile(r"Zero point energy\s*\.\.\.\s*(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.IGNORECASE)
_ORCA_NMR_RE = re.compile(r"^\s*\d+\s+[A-Za-z]+\s+(-?\d+(?:\.\d+)?)\s+ppm", re.IGNORECASE)


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _try_read_last_xyz_frame(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        atoms = ase_read(str(path), index=-1 if path.suffix == ".trj" else 0)
    except Exception:
        return None
    temp_path = path.with_suffix(".last.xyz")
    try:
        from ase.io import write as ase_write

        ase_write(str(temp_path), atoms, format="xyz")
        return workspace_relpath(temp_path)
    except Exception:
        return workspace_relpath(path)


def _discover_xtb_runs(root: Path) -> list[Path]:
    if root.is_file():
        _tool_error("analyze_xtb_results", "result_root must be a directory", error_code="invalid_result_root")
    if (root / "xtb_summary.json").is_file() or (root / "crest_summary.json").is_file():
        return [root]
    runs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if (path / "xtb_summary.json").is_file() or (path / "crest_summary.json").is_file():
            runs.append(path)
    return runs


def _discover_orca_runs(root: Path) -> list[Path]:
    if root.is_file():
        _tool_error("analyze_orca_results", "result_root must be a directory", error_code="invalid_result_root")
    if (root / "job.out").is_file() or (root / "orca_summary.json").is_file():
        return [root]
    runs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if (path / "job.out").is_file() or (path / "orca_summary.json").is_file():
            runs.append(path)
    return runs


def _extract_first(patterns: tuple[re.Pattern[str], ...], text: str) -> float | None:
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


def _parse_xtb_frequencies(run_dir: Path) -> tuple[int, list[float]]:
    values: list[float] = []
    for name in ("g98.out", "vibspectrum"):
        text = _read_text(run_dir / name)
        if not text:
            continue
        for line in text.splitlines():
            match = _XTB_FREQ_RE.search(line)
            if not match:
                continue
            try:
                values.append(float(match.group(1)))
            except Exception:
                continue
    imag = [value for value in values if value < 0.0]
    return len(imag), values[:20]


def _parse_xtb_run(run_dir: Path) -> dict[str, Any]:
    xtb_summary = _read_json(run_dir / "xtb_summary.json") or {}
    crest_summary = _read_json(run_dir / "crest_summary.json") or {}
    log_name = str(xtb_summary.get("log_file") or crest_summary.get("log_file") or "xtb_stdout.out")
    text = _read_text(run_dir / log_name)
    energy = _extract_first(_XTB_ENERGY_PATTERNS, text)
    free_energy = _extract_first(_XTB_FREE_ENERGY_PATTERNS, text)
    imag_count, freq_values = _parse_xtb_frequencies(run_dir)
    completed = bool(xtb_summary.get("completed") or crest_summary.get("completed"))
    final_structure = None
    for name in ("xtbopt.xyz", "xtblast.xyz", "crest_best.xyz", "crest_conformers.xyz", "xtb.trj"):
        path = run_dir / name
        if not path.exists():
            continue
        final_structure = _try_read_last_xyz_frame(path) if name.endswith(".trj") else workspace_relpath(path)
        break
    record: dict[str, Any] = {
        "result_dir_rel": workspace_relpath(run_dir),
        "state": "completed" if completed else "incomplete",
        "energy_hartree": energy,
        "free_energy_hartree": free_energy,
        "imaginary_frequency_count": imag_count,
        "frequency_sample_cm-1": freq_values,
        "final_structure_rel": final_structure,
        "engine": "crest" if crest_summary else "xtb",
        "issues": [],
    }
    if energy is None:
        record["issues"].append("missing_total_energy")
    if xtb_summary and int(xtb_summary.get("returncode", 0)) != 0:
        record["issues"].append(f"nonzero_returncode:{xtb_summary.get('returncode')}")
    if crest_summary and int(crest_summary.get("returncode", 0)) != 0:
        record["issues"].append(f"nonzero_returncode:{crest_summary.get('returncode')}")
    return record


def _recursive_extract_numeric(payload: Any, key_hint: str) -> float | None:
    target = key_hint.lower()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if target in str(key).lower():
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except Exception:
                        pass
            nested = _recursive_extract_numeric(value, key_hint)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _recursive_extract_numeric(item, key_hint)
            if nested is not None:
                return nested
    return None


def _recursive_extract_list(payload: Any, key_hint: str) -> list[float]:
    target = key_hint.lower()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if target in str(key).lower() and isinstance(value, list):
                out: list[float] = []
                for item in value:
                    try:
                        out.append(float(item))
                    except Exception:
                        continue
                if out:
                    return out
            nested = _recursive_extract_list(value, key_hint)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _recursive_extract_list(item, key_hint)
            if nested:
                return nested
    return []


def _try_cclib_parse(out_path: Path) -> dict[str, Any]:
    try:
        import cclib  # type: ignore
    except Exception:
        return {}
    try:
        data = cclib.io.ccread(str(out_path))
    except Exception:
        return {}
    out: dict[str, Any] = {}
    try:
        if getattr(data, "scfenergies", None) is not None and len(data.scfenergies) > 0:
            out["final_energy_hartree"] = float(data.scfenergies[-1]) / 27.211386245988
    except Exception:
        pass
    try:
        if getattr(data, "vibfreqs", None) is not None:
            freqs = [float(value) for value in data.vibfreqs]
            out["frequency_sample_cm-1"] = freqs[:20]
            out["imaginary_frequency_count"] = sum(1 for value in freqs if value < 0.0)
    except Exception:
        pass
    return out


def _parse_orca_run(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "orca_summary.json") or {}
    property_json = _read_json(run_dir / "job.property.json") or {}
    out_path = run_dir / "job.out"
    text = _read_text(out_path)
    cclib_data = _try_cclib_parse(out_path)
    final_energy = _recursive_extract_numeric(property_json, "energy")
    if final_energy is None:
        final_energy = cclib_data.get("final_energy_hartree")
    if final_energy is None:
        final_energy = _extract_first((_ORCA_FINAL_ENERGY_RE,), text)
    gibbs = _recursive_extract_numeric(property_json, "gibbs")
    if gibbs is None:
        gibbs = _extract_first((_ORCA_GIBBS_RE,), text)
    enthalpy = _recursive_extract_numeric(property_json, "enthalpy")
    if enthalpy is None:
        enthalpy = _extract_first((_ORCA_ENTHALPY_RE,), text)
    zpe = _recursive_extract_numeric(property_json, "zero_point")
    if zpe is None:
        zpe = _extract_first((_ORCA_ZPE_RE,), text)
    freq_values = _recursive_extract_list(property_json, "freq")
    if not freq_values:
        freq_values = list(cclib_data.get("frequency_sample_cm-1") or [])
    if not freq_values:
        for line in text.splitlines():
            match = _ORCA_FREQ_RE.search(line)
            if not match:
                continue
            try:
                freq_values.append(float(match.group(1)))
            except Exception:
                continue
    imag_count = cclib_data.get("imaginary_frequency_count")
    if imag_count is None:
        imag_count = sum(1 for value in freq_values if value < 0.0)
    nmr_values: list[float] = []
    if property_json:
        nmr_values = _recursive_extract_list(property_json, "ppm")
    if not nmr_values:
        for line in text.splitlines():
            match = _ORCA_NMR_RE.search(line)
            if not match:
                continue
            try:
                nmr_values.append(float(match.group(1)))
            except Exception:
                continue
    completed = bool(summary.get("completed")) or ("ORCA TERMINATED NORMALLY" in text)
    final_structure = None
    for name in ("job.xyz", "job_IRC_Full_trj.xyz", "job_IRC_F.xyz", "job_IRC_B.xyz", "input.xyz"):
        path = run_dir / name
        if not path.exists():
            continue
        final_structure = _try_read_last_xyz_frame(path) if name.endswith(".trj") else workspace_relpath(path)
        break
    record = {
        "result_dir_rel": workspace_relpath(run_dir),
        "state": "completed" if completed else "incomplete",
        "final_energy_hartree": final_energy,
        "gibbs_free_energy_hartree": gibbs,
        "enthalpy_hartree": enthalpy,
        "zpe_hartree": zpe,
        "imaginary_frequency_count": int(imag_count or 0),
        "frequency_sample_cm-1": freq_values[:20],
        "nmr_shift_sample_ppm": nmr_values[:20],
        "final_structure_rel": final_structure,
        "issues": [],
    }
    if final_energy is None:
        record["issues"].append("missing_final_energy")
    if not completed:
        record["issues"].append("orca_not_normal_termination")
    return record


class AnalyzeXtbResultsInput(BaseModel):
    """[xtb/analyze] Summarize xTB/CREST result folders into JSON/CSV evidence."""

    result_root: str = Field(..., description="xTB or CREST result directory or a batch root containing many runs.")
    output_dir: str | None = Field(None, description="Optional directory for summary artifacts. Defaults to <result_root>/analysis.")


class AnalyzeOrcaResultsInput(BaseModel):
    """[orca/analyze] Summarize ORCA result folders into JSON/CSV evidence."""

    result_root: str = Field(..., description="ORCA result directory or a batch root containing many ORCA runs.")
    output_dir: str | None = Field(None, description="Optional directory for summary artifacts. Defaults to <result_root>/analysis.")


def analyze_xtb_results(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = AnalyzeXtbResultsInput(**payload)
    result_root = resolve_workspace_path(params.result_root, must_exist=True)
    runs = _discover_xtb_runs(result_root)
    if not runs:
        _tool_error("analyze_xtb_results", f"No xTB/CREST result folders found under {result_root}", error_code="no_runs")
    records = [_parse_xtb_run(run_dir) for run_dir in runs]
    finite = [rec["energy_hartree"] for rec in records if rec.get("energy_hartree") is not None]
    if finite:
        emin = min(float(value) for value in finite)
        for record in records:
            energy = record.get("energy_hartree")
            record["relative_energy_kcal_mol"] = None if energy is None else (float(energy) - emin) * _HARTREE_TO_KCAL
    else:
        for record in records:
            record["relative_energy_kcal_mol"] = None
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else result_root / "analysis"
    summary_json = output_dir / "xtb_results_summary.json"
    summary_csv = output_dir / "xtb_results_summary.csv"
    _write_json(summary_json, {"result_root_rel": workspace_relpath(result_root), "records": records})
    _write_csv(summary_csv, records)
    data = {
        "result_root_rel": workspace_relpath(result_root),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_json),
        "summary_csv_rel": workspace_relpath(summary_csv),
        "count": len(records),
    }
    content = (
        "analyze_xtb_results completed.\n"
        f"count={len(records)} output_dir_rel={data['output_dir_rel']}\n"
        f"summary_json_rel={data['summary_json_rel']} summary_csv_rel={data['summary_csv_rel']}"
    )
    return content, {"tool_name": "analyze_xtb_results", "data": data}


def analyze_orca_results(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = AnalyzeOrcaResultsInput(**payload)
    result_root = resolve_workspace_path(params.result_root, must_exist=True)
    runs = _discover_orca_runs(result_root)
    if not runs:
        _tool_error("analyze_orca_results", f"No ORCA result folders found under {result_root}", error_code="no_runs")
    records = [_parse_orca_run(run_dir) for run_dir in runs]
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else result_root / "analysis"
    summary_json = output_dir / "orca_results_summary.json"
    summary_csv = output_dir / "orca_results_summary.csv"
    _write_json(summary_json, {"result_root_rel": workspace_relpath(result_root), "records": records})
    _write_csv(summary_csv, records)
    data = {
        "result_root_rel": workspace_relpath(result_root),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_json),
        "summary_csv_rel": workspace_relpath(summary_csv),
        "count": len(records),
    }
    content = (
        "analyze_orca_results completed.\n"
        f"count={len(records)} output_dir_rel={data['output_dir_rel']}\n"
        f"summary_json_rel={data['summary_json_rel']} summary_csv_rel={data['summary_csv_rel']}"
    )
    return content, {"tool_name": "analyze_orca_results", "data": data}


__all__ = [
    "AnalyzeXtbResultsInput",
    "AnalyzeOrcaResultsInput",
    "analyze_xtb_results",
    "analyze_orca_results",
]
