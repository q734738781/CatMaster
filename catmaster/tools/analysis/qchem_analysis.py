from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from .output_sources import analysis_sources, discover_directories, select_output

_NUMBER = r"-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
_XTB_ENERGY_PATTERNS = (
    re.compile(rf"TOTAL ENERGY\s+({_NUMBER})", re.IGNORECASE),
    re.compile(rf"total energy\s+({_NUMBER})\s+Eh", re.IGNORECASE),
    re.compile(rf":: total energy\s+({_NUMBER})\s+Eh ::", re.IGNORECASE),
)
_XTB_FREE_ENERGY_PATTERNS = (
    re.compile(rf"TOTAL FREE ENERGY\s+({_NUMBER})", re.IGNORECASE),
    re.compile(rf":: total free energy\s+({_NUMBER})\s+Eh ::", re.IGNORECASE),
)
_XTB_FREQ_RE = re.compile(rf"^\s*\d+\s+({_NUMBER})\s+cm-1", re.IGNORECASE)
_ORCA_FINAL_ENERGY_RE = re.compile(rf"FINAL SINGLE POINT ENERGY\s+({_NUMBER})")
_ORCA_FREQ_RE = re.compile(rf"^\s*\d+:\s+({_NUMBER})\s+cm\*\*-1", re.IGNORECASE)
_ORCA_GIBBS_RE = re.compile(rf"Final Gibbs free energy\s*\.\.\.\s*({_NUMBER})", re.IGNORECASE)
_ORCA_ENTHALPY_RE = re.compile(rf"Total Enthalpy\s*\.\.\.\s*({_NUMBER})", re.IGNORECASE)
_ORCA_ZPE_RE = re.compile(rf"Zero point energy\s*\.\.\.\s*({_NUMBER})", re.IGNORECASE)
_ORCA_SHIELDING_ROW_RE = re.compile(rf"^\s*(\d+)\s+([A-Za-z]{{1,2}})\s+({_NUMBER})(?:\s+|$)")


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
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _extract_last(patterns: tuple[re.Pattern[str], ...], text: str) -> float | None:
    matches: list[tuple[int, float]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                matches.append((match.start(), float(match.group(1))))
            except Exception:
                continue
    return max(matches, key=lambda item: item[0])[1] if matches else None


def _last_xyz_frame(path: Path, output_dir: Path | None = None) -> str | None:
    if not path.is_file():
        return None
    if output_dir is None:
        return workspace_relpath(path)
    atoms = ase_read(str(path), index=-1, format="xyz" if path.suffix == ".trj" else None)
    from ase.io import write as ase_write
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (path.stem + ".last.xyz")
    ase_write(str(destination), atoms, format="xyz")
    return workspace_relpath(destination)


def _xyz_frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                first = handle.readline()
                if not first:
                    break
                atoms = int(first.strip())
                handle.readline()
                for _ in range(atoms):
                    if not handle.readline():
                        return count
                count += 1
    except Exception:
        return None
    return count


def _find_run_output(directory: Path, engine: str) -> Path | None:
    if engine == "orca":
        return select_output(directory, names=("job.out",), signals=("O   R   C   A", "ORCA TERMINATED", "FINAL SINGLE POINT ENERGY"))
    return select_output(directory, names=("xtb_stdout.out", "crest_stdout.out"), signals=("NORMAL TERMINATION OF XTB", "CREST TERMINATED", "* X T B *"))


def _discover_runs(root: Path, markers: tuple[str, ...], *, tool_name: str) -> list[Path]:
    engine = "orca" if "orca" in tool_name else "xtb"
    return discover_directories(root, markers, lambda directory: _find_run_output(directory, engine))


def _parse_frequency_files(run_dir: Path) -> tuple[str, list[float]]:
    values: list[float] = []
    found_source = False
    for name in ("g98.out", "vibspectrum"):
        path = run_dir / name
        if not path.is_file():
            continue
        found_source = True
        for line in _read_text(path).splitlines():
            match = _XTB_FREQ_RE.search(line)
            if match:
                try:
                    values.append(float(match.group(1)))
                except Exception:
                    pass
    if values:
        return "calculated", values
    return ("unparsed" if found_source else "not_calculated"), []


def _parse_crest_energy_table(run_dir: Path) -> dict[str, Any]:
    energy_path = run_dir / "crest.energies"
    if not energy_path.is_file():
        return {"property_state": "not_calculated", "path": "", "rows": []}
    rows: list[dict[str, Any]] = []
    for line in _read_text(energy_path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        numbers: list[float] = []
        for token in re.findall(_NUMBER, stripped):
            try:
                numbers.append(float(token))
            except Exception:
                pass
        if numbers:
            rows.append(
                {
                    "conformer_index": len(rows) + 1,
                    "frame_index": len(rows),
                    "values": numbers,
                    "energy": numbers[-1],
                }
            )
    candidates: list[Path] = []
    for name in ("crest_conformers.xyz", "crest_ensemble.xyz"):
        candidate = run_dir / name
        if candidate.is_file():
            candidates.append(candidate)
    counts = {path: _xyz_frame_count(path) for path in candidates}
    matches = [path for path in candidates if counts[path] == len(rows)]
    ensemble_path = matches[0] if len(matches) == 1 else None
    state = "calculated" if rows and ensemble_path is not None else "invalid"
    return {
        "property_state": state,
        "path": workspace_relpath(energy_path),
        "ensemble_path": workspace_relpath(ensemble_path) if ensemble_path is not None else "",
        "ensemble_frames": counts.get(ensemble_path) if ensemble_path is not None else None,
        "rows": rows,
    }


def _parse_xtb_run(run_dir: Path, output_file: Path | None = None, output_dir: Path | None = None, geometry_file: Path | None = None) -> dict[str, Any]:
    xtb_summary = _read_json(run_dir / "xtb_summary.json")
    crest_summary = _read_json(run_dir / "crest_summary.json")
    summary = crest_summary or xtb_summary
    engine = "crest" if crest_summary else "xtb"
    log_name = str(summary.get("log_file") or ("crest_stdout.out" if engine == "crest" else "xtb_stdout.out"))
    log_path = output_file or ((run_dir / log_name) if (run_dir / log_name).is_file() else _find_run_output(run_dir, "xtb"))
    text = _read_text(log_path) if log_path else ""
    if not summary and ("CREST" in text.upper() or (log_path and "crest" in log_path.name.lower())):
        engine = "crest"
    returncode = summary.get("returncode")
    normal_termination = bool(summary.get("normal_termination")) or (
        "normal termination of xtb" in text.lower() or "crest terminated normally" in text.lower()
    )
    execution_state = (
        "failed"
        if returncode not in (None, 0)
        else ("completed" if returncode == 0 or normal_termination else "unknown")
    )
    not_converged = "NOT_CONVERGED" in text.upper() or bool(summary.get("not_converged"))
    optimization_converged = engine == "xtb" and (
        "GEOMETRY OPTIMIZATION CONVERGED" in text.upper() or (run_dir / "xtbopt.xyz").is_file()
    )
    task_state = (
        "not_converged"
        if not_converged
        else (
            "converged"
            if optimization_converged
            else ("incomplete" if execution_state == "failed" else ("not_applicable" if execution_state == "completed" else "unknown"))
        )
    )
    frequency_state, frequencies = _parse_frequency_files(run_dir)
    final_structure = None
    structure_names = (
        ("crest_best.xyz", "crest_conformers.xyz", "crest_ensemble.xyz", "xtbopt.xyz")
        if engine == "crest"
        else ("xtbopt.xyz", "xtblast.xyz", "xtb.trj")
    )
    for path in ([geometry_file] if geometry_file else [run_dir / name for name in structure_names]):
        if path.is_file():
            final_structure = _last_xyz_frame(path, output_dir)
            break
    energy = _extract_last(_XTB_ENERGY_PATTERNS, text)
    free_energy = _extract_last(_XTB_FREE_ENERGY_PATTERNS, text)
    issues: list[str] = []
    if energy is None and engine == "xtb":
        issues.append("total_energy_unparsed")
    if not_converged:
        issues.append("not_converged")
    if returncode not in (None, 0):
        issues.append(f"nonzero_returncode:{returncode}")
    return {
        "result_path": workspace_relpath(run_dir),
        "engine": engine,
        "output_file": workspace_relpath(log_path) if log_path else "",
        "execution_state": execution_state,
        "task_state": task_state,
        "normal_termination": normal_termination,
        "energy_state": "calculated" if energy is not None else "unparsed",
        "energy_hartree": energy,
        "free_energy_state": "calculated" if free_energy is not None else "not_calculated",
        "free_energy_hartree": free_energy,
        "frequency_state": frequency_state,
        "imaginary_frequency_count": sum(value < 0.0 for value in frequencies) if frequencies else None,
        "frequencies_cm-1": frequencies,
        "conformer_energies": _parse_crest_energy_table(run_dir) if engine == "crest" else None,
        "final_structure": final_structure,
        "final_structure_frame": -1,
        "structure_state": "available" if final_structure else "unparsed",
        "issues": issues,
    }


def _try_cclib_parse(out_path: Path) -> dict[str, Any]:
    try:
        import cclib  # type: ignore

        data = cclib.io.ccread(str(out_path))
    except Exception:
        return {}
    out: dict[str, Any] = {}
    try:
        if getattr(data, "scfenergies", None) is not None and len(data.scfenergies) > 0:
            out["scf_energy_hartree"] = float(data.scfenergies[-1]) / 27.211386245988
    except Exception:
        pass
    try:
        if getattr(data, "vibfreqs", None) is not None:
            out["frequencies_cm-1"] = [float(value) for value in data.vibfreqs]
    except Exception:
        pass
    return out


def _orca_property_result(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Read the documented final-geometry ORCA property-JSON energy fields."""

    candidates: list[Path] = []
    runtime_input = str(summary.get("runtime_input") or "").strip()
    if runtime_input:
        expected = run_dir / f"{Path(runtime_input).stem}.property.json"
        if expected.is_file():
            candidates.append(expected)
    if not candidates:
        discovered = sorted(run_dir.glob("*.property.json"))
        if len(discovered) == 1:
            candidates = discovered
    if not candidates:
        return {}

    path = candidates[0]
    payload = _read_json(path)
    geometries = payload.get("Geometries")
    if not isinstance(geometries, list):
        return {"property_json_path": workspace_relpath(path)}

    final_energy: float | None = None
    final_converged: bool | None = None
    final_scf_energy: float | None = None
    for geometry in geometries:
        if not isinstance(geometry, dict):
            continue
        single_point = geometry.get("Single_Point_Data")
        if isinstance(single_point, dict):
            value = single_point.get("finalenergy")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                final_energy = float(value)
            converged = single_point.get("converged")
            if isinstance(converged, bool):
                final_converged = converged
        scf_energy = geometry.get("SCF_Energy")
        if isinstance(scf_energy, dict):
            value = scf_energy.get("totalEnergy")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                final_scf_energy = float(value)
    return {
        "property_json_path": workspace_relpath(path),
        "final_energy_hartree": final_energy,
        "single_point_converged": final_converged,
        "scf_energy_hartree": final_scf_energy,
    }


def _orca_simple_keywords(run_dir: Path, input_file: Path | None = None) -> set[str]:
    text = _read_text(input_file or (run_dir / "job.inp"))
    keywords: set[str] = set()
    for line in text.splitlines():
        if line.lstrip().startswith("!"):
            keywords.update(token.lower() for token in line.lstrip()[1:].split())
    return keywords


def _parse_orca_shieldings(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active = False
    remaining = 0
    for line in text.splitlines():
        upper = line.upper()
        if "CHEMICAL SHIELDING SUMMARY" in upper:
            active = True
            remaining = 200
            continue
        if not active:
            continue
        remaining -= 1
        if remaining <= 0:
            break
        match = _ORCA_SHIELDING_ROW_RE.match(line)
        if match:
            rows.append(
                {
                    "atom_index": int(match.group(1)),
                    "element": match.group(2),
                    "isotropic_shielding_ppm": float(match.group(3)),
                }
            )
        elif rows and not line.strip():
            break
    return rows


def _parse_orca_run(run_dir: Path, output_file: Path | None = None, output_dir: Path | None = None, geometry_file: Path | None = None, input_file: Path | None = None) -> dict[str, Any]:
    summary = _read_json(run_dir / "orca_summary.json")
    out_path = output_file or _find_run_output(run_dir, "orca") or (run_dir / "job.out")
    text = _read_text(out_path)
    cclib_data = _try_cclib_parse(out_path)
    property_data = _orca_property_result(run_dir, summary)
    final_energy = property_data.get("final_energy_hartree")
    energy_source = "property_json"
    if final_energy is None:
        final_energy = _extract_last((_ORCA_FINAL_ENERGY_RE,), text)
        energy_source = "output_text"
    if final_energy is None:
        final_energy = cclib_data.get("scf_energy_hartree")
        energy_source = "cclib_scf"
    if final_energy is None:
        final_energy = property_data.get("scf_energy_hartree")
        energy_source = "property_json_scf"
    if final_energy is None:
        energy_source = ""
    frequencies = list(cclib_data.get("frequencies_cm-1") or [])
    if not frequencies:
        for line in text.splitlines():
            match = _ORCA_FREQ_RE.search(line)
            if match:
                try:
                    frequencies.append(float(match.group(1)))
                except Exception:
                    pass
    keywords = _orca_simple_keywords(run_dir, input_file or (out_path.with_suffix(".inp") if out_path.with_suffix(".inp").is_file() else None))
    frequency_requested = any(token in keywords for token in {"freq", "numfreq", "anfreq"})
    optimization_requested = any(
        token.startswith("opt") or token in {"tightopt", "verytightopt"}
        for token in keywords
    )
    path_task_requested = any(token.startswith("neb-") or token == "irc" for token in keywords)
    returncode = summary.get("returncode")
    normal_termination = bool(summary.get("normal_termination")) or "ORCA TERMINATED NORMALLY" in text
    execution_state = (
        "failed"
        if returncode not in (None, 0)
        else ("completed" if returncode == 0 or normal_termination else "unknown")
    )
    property_converged = property_data.get("single_point_converged")
    scf_failed = property_converged is False or "SCF NOT CONVERGED" in text.upper()
    scf_converged = bool(
        property_converged is True
        or "SCF CONVERGED AFTER" in text.upper()
        or (normal_termination and final_energy is not None and not scf_failed)
    )
    optimization_converged = (
        "THE OPTIMIZATION HAS CONVERGED" in text.upper()
        or "OPTIMIZATION RUN DONE" in text.upper()
        or "HURRAY" in text.upper()
    )
    if optimization_requested or optimization_converged:
        task_state = "converged" if optimization_converged else (
            "not_converged" if execution_state == "completed" else ("incomplete" if execution_state == "failed" else "unknown")
        )
    elif path_task_requested:
        task_state = "incomplete" if execution_state == "failed" else "unknown"
    else:
        task_state = "not_converged" if scf_failed else "not_applicable"
    shielding_rows = _parse_orca_shieldings(text)
    final_structure = None
    geometry_candidates = [geometry_file] if geometry_file else list(dict.fromkeys([
        out_path.with_suffix(".xyz"), run_dir / f"{out_path.stem}_trj.xyz",
        run_dir / "job.runtime.xyz", run_dir / "job.xyz", run_dir / "job.runtime_trj.xyz", run_dir / "job_trj.xyz",
        run_dir / "job.runtime_IRC_Full_trj.xyz", run_dir / "job_IRC_Full_trj.xyz",
    ]))
    for path in geometry_candidates:
        if path.is_file():
            final_structure = _last_xyz_frame(path, output_dir)
            break
    issues: list[str] = []
    if final_energy is None:
        issues.append("final_energy_unparsed")
    if execution_state != "completed":
        issues.append("orca_not_normal_termination")
    if optimization_requested and not optimization_converged:
        issues.append("optimization_not_converged")
    return {
        "result_path": workspace_relpath(run_dir),
        "output_file": workspace_relpath(out_path),
        "execution_state": execution_state,
        "task_state": task_state,
        "normal_termination": normal_termination,
        "scf_state": "not_converged" if scf_failed else ("converged" if scf_converged else "unknown"),
        "energy_state": "calculated" if final_energy is not None else "unparsed",
        "final_energy_hartree": final_energy,
        "energy_source": energy_source,
        "property_json_path": str(property_data.get("property_json_path") or ""),
        "gibbs_free_energy_hartree": _extract_last((_ORCA_GIBBS_RE,), text),
        "enthalpy_hartree": _extract_last((_ORCA_ENTHALPY_RE,), text),
        "zpe_hartree": _extract_last((_ORCA_ZPE_RE,), text),
        "frequency_state": "calculated" if frequencies else ("unparsed" if frequency_requested else "not_calculated"),
        "imaginary_frequency_count": sum(value < 0.0 for value in frequencies) if frequencies else None,
        "frequencies_cm-1": frequencies,
        "nmr_state": "calculated" if shielding_rows else ("unparsed" if "nmr" in keywords else "not_calculated"),
        "nmr_isotropic_shieldings": shielding_rows,
        "final_structure": final_structure,
        "final_structure_frame": -1,
        "structure_state": "available" if final_structure else "unparsed",
        "issues": issues,
    }


class AnalyzeXtbResultsInput(BaseModel):
    """[xtb/analyze] Summarize xTB/CREST execution, convergence, energy, frequency, and ensemble states."""

    model_config = ConfigDict(extra="forbid")

    result_root: str = Field("", description="xTB/CREST output file, run directory or recursive batch root; omit with result_files.")
    result_files: list[str] = Field(default_factory=list, description="Explicit native output files of any basename, independently parsed; use instead of result_root.")
    geometry_file: str = Field("", description="Explicit final geometry/trajectory for a single run; omit for known engine filenames. Extraction writes only to output_dir.")
    output_dir: str = Field("", description="Summary/extracted-geometry directory; omit for analysis next to the source. Use independent destinations for concurrent analyses.")

    @model_validator(mode="before")
    @classmethod
    def _legacy_null_output(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("output_dir") is None:
            return {**data, "output_dir": ""}
        return data


class AnalyzeOrcaResultsInput(BaseModel):
    """[orca/analyze] Summarize ORCA process, task, energy, frequency, and shielding states without invented values."""

    model_config = ConfigDict(extra="forbid")

    result_root: str = Field("", description="ORCA output file, run directory or recursive batch root; omit with result_files.")
    input_file: str = Field("", description="Explicit native input for task keywords in a single-run analysis; otherwise try the output basename and job.inp.")
    result_files: list[str] = Field(default_factory=list, description="Explicit native output files of any basename, independently parsed; use instead of result_root.")
    geometry_file: str = Field("", description="Explicit final geometry/trajectory for a single run; omit for known engine filenames. Extraction writes only to output_dir.")
    output_dir: str = Field("", description="Summary/extracted-geometry directory; omit for analysis next to the source. Use independent destinations for concurrent analyses.")

    @model_validator(mode="before")
    @classmethod
    def _legacy_null_output(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("output_dir") is None:
            return {**data, "output_dir": ""}
        return data


def _analyze_results(params, engine: str) -> tuple[str, dict[str, Any]]:
    tool_name = f"analyze_{engine}_results"
    markers = ("orca_summary.json",) if engine == "orca" else ("xtb_summary.json", "crest_summary.json")
    root, sources = analysis_sources(params.result_root, params.result_files, lambda path: _discover_runs(path, markers, tool_name=tool_name))
    if not sources:
        _tool_error(tool_name, f"No {engine} results found; provide explicit result_files for custom output layouts", error_code="no_runs")
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else ((root if root.is_dir() else root.parent) / "analysis")
    if len(sources) > 1 and (params.geometry_file or getattr(params, "input_file", "")):
        raise ValueError("geometry_file/input_file apply to one run; split calls for different per-run overrides")
    geometry_file = resolve_workspace_path(params.geometry_file, must_exist=True) if params.geometry_file else None
    input_file = resolve_workspace_path(params.input_file, must_exist=True) if getattr(params, "input_file", "") else None
    records = []
    for index, (directory, output_file) in enumerate(sources):
        try:
            if output_file and not output_file.is_file():
                raise ValueError(f"Output file does not exist: {workspace_relpath(output_file)}")
            kwargs = {"output_file": output_file, "output_dir": output_dir / f"run_{index:04d}", "geometry_file": geometry_file}
            record = _parse_orca_run(directory, input_file=input_file, **kwargs) if engine == "orca" else _parse_xtb_run(directory, **kwargs)
            records.append(record)
        except Exception as exc:
            records.append({"result_path": workspace_relpath(directory), "output_file": workspace_relpath(output_file) if output_file else "",
                            "parse_state": "failed", "issues": [str(exc)]})
    failed_count = sum(record.get("parse_state") == "failed" for record in records)
    state = "failed" if failed_count == len(records) else "partial" if failed_count else "completed"
    summary_json = output_dir / f"{engine}_results_summary.json"
    summary_csv = output_dir / f"{engine}_results_summary.csv"
    _write_json(summary_json, {"result_root": workspace_relpath(root), "state": state, "records": records})
    _write_csv(summary_csv, records)
    data = {"result_root": workspace_relpath(root), "summary_json": workspace_relpath(summary_json),
            "summary_csv": workspace_relpath(summary_csv), "count": len(records), "parse_failed_count": failed_count, "state": state}
    content = (f"{tool_name} {state}.\ncount={len(records)} parse_failed_count={failed_count} "
               f"summary_json={data['summary_json']} summary_csv={data['summary_csv']}")
    return content, {"tool_name": tool_name, "data": data}


def analyze_xtb_results(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[xtb/analyze] Summarize xTB/CREST result states without automatic cross-run relative energies."""
    return _analyze_results(AnalyzeXtbResultsInput(**payload), "xtb")


def analyze_orca_results(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[orca/analyze] Summarize ORCA result states without turning absent properties into zero."""
    return _analyze_results(AnalyzeOrcaResultsInput(**payload), "orca")


__all__ = [
    "AnalyzeXtbResultsInput",
    "AnalyzeOrcaResultsInput",
    "analyze_xtb_results",
    "analyze_orca_results",
]
