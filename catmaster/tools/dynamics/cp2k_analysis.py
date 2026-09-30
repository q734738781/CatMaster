from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.analysis.output_sources import analysis_sources, discover_directories, select_output


class Cp2kOutputSummaryInput(BaseModel):
    """[cp2k/analysis] Summarize reusable CP2K run evidence from result directories without task-specific property interpretation."""

    model_config = ConfigDict(extra="forbid")

    result_root: str = Field("", description="Explicit CP2K output file, single result directory or recursive batch root. Omit when using result_files.")
    result_files: list[str] = Field(default_factory=list, description="Explicit output files of any basename; each is parsed independently. Use instead of result_root.")
    output_dir: str = Field("", description="Summary output directory; leave empty to create one next to result_root.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_null_output_dir(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("output_dir") is None:
            return {**data, "output_dir": ""}
        return data


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
_RUN_TYPE_RE = re.compile(r"\bRUN_TYPE\s+([A-Za-z0-9_]+)", re.IGNORECASE)
_ENERGY_SUFFIXES = {".ener", ".energy"}
_FREQUENCY_UNIT_RE = re.compile(r"CM\s*\^?\s*-?1|CM\*\*-1|CM-1", re.IGNORECASE)
_OPTIMIZATION_METRIC_PATTERNS = {
    "max_gradient": re.compile(rf"^\s*Max\.\s+gradient\s*=\s*({_FLOAT_RE.pattern})", re.IGNORECASE),
    "rms_gradient": re.compile(rf"^\s*RMS\s+gradient\s*=\s*({_FLOAT_RE.pattern})", re.IGNORECASE),
    "max_step": re.compile(rf"^\s*Max\.\s+step\s+size\s*=\s*({_FLOAT_RE.pattern})", re.IGNORECASE),
    "rms_step": re.compile(rf"^\s*RMS\s+step\s+size\s*=\s*({_FLOAT_RE.pattern})", re.IGNORECASE),
}


def _tool_error(tool_name: str, message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _last_float(text: str) -> float | None:
    matches = _FLOAT_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _numeric_values(text: str) -> list[float]:
    out: list[float] = []
    for token in _FLOAT_RE.findall(text):
        try:
            out.append(float(token))
        except Exception:
            continue
    return out


def _find_cp2k_output_file(run_dir: Path) -> Path | None:
    return select_output(run_dir, names=("job.out", "cp2k.out", "cp2k_stdout.out"),
                         signals=("CP2K|", "ENERGY|", "PROGRAM STARTED AT", "PROGRAM ENDED AT"))


def _discover_cp2k_result_dirs(root: Path) -> list[Path]:
    return discover_directories(root, ("cp2k_summary.json",), _find_cp2k_output_file)


def _run_type_from_input(run_dir: Path) -> str:
    input_path = run_dir / "job.inp"
    if not input_path.is_file():
        return ""
    text = input_path.read_text(encoding="utf-8", errors="replace")
    match = _RUN_TYPE_RE.search(text)
    return match.group(1).upper() if match else ""


def _run_type_from_output(lines: list[str]) -> str:
    for line in lines:
        if "Run type" in line:
            match = re.search(r"Run type\s+([A-Za-z0-9_]+)", line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    return ""


def _parse_cp2k_energy_lines(lines: list[str]) -> dict[str, Any]:
    energies: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "ENERGY|" not in upper:
            continue
        value = _last_float(line)
        if value is None:
            continue
        energies.append({"line": idx + 1, "hartree": value, "text": line.strip()[:240]})
    out: dict[str, Any] = {"count": len(energies)}
    if energies:
        out["last"] = energies[-1]
        out["first"] = energies[0]
    return out


def _parse_cp2k_optimization(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines).upper()
    metrics: dict[str, float] = {}
    for line in lines:
        for key, pattern in _OPTIMIZATION_METRIC_PATTERNS.items():
            match = pattern.match(line)
            if match is not None:
                metrics[key] = float(match.group(1))
    converged = (
        "OPTIMIZATION COMPLETED" in text
        or "GEOMETRY OPTIMIZATION COMPLETED" in text
        or "CELL OPTIMIZATION COMPLETED" in text
    )
    requested = any(token in text for token in ("GEOMETRY OPTIMIZATION", "CELL OPTIMIZATION")) or bool(metrics)
    return {
        "state": "converged" if converged else ("not_converged" if requested else "not_applicable"),
        "convergence_metrics": metrics,
    }


def _parse_cp2k_frequencies(lines: list[str], *, requested: bool) -> dict[str, Any]:
    values: list[float] = []
    for line in lines:
        unit = _FREQUENCY_UNIT_RE.search(line)
        if unit is None:
            continue
        values.extend(_numeric_values(line[unit.end() :]))
    if not values:
        return {
            "property_state": "unparsed" if requested else "not_calculated",
            "count": None,
            "imaginary_count": None,
            "values_cm-1": [],
        }
    return {
        "property_state": "calculated",
        "count": len(values),
        "imaginary_count": sum(1 for value in values if value < 0),
        "min_cm-1": min(values),
        "max_cm-1": max(values),
        "values_cm-1": values,
    }


def parse_cp2k_energy_file(path: Path) -> dict[str, Any]:
    rows: list[list[float]] = []
    comments: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line.lstrip("#").strip())
            continue
        values = _numeric_values(line)
        if values:
            rows.append(values)
    out: dict[str, Any] = {
        "path_rel": workspace_relpath(path),
        "rows": len(rows),
        "comments": comments[:3],
    }
    if not rows:
        return out
    first = rows[0]
    last = rows[-1]
    out["first_row"] = first
    out["last_row"] = last
    if len(last) >= 1:
        out["step_start"] = first[0]
        out["step_end"] = last[0]
    if len(last) >= 2:
        out["time_start"] = first[1]
        out["time_end"] = last[1]
        out["time_span"] = last[1] - first[1]
    if len(last) >= 4:
        out["temperature_start"] = first[3]
        out["temperature_end"] = last[3]
        out["temperature_drift"] = last[3] - first[3]
    if len(last) >= 5:
        out["potential_start"] = first[4]
        out["potential_end"] = last[4]
        out["potential_drift"] = last[4] - first[4]
    if len(last) >= 6:
        out["conserved_start"] = first[5]
        out["conserved_end"] = last[5]
        out["conserved_drift"] = last[5] - first[5]
    return out


def _energy_file_summaries(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and (path.suffix.lower() in _ENERGY_SUFFIXES or path.name.lower().endswith(".ener")):
            summaries.append(parse_cp2k_energy_file(path))
    return summaries


def _output_files(run_dir: Path) -> list[str]:
    suffixes = {".out", ".xyz", ".ener", ".restart", ".wfn", ".pdos", ".cube", ".bs", ".dat", ".log"}
    names: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and (path.suffix.lower() in suffixes or path.name in {"cp2k_summary.json", "status.json"}):
            names.append(workspace_relpath(path))
    return names


def _parse_cp2k_run(run_dir: Path, output_file: Path | None = None) -> dict[str, Any]:
    wrapper = _read_json(run_dir / "cp2k_summary.json")
    output_file = output_file or _find_cp2k_output_file(run_dir)
    lines = output_file.read_text(encoding="utf-8", errors="replace").splitlines() if output_file else []
    text_upper = "\n".join(lines).upper()
    warnings = [line.strip() for line in lines if "WARNING" in line.upper()]
    errors = [line.strip() for line in lines if "ERROR" in line.upper() or "ABORT" in line.upper()]
    normal_termination = bool(lines and "PROGRAM ENDED AT" in text_upper and "ABORT" not in text_upper)
    returncode = wrapper.get("returncode")
    if normal_termination:
        execution_state = "completed"
    elif returncode not in (None, 0) or "ABORT" in text_upper:
        execution_state = "failed"
    else:
        execution_state = "unknown"
    run_type = _run_type_from_output(lines) or _run_type_from_input(run_dir)
    scf_converged = sum(1 for line in lines if "SCF RUN CONVERGED" in line.upper())
    scf_not_converged = sum(1 for line in lines if "SCF RUN NOT CONVERGED" in line.upper())
    optimization = _parse_cp2k_optimization(lines)
    if run_type in {"GEO_OPT", "CELL_OPT"}:
        if optimization["state"] == "converged":
            task_state = "converged"
        elif execution_state == "failed":
            task_state = "incomplete"
        elif execution_state == "completed":
            task_state = "not_converged"
        else:
            task_state = "unknown"
    elif execution_state == "failed":
        task_state = "incomplete"
    elif scf_not_converged:
        task_state = "not_converged"
    elif execution_state == "completed":
        task_state = "not_applicable"
    else:
        task_state = "unknown"
    record: dict[str, Any] = {
        "result_dir_rel": workspace_relpath(run_dir),
        "output_file_rel": workspace_relpath(output_file) if output_file else "",
        "execution_state": execution_state,
        "task_state": task_state,
        "normal_termination": normal_termination,
        "returncode": returncode,
        "run_type": run_type,
        "warnings": warnings,
        "errors": errors,
        "scf": {
            "state": "not_converged" if scf_not_converged else ("converged" if scf_converged else "unknown"),
            "converged_count": scf_converged,
            "not_converged_count": scf_not_converged,
        },
        "energies": _parse_cp2k_energy_lines(lines),
        "optimization": optimization,
        "frequencies": _parse_cp2k_frequencies(
            lines,
            requested=run_type == "VIBRATIONAL_ANALYSIS" or "VIBRATIONAL ANALYSIS" in text_upper,
        ),
        "energy_files": _energy_file_summaries(run_dir),
        "output_files": _output_files(run_dir),
    }
    return record


def cp2k_output_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "cp2k_output_summary"
    params = Cp2kOutputSummaryInput(**payload)
    result_root, sources = analysis_sources(params.result_root, params.result_files, _discover_cp2k_result_dirs)
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir.strip() else result_root.parent / f"{result_root.name}_cp2k_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not sources:
        _tool_error(
            tool_name,
            "No CP2K result directories found.",
            data={"result_root_rel": workspace_relpath(result_root)},
            error_code="no_cp2k_runs",
        )
    records = []
    for directory, source_file in sources:
        try:
            records.append(_parse_cp2k_run(directory, source_file))
        except Exception as exc:
            records.append({"result_path": workspace_relpath(directory), "output_file_rel": workspace_relpath(source_file) if source_file else "",
                            "parse_state": "failed", "errors": [str(exc)]})
    failed_count = sum(record.get("parse_state") == "failed" for record in records)
    summary = {
        "result_root_rel": workspace_relpath(result_root),
        "runs_analyzed": len(records),
        "execution_completed_count": sum(1 for record in records if record.get("execution_state") == "completed"),
        "records": records,
    }
    summary_path = output_dir / "cp2k_output_summary.json"
    _write_json(summary_path, summary)
    data = {
        "result_root_rel": workspace_relpath(result_root),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_path),
        "runs_analyzed": len(records),
        "execution_completed_count": summary["execution_completed_count"],
    }
    content = (
        f"cp2k_output_summary {'failed' if failed_count == len(records) else 'partial' if failed_count else 'completed'}; parse_failed_count={failed_count}.\n"
        f"runs_analyzed={len(records)} execution_completed_count={summary['execution_completed_count']} "
        f"summary_json_rel={data['summary_json_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


__all__ = ["Cp2kOutputSummaryInput", "cp2k_output_summary", "parse_cp2k_energy_file"]
