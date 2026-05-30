from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.geometry_inputs.cp2k_common import CP2K_REFERENCE_URLS, write_json


class Cp2kOutputSummaryInput(BaseModel):
    """[cp2k/analysis] Summarize reusable CP2K run evidence from result directories without task-specific property interpretation."""

    model_config = ConfigDict(extra="forbid")

    result_root: str = Field(..., description="CP2K result directory or batch result root.")
    output_dir: str | None = Field(None, description="Summary output directory.")


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
_RUN_TYPE_RE = re.compile(r"\bRUN_TYPE\s+([A-Za-z0-9_]+)", re.IGNORECASE)
_PROGRAM_MARKERS = ("PROGRAM ENDED AT", "CP2K| version string:", "CP2K")
_ENERGY_SUFFIXES = {".ener", ".energy"}


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
    for name in ("job.out", "cp2k.out", "cp2k_stdout.out"):
        path = run_dir / name
        if path.is_file():
            return path
    outputs = sorted(path for path in run_dir.glob("*.out") if path.is_file())
    if outputs:
        return outputs[0]
    return None


def _discover_cp2k_result_dirs(root: Path) -> list[Path]:
    if (root / "cp2k_summary.json").is_file() or _find_cp2k_output_file(root) is not None:
        return [root]
    out: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and ((path / "cp2k_summary.json").is_file() or _find_cp2k_output_file(path) is not None):
            out.append(path)
    return sorted(out)


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
    labels = {
        "max_gradient": ("MAX. GRADIENT", "MAXIMUM GRADIENT", "MAX_FORCE", "MAXIMUM FORCE"),
        "rms_gradient": ("RMS GRADIENT", "RMS_FORCE"),
        "max_step": ("MAX. STEP", "MAX_DR"),
        "rms_step": ("RMS STEP", "RMS_DR"),
    }
    for line in lines:
        upper = line.upper()
        for key, needles in labels.items():
            if key in metrics:
                continue
            if any(needle in upper for needle in needles):
                value = _last_float(line)
                if value is not None:
                    metrics[key] = value
    return {
        "completed": "OPTIMIZATION COMPLETED" in text or "GEOMETRY OPTIMIZATION COMPLETED" in text or "CELL OPTIMIZATION COMPLETED" in text,
        "convergence_metrics": metrics,
    }


def _parse_cp2k_frequencies(lines: list[str]) -> dict[str, Any]:
    values: list[float] = []
    for line in lines:
        upper = line.upper()
        if "CM" not in upper and "FREQUENCY" not in upper and "VIB" not in upper:
            continue
        if "FREQ" not in upper and "CM" not in upper:
            continue
        for value in _numeric_values(line):
            if -10000.0 < value < 10000.0:
                values.append(value)
    if not values:
        return {"count": 0, "imaginary_count": 0}
    return {
        "count": len(values),
        "imaginary_count": sum(1 for value in values if value < 0),
        "min_cm-1": min(values),
        "max_cm-1": max(values),
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


def _parse_cp2k_run(run_dir: Path) -> dict[str, Any]:
    wrapper = _read_json(run_dir / "cp2k_summary.json")
    output_file = _find_cp2k_output_file(run_dir)
    lines = output_file.read_text(encoding="utf-8", errors="replace").splitlines() if output_file else []
    text_upper = "\n".join(lines).upper()
    warnings = [line.strip() for line in lines if "WARNING" in line.upper()]
    errors = [line.strip() for line in lines if "ERROR" in line.upper() or "ABORT" in line.upper()]
    completed = bool(wrapper.get("completed")) if wrapper else False
    if lines:
        completed = completed or ("PROGRAM ENDED AT" in text_upper and "ABORT" not in text_upper)
    run_type = _run_type_from_output(lines) or _run_type_from_input(run_dir)
    record: dict[str, Any] = {
        "result_dir_rel": workspace_relpath(run_dir),
        "output_file_rel": workspace_relpath(output_file) if output_file else "",
        "completed": completed,
        "returncode": wrapper.get("returncode"),
        "run_type": run_type,
        "warnings": warnings[:20],
        "errors": errors[:20],
        "scf": {
            "converged_count": sum(1 for line in lines if "SCF RUN CONVERGED" in line.upper()),
            "not_converged_count": sum(1 for line in lines if "SCF RUN NOT CONVERGED" in line.upper()),
        },
        "energies": _parse_cp2k_energy_lines(lines),
        "optimization": _parse_cp2k_optimization(lines),
        "frequencies": _parse_cp2k_frequencies(lines),
        "energy_files": _energy_file_summaries(run_dir),
        "output_files": _output_files(run_dir),
        "references": CP2K_REFERENCE_URLS,
    }
    if wrapper:
        record["wrapper"] = {
            key: wrapper.get(key)
            for key in ("normal_completion", "mpi_ranks", "omp_num_threads", "log_file")
            if key in wrapper
        }
    return record


def cp2k_output_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "cp2k_output_summary"
    params = Cp2kOutputSummaryInput(**payload)
    result_root = resolve_workspace_path(params.result_root, must_exist=True)
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else result_root.parent / f"{result_root.name}_cp2k_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_cp2k_result_dirs(result_root)
    if not runs:
        _tool_error(
            tool_name,
            "No CP2K result directories found.",
            data={"result_root_rel": workspace_relpath(result_root)},
            error_code="no_cp2k_runs",
        )
    records = [_parse_cp2k_run(run) for run in runs]
    summary = {
        "result_root_rel": workspace_relpath(result_root),
        "runs_analyzed": len(records),
        "completed_count": sum(1 for record in records if record.get("completed")),
        "records": records,
    }
    summary_path = output_dir / "cp2k_output_summary.json"
    write_json(summary_path, summary)
    data = {
        "result_root_rel": workspace_relpath(result_root),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_path),
        "runs_analyzed": len(records),
        "completed_count": summary["completed_count"],
    }
    content = (
        "cp2k_output_summary completed.\n"
        f"runs_analyzed={len(records)} completed_count={summary['completed_count']} "
        f"summary_json_rel={data['summary_json_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


__all__ = ["Cp2kOutputSummaryInput", "cp2k_output_summary", "parse_cp2k_energy_file"]
