from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ase.io.trajectory import Trajectory
from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.analysis.output_sources import analysis_sources, discover_directories, select_output
from catmaster.tools.dynamics.cp2k_analysis import parse_cp2k_energy_file
from catmaster.tools.geometry_inputs.native_stage import (
    StageAssetMapping,
    copy_stage_assets,
    require_fresh_stage,
    resolve_stage_assets,
    write_stage_manifest,
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


class LammpsPrepareInput(BaseModel):
    """[lammps/prepare] Stage one complete agent-authored LAMMPS script and its explicit input assets."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        ...,
        description="Workspace-relative path to the complete native LAMMPS script that will become in.lammps.",
    )
    output_root: str = Field(..., description="Fresh empty directory for the prepared LAMMPS stage.")
    asset_mappings: list[StageAssetMapping] = Field(
        default_factory=list,
        description=(
            "Explicit data, restart, potential, table, and other input files copied as source_path to stage_path. "
            "Omit or pass [] when the script has no external files."
        ),
    )


class LammpsLogSummaryInput(BaseModel):
    """[lammps/analysis] Separate LAMMPS process completion, minimizer state, and parsed thermo availability."""

    model_config = ConfigDict(extra="forbid")

    result_root: str = Field("", description="Explicit LAMMPS log file, single result directory or recursive batch root. Omit when using result_files.")
    result_files: list[str] = Field(default_factory=list, description="Explicit log files of any basename, parsed independently; use instead of result_root.")
    output_dir: str = Field("", description="Summary directory; leave empty to create one next to result_root.")


class MdTrajectorySummaryInput(BaseModel):
    """[md/analysis] Inventory MD trajectory frames and native observable files without inferring diffusion or RDF."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "One LAMMPS dump, XYZ, or ASE .traj file, or a result directory containing exactly one such trajectory."
        ),
    )
    observable_files: list[str] = Field(default_factory=list, description="Explicit native numeric tables of any basename; omit to discover rdf.dat/msd.dat next to the trajectory.")
    output_dir: str = Field("", description="Summary directory; leave empty to create one next to the trajectory result.")


def lammps_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[lammps/prepare] Stage one complete agent-authored LAMMPS script and its explicit input assets."""
    tool_name = "lammps_prepare"
    try:
        params = LammpsPrepareInput(**payload)
        source = resolve_workspace_path(params.input_path, must_exist=True)
        if not source.is_file():
            raise ValueError(f"input_path must name a file: {workspace_relpath(source)}")
        stage_dir = resolve_workspace_path(params.output_root)
        assets = resolve_stage_assets(
            params.asset_mappings,
            reserved_paths=("in.lammps", "manifest.json"),
        )
        require_fresh_stage(stage_dir, source_paths=[source, *(item[0] for item in assets)])
        stage_dir.mkdir(parents=True, exist_ok=True)
        canonical_input = stage_dir / "in.lammps"
        shutil.copy2(source, canonical_input)
        copied_assets = copy_stage_assets(stage_dir, assets)
        manifest = write_stage_manifest(
            stage_dir,
            {"input_file": "in.lammps", "input_assets": copied_assets},
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _tool_error(
            tool_name,
            f"lammps_prepare failed: {exc}",
            data={"input_path": str(payload.get("input_path") or ""), "output_root": str(payload.get("output_root") or "")},
            error_code="lammps_prepare_failed",
        )

    data = {
        "stage_path": workspace_relpath(stage_dir),
        "input_path": workspace_relpath(canonical_input),
        "manifest_path": workspace_relpath(manifest),
        "input_assets": copied_assets,
    }
    content = (
        "lammps_prepare completed.\n"
        f"stage_path={data['stage_path']} input_path={data['input_path']}\n"
        f"manifest_path={data['manifest_path']} input_assets={len(copied_assets)}"
    )
    return content, {"tool_name": tool_name, "data": data}


_THERMO_START_RE = re.compile(r"^\s*Step(?:\s+|$)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for token in _NUMBER_RE.findall(text):
        try:
            values.append(float(token))
        except Exception:
            continue
    return values


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _thermo_segment(rows: list[dict[str, float]]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    out: dict[str, Any] = {"rows": len(rows), "first": first, "last": last}
    if "Step" in first and "Step" in last:
        out["step_start"] = first["Step"]
        out["step_end"] = last["Step"]
    return out


def _thermo_drift(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) < 2:
        return {}
    first = rows[0]
    last = rows[-1]
    aliases = {
        "temperature": ("Temp", "temp"),
        "potential_energy": ("PotEng", "pe", "PE"),
        "total_energy": ("TotEng", "etotal", "E_pair"),
        "pressure": ("Press", "press"),
        "volume": ("Vol", "vol"),
    }
    drift: dict[str, float] = {}
    for output_key, candidates in aliases.items():
        key = next((candidate for candidate in candidates if candidate in first and candidate in last), None)
        if key is not None:
            drift[output_key] = last[key] - first[key]
    return drift


def _parse_minimization_stats(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Stopping criterion"):
            _, _, value = stripped.partition("=")
            out["stopping_criterion"] = value.strip()
        elif stripped.startswith("Energy initial, next-to-last, final"):
            values = _numeric_values(lines[index + 1] if index + 1 < len(lines) else "")
            if len(values) >= 3:
                out.update(
                    {
                        "energy_initial": values[0],
                        "energy_next_to_last": values[1],
                        "energy_final": values[2],
                        "energy_change": values[2] - values[0],
                    }
                )
        elif stripped.startswith("Force two-norm initial, final"):
            values = _numeric_values(line)
            if len(values) >= 2:
                out["force_two_norm_initial"], out["force_two_norm_final"] = values[-2:]
        elif stripped.startswith("Force max component initial, final"):
            values = _numeric_values(line)
            if len(values) >= 2:
                out["force_max_initial"], out["force_max_final"] = values[-2:]
        elif stripped.startswith("Iterations, force evaluations"):
            values = _numeric_values(line)
            if len(values) >= 2:
                out["iterations"], out["force_evaluations"] = int(values[-2]), int(values[-1])
    if out:
        criterion = str(out.get("stopping_criterion") or "").lower()
        if "force tolerance" in criterion or "energy tolerance" in criterion:
            out["state"] = "converged"
        elif "maximum" in criterion or "max iterations" in criterion or "max force evaluations" in criterion:
            out["state"] = "not_converged"
        else:
            out["state"] = "unknown"
    return out


def _parse_lammps_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    warnings = [line.strip() for line in lines if line.strip().startswith("WARNING:")]
    errors = [line.strip() for line in lines if line.strip().startswith("ERROR:")]
    rows: list[dict[str, float]] = []
    segments: list[dict[str, Any]] = []
    headers: list[str] = []
    current_rows: list[dict[str, float]] = []
    in_table = False
    for line in lines:
        if _THERMO_START_RE.match(line):
            if current_rows:
                segments.append(_thermo_segment(current_rows))
                current_rows = []
            headers = line.split()
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) != len(headers):
            if parts and not _looks_numeric(parts[0]):
                if current_rows:
                    segments.append(_thermo_segment(current_rows))
                    current_rows = []
                in_table = False
            continue
        try:
            row = {key: float(value) for key, value in zip(headers, parts)}
        except Exception:
            continue
        rows.append(row)
        current_rows.append(row)
    if current_rows:
        segments.append(_thermo_segment(current_rows))
    normal_script_end = bool(not errors and any("Total wall time:" in line or "Loop time of" in line for line in lines))
    minimization = _parse_minimization_stats(lines)
    if minimization and not errors:
        normal_script_end = True
    execution_state = "completed" if normal_script_end else ("failed" if errors else "unknown")
    return {
        "log_path": workspace_relpath(path),
        "execution_state": execution_state,
        "task_state": minimization.get("state", "not_applicable"),
        "thermo_state": "calculated" if rows else "not_calculated",
        "warnings": warnings,
        "errors": errors,
        "thermo_rows": len(rows),
        "thermo_segments": segments,
        "final_thermo": rows[-1] if rows else {},
        "thermo_drift": _thermo_drift(rows),
        "minimization": minimization,
    }


def _find_lammps_log(directory: Path) -> Path | None:
    return select_output(directory, names=("log.lammps", "lammps_stdout.out"),
                         signals=("LAMMPS (", "LOOP TIME OF", "TOTAL WALL TIME:"))


def _discover_lammps_result_dirs(root: Path) -> list[Path]:
    return discover_directories(root, ("lammps_summary.json",), _find_lammps_log)


def lammps_log_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[lammps/analysis] Separate LAMMPS process completion, minimizer state, and parsed thermo availability."""
    tool_name = "lammps_log_summary"
    params = LammpsLogSummaryInput(**payload)
    result_root, sources = analysis_sources(params.result_root, params.result_files, _discover_lammps_result_dirs)
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir.strip() else result_root.parent / f"{result_root.name}_analysis"
    if not sources:
        _tool_error(
            tool_name,
            "No LAMMPS result directories found.",
            data={"result_root": workspace_relpath(result_root)},
            error_code="no_lammps_runs",
        )
    records: list[dict[str, Any]] = []
    for run, source_file in sources:
        try:
            log_path = source_file or _find_lammps_log(run)
            if log_path is None:
                raise ValueError("No identifiable LAMMPS log; pass an explicit file")
            record = _parse_lammps_log(log_path)
            record["result_path"] = workspace_relpath(run)
            records.append(record)
        except Exception as exc:
            records.append({"result_path": workspace_relpath(run), "log_path": workspace_relpath(source_file) if source_file else "",
                            "parse_state": "failed", "errors": [str(exc)]})
    failed_count = sum(record.get("parse_state") == "failed" for record in records)
    summary_path = output_dir / "lammps_log_summary.json"
    _write_json(summary_path, {"result_root": workspace_relpath(result_root), "records": records})
    data = {
        "result_root": workspace_relpath(result_root),
        "summary_path": workspace_relpath(summary_path),
        "runs_analyzed": len(records),
    }
    content = (
        f"lammps_log_summary {'failed' if failed_count == len(records) else 'partial' if failed_count else 'completed'}; parse_failed_count={failed_count}.\n"
        f"runs_analyzed={len(records)} summary_path={data['summary_path']}"
    )
    return content, {"tool_name": tool_name, "data": data}


def _trajectory_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates: set[Path] = set()
    for pattern in ("*.lammpstrj", "*.xyz", "*.traj"):
        candidates.update(candidate for candidate in path.glob(pattern) if candidate.is_file())
    return sorted(candidates)


def _scan_text_trajectory(path: Path, output_dir: Path) -> tuple[int, int | None, str, str]:
    """Read one frame at a time and retain only the last complete frame."""
    frames, natoms, last_block, error = 0, None, [], ""
    is_dump = path.suffix.lower() == ".lammpstrj"
    with path.open(encoding="utf-8", errors="strict") as stream:
        while True:
            first = stream.readline()
            if not first:
                break
            if not first.strip():
                continue
            block = [first]
            def take():
                line = stream.readline()
                if not line:
                    raise ValueError("truncated frame")
                block.append(line)
                return line
            try:
                if is_dump:
                    if first.strip() != "ITEM: TIMESTEP":
                        raise ValueError("missing TIMESTEP header")
                    int(take().strip())
                    if take().strip() != "ITEM: NUMBER OF ATOMS":
                        raise ValueError("missing NUMBER OF ATOMS header")
                    count = int(take().strip())
                    if not take().startswith("ITEM: BOX BOUNDS"):
                        raise ValueError("missing BOX BOUNDS header")
                    for _ in range(3):
                        values = take().split()
                        if len(values) < 2:
                            raise ValueError("incomplete box bounds")
                        [float(value) for value in values]
                    header = take()
                    if not header.startswith("ITEM: ATOMS "):
                        raise ValueError("missing ATOMS header")
                    columns = len(header.split()) - 2
                else:
                    count = int(first.strip())
                    take()  # XYZ comment may be empty but must exist.
                    columns = 4
                if count <= 0:
                    raise ValueError("non-positive atom count")
                for _ in range(count):
                    values = take().split()
                    if len(values) < columns:
                        raise ValueError("incomplete atom row")
                    if not is_dump:
                        [float(value) for value in values[1:4]]
                last_block, natoms = block, count
                frames += 1
            except (ValueError, UnicodeError) as exc:
                error = f"frame_index={frames}: {exc}"
                break
    destination = ""
    if last_block:
        output = output_dir / ("last_complete_frame.lammpstrj" if is_dump else "last_complete_frame.xyz")
        output.write_text("".join(last_block), encoding="utf-8")
        destination = workspace_relpath(output)
    return frames, natoms, destination, error


def _summarize_ase_trajectory(path: Path, output_dir: Path) -> tuple[int, int | None, str]:
    with Trajectory(str(path), mode="r") as trajectory:
        frames = int(len(trajectory))
        if not frames:
            return 0, None, ""
        final_atoms = trajectory[-1]

    output = output_dir / "final_frame.traj"
    with Trajectory(str(output), mode="w") as final_trajectory:
        final_trajectory.write(final_atoms)
    return frames, len(final_atoms), workspace_relpath(output)


def _numeric_table_summary(path: Path) -> dict[str, Any]:
    rows = [
        _numeric_values(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = [row for row in rows if row]
    return {
        "path": workspace_relpath(path),
        "rows": len(rows),
        "first_row": rows[0] if rows else [],
        "last_row": rows[-1] if rows else [],
    }


def md_trajectory_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[md/analysis] Inventory trajectory frames and native observables without inferring diffusion or RDF."""
    tool_name = "md_trajectory_summary"
    params = MdTrajectorySummaryInput(**payload)
    source = resolve_workspace_path(params.path, must_exist=True)
    candidates = _trajectory_candidates(source)
    if not candidates:
        _tool_error(
            tool_name,
            "No supported MD trajectory candidate found (*.lammpstrj, *.xyz, or *.traj).",
            error_code="missing_trajectory",
        )
    if len(candidates) > 1:
        _tool_error(
            tool_name,
            "Multiple trajectory candidates found; pass the intended trajectory file explicitly.",
            data={"candidates": [workspace_relpath(path) for path in candidates]},
            error_code="ambiguous_trajectory",
        )
    trajectory = candidates[0]
    result_dir = source if source.is_dir() else source.parent
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir.strip() else result_dir.parent / f"{result_dir.name}_trajectory_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = trajectory.suffix.lower()
    parse_error = ""
    if suffix in {".lammpstrj", ".xyz"}:
        frames, natoms, last_complete_frame, parse_error = _scan_text_trajectory(trajectory, output_dir)
        trajectory_format = "lammps-dump" if suffix == ".lammpstrj" else "xyz"
    elif suffix == ".traj":
        frames, natoms, last_complete_frame = _summarize_ase_trajectory(trajectory, output_dir)
        trajectory_format = "ase-trajectory"
    else:
        raise ValueError(f"Unsupported trajectory format: {suffix}")
    final_frame = "" if parse_error else last_complete_frame
    state = "partial" if parse_error and frames else "failed" if parse_error or not frames else "completed"
    observable_paths = [resolve_workspace_path(path, must_exist=True) for path in params.observable_files] if params.observable_files else [
        result_dir / name for name in ("rdf.dat", "msd.dat") if (result_dir / name).is_file()
    ]
    native_observables = {(workspace_relpath(path) if params.observable_files else path.name): _numeric_table_summary(path) for path in observable_paths}
    cp2k_energy_files = [parse_cp2k_energy_file(path) for path in sorted(result_dir.glob("*.ener"))]
    summary = {
        "source": workspace_relpath(source),
        "trajectory": workspace_relpath(trajectory),
        "format": trajectory_format,
        "nframes": frames,
        "natoms": natoms,
        "final_frame": final_frame,
        "last_complete_frame": last_complete_frame,
        "parse_error": parse_error,
        "state": state,
        "native_observables": native_observables,
        "cp2k_energy_files": cp2k_energy_files,
        "restart_files": [
            workspace_relpath(path)
            for path in sorted({*result_dir.glob("restart*"), *result_dir.glob("*RESTART*"), *result_dir.glob("*.restart")})
            if path.is_file()
        ],
    }
    summary_path = output_dir / "md_trajectory_summary.json"
    _write_json(summary_path, summary)
    data = {
        "trajectory": workspace_relpath(trajectory),
        "summary_path": workspace_relpath(summary_path),
        "final_frame": final_frame,
        "last_complete_frame": last_complete_frame,
        "parse_error": parse_error,
        "state": state,
        "nframes": frames,
        "natoms": natoms,
    }
    content = (
        f"md_trajectory_summary {state}.\n"
        f"trajectory={data['trajectory']} nframes={frames} summary_path={data['summary_path']}"
    )
    if parse_error:
        content += f"\nparse_error={parse_error} last_complete_frame={last_complete_frame}"
    if final_frame:
        content += f"\nfinal_frame={final_frame}"
    return content, {"tool_name": tool_name, "data": data}


__all__ = [
    "LammpsPrepareInput",
    "LammpsLogSummaryInput",
    "MdTrajectorySummaryInput",
    "lammps_prepare",
    "lammps_log_summary",
    "md_trajectory_summary",
]
