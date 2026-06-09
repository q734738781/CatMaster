from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_records_for_artifact, resolve_workspace_path, workspace_relpath
from catmaster.tools.dynamics.cp2k_analysis import parse_cp2k_energy_file
from catmaster.tools.geometry_inputs.cp2k_common import discover_structure_paths, safe_stage_name

LAMMPS_REFERENCE_URLS = [
    "https://docs.lammps.org/",
    "https://docs.lammps.org/units.html",
    "https://docs.lammps.org/read_data.html",
    "https://docs.lammps.org/pair_style.html",
    "https://docs.lammps.org/fix.html",
    "https://docs.lammps.org/fix_nh.html",
    "https://docs.lammps.org/minimize.html",
    "https://docs.lammps.org/thermo_style.html",
    "https://docs.lammps.org/dump.html",
    "https://docs.lammps.org/write_restart.html",
    "https://docs.lammps.org/read_restart.html",
    "https://docs.lammps.org/compute_rdf.html",
    "https://docs.lammps.org/compute_msd.html",
    "https://pymatgen.org/pymatgen.io.lammps.html",
    "https://lammpsio.readthedocs.io/",
]
_SUPPORTED_STRUCTURE_EXTS = {".xyz", ".cif", ".vasp", ".poscar", ".pdb"}
_COMMON_PAIR_STYLE_PREFIXES = {
    "lj/cut",
    "lj/cut/coul",
    "buck",
    "morse",
    "eam",
    "eam/alloy",
    "eam/fs",
    "tersoff",
    "sw",
    "meam",
    "reaxff",
    "reax/c",
    "hybrid",
    "hybrid/overlay",
    "table",
    "pace",
    "mliap",
}
_ALLOWED_SETTINGS = {
    "cell_abc",
    "temperature",
    "temperature_start",
    "temperature_stop",
    "pressure",
    "pressure_start",
    "pressure_stop",
    "timestep",
    "steps",
    "thermo",
    "dump_stride",
    "restart_stride",
    "seed",
    "tdamp",
    "pdamp",
    "etol",
    "ftol",
    "maxiter",
    "maxeval",
    "min_style",
    "rdf",
    "rdf_bins",
    "msd",
    "restart_file",
    "create_velocities",
}


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read JSON {workspace_relpath(path)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")
    return payload


def _normalize_settings(settings: dict[str, Any] | None, *, tool_name: str) -> dict[str, Any]:
    out = dict(settings or {})
    unknown = sorted(set(out) - _ALLOWED_SETTINGS)
    if unknown:
        _tool_error(
            tool_name,
            f"Unsupported LAMMPS settings key(s): {', '.join(unknown)}",
            data={"unsupported_settings": unknown},
            error_code="unsupported_settings",
        )
    if out.get("cell_abc") is not None:
        raw = out["cell_abc"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError("settings.cell_abc must be a three-item list in angstrom.")
        cell = [float(item) for item in raw]
        if any(item <= 0 for item in cell):
            raise ValueError("settings.cell_abc values must be positive.")
        out["cell_abc"] = cell
    return out


def _normalize_forcefield_card(card: dict[str, Any], *, root: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(card or {})
    warnings: list[str] = []
    required = ["units", "atom_style", "pair_style", "pair_coeff"]
    missing = [key for key in required if normalized.get(key) in (None, "", [], {})]
    if missing:
        raise ValueError(f"Missing force-field card field(s): {', '.join(missing)}")
    normalized["units"] = str(normalized["units"]).strip()
    normalized["atom_style"] = str(normalized["atom_style"]).strip()
    normalized["pair_style"] = str(normalized["pair_style"]).strip()
    pair_prefix = normalized["pair_style"].split()[0]
    if pair_prefix not in _COMMON_PAIR_STYLE_PREFIXES:
        warnings.append(f"Unrecognized pair_style prefix '{pair_prefix}'; lammps_prepare will pass it through unchanged.")
    pair_coeff = normalized.get("pair_coeff")
    if isinstance(pair_coeff, str):
        pair_coeff = [pair_coeff]
    if not isinstance(pair_coeff, list) or not all(str(item).strip() for item in pair_coeff):
        raise ValueError("forcefield_card.pair_coeff must be a non-empty string list.")
    normalized["pair_coeff"] = [str(item).strip() for item in pair_coeff]
    potential_files = normalized.get("potential_files") or []
    if isinstance(potential_files, str):
        potential_files = [potential_files]
    if not isinstance(potential_files, list):
        raise ValueError("forcefield_card.potential_files must be a list when provided.")
    normalized_files: list[str] = []
    for raw in potential_files:
        rel = str(raw).strip()
        if not rel:
            continue
        path = resolve_workspace_path(rel, must_exist=True)
        if not path.is_file():
            raise ValueError(f"Potential file is not a file: {workspace_relpath(path)}")
        normalized_files.append(rel)
    normalized["potential_files"] = normalized_files
    masses = normalized.get("masses") or {}
    if masses and not isinstance(masses, dict):
        raise ValueError("forcefield_card.masses must be an object when provided.")
    normalized["masses"] = {str(key): float(value) for key, value in dict(masses).items()}
    element_map = normalized.get("element_map") or {}
    if element_map and not isinstance(element_map, dict):
        raise ValueError("forcefield_card.element_map must be an object when provided.")
    normalized["element_map"] = {str(key): int(value) for key, value in dict(element_map).items()}
    if normalized["atom_style"] not in {"atomic", "charge"}:
        warnings.append(
            "Only atom_style atomic/charge can be generated from structures; other styles require a prebuilt system.data."
        )
    return normalized, warnings


class LammpsForcefieldValidateInput(BaseModel):
    """[lammps/prepare] Validate and normalize a LAMMPS force-field card without preparing or running a simulation."""

    model_config = ConfigDict(extra="forbid")

    forcefield_card: dict[str, Any] | None = Field(None, description="Inline LAMMPS force-field card.")
    forcefield_card_path: str | None = Field(None, description="Workspace-relative JSON force-field card path.")
    output_path: str | None = Field(
        None,
        description="Output JSON path for the normalized card. Defaults to forcefields/lammps_forcefield_card.normalized.json.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> "LammpsForcefieldValidateInput":
        has_card = self.forcefield_card not in (None, {})
        has_path = bool(str(self.forcefield_card_path or "").strip())
        if has_card == has_path:
            raise ValueError("Exactly one of forcefield_card or forcefield_card_path is required.")
        return self


class LammpsPrepareInput(BaseModel):
    """[lammps/prepare] Prepare LAMMPS minimization, MD, annealing, or restart stages from a validated force-field card."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description="Structure file/directory or prior result directory for restart.")
    output_root: str = Field(..., description="Output root for prepared LAMMPS stage directories.")
    recipe: Literal["minimize", "anneal", "nve", "nvt", "npt", "restart"] = Field(..., description="LAMMPS recipe.")
    forcefield_card_path: str = Field(..., description="Workspace-relative normalized force-field card JSON path.")
    settings: dict[str, Any] = Field(default_factory=dict, description="Restricted LAMMPS settings override map.")


class LammpsLogSummaryInput(BaseModel):
    """[lammps/analysis] Summarize generic LAMMPS log files into JSON without task-specific scientific interpretation."""

    model_config = ConfigDict(extra="forbid")

    result_root: str = Field(..., description="LAMMPS result directory or batch root.")
    output_dir: str | None = Field(None, description="Summary output directory.")


class MdTrajectorySummaryInput(BaseModel):
    """[md/analysis] Summarize CP2K/LAMMPS trajectory health with frame counts and optional thermo drift."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Trajectory file or result directory containing trajectory/log outputs.")
    output_dir: str | None = Field(None, description="Summary output directory.")


def lammps_forcefield_validate(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "lammps_forcefield_validate"
    params = LammpsForcefieldValidateInput(**payload)
    if params.forcefield_card_path:
        source_path = resolve_workspace_path(params.forcefield_card_path, must_exist=True)
        card = _read_json(source_path)
    else:
        source_path = None
        card = dict(params.forcefield_card or {})
    normalized, warnings = _normalize_forcefield_card(card, root=source_path.parent if source_path else None)
    output_path = resolve_workspace_path(params.output_path or "forcefields/lammps_forcefield_card.normalized.json")
    payload_out = {"forcefield_card": normalized, "references": LAMMPS_REFERENCE_URLS, "warnings": warnings}
    _write_json(output_path, payload_out)
    data = {
        "output_path_rel": workspace_relpath(output_path),
        "units": normalized["units"],
        "atom_style": normalized["atom_style"],
        "pair_style": normalized["pair_style"],
        "potential_files": normalized.get("potential_files", []),
    }
    content = (
        "lammps_forcefield_validate completed.\n"
        f"output_path_rel={data['output_path_rel']} pair_style={data['pair_style']}"
    )
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    return content, artifact


def _load_forcefield_card(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    card = payload.get("forcefield_card") if "forcefield_card" in payload else payload
    if not isinstance(card, dict):
        raise ValueError("forcefield card JSON must contain an object or {'forcefield_card': object}.")
    normalized, _ = _normalize_forcefield_card(card, root=path.parent)
    return normalized


def _read_atoms(path: Path) -> Atoms:
    try:
        return ase_read(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to read structure {workspace_relpath(path)}: {exc}") from exc


def _ensure_cell(atoms: Atoms, settings: dict[str, Any]) -> Atoms:
    out = atoms.copy()
    if out.cell.rank == 3 and min(out.cell.lengths()) > 1.0e-8:
        return out
    cell = settings.get("cell_abc") or [50.0, 50.0, 50.0]
    out.set_cell(cell)
    out.center()
    out.set_pbc([False, False, False])
    return out


def _element_type_map(symbols: list[str], card: dict[str, Any]) -> dict[str, int]:
    explicit = card.get("element_map") or {}
    mapping: dict[str, int] = {}
    for symbol in sorted(set(symbols), key=symbols.index):
        if symbol in explicit:
            mapping[symbol] = int(explicit[symbol])
        else:
            mapping[symbol] = len(mapping) + 1
    return mapping


def _atomic_mass(symbol: str, card: dict[str, Any]) -> float:
    masses = card.get("masses") or {}
    if symbol in masses:
        return float(masses[symbol])
    number = atomic_numbers.get(symbol)
    if number is None:
        return 1.0
    return float(atomic_masses[number])


def _write_lammps_data(path: Path, atoms: Atoms, card: dict[str, Any], settings: dict[str, Any]) -> dict[str, int]:
    atom_style = str(card.get("atom_style") or "atomic")
    atoms = _ensure_cell(atoms, settings)
    symbols = atoms.get_chemical_symbols()
    mapping = _element_type_map(symbols, card)
    cell = atoms.cell.array
    if abs(cell[0][1]) > 1e-8 or abs(cell[0][2]) > 1e-8 or abs(cell[1][2]) > 1e-8:
        raise ValueError("lammps_prepare currently writes orthogonal data boxes; prebuild system.data for triclinic cells.")
    xhi, yhi, zhi = float(cell[0][0]), float(cell[1][1]), float(cell[2][2])
    lines = [
        "LAMMPS data file written by CatMaster",
        "",
        f"{len(atoms)} atoms",
        f"{len(mapping)} atom types",
        "",
        f"0.0 {xhi:.10f} xlo xhi",
        f"0.0 {yhi:.10f} ylo yhi",
        f"0.0 {zhi:.10f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for symbol, type_id in sorted(mapping.items(), key=lambda item: item[1]):
        lines.append(f"{type_id} {_atomic_mass(symbol, card):.10f} # {symbol}")
    lines.extend(["", "Atoms # " + atom_style, ""])
    for idx, (symbol, position) in enumerate(zip(symbols, atoms.get_positions()), start=1):
        type_id = mapping[symbol]
        if atom_style == "charge":
            lines.append(
                f"{idx} {type_id} 0.0 {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}"
            )
        elif atom_style == "atomic":
            lines.append(f"{idx} {type_id} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}")
        else:
            raise ValueError("Cannot generate system.data for atom_style other than atomic/charge.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mapping


def _copy_potential_files(card: dict[str, Any], stage_dir: Path, card_root: Path) -> list[str]:
    copied: list[str] = []
    for rel in card.get("potential_files") or []:
        src = resolve_workspace_path(str(rel), must_exist=True)
        dst = stage_dir / Path(str(rel)).name
        shutil.copy2(src, dst)
        copied.append(dst.name)
    return copied


def _pair_lines(card: dict[str, Any]) -> list[str]:
    lines = [f"pair_style {card['pair_style']}"]
    for coeff in card.get("pair_coeff") or []:
        lines.append(f"pair_coeff {coeff}")
    return lines


def _common_output_lines(settings: dict[str, Any], *, minimize: bool = False) -> list[str]:
    thermo = int(settings.get("thermo", 100))
    dump_stride = int(settings.get("dump_stride", max(1, thermo)))
    thermo_style = "step pe etotal press fmax fnorm" if minimize else "step temp pe ke etotal press vol"
    lines = [
        f"thermo {thermo}",
        f"thermo_style custom {thermo_style}",
        f"dump traj all custom {dump_stride} trajectory.lammpstrj id type x y z vx vy vz",
        "dump_modify traj sort id",
    ]
    if settings.get("rdf"):
        bins = int(settings.get("rdf_bins", 100))
        lines.extend(
            [
                f"compute rdf_all all rdf {bins}",
                f"fix rdf_out all ave/time {thermo} 1 {thermo} c_rdf_all[*] file rdf.dat mode vector",
            ]
        )
    if settings.get("msd"):
        lines.extend(
            [
                "compute msd_all all msd",
                f"fix msd_out all ave/time {thermo} 1 {thermo} c_msd_all[4] file msd.dat",
            ]
        )
    return lines


def _lammps_input_text(recipe: str, card: dict[str, Any], settings: dict[str, Any], *, restart_file: str | None = None) -> str:
    lines: list[str] = [
        "clear",
        f"units {card['units']}",
        f"atom_style {card['atom_style']}",
    ]
    if recipe == "restart":
        if not restart_file:
            raise ValueError("restart recipe requires settings.restart_file or a discovered restart file.")
        lines.append(f"read_restart {restart_file}")
    else:
        lines.append("read_data system.data")
    lines.extend(_pair_lines(card))
    lines.append("neighbor 2.0 bin")
    lines.append("neigh_modify delay 0 every 1 check yes")
    if recipe == "minimize":
        lines.extend(_common_output_lines(settings, minimize=True))
        lines.append(f"min_style {str(settings.get('min_style') or 'cg')}")
        lines.append(
            "minimize "
            f"{float(settings.get('etol', 1e-6)):.10g} "
            f"{float(settings.get('ftol', 1e-8)):.10g} "
            f"{int(settings.get('maxiter', 1000))} "
            f"{int(settings.get('maxeval', 10000))}"
        )
    else:
        timestep = float(settings.get("timestep", 1.0))
        steps = int(settings.get("steps", 1000))
        temp_start = float(settings.get("temperature_start", settings.get("temperature", 300.0)))
        temp_stop = float(settings.get("temperature_stop", settings.get("temperature", temp_start)))
        tdamp = float(settings.get("tdamp", max(100.0 * timestep, 1.0)))
        pdamp = float(settings.get("pdamp", max(1000.0 * timestep, 1.0)))
        seed = int(settings.get("seed", 12345))
        lines.extend(_common_output_lines(settings, minimize=False))
        lines.append(f"timestep {timestep:.10g}")
        if recipe != "restart" and bool(settings.get("create_velocities", True)) and recipe != "nve":
            lines.append(f"velocity all create {temp_start:.10g} {seed} mom yes rot yes dist gaussian")
        if recipe in {"nvt", "anneal"}:
            lines.append(f"fix int all nvt temp {temp_start:.10g} {temp_stop:.10g} {tdamp:.10g}")
        elif recipe == "npt":
            p_start = float(settings.get("pressure_start", settings.get("pressure", 1.0)))
            p_stop = float(settings.get("pressure_stop", settings.get("pressure", p_start)))
            lines.append(
                f"fix int all npt temp {temp_start:.10g} {temp_stop:.10g} {tdamp:.10g} "
                f"iso {p_start:.10g} {p_stop:.10g} {pdamp:.10g}"
            )
        else:
            lines.append("fix int all nve")
        lines.append(f"run {steps}")
        lines.append("unfix int")
    lines.append("write_data final.data")
    lines.append("write_restart restart.final")
    return "\n".join(lines) + "\n"


def _discover_restart_file(input_path: Path, settings: dict[str, Any]) -> Path | None:
    if settings.get("restart_file"):
        path = resolve_workspace_path(str(settings["restart_file"]), must_exist=True)
        if not path.is_file():
            raise ValueError(f"settings.restart_file is not a file: {workspace_relpath(path)}")
        return path
    if input_path.is_dir():
        candidates = sorted(input_path.glob("restart*")) + sorted(input_path.glob("*.restart"))
        files = [path for path in candidates if path.is_file()]
        if files:
            return files[-1]
    return None


def _stage_lammps_case(
    *,
    source: Path,
    stage_dir: Path,
    recipe: str,
    card: dict[str, Any],
    card_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    restart_file = None
    type_map: dict[str, int] = {}
    if recipe == "restart":
        restart_path = _discover_restart_file(source, settings)
        if restart_path is None:
            raise ValueError("restart recipe requires a restart file under input_path or settings.restart_file.")
        dst = stage_dir / restart_path.name
        shutil.copy2(restart_path, dst)
        restart_file = dst.name
    elif source.is_dir() and (source / "system.data").is_file():
        shutil.copy2(source / "system.data", stage_dir / "system.data")
    else:
        atoms = _read_atoms(source)
        type_map = _write_lammps_data(stage_dir / "system.data", atoms, card, settings)
        ase_write(str(stage_dir / "input.xyz"), atoms, format="xyz")
    copied = _copy_potential_files(card, stage_dir, card_root)
    (stage_dir / "in.lammps").write_text(
        _lammps_input_text(recipe, card, settings, restart_file=restart_file),
        encoding="utf-8",
    )
    manifest_path = stage_dir / "manifest.json"
    manifest = {
        "program": "lammps",
        "recipe": recipe,
        "source_rel": workspace_relpath(source),
        "stage_dir_rel": workspace_relpath(stage_dir),
        "input_file": "in.lammps",
        "data_file": "system.data" if (stage_dir / "system.data").exists() else None,
        "restart_file": restart_file,
        "forcefield_card": card,
        "element_type_map": type_map,
        "copied_potential_files": copied,
        "settings": settings,
        "references": LAMMPS_REFERENCE_URLS,
    }
    _write_json(manifest_path, manifest)
    return {
        "source_path_rel": workspace_relpath(source),
        "stage_dir_rel": workspace_relpath(stage_dir),
        "input_file_rel": workspace_relpath(stage_dir / "in.lammps"),
        "manifest_rel": workspace_relpath(manifest_path),
        "recipe": recipe,
    }


def lammps_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "lammps_prepare"
    params = LammpsPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    card_path = resolve_workspace_path(params.forcefield_card_path, must_exist=True)
    card = _load_forcefield_card(card_path)
    settings = _normalize_settings(params.settings, tool_name=tool_name)
    if params.recipe == "restart":
        sources = [input_path]
        input_root = None
    else:
        sources = discover_structure_paths(input_path)
        if input_path.is_dir() and (input_path / "system.data").is_file():
            sources = [input_path]
        input_root = input_path if input_path.is_dir() and len(sources) > 1 else None
    if not sources:
        _tool_error(
            tool_name,
            "No supported structure files found under input_path.",
            data={"input_path_rel": workspace_relpath(input_path)},
            error_code="no_structures",
        )
    records: list[dict[str, Any]] = []
    for source in sources:
        stage_dir = output_root if input_root is None else output_root / safe_stage_name(source, root=input_root)
        records.append(
            _stage_lammps_case(
                source=source,
                stage_dir=stage_dir,
                recipe=params.recipe,
                card=card,
                card_root=card_path.parent,
                settings=settings,
            )
        )
    manifest = output_root / "lammps_prepare_manifest.json"
    _write_json(manifest, {"input_path_rel": workspace_relpath(input_path), "recipe": params.recipe, "records": records})
    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "recipe": params.recipe,
        "prepared_count": len(records),
        "manifest_rel": workspace_relpath(manifest),
        **compact_records_for_artifact(records, full_records_rel=workspace_relpath(manifest)),
    }
    content = (
        "lammps_prepare completed.\n"
        f"recipe={params.recipe} prepared_count={len(records)} output_root_rel={data['output_root_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


_THERMO_START_RE = re.compile(r"^\s*Step(?:\s+|$)")


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
    final_row = rows[-1] if rows else {}
    min_stats = _parse_minimization_stats(lines)
    completed = bool(lines and not errors and any("Total wall time:" in line or "Loop time of" in line for line in lines))
    if min_stats:
        completed = bool(lines and not errors)
    return {
        "log_rel": workspace_relpath(path),
        "warnings": warnings[:20],
        "errors": errors[:20],
        "thermo_rows": len(rows),
        "thermo_segments": segments,
        "final_thermo": final_row,
        "thermo_drift": _thermo_drift(rows),
        "minimization": min_stats,
        "completed": completed,
    }


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
    for out_key, keys in aliases.items():
        key = next((item for item in keys if item in first and item in last), None)
        if key is not None:
            drift[out_key] = last[key] - first[key]
    return drift


def _parse_minimization_stats(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Stopping criterion"):
            _, _, value = stripped.partition("=")
            out["stopping_criterion"] = value.strip()
        elif stripped.startswith("Energy initial, next-to-last, final"):
            values = _numeric_values(lines[idx + 1] if idx + 1 < len(lines) else "")
            if len(values) >= 3:
                out["energy_initial"] = values[0]
                out["energy_next_to_last"] = values[1]
                out["energy_final"] = values[2]
                out["energy_change"] = values[2] - values[0]
        elif stripped.startswith("Force two-norm initial, final"):
            values = _numeric_values(line)
            if len(values) >= 2:
                out["force_two_norm_initial"] = values[-2]
                out["force_two_norm_final"] = values[-1]
        elif stripped.startswith("Force max component initial, final"):
            values = _numeric_values(line)
            if len(values) >= 2:
                out["force_max_initial"] = values[-2]
                out["force_max_final"] = values[-1]
        elif stripped.startswith("Iterations, force evaluations"):
            values = _numeric_values(line)
            if len(values) >= 2:
                out["iterations"] = int(values[-2])
                out["force_evaluations"] = int(values[-1])
    return out


def _numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", text):
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


def _discover_lammps_result_dirs(root: Path) -> list[Path]:
    if any((root / name).is_file() for name in ("log.lammps", "lammps_summary.json", "lammps_stdout.out")):
        return [root]
    out: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and any((path / name).is_file() for name in ("log.lammps", "lammps_summary.json", "lammps_stdout.out")):
            out.append(path)
    return sorted(out)


def lammps_log_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "lammps_log_summary"
    params = LammpsLogSummaryInput(**payload)
    result_root = resolve_workspace_path(params.result_root, must_exist=True)
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else result_root.parent / f"{result_root.name}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_lammps_result_dirs(result_root)
    if not runs:
        _tool_error(
            tool_name,
            "No LAMMPS result directories found.",
            data={"result_root_rel": workspace_relpath(result_root)},
            error_code="no_lammps_runs",
        )
    records: list[dict[str, Any]] = []
    for run in runs:
        log_path = next((run / name for name in ("log.lammps", "lammps_stdout.out") if (run / name).is_file()), None)
        if log_path is None:
            records.append({"result_dir_rel": workspace_relpath(run), "completed": False, "errors": ["missing log file"]})
            continue
        record = _parse_lammps_log(log_path)
        record["result_dir_rel"] = workspace_relpath(run)
        records.append(record)
    summary_path = output_dir / "lammps_log_summary.json"
    _write_json(summary_path, {"result_root_rel": workspace_relpath(result_root), "records": records})
    data = {
        "result_root_rel": workspace_relpath(result_root),
        "output_dir_rel": workspace_relpath(output_dir),
        "summary_json_rel": workspace_relpath(summary_path),
        "runs_analyzed": len(records),
    }
    content = (
        "lammps_log_summary completed.\n"
        f"runs_analyzed={len(records)} summary_json_rel={data['summary_json_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


def _trajectory_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates: list[Path] = []
    for name in ("trajectory.lammpstrj", "trajectory.xyz", "cp2k-pos-1.xyz", "input.xyz", "final.xyz"):
        candidate = path / name
        if candidate.is_file():
            candidates.append(candidate)
    candidates.extend(sorted(path.glob("*.lammpstrj")))
    candidates.extend(sorted(path.glob("*.xyz")))
    return candidates


def _count_lammps_dump_frames(path: Path) -> tuple[int, int | None]:
    frames = 0
    natoms = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                frames += 1
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                try:
                    natoms = int(next(fh).strip())
                except Exception:
                    pass
    return frames, natoms


def _count_xyz_frames(path: Path) -> tuple[int, int | None]:
    frames = 0
    natoms = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            first = fh.readline()
            if not first:
                break
            try:
                n = int(first.strip())
            except Exception:
                break
            natoms = n
            fh.readline()
            for _ in range(n):
                if not fh.readline():
                    break
            frames += 1
    return frames, natoms


def _extract_last_xyz_frame(path: Path, output_dir: Path) -> str:
    last_frame: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            first = fh.readline()
            if not first:
                break
            try:
                n = int(first.strip())
            except Exception:
                break
            frame = [first, fh.readline()]
            for _ in range(n):
                line = fh.readline()
                if not line:
                    break
                frame.append(line)
            if len(frame) >= n + 2:
                last_frame = frame
    if not last_frame:
        return ""
    out = output_dir / "final_frame.xyz"
    out.write_text("".join(last_frame), encoding="utf-8")
    return workspace_relpath(out)


def _extract_last_lammps_frame(path: Path, output_dir: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "ITEM: TIMESTEP"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    out = output_dir / "final_frame.lammpstrj"
    out.write_text(text[idx:], encoding="utf-8")
    return workspace_relpath(out)


def _numeric_table_summary(path: Path) -> dict[str, Any]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = _numeric_values(stripped)
        if values:
            rows.append(values)
    out: dict[str, Any] = {"path_rel": workspace_relpath(path), "rows": len(rows)}
    if rows:
        out["first_row"] = rows[0]
        out["last_row"] = rows[-1]
    return out


def _observable_summaries(result_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("rdf.dat", "msd.dat"):
        path = result_dir / name
        if path.is_file():
            out[name] = _numeric_table_summary(path)
    return out


def _cp2k_energy_summaries(result_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.ener")):
        summaries.append(parse_cp2k_energy_file(path))
    return summaries


def md_trajectory_summary(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "md_trajectory_summary"
    params = MdTrajectorySummaryInput(**payload)
    source = resolve_workspace_path(params.path, must_exist=True)
    output_dir = resolve_workspace_path(params.output_dir) if params.output_dir else (
        (source if source.is_dir() else source.parent).parent / f"{(source if source.is_dir() else source.parent).name}_trajectory_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _trajectory_candidates(source)
    if not candidates:
        _tool_error(
            tool_name,
            "No CP2K/LAMMPS trajectory candidate found.",
            data={"path_rel": workspace_relpath(source)},
            error_code="missing_trajectory",
        )
    traj = candidates[0]
    if traj.suffix == ".lammpstrj":
        nframes, natoms = _count_lammps_dump_frames(traj)
        fmt = "lammps-dump"
        final_frame_rel = _extract_last_lammps_frame(traj, output_dir)
    elif traj.suffix == ".xyz":
        nframes, natoms = _count_xyz_frames(traj)
        fmt = "xyz"
        final_frame_rel = _extract_last_xyz_frame(traj, output_dir)
    else:
        nframes, natoms = 0, None
        fmt = traj.suffix.lstrip(".") or "unknown"
        final_frame_rel = ""
    thermo_summary = None
    result_dir = source if source.is_dir() else source.parent
    log_path = next((result_dir / name for name in ("log.lammps", "lammps_stdout.out") if (result_dir / name).is_file()), None)
    if log_path is not None:
        thermo_summary = _parse_lammps_log(log_path)
    cp2k_energy_files = _cp2k_energy_summaries(result_dir)
    summary = {
        "source_rel": workspace_relpath(source),
        "trajectory_rel": workspace_relpath(traj),
        "format": fmt,
        "nframes": nframes,
        "natoms": natoms,
        "final_frame_rel": final_frame_rel,
        "thermo_summary": thermo_summary,
        "cp2k_energy_files": cp2k_energy_files,
        "observables": _observable_summaries(result_dir),
        "restart_files": [
            workspace_relpath(path)
            for path in sorted({*result_dir.glob("restart*"), *result_dir.glob("*RESTART*"), *result_dir.glob("*.restart")})
            if path.is_file()
        ],
    }
    summary_path = output_dir / "md_trajectory_summary.json"
    _write_json(summary_path, summary)
    data = {
        "path_rel": workspace_relpath(source),
        "summary_json_rel": workspace_relpath(summary_path),
        "trajectory_rel": summary["trajectory_rel"],
        "final_frame_rel": final_frame_rel,
        "nframes": nframes,
        "natoms": natoms,
    }
    content = (
        "md_trajectory_summary completed.\n"
        f"trajectory_rel={data['trajectory_rel']} nframes={nframes} summary_json_rel={data['summary_json_rel']}"
    )
    return content, {"tool_name": tool_name, "data": data}


__all__ = [
    "LammpsForcefieldValidateInput",
    "LammpsPrepareInput",
    "LammpsLogSummaryInput",
    "MdTrajectorySummaryInput",
    "lammps_forcefield_validate",
    "lammps_prepare",
    "lammps_log_summary",
    "md_trajectory_summary",
]
