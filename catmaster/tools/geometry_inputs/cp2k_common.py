from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

SUPPORTED_STRUCTURE_EXTS = {".xyz", ".cif", ".vasp", ".poscar", ".pdb"}
CP2K_REFERENCE_URLS = [
    "https://manual.cp2k.org/",
    "https://manual.cp2k.org/trunk/methods/optimization/geometry_and_cell_opt.html",
    "https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html",
    "https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html",
    "https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html",
    "https://pymatgen.org/pymatgen.io.cp2k.html",
]

_ALLOWED_SETTINGS = {
    "xc",
    "basis_set",
    "potential",
    "charge",
    "multiplicity",
    "cutoff",
    "rel_cutoff",
    "eps_scf",
    "max_scf",
    "outer_scf",
    "scf_guess",
    "extrapolation",
    "extrapolation_order",
    "kpoints",
    "periodic",
    "cell_abc",
    "dispersion",
    "smearing",
    "added_mos",
    "print_level",
    "optimizer",
    "max_iter",
    "max_force",
    "rms_force",
    "max_dr",
    "rms_dr",
    "external_pressure",
    "pressure_tolerance",
    "stress_tensor",
    "band_type",
    "k_spring",
    "nproc_rep",
    "optimize_endpoints",
    "band_optimizer",
    "band_max_iter",
    "dimer_dr",
    "dimer_rot_optimizer",
    "dimer_vector_file",
    "project",
    "properties",
    "trajectory_stride",
    "restart_stride",
    "energy_stride",
    "temperature",
    "pressure",
    "timestep_fs",
    "steps",
    "ensemble",
    "thermostat",
    "barostat",
    "plumed_input_path",
    "restart_file",
    "structure_file",
}
_PERIODIC_VALUES = {"NONE", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"}
_DISPERSION_VALUES = {"none", "d3bj"}


def tool_error(tool_name: str, message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_settings(settings: dict[str, Any] | None, *, tool_name: str) -> dict[str, Any]:
    raw = dict(settings or {})
    unknown = sorted(set(raw) - _ALLOWED_SETTINGS)
    if unknown:
        tool_error(
            tool_name,
            f"Unsupported CP2K settings key(s): {', '.join(unknown)}",
            data={"unsupported_settings": unknown},
            error_code="unsupported_settings",
        )
    out = dict(raw)
    if out.get("periodic") is not None:
        periodic = str(out["periodic"]).strip().upper()
        if periodic == "AUTO":
            out["periodic"] = "auto"
        elif periodic not in _PERIODIC_VALUES:
            raise ValueError(f"settings.periodic must be one of auto/{sorted(_PERIODIC_VALUES)}")
        else:
            out["periodic"] = periodic
    if out.get("dispersion") is not None:
        dispersion = str(out["dispersion"]).strip().lower()
        if dispersion not in _DISPERSION_VALUES:
            raise ValueError(f"settings.dispersion must be one of {sorted(_DISPERSION_VALUES)}")
        out["dispersion"] = dispersion
    if out.get("kpoints") is not None:
        kpoints = out["kpoints"]
        if not isinstance(kpoints, (list, tuple)) or len(kpoints) != 3:
            raise ValueError("settings.kpoints must be a three-item integer list.")
        parsed = [int(item) for item in kpoints]
        if any(item <= 0 for item in parsed):
            raise ValueError("settings.kpoints values must be positive.")
        out["kpoints"] = parsed
    if out.get("cell_abc") is not None:
        cell_abc = out["cell_abc"]
        if not isinstance(cell_abc, (list, tuple)) or len(cell_abc) != 3:
            raise ValueError("settings.cell_abc must be a three-item list in angstrom.")
        parsed_cell = [float(item) for item in cell_abc]
        if any(item <= 0 for item in parsed_cell):
            raise ValueError("settings.cell_abc values must be positive.")
        out["cell_abc"] = parsed_cell
    if out.get("properties") is not None and not isinstance(out["properties"], dict):
        raise ValueError("settings.properties must be an object when provided.")
    return out


def discover_structure_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in SUPPORTED_STRUCTURE_EXTS or candidate.name in {"POSCAR", "CONTCAR"}:
            candidates.append(candidate)
    return candidates


def read_atoms(path: Path) -> Atoms:
    try:
        return ase_read(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to read structure {workspace_relpath(path)}: {exc}") from exc


def safe_stage_name(path: Path, *, root: Path | None = None) -> str:
    if root is not None and path.is_relative_to(root):
        token = str(path.relative_to(root).with_suffix(""))
    else:
        token = path.stem
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("._")
    return token or "structure"


def infer_periodic(atoms: Atoms, settings: dict[str, Any]) -> str:
    requested = str(settings.get("periodic") or "auto").strip()
    if requested and requested.lower() != "auto":
        return requested.upper()
    pbc = list(bool(item) for item in atoms.get_pbc())
    labels = ["X", "Y", "Z"]
    periodic = "".join(label for label, enabled in zip(labels, pbc) if enabled)
    return periodic or "NONE"


def ensure_cell(atoms: Atoms, settings: dict[str, Any]) -> Atoms:
    out = atoms.copy()
    lengths = out.cell.lengths()
    has_cell = bool(out.cell.rank == 3 and min(lengths) > 1.0e-8)
    if has_cell:
        return out
    abc = settings.get("cell_abc") or [20.0, 20.0, 20.0]
    out.set_cell(abc)
    out.center()
    out.set_pbc([False, False, False])
    return out


def _symbol_value(raw: Any, symbol: str, default: str) -> str:
    if isinstance(raw, dict):
        return str(raw.get(symbol) or raw.get(str(symbol).upper()) or raw.get(str(symbol).lower()) or default)
    if raw is None:
        return default
    return str(raw)


def _fmt_float(value: Any) -> str:
    return f"{float(value):.10g}"


def cp2k_input_text(
    *,
    atoms: Atoms,
    project: str,
    run_type: str,
    settings: dict[str, Any],
    motion_lines: Iterable[str] | None = None,
    top_level_sections: Iterable[str] | None = None,
    stress_tensor: bool = False,
) -> str:
    atoms = ensure_cell(atoms, settings)
    periodic = infer_periodic(atoms, settings)
    cell = atoms.cell.array
    symbols = atoms.get_chemical_symbols()
    unique_symbols = sorted(set(symbols), key=symbols.index)
    xc = str(settings.get("xc") or "PBE")
    basis_raw = settings.get("basis_set")
    potential_raw = settings.get("potential")
    basis_default = "DZVP-MOLOPT-SR-GTH" if isinstance(basis_raw, dict) else str(basis_raw or "DZVP-MOLOPT-SR-GTH")
    potential_default = f"GTH-{xc.upper()}" if isinstance(potential_raw, dict) else str(potential_raw or f"GTH-{xc.upper()}")
    cutoff = settings.get("cutoff", 400)
    rel_cutoff = settings.get("rel_cutoff", 40)
    eps_scf = settings.get("eps_scf", 1.0e-6)
    max_scf = settings.get("max_scf", 100)
    outer_scf = settings.get("outer_scf", 20)
    print_level = str(settings.get("print_level") or "MEDIUM").upper()

    lines: list[str] = [
        "&GLOBAL",
        f"  PROJECT {project}",
        f"  RUN_TYPE {run_type}",
        f"  PRINT_LEVEL {print_level}",
        "&END GLOBAL",
        "",
        "&FORCE_EVAL",
        "  METHOD Quickstep",
    ]
    if stress_tensor or bool(settings.get("stress_tensor")):
        lines.append("  STRESS_TENSOR ANALYTICAL")
    lines.extend(
        [
            "  &DFT",
            "    BASIS_SET_FILE_NAME BASIS_MOLOPT",
            "    POTENTIAL_FILE_NAME GTH_POTENTIALS",
            f"    CHARGE {int(settings.get('charge', 0))}",
            f"    MULTIPLICITY {int(settings.get('multiplicity', 1))}",
            *([f"    EXTRAPOLATION {str(settings.get('extrapolation')).upper()}"] if settings.get("extrapolation") is not None else []),
            *([f"    EXTRAPOLATION_ORDER {int(settings.get('extrapolation_order'))}"] if settings.get("extrapolation_order") is not None else []),
            "    &MGRID",
            f"      CUTOFF {_fmt_float(cutoff)}",
            f"      REL_CUTOFF {_fmt_float(rel_cutoff)}",
            "    &END MGRID",
            "    &SCF",
            f"      EPS_SCF {_fmt_float(eps_scf)}",
            f"      MAX_SCF {int(max_scf)}",
            f"      SCF_GUESS {str(settings.get('scf_guess') or 'ATOMIC').upper()}",
            "      &OT",
            "        PRECONDITIONER FULL_SINGLE_INVERSE",
            "        MINIMIZER DIIS",
            "      &END OT",
            "      &OUTER_SCF",
            f"        MAX_SCF {int(outer_scf)}",
            f"        EPS_SCF {_fmt_float(eps_scf)}",
            "      &END OUTER_SCF",
        ]
    )
    if settings.get("added_mos") is not None:
        scf_guess_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("SCF_GUESS "))
        lines.insert(scf_guess_idx, f"      ADDED_MOS {int(settings['added_mos'])}")
    smearing = settings.get("smearing")
    if isinstance(smearing, dict):
        lines.extend(
            [
                "      &SMEAR ON",
                f"        METHOD {str(smearing.get('method') or 'FERMI_DIRAC')}",
                f"        ELECTRONIC_TEMPERATURE {_fmt_float(smearing.get('temperature', 300.0))}",
                "      &END SMEAR",
            ]
        )
    lines.extend(
        [
            "    &END SCF",
            "    &XC",
            f"      &XC_FUNCTIONAL {xc}",
            "      &END XC_FUNCTIONAL",
        ]
    )
    if str(settings.get("dispersion") or "none").lower() == "d3bj":
        lines.extend(
            [
                "      &VDW_POTENTIAL",
                "        POTENTIAL_TYPE PAIR_POTENTIAL",
                "        &PAIR_POTENTIAL",
                "          TYPE DFTD3(BJ)",
                "          PARAMETER_FILE_NAME dftd3.dat",
                f"          REFERENCE_FUNCTIONAL {xc}",
                "        &END PAIR_POTENTIAL",
                "      &END VDW_POTENTIAL",
            ]
        )
    lines.extend(["    &END XC"])
    if periodic == "NONE":
        lines.extend(
            [
                "    &POISSON",
                "      PERIODIC NONE",
                "      PSOLVER WAVELET",
                "    &END POISSON",
            ]
        )
    elif periodic != "XYZ":
        lines.extend(["    &POISSON", f"      PERIODIC {periodic}", "    &END POISSON"])
    kpoints = settings.get("kpoints")
    if kpoints is not None and periodic != "NONE":
        lines.extend(
            [
                "    &KPOINTS",
                f"      SCHEME MONKHORST-PACK {int(kpoints[0])} {int(kpoints[1])} {int(kpoints[2])}",
                "    &END KPOINTS",
            ]
        )
    lines.extend(
        [
            "  &END DFT",
            "  &SUBSYS",
            "    &CELL",
            f"      A {_fmt_float(cell[0][0])} {_fmt_float(cell[0][1])} {_fmt_float(cell[0][2])}",
            f"      B {_fmt_float(cell[1][0])} {_fmt_float(cell[1][1])} {_fmt_float(cell[1][2])}",
            f"      C {_fmt_float(cell[2][0])} {_fmt_float(cell[2][1])} {_fmt_float(cell[2][2])}",
            f"      PERIODIC {periodic}",
            "    &END CELL",
            "    &COORD",
        ]
    )
    for symbol, position in zip(symbols, atoms.get_positions()):
        lines.append(
            f"      {symbol} {_fmt_float(position[0])} {_fmt_float(position[1])} {_fmt_float(position[2])}"
        )
    lines.extend(["    &END COORD"])
    for symbol in unique_symbols:
        lines.extend(
            [
                f"    &KIND {symbol}",
                f"      BASIS_SET {_symbol_value(settings.get('basis_set'), symbol, basis_default)}",
                f"      POTENTIAL {_symbol_value(settings.get('potential'), symbol, potential_default)}",
                f"    &END KIND",
            ]
        )
    lines.extend(["  &END SUBSYS", ""])
    properties = settings.get("properties")
    if isinstance(properties, dict) and properties:
        lines.extend(_properties_lines(properties))
    lines.append("&END FORCE_EVAL")
    if motion_lines:
        lines.extend(["", *motion_lines])
    if top_level_sections:
        lines.extend(["", *top_level_sections])
    return "\n".join(lines).rstrip() + "\n"


def _properties_lines(properties: dict[str, Any]) -> list[str]:
    lines: list[str] = ["  &PROPERTIES"]
    if bool(properties.get("dos")):
        energy_window = properties.get("energy_window", 10.0)
        energy_step = properties.get("energy_step", 0.01)
        broadening = properties.get("broadening", 0.01)
        lines.extend(
            [
                "    &BANDSTRUCTURE",
                "      &DOS T",
                f"        ENERGY_WINDOW {_fmt_float(energy_window)}",
                f"        ENERGY_STEP {_fmt_float(energy_step)}",
                f"        BROADENING {_fmt_float(broadening)}",
                "      &END DOS",
                "    &END BANDSTRUCTURE",
            ]
        )
    if bool(properties.get("population")):
        lines.extend(["    &FIT_CHARGE", "    &END FIT_CHARGE"])
    lines.append("  &END PROPERTIES")
    return lines


def geo_opt_motion_lines(settings: dict[str, Any]) -> list[str]:
    return [
        "&MOTION",
        "  &GEO_OPT",
        "    TYPE MINIMIZATION",
        f"    OPTIMIZER {str(settings.get('optimizer') or 'BFGS').upper()}",
        f"    MAX_ITER {int(settings.get('max_iter', 200))}",
        f"    MAX_FORCE {_fmt_float(settings.get('max_force', 4.5e-4))}",
        f"    RMS_FORCE {_fmt_float(settings.get('rms_force', 3.0e-4))}",
        f"    MAX_DR {_fmt_float(settings.get('max_dr', 3.0e-3))}",
        f"    RMS_DR {_fmt_float(settings.get('rms_dr', 1.5e-3))}",
        "  &END GEO_OPT",
        "&END MOTION",
    ]


def cell_opt_motion_lines(settings: dict[str, Any]) -> list[str]:
    return [
        "&MOTION",
        "  &CELL_OPT",
        f"    OPTIMIZER {str(settings.get('optimizer') or 'BFGS').upper()}",
        f"    MAX_ITER {int(settings.get('max_iter', 200))}",
        f"    EXTERNAL_PRESSURE {_fmt_float(settings.get('external_pressure', 1.01325))}",
        f"    PRESSURE_TOLERANCE {_fmt_float(settings.get('pressure_tolerance', 100.0))}",
        f"    MAX_FORCE {_fmt_float(settings.get('max_force', 4.5e-4))}",
        f"    RMS_FORCE {_fmt_float(settings.get('rms_force', 3.0e-4))}",
        f"    MAX_DR {_fmt_float(settings.get('max_dr', 3.0e-3))}",
        f"    RMS_DR {_fmt_float(settings.get('rms_dr', 1.5e-3))}",
        "  &END CELL_OPT",
        "&END MOTION",
    ]


def vibrational_top_level_lines(settings: dict[str, Any]) -> list[str]:
    vib = settings.get("properties") if isinstance(settings.get("properties"), dict) else {}
    dx = vib.get("dx", 0.01) if isinstance(vib, dict) else 0.01
    lines = ["&VIBRATIONAL_ANALYSIS", f"  DX {_fmt_float(dx)}"]
    if isinstance(vib, dict) and bool(vib.get("thermochemistry")):
        lines.append("  THERMOCHEMISTRY")
    lines.append("&END VIBRATIONAL_ANALYSIS")
    return lines


def band_motion_lines(settings: dict[str, Any], replica_files: list[str]) -> list[str]:
    lines = [
        "&MOTION",
        "  &BAND",
        f"    BAND_TYPE {str(settings.get('band_type') or 'CI-NEB').upper()}",
        f"    NUMBER_OF_REPLICA {len(replica_files)}",
        f"    K_SPRING {_fmt_float(settings.get('k_spring', 0.02))}",
    ]
    if settings.get("nproc_rep") is not None:
        lines.append(f"    NPROC_REP {int(settings['nproc_rep'])}")
    lines.extend(
        [
            "    &OPTIMIZE_BAND",
            f"      OPT_TYPE {str(settings.get('band_optimizer') or 'DIIS').upper()}",
            f"      OPTIMIZE_END_POINTS {'.TRUE.' if bool(settings.get('optimize_endpoints', False)) else '.FALSE.'}",
            f"      MAX_ITER {int(settings.get('band_max_iter', settings.get('max_iter', 200)))}",
            "    &END OPTIMIZE_BAND",
        ]
    )
    for rel in replica_files:
        lines.extend(["    &REPLICA", f"      COORD_FILE_NAME {rel}", "    &END REPLICA"])
    lines.extend(["  &END BAND", "&END MOTION"])
    return lines


def dimer_motion_lines(settings: dict[str, Any], *, vector_lines: list[str] | None = None) -> list[str]:
    lines = [
        "&MOTION",
        "  &GEO_OPT",
        "    TYPE TRANSITION_STATE",
        f"    OPTIMIZER {str(settings.get('optimizer') or 'CG').upper()}",
        f"    MAX_ITER {int(settings.get('max_iter', 200))}",
        f"    MAX_FORCE {_fmt_float(settings.get('max_force', 8.0e-4))}",
        "    &TRANSITION_STATE",
        "      METHOD DIMER",
        "      &DIMER",
        f"        DR {_fmt_float(settings.get('dimer_dr', 0.01))}",
        "        &ROT_OPT",
        f"          OPTIMIZER {str(settings.get('dimer_rot_optimizer') or 'CG').upper()}",
        "        &END ROT_OPT",
    ]
    if vector_lines:
        lines.append("        &DIMER_VECTOR")
        lines.extend(f"          {line.strip()}" for line in vector_lines if line.strip())
        lines.append("        &END DIMER_VECTOR")
    lines.extend(
        [
            "      &END DIMER",
            "    &END TRANSITION_STATE",
            "  &END GEO_OPT",
            "  &PRINT",
            "    &TRAJECTORY",
            "      &EACH",
            "        GEO_OPT 1",
            "      &END EACH",
            "    &END TRAJECTORY",
            "  &END PRINT",
            "&END MOTION",
        ]
    )
    return lines


def write_cp2k_stage(
    *,
    source_path: Path,
    output_dir: Path,
    recipe: str,
    run_type: str,
    settings: dict[str, Any],
    motion_lines: Iterable[str] | None = None,
    top_level_sections: Iterable[str] | None = None,
    stress_tensor: bool = False,
    extra_files: dict[Path, str] | None = None,
) -> dict[str, Any]:
    atoms = read_atoms(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    project = str(settings.get("project") or safe_stage_name(source_path)).strip() or "cp2k_job"
    inp_path = output_dir / "job.inp"
    structure_path = output_dir / "input.xyz"
    ase_write(str(structure_path), atoms, format="xyz")
    inp_path.write_text(
        cp2k_input_text(
            atoms=atoms,
            project=project,
            run_type=run_type,
            settings=settings,
            motion_lines=motion_lines,
            top_level_sections=top_level_sections,
            stress_tensor=stress_tensor,
        ),
        encoding="utf-8",
    )
    copied_files: list[str] = []
    for src, rel_dst in (extra_files or {}).items():
        dst = output_dir / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(dst.name if dst.parent == output_dir else dst.relative_to(output_dir).as_posix())
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "program": "cp2k",
        "recipe": recipe,
        "run_type": run_type,
        "source_structure_rel": workspace_relpath(source_path),
        "stage_dir_rel": workspace_relpath(output_dir),
        "input_file": "job.inp",
        "structure_file": "input.xyz",
        "copied_files": copied_files,
        "settings": settings,
        "references": CP2K_REFERENCE_URLS,
    }
    write_json(manifest_path, manifest)
    return {
        "source_path_rel": workspace_relpath(source_path),
        "stage_dir_rel": workspace_relpath(output_dir),
        "input_file_rel": workspace_relpath(inp_path),
        "manifest_rel": workspace_relpath(manifest_path),
        "recipe": recipe,
        "run_type": run_type,
    }


def resolve_optional_workspace_file(raw_path: str | None, *, tool_name: str) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = resolve_workspace_path(str(raw_path), must_exist=True)
    if not path.is_file():
        tool_error(
            tool_name,
            f"Expected a file path, got {workspace_relpath(path)}.",
            data={"path_rel": workspace_relpath(path)},
            error_code="not_a_file",
        )
    return path
