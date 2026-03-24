from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_SUPPORTED_EXTS = {".xyz", ".mol", ".sdf", ".mol2", ".pdb", ".vasp", ".cif"}
_TIGHTNESS_KEYWORDS = {
    "loose": ("LooseSCF", "LooseOpt"),
    "normal": ("TightSCF", ""),
    "tight": ("VeryTightSCF", "TightOpt"),
}
_DISPERSION_KEYWORDS = {
    "none": "",
    "d3bj": "D3BJ",
    "d4": "D4",
}
_ORCA_METHOD_ALIASES = {
    "gfn2-xtb": "XTB2",
    "gfn1-xtb": "XTB1",
}
_SAFE_PATCH_KEYS = {"ri", "grid", "final_grid", "scf_maxiter", "calc_hess", "recalc_hess", "nroots"}


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


def _read_atoms(path: Path) -> Atoms:
    try:
        return ase_read(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to read structure {path}: {exc}") from exc


def _discover_structures(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in _SUPPORTED_EXTS and candidate.name not in {"POSCAR", "CONTCAR"}:
            continue
        files.append(candidate)
    return files


def _orca_method_name(method: str) -> str:
    normalized = str(method or "").strip()
    if not normalized:
        raise ValueError("method is required")
    return _ORCA_METHOD_ALIASES.get(normalized.lower(), normalized)


def _orca_solvent_keywords(solvent_model: str, solvent: str | None) -> list[str]:
    model = str(solvent_model or "none").strip().lower()
    solvent_name = str(solvent or "").strip()
    if model == "none":
        return []
    if not solvent_name:
        raise ValueError("solvent must be provided when solvent_model is enabled")
    if model == "cpcm":
        return [f"CPCM({solvent_name})"]
    if model == "smd":
        return [f"CPCM({solvent_name})", "SMD"]
    raise ValueError(f"Unsupported solvent_model: {solvent_model}")


def _orca_task_keywords(task: str, *, tightness: str, nroots: int | None = None) -> tuple[list[str], list[str]]:
    task_name = str(task).strip().lower()
    scf_kw, opt_kw = _TIGHTNESS_KEYWORDS[tightness]
    keywords: list[str] = [scf_kw]
    if task_name == "sp":
        pass
    elif task_name == "opt":
        keywords.append(opt_kw or "Opt")
    elif task_name == "freq":
        keywords.append("Freq")
    elif task_name == "optfreq":
        keywords.extend([opt_kw or "Opt", "Freq"])
    elif task_name == "td":
        # ORCA 5 accepts the TD setup through the %tddft block; adding a
        # simple-input TDDFT keyword causes an input parse error there.
        pass
    elif task_name == "nmr":
        keywords.append("NMR")
    else:
        raise ValueError(f"Unsupported ORCA task: {task}")
    blocks: list[str] = []
    if task_name == "td":
        blocks.extend(["%tddft", f"  NRoots {int(nroots or 10)}", "end"])
    return keywords, blocks


def _safe_patch_lines(safe_patch: dict[str, Any]) -> tuple[list[str], list[str]]:
    if not safe_patch:
        return [], []
    unknown = sorted(set(safe_patch) - _SAFE_PATCH_KEYS)
    if unknown:
        raise ValueError(f"Unsupported safe_patch keys: {', '.join(unknown)}")
    keywords: list[str] = []
    blocks: list[str] = []
    if bool(safe_patch.get("ri")):
        keywords.append("RIJCOSX")
    if safe_patch.get("grid") is not None or safe_patch.get("final_grid") is not None:
        blocks.append("%method")
        if safe_patch.get("grid") is not None:
            blocks.append(f"  Grid {safe_patch['grid']}")
        if safe_patch.get("final_grid") is not None:
            blocks.append(f"  FinalGrid {safe_patch['final_grid']}")
        blocks.append("end")
    if safe_patch.get("scf_maxiter") is not None:
        blocks.extend(["%scf", f"  MaxIter {int(safe_patch['scf_maxiter'])}", "end"])
    return keywords, blocks


def _format_constraints(*, frozen_atom_indices: list[int]) -> list[str]:
    if not frozen_atom_indices:
        return []
    lines = ["%geom", "  Constraints"]
    for index in frozen_atom_indices:
        lines.append(f"    {{ C {int(index)} C }}")
    lines.extend(["  end", "end"])
    return lines


def _format_scan_block(scan_type: str, atom_indices: list[int], start_value: float, end_value: float, steps: int) -> list[str]:
    kind = {"distance": "B", "angle": "A", "dihedral": "D"}[scan_type]
    joined = " ".join(str(int(idx)) for idx in atom_indices)
    return [
        "%geom",
        "  Scan",
        f"    {kind} {joined} = {float(start_value):.6f}, {float(end_value):.6f}, {int(steps)}",
        "  end",
        "end",
    ]


def _format_ts_block(*, calc_hess: bool, recalc_hess: int | None, ts_mode: str | None, active_atoms: list[int] | None) -> list[str]:
    lines = ["%geom"]
    if calc_hess:
        lines.append("  Calc_Hess true")
    if recalc_hess is not None:
        lines.append(f"  Recalc_Hess {int(recalc_hess)}")
    if ts_mode:
        lines.append(f"  TS_Mode {ts_mode}")
    if active_atoms:
        joined = " ".join(str(int(idx)) for idx in active_atoms)
        lines.append(f"  TS_Active_Atoms {{ {joined} }} end")
    lines.append("end")
    return lines


def _format_neb_block(product_xyz_name: str, *, nimages: int, variant: str, preopt_method: str | None) -> tuple[list[str], str]:
    variant_name = variant.strip().lower()
    keyword_map = {
        "default": "NEB-TS",
        "loose": "Loose-NEB-TS",
        "fast": "Fast-NEB-TS",
        "zoom": "Zoom-NEB-TS",
    }
    run_keyword = keyword_map[variant_name]
    lines = ["%neb", f"  neb_end_xyzfile \"{product_xyz_name}\"", f"  NImages {int(nimages)}"]
    if preopt_method:
        lines.append(f"  Interpolation {preopt_method}")
    lines.append("end")
    return lines, run_keyword


@dataclass(frozen=True)
class _PreparedDir:
    input_path: Path
    run_dir: Path
    xyz_path: Path
    inp_path: Path


class OrcaPrepareInput(BaseModel):
    """[orca/prepare] Prepare ORCA input directories for molecular single-point, optimization, frequency, TDDFT, or NMR work."""

    input_path: str = Field(..., description="Single molecular structure file or a directory of molecular structures.")
    output_root: str = Field(..., description="Output root for prepared ORCA job directories.")
    task: Literal["sp", "opt", "freq", "optfreq", "td", "nmr"] = Field(..., description="Canonical ORCA task kind.")
    method: str = Field("r2SCAN-3c", description="ORCA method keyword, e.g. r2SCAN-3c, B3LYP, PBE0, XTB2.")
    basis: str = Field("def2-SVP", description="Primary basis-set keyword.")
    aux_basis: str | None = Field(None, description="Optional auxiliary basis keyword.")
    dispersion: Literal["none", "d3bj", "d4"] = Field("none", description="Dispersion correction keyword.")
    solvent_model: Literal["none", "cpcm", "smd"] = Field("none", description="Implicit-solvation model.")
    solvent: str | None = Field(None, description="Solvent name for CPCM/SMD.")
    charge: int = Field(0, description="Total molecular charge.")
    multiplicity: int = Field(1, ge=1, description="Spin multiplicity.")
    nprocs: int = Field(8, ge=1, description="Requested CPU processes for %pal.")
    maxcore_mb: int = Field(2000, ge=1, description="ORCA %maxcore value in MB.")
    tightness: Literal["loose", "normal", "tight"] = Field("normal", description="Optimization/SCF tightness preset.")
    safe_patch: dict[str, Any] = Field(default_factory=dict, description="Restricted patch map for a small whitelist of ORCA settings.")
    frozen_atom_indices: list[int] = Field(default_factory=list, description="Optional 0-based atom indices to freeze via %geom Constraints.")


class OrcaScanPrepareInput(OrcaPrepareInput):
    task: Literal["opt"] = Field("opt", description="Relaxed scan jobs are built on top of geometry optimization.")
    scan_type: Literal["distance", "angle", "dihedral"] = Field(..., description="Internal coordinate type to scan.")
    atom_indices: list[int] = Field(..., description="0-based atom indices for the scanned coordinate.")
    start_value: float = Field(..., description="Initial value for the scanned coordinate.")
    end_value: float = Field(..., description="Final value for the scanned coordinate.")
    steps: int = Field(..., ge=1, description="Number of scan intervals.")

    @model_validator(mode="after")
    def _validate_indices(self) -> "OrcaScanPrepareInput":
        expected = {"distance": 2, "angle": 3, "dihedral": 4}[self.scan_type]
        if len(self.atom_indices) != expected:
            raise ValueError(f"{self.scan_type} scans require {expected} atom indices")
        return self


class OrcaOptTSPrepareInput(OrcaPrepareInput):
    task: Literal["opt"] = Field("opt", description="TS optimization starts from one TS guess structure.")
    calc_hess: bool = Field(True, description="Whether to compute an initial Hessian.")
    recalc_hess: int | None = Field(5, ge=1, description="Optional Hessian refresh interval in optimization steps.")
    ts_mode: str | None = Field(
        None,
        description="Optional ORCA TS_Mode expression such as `{M 0}` or `{B 1 5}`. Uses ORCA syntax directly because the grammar is compact and stable.",
    )
    active_atoms: list[int] = Field(default_factory=list, description="Optional 0-based active-atom list for TS search guidance.")


class OrcaNebTSPrepareInput(BaseModel):
    """[orca/prepare] Prepare an ORCA NEB-TS job from reactant and product geometries."""

    reactant_path: str = Field(..., description="Reactant structure file.")
    product_path: str = Field(..., description="Product structure file.")
    output_root: str = Field(..., description="Prepared NEB-TS run directory.")
    method: str = Field("XTB2", description="Method keyword for the NEB-TS run.")
    basis: str = Field("", description="Optional basis keyword; leave empty for semiempirical methods such as XTB2.")
    aux_basis: str | None = Field(None, description="Optional auxiliary basis keyword.")
    dispersion: Literal["none", "d3bj", "d4"] = Field("none", description="Dispersion correction keyword.")
    solvent_model: Literal["none", "cpcm", "smd"] = Field("none", description="Implicit-solvation model.")
    solvent: str | None = Field(None, description="Solvent name for CPCM/SMD.")
    charge: int = Field(0, description="Total molecular charge.")
    multiplicity: int = Field(1, ge=1, description="Spin multiplicity.")
    nprocs: int = Field(8, ge=1, description="Requested CPU processes for %pal.")
    maxcore_mb: int = Field(2000, ge=1, description="ORCA %maxcore value in MB.")
    tightness: Literal["loose", "normal", "tight"] = Field("normal", description="SCF/optimization tightness.")
    nimages: int = Field(8, ge=2, description="Number of interpolated NEB images.")
    variant: Literal["default", "loose", "fast", "zoom"] = Field("default", description="NEB-TS flavor.")
    preopt_method: Literal["linear", "idpp", "xtb", "xtb0", "xtb1", "xtb2"] | None = Field(
        "idpp",
        description="Interpolation/preoptimization method written into the %neb block when supported.",
    )
    safe_patch: dict[str, Any] = Field(default_factory=dict, description="Restricted patch map for supported ORCA settings.")


class OrcaIRCPrepareInput(OrcaPrepareInput):
    task: Literal["freq"] = Field("freq", description="IRC runs are typically launched from a TS/frequency structure.")


def _build_simple_keywords(*, task: str, method: str, basis: str, aux_basis: str | None, dispersion: str, solvent_model: str, solvent: str | None, tightness: str, safe_patch: dict[str, Any]) -> tuple[list[str], list[str]]:
    keywords, blocks = _orca_task_keywords(task, tightness=tightness, nroots=safe_patch.get("nroots"))
    method_name = _orca_method_name(method)
    all_keywords = [method_name]
    basis_text = str(basis or "").strip()
    if basis_text:
        all_keywords.append(basis_text)
    aux_text = str(aux_basis or "").strip()
    if aux_text:
        all_keywords.append(aux_text)
    dispersion_kw = _DISPERSION_KEYWORDS[str(dispersion)]
    if dispersion_kw:
        all_keywords.append(dispersion_kw)
    all_keywords.extend(_orca_solvent_keywords(solvent_model, solvent))
    safe_keywords, safe_blocks = _safe_patch_lines(safe_patch)
    all_keywords.extend(keywords)
    all_keywords.extend(safe_keywords)
    blocks.extend(safe_blocks)
    seen: set[str] = set()
    deduped: list[str] = []
    for item in all_keywords:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped, blocks


def _render_orca_input(
    *,
    task: str,
    xyz_name: str,
    method: str,
    basis: str,
    aux_basis: str | None,
    dispersion: str,
    solvent_model: str,
    solvent: str | None,
    charge: int,
    multiplicity: int,
    nprocs: int,
    maxcore_mb: int,
    tightness: str,
    safe_patch: dict[str, Any],
    extra_blocks: list[str],
    frozen_atom_indices: list[int],
) -> str:
    simple_keywords, blocks = _build_simple_keywords(
        task=task,
        method=method,
        basis=basis,
        aux_basis=aux_basis,
        dispersion=dispersion,
        solvent_model=solvent_model,
        solvent=solvent,
        tightness=tightness,
        safe_patch=safe_patch,
    )
    lines = [
        "! " + " ".join(simple_keywords),
        "%pal",
        f"  nprocs {int(nprocs)}",
        "end",
        f"%maxcore {int(maxcore_mb)}",
    ]
    lines.extend(blocks)
    lines.extend(_format_constraints(frozen_atom_indices=frozen_atom_indices))
    lines.extend(extra_blocks)
    lines.append(f"* xyzfile {int(charge)} {int(multiplicity)} {xyz_name}")
    return "\n".join(lines) + "\n"


def _prepare_job_dir(
    *,
    structure_path: Path,
    output_root: Path,
    relative_name: str,
    input_text: str,
) -> _PreparedDir:
    run_dir = output_root / relative_name
    run_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = run_dir / "input.xyz"
    atoms = _read_atoms(structure_path)
    ase_write(str(xyz_path), atoms, format="xyz")
    inp_path = run_dir / "job.inp"
    inp_path.write_text(input_text, encoding="utf-8")
    return _PreparedDir(input_path=structure_path, run_dir=run_dir, xyz_path=xyz_path, inp_path=inp_path)


def orca_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = OrcaPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    structures = _discover_structures(input_path)
    if not structures:
        _tool_error("orca_prepare", f"No supported structures found under {input_path}", error_code="no_structures")
    prepared: list[dict[str, Any]] = []
    for structure in structures:
        relative_name = structure.stem if input_path.is_file() else str(structure.relative_to(input_path).with_suffix(""))
        input_text = _render_orca_input(
            task=params.task,
            xyz_name="input.xyz",
            method=params.method,
            basis=params.basis,
            aux_basis=params.aux_basis,
            dispersion=params.dispersion,
            solvent_model=params.solvent_model,
            solvent=params.solvent,
            charge=params.charge,
            multiplicity=params.multiplicity,
            nprocs=params.nprocs,
            maxcore_mb=params.maxcore_mb,
            tightness=params.tightness,
            safe_patch=params.safe_patch,
            extra_blocks=[],
            frozen_atom_indices=params.frozen_atom_indices,
        )
        built = _prepare_job_dir(structure_path=structure, output_root=output_root, relative_name=relative_name, input_text=input_text)
        prepared.append(
            {
                "input_rel": workspace_relpath(structure),
                "run_dir_rel": workspace_relpath(built.run_dir),
                "xyz_rel": workspace_relpath(built.xyz_path),
                "inp_rel": workspace_relpath(built.inp_path),
            }
        )
    manifest = output_root / "orca_prepare_manifest.json"
    _write_json(
        manifest,
        {
            "task": params.task,
            "method": params.method,
            "basis": params.basis,
            "records": prepared,
        },
    )
    data = {
        "task": params.task,
        "output_root_rel": workspace_relpath(output_root),
        "manifest_rel": workspace_relpath(manifest),
        "prepared_count": len(prepared),
        "records": prepared,
    }
    content = (
        "orca_prepare completed.\n"
        f"task={params.task} prepared_count={len(prepared)} output_root_rel={data['output_root_rel']}\n"
        f"manifest_rel={data['manifest_rel']}"
    )
    return content, {"tool_name": "orca_prepare", "data": data}


def orca_scan_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = OrcaScanPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    structures = _discover_structures(input_path)
    if len(structures) != 1:
        _tool_error("orca_scan_prepare", "orca_scan_prepare currently expects one input structure.", error_code="single_structure_required")
    extra_blocks = _format_scan_block(
        params.scan_type,
        params.atom_indices,
        params.start_value,
        params.end_value,
        params.steps,
    )
    input_text = _render_orca_input(
        task="opt",
        xyz_name="input.xyz",
        method=params.method,
        basis=params.basis,
        aux_basis=params.aux_basis,
        dispersion=params.dispersion,
        solvent_model=params.solvent_model,
        solvent=params.solvent,
        charge=params.charge,
        multiplicity=params.multiplicity,
        nprocs=params.nprocs,
        maxcore_mb=params.maxcore_mb,
        tightness=params.tightness,
        safe_patch=params.safe_patch,
        extra_blocks=extra_blocks,
        frozen_atom_indices=params.frozen_atom_indices,
    )
    built = _prepare_job_dir(structure_path=structures[0], output_root=output_root, relative_name=structures[0].stem, input_text=input_text)
    manifest = output_root / "orca_scan_manifest.json"
    _write_json(
        manifest,
        {
            "scan_type": params.scan_type,
            "atom_indices": params.atom_indices,
            "start_value": params.start_value,
            "end_value": params.end_value,
            "steps": params.steps,
            "run_dir_rel": workspace_relpath(built.run_dir),
            "inp_rel": workspace_relpath(built.inp_path),
        },
    )
    data = {
        "output_root_rel": workspace_relpath(output_root),
        "run_dir_rel": workspace_relpath(built.run_dir),
        "inp_rel": workspace_relpath(built.inp_path),
        "manifest_rel": workspace_relpath(manifest),
    }
    content = (
        "orca_scan_prepare completed.\n"
        f"run_dir_rel={data['run_dir_rel']} inp_rel={data['inp_rel']}\n"
        f"manifest_rel={data['manifest_rel']}"
    )
    return content, {"tool_name": "orca_scan_prepare", "data": data}


def orca_optts_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = OrcaOptTSPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    structures = _discover_structures(input_path)
    if len(structures) != 1:
        _tool_error("orca_optts_prepare", "orca_optts_prepare currently expects one input structure.", error_code="single_structure_required")
    extra_blocks = _format_ts_block(
        calc_hess=params.calc_hess or bool(params.safe_patch.get("calc_hess")),
        recalc_hess=params.recalc_hess if params.recalc_hess is not None else params.safe_patch.get("recalc_hess"),
        ts_mode=params.ts_mode,
        active_atoms=params.active_atoms,
    )
    input_text = _render_orca_input(
        task="opt",
        xyz_name="input.xyz",
        method=params.method,
        basis=params.basis,
        aux_basis=params.aux_basis,
        dispersion=params.dispersion,
        solvent_model=params.solvent_model,
        solvent=params.solvent,
        charge=params.charge,
        multiplicity=params.multiplicity,
        nprocs=params.nprocs,
        maxcore_mb=params.maxcore_mb,
        tightness=params.tightness,
        safe_patch=params.safe_patch,
        extra_blocks=extra_blocks,
        frozen_atom_indices=params.frozen_atom_indices,
    ).replace(" Opt", " OptTS", 1)
    built = _prepare_job_dir(structure_path=structures[0], output_root=output_root, relative_name=structures[0].stem, input_text=input_text)
    manifest = output_root / "orca_optts_manifest.json"
    _write_json(
        manifest,
        {
            "run_dir_rel": workspace_relpath(built.run_dir),
            "inp_rel": workspace_relpath(built.inp_path),
            "calc_hess": params.calc_hess,
            "recalc_hess": params.recalc_hess,
            "ts_mode": params.ts_mode,
            "active_atoms": params.active_atoms,
        },
    )
    data = {
        "output_root_rel": workspace_relpath(output_root),
        "run_dir_rel": workspace_relpath(built.run_dir),
        "inp_rel": workspace_relpath(built.inp_path),
        "manifest_rel": workspace_relpath(manifest),
    }
    content = (
        "orca_optts_prepare completed.\n"
        f"run_dir_rel={data['run_dir_rel']} inp_rel={data['inp_rel']}\n"
        f"manifest_rel={data['manifest_rel']}"
    )
    return content, {"tool_name": "orca_optts_prepare", "data": data}


def orca_nebts_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = OrcaNebTSPrepareInput(**payload)
    reactant = resolve_workspace_path(params.reactant_path, must_exist=True)
    product = resolve_workspace_path(params.product_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reactant_xyz = output_root / "reactant.xyz"
    product_xyz = output_root / "product.xyz"
    ase_write(str(reactant_xyz), _read_atoms(reactant), format="xyz")
    ase_write(str(product_xyz), _read_atoms(product), format="xyz")
    neb_block, run_keyword = _format_neb_block(
        product_xyz_name=product_xyz.name,
        nimages=params.nimages,
        variant=params.variant,
        preopt_method=params.preopt_method,
    )
    input_text = _render_orca_input(
        task="sp",
        xyz_name=reactant_xyz.name,
        method=params.method,
        basis=params.basis,
        aux_basis=params.aux_basis,
        dispersion=params.dispersion,
        solvent_model=params.solvent_model,
        solvent=params.solvent,
        charge=params.charge,
        multiplicity=params.multiplicity,
        nprocs=params.nprocs,
        maxcore_mb=params.maxcore_mb,
        tightness=params.tightness,
        safe_patch=params.safe_patch,
        extra_blocks=neb_block,
        frozen_atom_indices=[],
    ).replace("! ", f"! {run_keyword} ", 1)
    inp_path = output_root / "job.inp"
    inp_path.write_text(input_text, encoding="utf-8")
    manifest = output_root / "orca_nebts_manifest.json"
    _write_json(
        manifest,
        {
            "reactant_rel": workspace_relpath(reactant),
            "product_rel": workspace_relpath(product),
            "run_dir_rel": workspace_relpath(output_root),
            "inp_rel": workspace_relpath(inp_path),
            "nimages": params.nimages,
            "variant": params.variant,
            "preopt_method": params.preopt_method,
        },
    )
    data = {
        "output_root_rel": workspace_relpath(output_root),
        "run_dir_rel": workspace_relpath(output_root),
        "inp_rel": workspace_relpath(inp_path),
        "manifest_rel": workspace_relpath(manifest),
        "reactant_xyz_rel": workspace_relpath(reactant_xyz),
        "product_xyz_rel": workspace_relpath(product_xyz),
    }
    content = (
        "orca_nebts_prepare completed.\n"
        f"run_dir_rel={data['run_dir_rel']} inp_rel={data['inp_rel']}\n"
        f"manifest_rel={data['manifest_rel']}"
    )
    return content, {"tool_name": "orca_nebts_prepare", "data": data}


def orca_irc_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = OrcaIRCPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    structures = _discover_structures(input_path)
    if len(structures) != 1:
        _tool_error("orca_irc_prepare", "orca_irc_prepare currently expects one input structure.", error_code="single_structure_required")
    input_text = _render_orca_input(
        task="freq",
        xyz_name="input.xyz",
        method=params.method,
        basis=params.basis,
        aux_basis=params.aux_basis,
        dispersion=params.dispersion,
        solvent_model=params.solvent_model,
        solvent=params.solvent,
        charge=params.charge,
        multiplicity=params.multiplicity,
        nprocs=params.nprocs,
        maxcore_mb=params.maxcore_mb,
        tightness=params.tightness,
        safe_patch=params.safe_patch,
        extra_blocks=[],
        frozen_atom_indices=params.frozen_atom_indices,
    ).replace(" Freq", " Freq IRC", 1)
    built = _prepare_job_dir(structure_path=structures[0], output_root=output_root, relative_name=structures[0].stem, input_text=input_text)
    manifest = output_root / "orca_irc_manifest.json"
    _write_json(
        manifest,
        {
            "run_dir_rel": workspace_relpath(built.run_dir),
            "inp_rel": workspace_relpath(built.inp_path),
        },
    )
    data = {
        "output_root_rel": workspace_relpath(output_root),
        "run_dir_rel": workspace_relpath(built.run_dir),
        "inp_rel": workspace_relpath(built.inp_path),
        "manifest_rel": workspace_relpath(manifest),
    }
    content = (
        "orca_irc_prepare completed.\n"
        f"run_dir_rel={data['run_dir_rel']} inp_rel={data['inp_rel']}\n"
        f"manifest_rel={data['manifest_rel']}"
    )
    return content, {"tool_name": "orca_irc_prepare", "data": data}


__all__ = [
    "OrcaPrepareInput",
    "OrcaScanPrepareInput",
    "OrcaOptTSPrepareInput",
    "OrcaNebTSPrepareInput",
    "OrcaIRCPrepareInput",
    "orca_prepare",
    "orca_scan_prepare",
    "orca_optts_prepare",
    "orca_nebts_prepare",
    "orca_irc_prepare",
]
