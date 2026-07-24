from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Annotated, Any, Literal

from ase.io import read as ase_read
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_records_for_artifact, resolve_workspace_path, workspace_relpath


_MOLECULE_EXTS = {".xyz", ".mol", ".sdf", ".mol2", ".pdb", ".vasp", ".cif"}
_INTERNAL_DIRS = {"metadata", ".catmaster"}
_SKIP_PREFIXES = ("xtb_batch_", "crest_batch_", "orca_batch_", "vasp_batch_", "mace_batch_")
_AtomIndex = Annotated[int, Field(ge=0)]


class XtbDistanceConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom1: int = Field(..., ge=0, description="First atom index, using 0-based indexing.")
    atom2: int = Field(..., ge=0, description="Second atom index, using 0-based indexing.")
    value_angstrom: float = Field(..., gt=0.0, description="Target distance in angstrom.")

    @model_validator(mode="after")
    def _distinct_atoms(self) -> "XtbDistanceConstraint":
        if self.atom1 == self.atom2:
            raise ValueError("Distance constraints require two distinct atoms.")
        return self


class XtbAngleConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom1: int = Field(..., ge=0, description="First atom index, using 0-based indexing.")
    atom2: int = Field(..., ge=0, description="Vertex atom index, using 0-based indexing.")
    atom3: int = Field(..., ge=0, description="Third atom index, using 0-based indexing.")
    value_degree: float = Field(..., gt=0.0, le=180.0, description="Target angle in degrees.")

    @model_validator(mode="after")
    def _distinct_atoms(self) -> "XtbAngleConstraint":
        if len({self.atom1, self.atom2, self.atom3}) != 3:
            raise ValueError("Angle constraints require three distinct atoms.")
        return self


class XtbDihedralConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom1: int = Field(..., ge=0, description="First atom index, using 0-based indexing.")
    atom2: int = Field(..., ge=0, description="Second atom index, using 0-based indexing.")
    atom3: int = Field(..., ge=0, description="Third atom index, using 0-based indexing.")
    atom4: int = Field(..., ge=0, description="Fourth atom index, using 0-based indexing.")
    value_degree: float = Field(..., ge=-360.0, le=360.0, description="Target dihedral angle in degrees.")

    @model_validator(mode="after")
    def _distinct_atoms(self) -> "XtbDihedralConstraint":
        if len({self.atom1, self.atom2, self.atom3, self.atom4}) != 4:
            raise ValueError("Dihedral constraints require four distinct atoms.")
        return self


class XtbPrepareInput(BaseModel):
    """[xtb/prepare] Prepare complete xTB stages without executing xTB."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        ...,
        description=(
            "Single molecular structure or directory of structures; one scientific setting and constraint set "
            "is applied to every discovered structure."
        ),
    )
    output_root: str = Field(
        ...,
        description=(
            "Prepared stage directory for one input, or parent directory containing one first-level stage per input."
        ),
    )
    mode: Literal["sp", "opt", "hess", "md"] = Field("opt", description="xTB run type recorded in manifest.json.")
    gfn: Literal["gfn2", "gfn1", "gfnff"] = Field("gfn2", description="xTB Hamiltonian family.")
    solvent_model: Literal["none", "alpb", "gbsa"] = Field("none", description="Implicit-solvation model.")
    solvent: str = Field("", description="Solvent name; leave empty when solvent_model=none.")
    charge: int = Field(0, description="Molecular charge.")
    uhf: int = Field(0, ge=0, description="Number of unpaired electrons.")
    opt_level: Literal["crude", "sloppy", "loose", "normal", "tight", "vtight", "extreme"] = Field(
        "normal",
        description="Optimization tightness recorded for mode=opt.",
    )
    temperature: float = Field(298.15, gt=0.0, description="Generated xcontrol MD temperature in Kelvin.")
    md_time_ps: float = Field(5.0, gt=0.0, description="Generated xcontrol MD duration in ps.")
    timestep_fs: float = Field(1.0, gt=0.0, description="Generated xcontrol MD timestep in fs.")
    md_dump_fs: float = Field(50.0, gt=0.0, description="Generated xcontrol MD trajectory interval in fs.")
    xcontrol_path: str = Field(
        "",
        description=(
            "Optional complete xTB detailed-input file to copy verbatim as xtb.inp. Leave empty to let this tool "
            "generate MD and constraint blocks."
        ),
    )
    fixed_atom_indices: list[_AtomIndex] = Field(
        default_factory=list,
        description="0-based atom indices written as exact Cartesian fixes in a $fix block.",
    )
    constrained_atom_indices: list[_AtomIndex] = Field(
        default_factory=list,
        description="0-based atom indices written as atomic-position constraints in a $constrain block.",
    )
    distance_constraints: list[XtbDistanceConstraint] = Field(
        default_factory=list,
        description="Internal-coordinate distance constraints.",
    )
    angle_constraints: list[XtbAngleConstraint] = Field(
        default_factory=list,
        description="Internal-coordinate angle constraints.",
    )
    dihedral_constraints: list[XtbDihedralConstraint] = Field(
        default_factory=list,
        description="Internal-coordinate dihedral constraints.",
    )
    constraint_force_constant: float = Field(
        0.0,
        ge=0.0,
        description="Constraint force constant in Hartree/Bohr^2; 0 leaves the xTB default.",
    )

    @model_validator(mode="after")
    def _validate_control_ownership(self) -> "XtbPrepareInput":
        solvent = self.solvent.strip()
        if self.solvent_model == "none" and solvent:
            raise ValueError("solvent must be empty when solvent_model=none.")
        if self.solvent_model != "none" and not solvent:
            raise ValueError(f"solvent is required when solvent_model={self.solvent_model}.")
        if self.mode == "md" and self.fixed_atom_indices:
            raise ValueError("xTB exact $fix constraints are deactivated for MD; use $constrain instead.")
        constrain_targets = any(
            (
                self.constrained_atom_indices,
                self.distance_constraints,
                self.angle_constraints,
                self.dihedral_constraints,
            )
        )
        if self.constraint_force_constant > 0.0 and not constrain_targets:
            raise ValueError("constraint_force_constant requires at least one $constrain target.")
        generated_constraints = bool(self.fixed_atom_indices) or constrain_targets
        if self.xcontrol_path.strip() and generated_constraints:
            raise ValueError(
                "xcontrol_path owns the complete detailed input; do not combine it with generated constraint fields."
            )
        if self.xcontrol_path.strip() and self.mode == "md":
            explicit_md_fields = {"temperature", "md_time_ps", "timestep_fs", "md_dump_fs"} & self.model_fields_set
            if explicit_md_fields:
                raise ValueError(
                    "When xcontrol_path is provided for MD, put MD controls in that file instead of passing "
                    f"{', '.join(sorted(explicit_md_fields))}."
                )
        return self


def _tool_error(message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name="xtb_prepare",
        public_message=str(message).strip(),
        artifact={"tool_name": "xtb_prepare", "data": data or {}},
        error_code=error_code,
    )


def _discover_molecule_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part.startswith(_SKIP_PREFIXES) for part in path.parts):
            dirnames[:] = []
            continue
        if any(part in _INTERNAL_DIRS for part in path.parts):
            dirnames[:] = []
            continue
        dirnames[:] = [
            name for name in dirnames if name not in _INTERNAL_DIRS and not name.startswith(_SKIP_PREFIXES)
        ]
        for filename in filenames:
            candidate = path / filename
            if candidate.suffix.lower() in _MOLECULE_EXTS or filename in {"POSCAR", "CONTCAR"}:
                files.append(candidate)
    return sorted(files, key=lambda item: str(item))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _stage_name(source: Path, *, input_root: Path) -> str:
    rel = source.relative_to(input_root)
    if source.name in {"POSCAR", "CONTCAR"}:
        rel = rel.parent / source.name.lower()
    else:
        rel = rel.with_suffix("")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel)).strip("._")
    return token or "structure"


def _coordinate_name(source: Path) -> str:
    if source.name in {"POSCAR", "CONTCAR"}:
        return "coord.vasp"
    suffix = source.suffix.lower()
    return f"coord{suffix or '.xyz'}"


def _all_constraint_indices(params: XtbPrepareInput) -> list[int]:
    indices = [*params.fixed_atom_indices, *params.constrained_atom_indices]
    for item in params.distance_constraints:
        indices.extend((item.atom1, item.atom2))
    for item in params.angle_constraints:
        indices.extend((item.atom1, item.atom2, item.atom3))
    for item in params.dihedral_constraints:
        indices.extend((item.atom1, item.atom2, item.atom3, item.atom4))
    return indices


def _validate_constraint_indices(source: Path, params: XtbPrepareInput) -> int | None:
    indices = _all_constraint_indices(params)
    if not indices:
        return None
    try:
        atoms = ase_read(str(source), index=0)
    except Exception as exc:
        _tool_error(
            f"Could not read {workspace_relpath(source)} to validate xTB constraint indices: {exc}",
            data={"input_rel": workspace_relpath(source)},
            error_code="constraint_index_validation_failed",
        )
    atom_count = len(atoms)
    invalid = sorted({index for index in indices if index >= atom_count})
    if invalid:
        _tool_error(
            f"xTB constraint indices exceed the {atom_count}-atom structure: {invalid}",
            data={"input_rel": workspace_relpath(source), "atom_count": atom_count, "invalid_indices": invalid},
            error_code="constraint_index_out_of_range",
        )
    return atom_count


def _index_list(indices: list[int]) -> str:
    return ", ".join(str(int(index) + 1) for index in indices)


def _render_generated_xcontrol(params: XtbPrepareInput) -> str:
    lines: list[str] = []
    if params.fixed_atom_indices:
        lines.extend(["$fix", f"  atoms: {_index_list(params.fixed_atom_indices)}", "$end"])

    has_constraints = any(
        (
            params.constrained_atom_indices,
            params.distance_constraints,
            params.angle_constraints,
            params.dihedral_constraints,
        )
    ) or params.constraint_force_constant > 0.0
    if has_constraints:
        lines.append("$constrain")
        if params.constraint_force_constant > 0.0:
            lines.append(f"  force constant={float(params.constraint_force_constant):.10g}")
        if params.constrained_atom_indices:
            lines.append(f"  atoms: {_index_list(params.constrained_atom_indices)}")
        for item in params.distance_constraints:
            lines.append(
                f"  distance: {item.atom1 + 1}, {item.atom2 + 1}, {float(item.value_angstrom):.10g}"
            )
        for item in params.angle_constraints:
            lines.append(
                f"  angle: {item.atom1 + 1}, {item.atom2 + 1}, {item.atom3 + 1}, "
                f"{float(item.value_degree):.10g}"
            )
        for item in params.dihedral_constraints:
            lines.append(
                f"  dihedral: {item.atom1 + 1}, {item.atom2 + 1}, {item.atom3 + 1}, {item.atom4 + 1}, "
                f"{float(item.value_degree):.10g}"
            )
        lines.append("$end")

    if params.mode == "md":
        lines.extend(
            [
                "$md",
                f"  temp={float(params.temperature):.10g}",
                f"  time={float(params.md_time_ps):.10g}",
                f"  step={float(params.timestep_fs):.10g}",
                f"  dump={float(params.md_dump_fs):.10g}",
                "$end",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_stage_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _prepare_stage(
    *,
    source: Path,
    stage_dir: Path,
    params: XtbPrepareInput,
    xcontrol_source: Path | None,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    atom_count = _validate_constraint_indices(source, params)
    coordinate_name = _coordinate_name(source)
    _copy_stage_file(source, stage_dir / coordinate_name)

    xcontrol_name = ""
    if xcontrol_source is not None:
        xcontrol_name = "xtb.inp"
        _copy_stage_file(xcontrol_source, stage_dir / xcontrol_name)
    else:
        generated = _render_generated_xcontrol(params)
        if generated:
            xcontrol_name = "xtb.inp"
            (stage_dir / xcontrol_name).write_text(generated, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "program": "xtb",
        "coordinate_file": coordinate_name,
        "xcontrol_file": xcontrol_name,
        "mode": params.mode,
        "gfn": params.gfn,
        "solvent_model": params.solvent_model,
        "solvent": params.solvent.strip(),
        "charge": params.charge,
        "uhf": params.uhf,
        "opt_level": params.opt_level,
        "source_rel": workspace_relpath(source),
        "xcontrol_source_rel": workspace_relpath(xcontrol_source) if xcontrol_source is not None else "",
        "atom_count": atom_count,
    }
    manifest_path = stage_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "source_rel": workspace_relpath(source),
        "stage_dir_rel": workspace_relpath(stage_dir),
        "coordinate_rel": workspace_relpath(stage_dir / coordinate_name),
        "xcontrol_rel": workspace_relpath(stage_dir / xcontrol_name) if xcontrol_name else "",
        "manifest_rel": workspace_relpath(manifest_path),
        "mode": params.mode,
    }


def xtb_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[xtb/prepare] Prepare complete xTB stages without executing xTB."""
    params = XtbPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    if input_path.is_dir() and _is_within(output_root, input_path):
        _tool_error(
            "output_root must not be inside an input directory.",
            data={"input_path_rel": workspace_relpath(input_path), "output_root_rel": workspace_relpath(output_root)},
            error_code="output_inside_input",
        )
    xcontrol_source = (
        resolve_workspace_path(params.xcontrol_path, must_exist=True) if params.xcontrol_path.strip() else None
    )
    if xcontrol_source is not None and not xcontrol_source.is_file():
        _tool_error(
            "xcontrol_path must name a file.",
            data={"xcontrol_path_rel": workspace_relpath(xcontrol_source)},
            error_code="xcontrol_not_file",
        )

    structures = _discover_molecule_files(input_path)
    if not structures:
        _tool_error(
            "No supported molecular structures found under input_path.",
            data={"input_path_rel": workspace_relpath(input_path)},
            error_code="no_structures",
        )

    input_root = input_path if input_path.is_dir() else None
    claimed_stage_names: dict[str, Path] = {}
    stage_pairs: list[tuple[Path, Path]] = []
    for source in structures:
        if input_root is None:
            stage_dir = output_root
        else:
            stage_name = _stage_name(source, input_root=input_root)
            previous_source = claimed_stage_names.get(stage_name)
            if previous_source is not None:
                _tool_error(
                    "Two input structures map to the same first-level xTB stage name.",
                    data={
                        "stage_name": stage_name,
                        "first_input_rel": workspace_relpath(previous_source),
                        "second_input_rel": workspace_relpath(source),
                    },
                    error_code="stage_name_collision",
                )
            claimed_stage_names[stage_name] = source
            stage_dir = output_root / stage_name
        stage_pairs.append((source, stage_dir))

    records: list[dict[str, Any]] = []
    for source, stage_dir in stage_pairs:
        records.append(
            _prepare_stage(
                source=source,
                stage_dir=stage_dir,
                params=params,
                xcontrol_source=xcontrol_source,
            )
        )

    aggregate_manifest = output_root / "xtb_prepare_manifest.json"
    _write_json(
        aggregate_manifest,
        {
            "input_path_rel": workspace_relpath(input_path),
            "output_root_rel": workspace_relpath(output_root),
            "mode": params.mode,
            "prepared_count": len(records),
            "records": records,
        },
    )
    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "mode": params.mode,
        "prepared_count": len(records),
        "prepare_manifest_rel": workspace_relpath(aggregate_manifest),
        **compact_records_for_artifact(records, full_records_rel=workspace_relpath(aggregate_manifest)),
    }
    content = (
        "xtb_prepare completed.\n"
        f"mode={params.mode} prepared_count={len(records)} output_root_rel={data['output_root_rel']}\n"
        f"prepare_manifest_rel={data['prepare_manifest_rel']}"
    )
    return content, {"tool_name": "xtb_prepare", "data": data}


__all__ = [
    "XtbDistanceConstraint",
    "XtbAngleConstraint",
    "XtbDihedralConstraint",
    "XtbPrepareInput",
    "xtb_prepare",
]
