from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from ase.io import read as ase_read, write as ase_write

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath

_CELL_TOL = 1e-5


class MakeNebGeometryInput(BaseModel):
    """
    Generate NEB interpolation geometries in nebmake.pl style.

    The tool validates that initial and final structures have the same element order, atom count,
    and lattice (within tolerance). Images are written under output_dir/00..(NI+1) with output_filename.
    Selective dynamics constraints are preserved if present in the input files (ASE handles constraints).
    """

    initial_path: str = Field(..., description="Initial structure file (POSCAR/CONTCAR/.vasp/.cif).")
    final_path: str = Field(..., description="Final structure file (POSCAR/CONTCAR/.vasp/.cif).")
    n_images: int = Field(..., ge=1, description="Number of intermediate images (NI).")
    output_dir: str = Field("neb_images", description="Output directory for image folders (workspace-relative).")
    output_filename: str = Field("POSCAR", description="Filename for each image output (e.g., POSCAR).")
    interp_mode: str = Field(
        "direct",
        description="Output coordinate style when writing POSCAR: direct (fractional) or cartesian.",
        pattern="^(direct|cartesian)$",
    )
    interp_method: str = Field(
        "linear",
        description="Interpolation method: linear or idpp.",
        pattern="^(linear|idpp)$",
    )
    vtst_wrap: bool = Field(
        True,
        description="If true, wrap fractional coordinates into [0,1) before interpolation (VTST nebmake.pl behavior).",
    )
    overwrite: bool = Field(False, description="If true, overwrite output_dir if it exists.")


class MakeNebIncarInput(BaseModel):
    """
    Generate a VTST NEB/CI-NEB INCAR based on a template INCAR.

    The tool parses the template INCAR (ignores comments/blank lines, case-insensitive keys),
    applies NEB overrides, and writes the final INCAR along with a patch JSON.
    """

    template_incar_path: str = Field(..., description="Template INCAR path (workspace-relative).")
    output_incar_path: Optional[str] = Field(
        None,
        description="Output INCAR path. If omitted, uses output_dir/INCAR.",
    )
    output_dir: Optional[str] = Field(
        "neb_inputs",
        description="Output directory for INCAR and patch JSON (workspace-relative).",
    )
    images: int = Field(..., ge=1, description="Number of intermediate images (IMAGES).")
    climb: bool = Field(False, description="Enable CI-NEB with LCLIMB=.TRUE. if true.")
    iopt: int = Field(7, description="IOPT value for VTST optimizer (allowed: 7/2/1).")
    ediff: Optional[float] = Field(None, description="Override EDIFF if provided.")
    ediffg: Optional[float] = Field(None, description="Override EDIFFG if provided.")
    spring: Optional[float] = Field(None, description="Override SPRING if provided.")
    potim: Optional[float] = Field(0.0, description="Override POTIM (default 0).")
    additional_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional INCAR overrides; highest priority and can override required defaults.",
    )


def _read_atoms(path: Path):
    return ase_read(str(path))


def _validate_structures(initial, final) -> Optional[str]:
    if len(initial) != len(final):
        return "Initial and final structures have different atom counts."
    if not np.array_equal(initial.get_atomic_numbers(), final.get_atomic_numbers()):
        return "Initial and final structures have different element sequences."
    if not np.allclose(initial.cell.array, final.cell.array, rtol=_CELL_TOL, atol=_CELL_TOL):
        return "Initial and final lattices differ beyond tolerance."
    return None


def _build_images(
    initial,
    final,
    n_images: int,
    *,
    interp_method: str,
    vtst_wrap: bool,
) -> Tuple[List[Any], List[str]]:
    from ase.neb import NEB

    warnings: List[str] = []
    init_use = initial.copy()
    final_use = final.copy()
    mic = bool(vtst_wrap)

    images_atoms = [init_use]
    for _ in range(n_images):
        images_atoms.append(init_use.copy())
    images_atoms.append(final_use)

    neb = NEB(images_atoms)
    neb.interpolate(method=interp_method, mic=mic)

    images: List[Any] = []
    for img in images_atoms:
        images.append(img)

    return images, warnings


def make_neb_geometry(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MakeNebGeometryInput(**payload)
    init_path = resolve_workspace_path(params.initial_path, must_exist=True)
    final_path = resolve_workspace_path(params.final_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_dir)

    try:
        initial = _read_atoms(init_path)
        final = _read_atoms(final_path)
    except Exception as exc:
        return create_tool_output("make_neb_geometry", success=False, error=str(exc))
    error = _validate_structures(initial, final)
    if error:
        return create_tool_output("make_neb_geometry", success=False, error=error)
    warnings: List[str] = []

    if output_root.exists():
        if output_root.is_file():
            return create_tool_output(
                "make_neb_geometry",
                success=False,
                error=f"output_dir is a file: {output_root}",
            )
        if not params.overwrite:
            return create_tool_output(
                "make_neb_geometry",
                success=False,
                error=f"output_dir already exists: {output_root}. Set overwrite=true to replace.",
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images, interp_warnings = _build_images(
        initial,
        final,
        params.n_images,
        interp_method=params.interp_method,
        vtst_wrap=params.vtst_wrap,
    )
    warnings.extend(interp_warnings)

    image_dirs: List[str] = []
    for idx, atoms in enumerate(images):
        img_dir = output_root / f"{idx:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        out_path = img_dir / params.output_filename
        ase_write(
            str(out_path),
            atoms,
            format="vasp",
            direct=(params.interp_mode == "direct"),
            vasp5=True,
        )
        image_dirs.append(workspace_relpath(img_dir))

    data = {
        "output_dir": workspace_relpath(output_root),
        "num_intermediate_images": params.n_images,
        "num_total_images": params.n_images + 2,
        "image_dirs": image_dirs,
    }
    return create_tool_output("make_neb_geometry", success=True, data=data, warnings=warnings)


def _strip_incar_comment(line: str) -> str:
    for sep in ("!", "#"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.strip()


def _parse_incar(path: Path) -> Tuple[List[str], Dict[str, str]]:
    order: List[str] = []
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = _strip_incar_comment(raw)
        if not line:
            continue
        if "=" in line:
            key, val = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
        key = key.strip().upper()
        val = val.strip()
        values[key] = val
        if key not in order:
            order.append(key)
    return order, values


def _format_incar_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.16g}"
    if isinstance(value, (list, tuple)):
        return " ".join(_format_incar_value(v) for v in value)
    if value is None:
        return ""
    return str(value).strip()


def make_neb_incar(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = MakeNebIncarInput(**payload)
    if params.iopt not in {7, 2, 1}:
        return create_tool_output(
            "make_neb_incar",
            success=False,
            error="iopt must be one of 7, 2, 1.",
        )
    template_path = resolve_workspace_path(params.template_incar_path, must_exist=True)
    output_dir = resolve_workspace_path(params.output_dir or "neb_inputs")
    if output_dir.exists() and output_dir.is_file():
        return create_tool_output(
            "make_neb_incar",
            success=False,
            error=f"output_dir is a file: {output_dir}",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_workspace_path(params.output_incar_path) if params.output_incar_path else output_dir / "INCAR"

    order, template_vals = _parse_incar(template_path)
    final_vals: Dict[str, Any] = dict(template_vals)

    overrides: Dict[str, Any] = {
        "IBRION": 3,
        "POTIM": params.potim if params.potim is not None else 0.0,
        "ICHAIN": 0,
        "IMAGES": params.images,
        "IOPT": params.iopt,
    }
    if params.climb:
        overrides["LCLIMB"] = True
    if params.ediff is not None:
        overrides["EDIFF"] = params.ediff
    if params.ediffg is not None:
        overrides["EDIFFG"] = params.ediffg
    if params.spring is not None:
        overrides["SPRING"] = params.spring

    default_suggestions = {"ISYM": 0, "LWAVE": False, "LCHARG": False}
    for key, value in default_suggestions.items():
        if key not in final_vals:
            final_vals[key] = value

    for key, value in overrides.items():
        final_vals[key] = value

    user_overrides = params.additional_overrides or {}
    for key, value in user_overrides.items():
        key_upper = str(key).upper()
        if value is None:
            final_vals.pop(key_upper, None)
        else:
            final_vals[key_upper] = value

    template_str = {k: _format_incar_value(v) for k, v in template_vals.items()}
    final_str = {k: _format_incar_value(v) for k, v in final_vals.items()}

    changed_keys = {
        key
        for key in set(template_str) | set(final_str)
        if template_str.get(key) != final_str.get(key)
    }
    applied_overrides = {k: final_str[k] for k in sorted(changed_keys) if k in final_str}

    diff = {
        k: {"old": template_str.get(k), "new": final_str.get(k)}
        for k in sorted(changed_keys)
    }
    diff_path = output_dir / "neb_incar_patch.json"
    try:
        diff_path.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    lines: List[str] = []
    for key in order:
        if key in final_str:
            lines.append(f"{key} = {final_str[key]}")
    for key in sorted(set(final_str) - set(order)):
        lines.append(f"{key} = {final_str[key]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = {
        "output_incar_path": workspace_relpath(out_path),
        "applied_overrides": applied_overrides,
        "diff_json_rel": workspace_relpath(diff_path) if diff_path.exists() else None,
    }
    return create_tool_output("make_neb_incar", success=True, data=data)


__all__ = [
    "MakeNebGeometryInput",
    "MakeNebIncarInput",
    "make_neb_geometry",
    "make_neb_incar",
]
