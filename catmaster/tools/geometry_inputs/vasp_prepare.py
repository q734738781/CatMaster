"""Canonical VASP input preparation tool."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Literal

from ase.io import read as ase_read
from pydantic import BaseModel, Field, field_validator, model_validator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

from .adsorbate_tool import propagate_adsorbate_metadata
from .vasp_inputs import StructWriter

SUPPORTED_EXTS = {".vasp", ".cif", ".xyz"}
ELEMENT_MAP_INCAR_KEYS = {"MAGMOM", "LDAUU", "LDAUJ"}


def _element_map_error_message(key: str) -> str:
    return (
        f"{key} must be an element-map in this tool due to pymatgen constraints, "
        'e.g. {"Fe": 2.2} or {"O": 1}.'
    )


def _coerce_element_map_value(key: str, raw_val: Any | None) -> Any | None:
    if raw_val is None:
        return None
    if not isinstance(raw_val, dict):
        raise ValueError(_element_map_error_message(key))
    normalized: Dict[str, Any] = {}
    for sym_raw, value in raw_val.items():
        symbol = str(sym_raw).strip()
        if not symbol:
            raise ValueError(_element_map_error_message(key))
        normalized[symbol] = value
    return normalized


def _normalize_user_incar_patch(value: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_val in value.items():
        key = str(raw_key).strip().upper()
        if not key:
            raise ValueError("INCAR key must be a non-empty string.")
        if key in ELEMENT_MAP_INCAR_KEYS:
            raw_val = _coerce_element_map_value(key, raw_val)
        normalized[key] = raw_val
    return normalized


class VaspPrepareInput(BaseModel):
    """[vasp/prepare] Prepare a canonical single-structure VASP input set for relax/static/freq/dos/md presets."""

    input_path: str = Field(
        ...,
        description="Single structure file path. Current canonical implementation supports one input file at a time.",
    )
    output_root: str = Field(
        ...,
        description="Target directory for generated VASP inputs. For a single structure, write directly into output_root/.",
    )
    preset: Literal["relax", "static", "freq", "dos", "md"] = Field(
        ...,
        description="Canonical VASP preset: relax, static, freq, dos, or md.",
    )
    regime: Literal["bulk", "slab", "gas"] = Field(
        ...,
        description="Scientific regime controlling k-mesh and regime-specific defaults.",
    )
    relax_cell: bool = Field(
        False,
        description="Only valid for bulk relax jobs. When true, use ISIF=3.",
    )
    k_product: int = Field(
        35,
        ge=1,
        description="Target k-mesh density; minimum 1 and odd Gamma-centered mesh.",
    )
    use_d3: bool = Field(False, description="Enable DFT-D3(BJ) correction (IVDW=12).")
    use_dft_plus_u: bool = Field(False, description="Enable DFT+U baseline toggle (LDAU=True).")
    enable_dipole: bool = Field(
        False,
        description="Enable dipole correction helper with IDIPOL=3 and center-of-mass DIPOL.",
    )
    compute_dos: bool = Field(
        False,
        description="Enable DOS/projection output for non-dos presets.",
    )
    dos_charge_density_path: str | None = Field(
        None,
        description="Optional CHGCAR path for preset='dos' non-self-consistent DOS.",
    )
    user_incar_patch: Dict[str, Any] = Field(
        default_factory=dict,
        description="Targeted INCAR patch. MAGMOM/LDAUU/LDAUJ must be element maps; null removes a key.",
    )
    patch_policy: Literal["safe", "force"] = Field(
        "safe",
        description="safe protects preset/regime identity keys; force applies the patch after canonical defaults.",
    )

    @field_validator("user_incar_patch")
    @classmethod
    def _validate_user_incar_patch(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_user_incar_patch(value)

    @model_validator(mode="after")
    def _validate_scope(self) -> "VaspPrepareInput":
        if self.relax_cell and not (self.preset == "relax" and self.regime == "bulk"):
            raise ValueError("relax_cell=True is only allowed when preset='relax' and regime='bulk'.")
        if self.preset != "dos" and self.dos_charge_density_path is not None:
            raise ValueError("dos_charge_density_path is only allowed when preset='dos'.")
        return self


def _load_structure(path: Path) -> Structure:
    if path.suffix.lower() == ".xyz":
        atoms = ase_read(path.as_posix())
        return AseAtomsAdaptor.get_structure(atoms)
    return Structure.from_file(path)


def _validate_single_structure_input(tool_name: str, input_path: Path, preset: str) -> None:
    if not input_path.exists():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} could not find input_path={workspace_relpath(input_path)}.",
            artifact={
                "tool_name": tool_name,
                "data": {"input_path_rel": workspace_relpath(input_path), "preset": preset},
            },
            error_code="missing_input",
        )
    if input_path.is_dir():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} currently supports only a single structure file input, but "
                f"{workspace_relpath(input_path)} is a directory."
            ),
            artifact={
                "tool_name": tool_name,
                "data": {"input_path_rel": workspace_relpath(input_path), "preset": preset},
            },
            error_code="directory_input_not_supported",
        )
    if input_path.suffix.lower() not in SUPPORTED_EXTS:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} does not support {input_path.suffix or '(no extension)'} for "
                f"{workspace_relpath(input_path)}. Supported: {', '.join(sorted(SUPPORTED_EXTS))}."
            ),
            artifact={
                "tool_name": tool_name,
                "data": {
                    "input_path_rel": workspace_relpath(input_path),
                    "supported_exts": sorted(SUPPORTED_EXTS),
                },
            },
            error_code="unsupported_input_extension",
        )


def vasp_prepare(payload: Dict[str, object]) -> tuple[str, dict[str, Any]]:
    """[vasp/prepare] Prepare a canonical VASP input directory for one structure and one preset."""
    tool_name = "vasp_prepare"
    params = VaspPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path)
    output_root = resolve_workspace_path(params.output_root)
    dos_charge_density_path = (
        resolve_workspace_path(params.dos_charge_density_path) if params.dos_charge_density_path is not None else None
    )

    _validate_single_structure_input(tool_name, input_path, params.preset)
    if dos_charge_density_path is not None:
        if not dos_charge_density_path.exists():
            raise CatMasterToolExecutionError(
                tool_name=tool_name,
                public_message=(
                    f"{tool_name} could not find dos_charge_density_path="
                    f"{workspace_relpath(dos_charge_density_path)}."
                ),
                artifact={
                    "tool_name": tool_name,
                    "data": {
                        "dos_charge_density_path_rel": workspace_relpath(dos_charge_density_path),
                        "preset": params.preset,
                    },
                },
                error_code="missing_dos_charge_density",
            )
        if dos_charge_density_path.is_dir():
            raise CatMasterToolExecutionError(
                tool_name=tool_name,
                public_message=(
                    f"{tool_name} expected a CHGCAR file for dos_charge_density_path, but "
                    f"{workspace_relpath(dos_charge_density_path)} is a directory."
                ),
                artifact={
                    "tool_name": tool_name,
                    "data": {
                        "dos_charge_density_path_rel": workspace_relpath(dos_charge_density_path),
                        "preset": params.preset,
                    },
                },
                error_code="invalid_dos_charge_density",
            )

    incar_path = output_root / "INCAR"
    if incar_path.exists():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} refused to overwrite existing VASP inputs in {workspace_relpath(output_root)} "
                "because INCAR already exists there. Choose a different output_root."
            ),
            artifact={
                "tool_name": tool_name,
                "data": {
                    "input_path_rel": workspace_relpath(input_path),
                    "output_root_rel": workspace_relpath(output_root),
                    "conflict_incar_rel": workspace_relpath(incar_path),
                    "preset": params.preset,
                    "regime": params.regime,
                },
            },
            error_code="prepare_target_exists",
        )

    try:
        structure = _load_structure(input_path)
        writer = StructWriter()
        plan = writer.write_vasp_inputs(
            structure=structure,
            output_dir=output_root,
            preset=params.preset,
            regime=params.regime,
            relax_cell=bool(params.relax_cell),
            k_product=int(params.k_product),
            use_d3=bool(params.use_d3),
            user_incar_patch=dict(params.user_incar_patch),
            use_dft_plus_u=bool(params.use_dft_plus_u),
            compute_dos=bool(params.compute_dos),
            enable_dipole=bool(params.enable_dipole),
            dos_use_chgcar=dos_charge_density_path is not None,
            patch_policy=params.patch_policy,
        )
        if dos_charge_density_path is not None:
            shutil.copy2(dos_charge_density_path, output_root / "CHGCAR")
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} failed for {workspace_relpath(input_path)}: {exc}\n"
                f"output_root_rel={workspace_relpath(output_root)}"
            ),
            artifact={
                "tool_name": tool_name,
                "data": {
                    "input_path_rel": workspace_relpath(input_path),
                    "output_root_rel": workspace_relpath(output_root),
                    "preset": params.preset,
                    "regime": params.regime,
                    "patch_policy": params.patch_policy,
                },
            },
            error_code="prepare_failed",
        ) from exc

    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "prepared_directory_rel": workspace_relpath(output_root),
        "preset": params.preset,
        "regime": params.regime,
        "relax_cell": bool(params.relax_cell),
        "k_product": int(params.k_product),
        "k_grid": list(plan.k_grid),
        "use_d3": bool(params.use_d3),
        "use_dft_plus_u": bool(params.use_dft_plus_u),
        "enable_dipole": bool(params.enable_dipole),
        "compute_dos": bool(params.compute_dos or params.preset == "dos"),
        "patch_policy": params.patch_policy,
        "protected_incar_keys": list(plan.protected_keys),
        "user_patch_keys": sorted(dict(params.user_incar_patch).keys()),
    }
    if params.preset == "dos":
        data.update(
            {
                "dos_nedos": plan.user_incar_settings.get("NEDOS"),
                "dos_ismear": plan.user_incar_settings.get("ISMEAR"),
                "dos_icharg": plan.user_incar_settings.get("ICHARG"),
                "dos_charge_density_mode": "fixed_chgcar" if dos_charge_density_path is not None else "self_consistent",
                "dos_charge_density_path_rel": workspace_relpath(dos_charge_density_path) if dos_charge_density_path is not None else None,
            }
        )
    if params.preset == "md":
        data.update(
            {
                "md_temperature_begin_k": plan.user_incar_settings.get("TEBEG"),
                "md_temperature_end_k": plan.user_incar_settings.get("TEEND"),
                "md_steps": plan.user_incar_settings.get("NSW"),
                "md_timestep_fs": plan.user_incar_settings.get("POTIM"),
                "md_smass": plan.user_incar_settings.get("SMASS"),
                "mdalgo": plan.user_incar_settings.get("MDALGO"),
            }
        )
    propagated, warnings = propagate_adsorbate_metadata(
        input_structure_path=input_path,
        output_structure_path=output_root / "POSCAR",
        tool_name=tool_name,
    )
    if propagated:
        data.update(propagated)
    lines = [
        f"{tool_name} completed.",
        f"preset={params.preset} regime={params.regime} k_grid={list(plan.k_grid)} patch_policy={params.patch_policy}",
        f"output_root_rel={data['output_root_rel']}",
    ]
    if params.preset == "dos":
        lines.insert(
            2,
            (
                "dos_mode="
                f"{data['dos_charge_density_mode']} nedos={data['dos_nedos']} "
                f"ismear={data['dos_ismear']} icharg={data['dos_icharg']} compute_dos={data['compute_dos']}"
            ),
        )
    if params.preset == "md":
        lines.insert(
            2,
            (
                f"mdalgo={data['mdalgo']} tebeg={data['md_temperature_begin_k']} "
                f"teend={data['md_temperature_end_k']} nsw={data['md_steps']} "
                f"potim={data['md_timestep_fs']} smass={data['md_smass']}"
            ),
        )
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    return "\n".join(lines), artifact


__all__ = [
    "VaspPrepareInput",
    "vasp_prepare",
]
