"""Dedicated VASP band-structure input preparation tool."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Literal

from ase.io import read as ase_read
from pydantic import BaseModel, Field, field_validator, model_validator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

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


class VaspBandPrepareInput(BaseModel):
    """[electronic/prepare] Prepare a dedicated line-mode VASP band-structure input set from a relaxed structure, explicit band KPOINTS, and optional fixed CHGCAR."""

    input_path: str = Field(..., description="Single relaxed structure file path.")
    output_root: str = Field(..., description="Target directory for the prepared band-structure input set.")
    kpoints_path: str = Field(..., description="Existing line-mode KPOINTS file, typically from generate_kpath.")
    regime: Literal["bulk", "slab", "gas"] = Field(
        "bulk",
        description="Scientific regime controlling baseline INCAR defaults before band-specific overrides.",
    )
    k_product: int = Field(
        35,
        ge=1,
        description="Reference SCF mesh density used only for the underlying template generation before replacing KPOINTS.",
    )
    charge_density_path: str | None = Field(
        None,
        description=(
            "Optional CHGCAR path for a non-self-consistent band run. When provided, the tool copies it to output_root/CHGCAR, "
            "sets ICHARG=11, and applies a recommended LMAXMIX baseline when absent."
        ),
    )
    use_d3: bool = Field(False, description="Carry a DFT-D3(BJ) baseline toggle into the generated INCAR.")
    use_dft_plus_u: bool = Field(False, description="Carry a DFT+U baseline toggle into the generated INCAR.")
    enable_dipole: bool = Field(False, description="Carry dipole-correction helpers into the generated INCAR.")
    user_incar_patch: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Targeted INCAR patch object for band-specific knobs such as SIGMA, NBANDS, LORBIT, or spin settings. "
            "MAGMOM/LDAUU/LDAUJ must be symbol:value maps. Use null to request removal of a key from the final INCAR."
        ),
    )
    patch_policy: Literal["safe", "force"] = Field(
        "safe",
        description="safe protects only band-identity INCAR keys; force applies the user patch after canonical band defaults.",
    )

    @field_validator("user_incar_patch")
    @classmethod
    def _validate_user_incar_patch(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_user_incar_patch(value)

    @model_validator(mode="after")
    def _validate_scope(self) -> "VaspBandPrepareInput":
        if self.regime == "gas" and self.charge_density_path is not None:
            raise ValueError("charge_density_path is not meaningful for gas band-structure preparation.")
        return self


def _load_structure(path: Path) -> Structure:
    if path.suffix.lower() == ".xyz":
        atoms = ase_read(path.as_posix())
        return AseAtomsAdaptor.get_structure(atoms)
    return Structure.from_file(path)


def _validate_single_structure_input(tool_name: str, input_path: Path) -> None:
    if not input_path.exists():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} could not find input_path={workspace_relpath(input_path)}.",
            artifact={"tool_name": tool_name, "data": {"input_path_rel": workspace_relpath(input_path)}},
            error_code="missing_input",
        )
    if input_path.is_dir():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} currently supports only a single structure file input, but "
                f"{workspace_relpath(input_path)} is a directory."
            ),
            artifact={"tool_name": tool_name, "data": {"input_path_rel": workspace_relpath(input_path)}},
            error_code="directory_input_not_supported",
        )
    if input_path.suffix.lower() not in SUPPORTED_EXTS:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} does not support {input_path.suffix or '(no extension)'} for "
                f"{workspace_relpath(input_path)}. Supported: {', '.join(sorted(SUPPORTED_EXTS))}."
            ),
            artifact={"tool_name": tool_name, "data": {"input_path_rel": workspace_relpath(input_path)}},
            error_code="unsupported_input_extension",
        )


def _load_line_mode_kpoints(path: Path) -> Kpoints:
    kpoints = Kpoints.from_file(path)
    if kpoints.style != Kpoints.supported_modes.Line_mode:
        raise ValueError("kpoints_path must point to a line-mode KPOINTS file.")
    return kpoints


def vasp_band_prepare(payload: Dict[str, object]) -> tuple[str, dict[str, Any]]:
    """[electronic/prepare] Prepare a dedicated VASP band-structure input directory with explicit line-mode KPOINTS."""
    tool_name = "vasp_band_prepare"
    params = VaspBandPrepareInput(**payload)
    input_path = resolve_workspace_path(params.input_path)
    output_root = resolve_workspace_path(params.output_root)
    kpoints_path = resolve_workspace_path(params.kpoints_path, must_exist=True)
    charge_density_path = (
        resolve_workspace_path(params.charge_density_path, must_exist=True) if params.charge_density_path is not None else None
    )

    _validate_single_structure_input(tool_name, input_path)
    if not kpoints_path.is_file():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} expected kpoints_path to be a file: {workspace_relpath(kpoints_path)}.",
            artifact={"tool_name": tool_name, "data": {"kpoints_rel": workspace_relpath(kpoints_path)}},
            error_code="invalid_kpoints_path",
        )
    if charge_density_path is not None and charge_density_path.is_dir():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} expected charge_density_path to be a CHGCAR file, but "
                f"{workspace_relpath(charge_density_path)} is a directory."
            ),
            artifact={"tool_name": tool_name, "data": {"charge_density_rel": workspace_relpath(charge_density_path)}},
            error_code="invalid_charge_density",
        )

    incar_path = output_root / "INCAR"
    if incar_path.exists():
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=(
                f"{tool_name} refused to overwrite existing VASP inputs in {workspace_relpath(output_root)} "
                "because INCAR already exists there. Choose a different output_root."
            ),
            artifact={"tool_name": tool_name, "data": {"output_root_rel": workspace_relpath(output_root)}},
            error_code="prepare_target_exists",
        )

    try:
        structure = _load_structure(input_path)
        band_defaults: Dict[str, Any] = {
            "IBRION": -1,
            "NSW": 0,
            "ISYM": 0,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LCHARG": False,
            "LWAVE": False,
            "LORBIT": 11,
        }
        if charge_density_path is not None:
            band_defaults["ICHARG"] = 11
        protected_keys = {"IBRION", "NSW", "ISYM"}
        if charge_density_path is not None:
            protected_keys.add("ICHARG")
        user_patch = dict(params.user_incar_patch)
        if params.patch_policy == "safe":
            for key in protected_keys:
                if key in user_patch and user_patch[key] != band_defaults.get(key):
                    raise ValueError(
                        f"user_incar_patch attempts to override protected INCAR key {key} under patch_policy='safe'. "
                        "Use patch_policy='force' if you really need to replace band-identity defaults."
                    )
        merged_patch = {**band_defaults, **user_patch}
        writer = StructWriter()
        plan = writer.write_vasp_inputs(
            structure=structure,
            output_dir=output_root,
            preset="static",
            regime=params.regime,
            k_product=int(params.k_product),
            use_d3=bool(params.use_d3),
            user_incar_patch=merged_patch,
            use_dft_plus_u=bool(params.use_dft_plus_u),
            enable_dipole=bool(params.enable_dipole),
            patch_policy="force",
        )
        line_kpoints = _load_line_mode_kpoints(kpoints_path)
        line_kpoints.write_file(str(output_root / "KPOINTS"))
        if charge_density_path is not None:
            shutil.copy2(charge_density_path, output_root / "CHGCAR")
        final_incar = Incar.from_file(output_root / "INCAR")
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
                    "kpoints_rel": workspace_relpath(kpoints_path),
                },
            },
            error_code="prepare_failed",
        ) from exc

    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "prepared_directory_rel": workspace_relpath(output_root),
        "kpoints_rel": workspace_relpath(output_root / "KPOINTS"),
        "source_kpoints_rel": workspace_relpath(kpoints_path),
        "charge_density_rel": workspace_relpath(charge_density_path) if charge_density_path is not None else None,
        "charge_density_mode": "fixed_chgcar" if charge_density_path is not None else "self_consistent",
        "regime": params.regime,
        "k_product": int(params.k_product),
        "patch_policy": params.patch_policy,
        "protected_incar_keys": sorted(protected_keys),
        "user_patch_keys": sorted(dict(params.user_incar_patch).keys()),
        "ismear": final_incar.get("ISMEAR"),
        "sigma": final_incar.get("SIGMA"),
        "icharg": final_incar.get("ICHARG"),
    }
    lines = [
        f"{tool_name} completed.",
        (
            f"regime={params.regime} charge_density_mode={data['charge_density_mode']} "
            f"ismear={data['ismear']} sigma={data['sigma']} icharg={data['icharg']}"
        ),
        f"kpoints_rel={data['kpoints_rel']}",
        f"output_root_rel={data['output_root_rel']}",
    ]
    return "\n".join(lines), {"tool_name": tool_name, "data": data}


__all__ = ["VaspBandPrepareInput", "vasp_band_prepare"]
