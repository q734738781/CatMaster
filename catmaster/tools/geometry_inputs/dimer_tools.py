from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
from ase.io import read as ase_read
from pydantic import BaseModel, Field, field_validator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Poscar

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.geometry_inputs.neb_tools import _classify_single_image_tree, _find_image_structure_file
from catmaster.tools.geometry_inputs.vasp_inputs import StructWriter

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_ELEMENT_MAP_INCAR_KEYS = {"MAGMOM", "LDAUU", "LDAUJ"}


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


def _normalize_incar_patch(value: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_val in value.items():
        key = str(raw_key).strip().upper()
        if not key:
            raise ValueError("INCAR key must be a non-empty string.")
        if key in _ELEMENT_MAP_INCAR_KEYS:
            raw_val = _coerce_element_map_value(key, raw_val)
        normalized[key] = raw_val
    return normalized


class VaspDimerPrepareInput(BaseModel):
    """[ts/dimer] Prepare an official VASP improved-dimer input directory (IBRION=44)."""

    input_path: str = Field(..., description="Transition-state guess structure file path.")
    output_root: str = Field(..., description="Target directory for the dimer-ready VASP input set.")
    mode_text_path: str = Field(
        ...,
        description=(
            "Plain-text file containing one raw Cartesian reaction-direction vector per atom. "
            "Preferred format is one `dx dy dz` row per atom. Lines copied from a vibration/OUTCAR block are also "
            "accepted when each atom line contains at least three trailing numeric displacement columns."
        ),
    )
    regime: Literal["bulk", "slab", "gas"] = Field(
        "slab",
        description="Scientific regime controlling k-mesh and shared relax-style defaults.",
    )
    relax_cell: bool = Field(False, description="Only valid for bulk dimer jobs. When true, use ISIF=3.")
    k_product: int = Field(35, ge=1, description="Target k-mesh density for the dimer KPOINTS.")
    use_d3: bool = Field(False, description="Enable DFT-D3(BJ) correction (IVDW=12).")
    use_dft_plus_u: bool = Field(False, description="Enable DFT+U baseline toggle (LDAU=True).")
    enable_dipole: bool = Field(False, description="Enable dipole correction helper with IDIPOL=3.")
    user_incar_patch: Dict[str, Any] = Field(
        default_factory=dict,
        description="Targeted INCAR patch object applied after canonical dimer overrides.",
    )
    patch_policy: Literal["safe", "force"] = Field(
        "safe",
        description="safe blocks overrides of protected dimer keys; force applies the patch after canonical dimer defaults.",
    )

    @field_validator("user_incar_patch")
    @classmethod
    def _validate_user_incar_patch(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_incar_patch(value)


class DimerModeFromNebInput(BaseModel):
    """[ts/dimer] Derive a raw dimer direction from the two images adjacent to a chosen TS image."""

    images_root: str = Field(
        ...,
        description=(
            "One NEB image tree using flat image files 00.vasp, 01.vasp, ... or legacy numbered directories "
            "00/POSCAR, 01/POSCAR, ... ."
        ),
    )
    output_root: str = Field(..., description="Directory for the derived dimer-mode artifacts.")
    ts_image_index: int | None = Field(
        None,
        description="Index of the image treated as TS. Defaults to the central non-endpoint image.",
    )
    use_mic: bool = Field(
        True,
        description="If true, apply minimum-image handling when subtracting the two neighboring images.",
    )
    overwrite: bool = Field(False, description="If true, replace output_root if it already exists.")


class DimerModeFromMaceInput(BaseModel):
    """[ts/dimer] Approximate a dimer direction by finite-difference ASE/MACE vibrations on a TS guess."""

    input_path: str = Field(..., description="Transition-state guess structure file path.")
    output_root: str = Field(..., description="Directory for vibration artifacts and the selected raw dimer mode.")
    model: str = Field(
        "mh-1",
        description=(
            "MACE model identifier or workspace-local trained-model path. "
            "Local paths may point to one model file or a directory containing a unique preferred model artifact."
        ),
    )
    head: Optional[str] = Field("omat_pbe", description="Model head for multi-head models; use empty string to disable.")
    dispersion: bool = Field(False, description="Enable dispersion correction when supported by the calculator.")
    device_preference: Literal["auto", "cuda", "cpu"] = Field("auto", description="Preferred execution device.")
    default_dtype: Literal["float32", "float64"] = Field("float64", description="Default dtype for local-file models.")
    active_atom_indices: list[int] | None = Field(
        None,
        description=(
            "Optional explicit 0-based atom indices for the finite-difference region. "
            "Default: use structurally free atoms from selective dynamics / FixAtoms. "
            "Set active_atom_indices as a shortcut to vibrationally analyze only those atoms. "
            "If unavailable, fall back to all atoms."
        ),
    )
    delta: float = Field(0.01, gt=0, description="Finite-displacement step in Angstrom.")
    nfree: Literal[2, 4] = Field(2, description="Finite-difference stencil width for ASE Vibrations.")
    mode_index: int | None = Field(
        None,
        description=(
            "Explicit mode index to export. If omitted, the tool selects the most negative/most imaginary frequency."
        ),
    )
    overwrite: bool = Field(False, description="If true, replace output_root if it already exists.")


class MaceAnalyzeFrequenciesInput(BaseModel):
    """[mace/analysis] Compute MACE vibrational frequencies and export all mode records."""

    input_path: str = Field(..., description="Structure file path for MACE vibrational analysis.")
    output_root: str = Field(..., description="Directory for frequency summary artifacts and per-mode vectors.")
    model: str = Field(
        "mh-1",
        description=(
            "MACE model identifier or workspace-local trained-model path. "
            "Local paths may point to one model file or a directory containing a unique preferred model artifact."
        ),
    )
    head: Optional[str] = Field("omat_pbe", description="Model head for multi-head models; use empty string to disable.")
    dispersion: bool = Field(False, description="Enable dispersion correction when supported by the calculator.")
    device_preference: Literal["auto", "cuda", "cpu"] = Field("auto", description="Preferred execution device.")
    default_dtype: Literal["float32", "float64"] = Field("float64", description="Default dtype for local-file models.")
    active_atom_indices: list[int] | None = Field(
        None,
        description=(
            "Optional explicit 0-based atom indices for the vibrational region. "
            "Default: use structurally free atoms from selective dynamics / FixAtoms. "
            "Set active_atom_indices as a shortcut to vibrationally analyze only those atoms. "
            "If unavailable, fall back to all atoms."
        ),
    )
    method: Literal["auto", "hessian", "finite_difference"] = Field(
        "auto",
        description=(
            "Vibrational backend choice. auto prefers analytic Hessian when supported by the installed MACE calculator "
            "and falls back to ASE finite differences otherwise."
        ),
    )
    delta: float = Field(0.01, gt=0, description="Finite-displacement step in Angstrom when finite differences are used.")
    nfree: Literal[2, 4] = Field(2, description="Finite-difference stencil width for ASE Vibrations.")
    overwrite: bool = Field(False, description="If true, replace output_root if it already exists.")


def _fail(tool_name: str, *, message: str, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    details: list[str] = [str(message).strip()]
    if isinstance(data, dict):
        for key in (
            "input_path_rel",
            "images_root_rel",
            "output_root_rel",
            "mode_text_path_rel",
            "raw_mode_rel",
            "mass_normalized_mode_rel",
            "summary_rel",
        ):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            details.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(details),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _success(tool_name: str, *, content: str, data: dict[str, Any], warnings: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    return content, artifact


def _prepare_output_root(tool_name: str, *, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if output_root.is_file():
            _fail(tool_name, message=f"output_root is a file: {output_root}", error_code="invalid_output_root")
        if not overwrite:
            _fail(
                tool_name,
                message=f"output_root already exists: {workspace_relpath(output_root)}. Set overwrite=true to replace.",
                data={"output_root_rel": workspace_relpath(output_root)},
                error_code="output_root_exists",
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_mode_vectors(path: Path, vectors: np.ndarray) -> None:
    lines = [f"{float(row[0]): .16e} {float(row[1]): .16e} {float(row[2]): .16e}" for row in vectors]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_mode_text(path: Path, *, atom_count: int) -> np.ndarray:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        raise ValueError(f"Failed to read mode_text_path: {exc}") from exc

    line_vectors: list[list[float]] = []
    all_numbers: list[float] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        numbers = [float(token) for token in _FLOAT_RE.findall(line)]
        if not numbers:
            continue
        all_numbers.extend(numbers)
        if len(numbers) == 3:
            line_vectors.append(numbers)
        elif len(numbers) >= 6:
            line_vectors.append(numbers[-3:])

    if len(line_vectors) == atom_count:
        arr = np.asarray(line_vectors, dtype=float)
    elif len(all_numbers) == atom_count * 3:
        arr = np.asarray(all_numbers, dtype=float).reshape(atom_count, 3)
    else:
        raise ValueError(
            f"Expected {atom_count} displacement rows (or {atom_count * 3} numeric values), "
            f"but parsed {len(line_vectors)} candidate rows and {len(all_numbers)} numeric values."
        )
    if not np.isfinite(arr).all():
        raise ValueError("Mode text contains non-finite values.")
    return arr


def _mass_normalize_mode(raw_mode: np.ndarray, masses: np.ndarray) -> np.ndarray:
    if raw_mode.shape != (len(masses), 3):
        raise ValueError("raw_mode shape does not match atom count.")
    sqrt_masses = np.sqrt(np.asarray(masses, dtype=float)).reshape(-1, 1)
    if np.any(sqrt_masses <= 0):
        raise ValueError("Atomic masses must be positive.")
    mass_divided = raw_mode / sqrt_masses
    norm = float(np.linalg.norm(mass_divided.reshape(-1)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Mass-normalized mode has zero or invalid norm.")
    return mass_divided / norm


def _append_dimer_vectors_to_poscar(poscar_path: Path, vectors: np.ndarray) -> None:
    poscar = Poscar.from_file(poscar_path)
    text = poscar.get_str(direct=False).rstrip()
    vector_block = "\n".join(
        f"{float(row[0]): .16e} {float(row[1]): .16e} {float(row[2]): .16e}" for row in vectors
    )
    poscar_path.write_text(f"{text}\n\n{vector_block}\n", encoding="utf-8")


def _validate_mode_shape(vectors: np.ndarray, atom_count: int) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.shape == (atom_count, 3):
        return arr
    if arr.shape == (atom_count * 3,):
        return arr.reshape(atom_count, 3)
    raise ValueError(f"Expected mode shape {(atom_count, 3)} or {(atom_count * 3,)}, got {arr.shape}.")


def _signed_frequency_cm1(value: Any) -> float:
    freq = complex(value)
    if abs(freq.imag) > 1e-8:
        return -abs(float(freq.imag))
    real = float(freq.real)
    return real


def _resolve_active_indices(atom_count: int, active_atom_indices: list[int] | None) -> list[int]:
    if active_atom_indices is None:
        return list(range(atom_count))
    indices = [int(idx) for idx in active_atom_indices]
    if len(set(indices)) != len(indices):
        raise ValueError("active_atom_indices contains duplicates.")
    for idx in indices:
        if idx < 0 or idx >= atom_count:
            raise ValueError(f"active_atom_indices contains out-of-range index {idx} for atom_count={atom_count}.")
    return sorted(indices)


def _infer_active_indices_from_structure(input_path: Path, atoms: Any) -> tuple[list[int] | None, str | None]:
    atom_count = int(len(atoms))

    try:
        poscar = Poscar.from_file(str(input_path))
    except Exception:
        poscar = None
    if poscar is not None:
        selective_dynamics = getattr(poscar, "selective_dynamics", None)
        if selective_dynamics is not None:
            mask = np.asarray(selective_dynamics, dtype=bool)
            if mask.shape == (atom_count, 3):
                active = [idx for idx, row in enumerate(mask) if bool(np.any(row))]
                return active, "selective_dynamics"

    raw_constraints = getattr(atoms, "constraints", None)
    if raw_constraints:
        constraints = raw_constraints if isinstance(raw_constraints, (list, tuple)) else [raw_constraints]
        fixed: set[int] = set()
        for constraint in constraints:
            if type(constraint).__name__ != "FixAtoms":
                continue
            indices = getattr(constraint, "index", None)
            if indices is None:
                continue
            fixed.update(int(idx) for idx in np.asarray(indices, dtype=int).reshape(-1))
        if fixed:
            active = [idx for idx in range(atom_count) if idx not in fixed]
            return active, "fixatoms_constraint"

    return None, None


def _resolve_active_indices_and_source(
    atom_count: int,
    active_atom_indices: list[int] | None,
    *,
    input_path: Path | None = None,
    atoms: Any | None = None,
) -> tuple[list[int], str]:
    if active_atom_indices is None:
        inferred: list[int] | None = None
        source: str | None = None
        if input_path is not None and atoms is not None:
            inferred, source = _infer_active_indices_from_structure(input_path, atoms)
        if inferred is None:
            inferred = list(range(atom_count))
            source = "all_atoms"
        if not inferred:
            raise ValueError(
                "No active atoms were resolved for vibrational analysis. "
                "Provide active_atom_indices explicitly or ensure the structure has mobile atoms in selective dynamics."
            )
        return sorted(int(idx) for idx in inferred), str(source or "all_atoms")

    indices = _resolve_active_indices(atom_count, active_atom_indices)
    if not indices:
        raise ValueError("active_atom_indices resolved to an empty set.")
    return indices, "explicit"


def _resolve_local_model(model_value: str) -> tuple[str, str, str]:
    from catmaster.tools.execution.mace_dispatch import _is_likely_local_model_path, _select_model_file_from_dir

    requested = str(model_value or "").strip()
    if not requested:
        raise ValueError("model is required")

    local_path: Path | None = None
    try:
        local_path = resolve_workspace_path(requested, must_exist=True)
    except (FileNotFoundError, ValueError):
        local_path = None

    if local_path is None:
        if _is_likely_local_model_path(requested):
            raise FileNotFoundError(f"Local MACE model path does not exist inside project files root: {requested}")
        return requested, "pretrained", requested

    if local_path.is_file():
        return str(local_path.resolve()), "local_file", workspace_relpath(local_path)
    if local_path.is_dir():
        selected = _select_model_file_from_dir(local_path)
        return str(selected.resolve()), "local_dir", workspace_relpath(local_path)
    raise ValueError(f"Unsupported model path type: {local_path}")


def _build_local_mace_calculator(*, model: str, head: Optional[str], dispersion: bool, device_preference: str, default_dtype: str):
    import torch
    from mace.calculators import mace_mp

    def _resolve_device(preference: str) -> str:
        if preference == "cpu":
            return "cpu"
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = _resolve_device(device_preference)
    model_arg, source_kind, source_ref = _resolve_local_model(model)
    kwargs = {"model": model_arg, "dispersion": dispersion, "device": device, "default_dtype": default_dtype}
    if head:
        kwargs["head"] = head
    return mace_mp(**kwargs), {"device": device, "source_kind": source_kind, "source_ref": source_ref}


def _active_cartesian_indices(active_indices: list[int]) -> np.ndarray:
    cart_indices: list[int] = []
    for atom_index in active_indices:
        cart_indices.extend((3 * atom_index, 3 * atom_index + 1, 3 * atom_index + 2))
    return np.asarray(cart_indices, dtype=int)


def _normalize_hessian_2d(raw_hessian: Any, *, atom_count: int, active_indices: list[int]) -> np.ndarray:
    hessian = np.asarray(raw_hessian, dtype=float)
    if hessian.shape == (atom_count, 3, atom_count, 3):
        hessian = hessian.reshape(atom_count * 3, atom_count * 3)
    elif hessian.shape != (atom_count * 3, atom_count * 3):
        raise ValueError(
            f"Unsupported Hessian shape {hessian.shape}; expected {(atom_count * 3, atom_count * 3)} "
            f"or {(atom_count, 3, atom_count, 3)}."
        )
    cart_indices = _active_cartesian_indices(active_indices)
    return np.asarray(hessian[np.ix_(cart_indices, cart_indices)], dtype=float)


def _vibrational_records_from_data(*, vib_data, atoms) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    frequencies = list(vib_data.get_frequencies())
    raw_modes = np.asarray(vib_data.get_modes(all_atoms=True), dtype=float)
    modes: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for mode_index, frequency in enumerate(frequencies):
        signed_cm1 = _signed_frequency_cm1(frequency)
        mode = _validate_mode_shape(raw_modes[mode_index], len(atoms))
        modes.append(mode)
        records.append(
            {
                "mode_index": mode_index,
                "frequency_cm1": signed_cm1,
                "imaginary": bool(signed_cm1 < 0),
            }
        )
    return records, modes


def _compute_mace_vibrational_modes_via_hessian(
    *,
    atoms,
    calc,
    active_indices: list[int],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    from ase.vibrations.data import VibrationsData

    getter = getattr(calc, "get_hessian", None)
    if getter is None:
        raise AttributeError("Installed MACE calculator does not expose get_hessian().")
    try:
        raw_hessian = getter(atoms=atoms)
    except TypeError:
        try:
            raw_hessian = getter(atoms)
        except TypeError:
            raw_hessian = getter()
    hessian_2d = _normalize_hessian_2d(raw_hessian, atom_count=len(atoms), active_indices=active_indices)
    vib_data = VibrationsData.from_2d(atoms, hessian_2d=hessian_2d, indices=active_indices)
    return _vibrational_records_from_data(vib_data=vib_data, atoms=atoms)


def _compute_mace_vibrational_modes_via_finite_difference(
    *,
    atoms,
    output_root: Path,
    calc,
    active_indices: list[int],
    delta: float,
    nfree: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    from ase.vibrations import Vibrations

    vib_root = output_root / "ase_vibrations"
    vib_root.mkdir(parents=True, exist_ok=True)
    atoms = atoms.copy()
    atoms.calc = calc
    vib = Vibrations(
        atoms,
        indices=active_indices,
        name=str(vib_root / "vib"),
        delta=delta,
        nfree=nfree,
    )
    vib.run()

    frequencies = list(vib.get_frequencies())
    modes: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for mode_index, frequency in enumerate(frequencies):
        signed_cm1 = _signed_frequency_cm1(frequency)
        mode = np.real(np.asarray(vib.get_mode(mode_index), dtype=float))
        mode = _validate_mode_shape(mode, len(atoms))
        modes.append(mode)
        records.append(
            {
                "mode_index": mode_index,
                "frequency_cm1": signed_cm1,
                "imaginary": bool(signed_cm1 < 0),
            }
        )
    return records, modes


def _compute_mace_vibrational_modes(
    *,
    atoms,
    output_root: Path,
    model: str,
    head: Optional[str],
    dispersion: bool,
    device_preference: str,
    default_dtype: str,
    active_indices: list[int],
    method: Literal["auto", "hessian", "finite_difference"] = "auto",
    delta: float,
    nfree: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    calc, calc_info = _build_local_mace_calculator(
        model=model,
        head=head,
        dispersion=dispersion,
        device_preference=device_preference,
        default_dtype=default_dtype,
    )
    atoms = atoms.copy()
    attempts: list[str] = []
    if method == "auto":
        # MACE's public get_hessian() computes the full-system Hessian and only later
        # can we slice to an active subspace. For subset analyses, finite differences
        # scale with the active region and are therefore the safer default.
        backend_order = ["hessian", "finite_difference"] if len(active_indices) == len(atoms) else ["finite_difference", "hessian"]
    elif method == "hessian":
        backend_order = ["hessian"]
    else:
        backend_order = ["finite_difference"]

    for backend in backend_order:
        if backend == "hessian":
            try:
                records, modes = _compute_mace_vibrational_modes_via_hessian(
                    atoms=atoms,
                    calc=calc,
                    active_indices=active_indices,
                )
                return records, modes, {
                    "calculator": calc_info,
                    "active_atom_indices": active_indices,
                    "method_used": "hessian",
                    "fallback_notes": attempts,
                }
            except Exception as exc:
                if method == "hessian":
                    raise
                attempts.append(f"hessian_failed={exc}")
            continue

        try:
            records, modes = _compute_mace_vibrational_modes_via_finite_difference(
                atoms=atoms,
                output_root=output_root,
                calc=calc,
                active_indices=active_indices,
                delta=delta,
                nfree=nfree,
            )
            return records, modes, {
                "calculator": calc_info,
                "active_atom_indices": active_indices,
                "method_used": "finite_difference",
                "fallback_notes": attempts,
            }
        except Exception as exc:
            if method == "finite_difference":
                raise
            attempts.append(f"finite_difference_failed={exc}")
    raise RuntimeError(
        "Failed to compute MACE vibrational modes via all available methods. " + "; ".join(attempts or ["no_backend_attempted"])
    )


def vasp_dimer_prepare(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ts/dimer] Prepare an official VASP improved-dimer input set with mass-normalized direction vectors appended to POSCAR."""
    params = VaspDimerPrepareInput(**payload)
    tool_name = "vasp_dimer_prepare"
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    mode_text_path = resolve_workspace_path(params.mode_text_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)

    if not input_path.is_file():
        _fail(tool_name, message=f"input_path is not a file: {workspace_relpath(input_path)}", error_code="invalid_input")
    if not mode_text_path.is_file():
        _fail(
            tool_name,
            message=f"mode_text_path is not a file: {workspace_relpath(mode_text_path)}",
            error_code="invalid_mode_text",
        )
    incar_path = output_root / "INCAR"
    if incar_path.exists():
        _fail(
            tool_name,
            message=(
                f"{tool_name} refused to overwrite existing VASP inputs in {workspace_relpath(output_root)} "
                "because INCAR already exists there. Choose a different output_root."
            ),
            data={"output_root_rel": workspace_relpath(output_root)},
            error_code="prepare_target_exists",
        )

    try:
        atoms = ase_read(str(input_path))
        structure = AseAtomsAdaptor.get_structure(atoms)
        raw_mode = _parse_mode_text(mode_text_path, atom_count=len(atoms))
        normalized_mode = _mass_normalize_mode(raw_mode, atoms.get_masses())
    except Exception as exc:
        _fail(
            tool_name,
            message=f"Failed to prepare dimer mode: {exc}",
            data={
                "input_path_rel": workspace_relpath(input_path),
                "mode_text_path_rel": workspace_relpath(mode_text_path),
                "output_root_rel": workspace_relpath(output_root),
            },
            error_code="mode_preparation_failed",
        )

    writer = StructWriter()
    try:
        plan = writer.write_vasp_inputs(
            structure=structure,
            output_dir=output_root,
            preset="dimer",
            regime=params.regime,
            relax_cell=params.relax_cell,
            k_product=params.k_product,
            use_d3=params.use_d3,
            user_incar_patch=params.user_incar_patch,
            use_dft_plus_u=params.use_dft_plus_u,
            enable_dipole=params.enable_dipole,
            patch_policy=params.patch_policy,
        )
    except Exception as exc:
        _fail(
            tool_name,
            message=f"Failed to write dimer VASP inputs: {exc}",
            data={"input_path_rel": workspace_relpath(input_path), "output_root_rel": workspace_relpath(output_root)},
            error_code="write_failed",
        )

    raw_copy = output_root / "dimer_mode_raw.txt"
    normalized_copy = output_root / "dimer_mode_mass_normalized.txt"
    _write_mode_vectors(raw_copy, raw_mode)
    _write_mode_vectors(normalized_copy, normalized_mode)
    _append_dimer_vectors_to_poscar(output_root / "POSCAR", normalized_mode)

    summary_path = output_root / "dimer_prepare_summary.json"
    _write_json(
        summary_path,
        {
            "input_path_rel": workspace_relpath(input_path),
            "mode_text_path_rel": workspace_relpath(mode_text_path),
            "raw_mode_rel": workspace_relpath(raw_copy),
            "mass_normalized_mode_rel": workspace_relpath(normalized_copy),
            "output_root_rel": workspace_relpath(output_root),
            "preset": "dimer",
            "regime": params.regime,
            "k_grid": list(plan.k_grid),
            "protected_incar_keys": list(plan.protected_keys),
        },
    )
    incar = Incar.from_file(output_root / "INCAR")
    data = {
        "input_path_rel": workspace_relpath(input_path),
        "mode_text_path_rel": workspace_relpath(mode_text_path),
        "output_root_rel": workspace_relpath(output_root),
        "poscar_rel": workspace_relpath(output_root / "POSCAR"),
        "incar_rel": workspace_relpath(output_root / "INCAR"),
        "raw_mode_rel": workspace_relpath(raw_copy),
        "mass_normalized_mode_rel": workspace_relpath(normalized_copy),
        "summary_rel": workspace_relpath(summary_path),
        "ibrion": int(incar["IBRION"]),
        "k_grid": list(plan.k_grid),
    }
    lines = [
        "vasp_dimer_prepare completed.",
        f"output_root_rel={data['output_root_rel']}",
        f"poscar_rel={data['poscar_rel']}",
        f"raw_mode_rel={data['raw_mode_rel']}",
        f"mass_normalized_mode_rel={data['mass_normalized_mode_rel']}",
        f"summary_rel={data['summary_rel']}",
    ]
    return _success(tool_name, content="\n".join(lines), data=data)


def _load_images_from_tree(images_root: Path) -> tuple[list[Any], list[str]]:
    classified = _classify_single_image_tree(images_root)
    if classified is None:
        raise ValueError(
            "images_root must be one image tree with contiguous numbering from 00 using either flat image files or legacy image directories."
        )
    image_sources, layout_kind = classified
    images = []
    labels: list[str] = []
    for idx, source in image_sources:
        structure_path = source if layout_kind == "flat_file" else _find_image_structure_file(source)
        images.append(ase_read(str(structure_path)))
        labels.append(workspace_relpath(structure_path))
    first = images[0]
    for idx, image in enumerate(images[1:], start=1):
        if len(image) != len(first):
            raise ValueError(f"Image {idx:02d} has a different atom count.")
        if not np.array_equal(image.get_atomic_numbers(), first.get_atomic_numbers()):
            raise ValueError(f"Image {idx:02d} has a different element ordering.")
        if not np.allclose(image.cell.array, first.cell.array, atol=1e-6, rtol=0):
            raise ValueError(f"Image {idx:02d} has a different cell.")
    return images, labels


def make_dimer_mode_from_neb(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ts/dimer] Approximate a dimer direction from the two images adjacent to a chosen TS image."""
    params = DimerModeFromNebInput(**payload)
    tool_name = "make_dimer_mode_from_neb"
    images_root = resolve_workspace_path(params.images_root, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    if not images_root.is_dir():
        _fail(tool_name, message=f"images_root is not a directory: {workspace_relpath(images_root)}", error_code="invalid_images_root")
    _prepare_output_root(tool_name, output_root=output_root, overwrite=params.overwrite)

    try:
        images, labels = _load_images_from_tree(images_root)
    except Exception as exc:
        _fail(
            tool_name,
            message=f"Failed to load NEB images: {exc}",
            data={"images_root_rel": workspace_relpath(images_root), "output_root_rel": workspace_relpath(output_root)},
            error_code="invalid_image_tree",
        )

    if len(images) < 3:
        _fail(tool_name, message="Need at least three total images to derive a TS-adjacent dimer direction.", error_code="too_few_images")
    ts_image_index = params.ts_image_index if params.ts_image_index is not None else len(images) // 2
    if ts_image_index <= 0 or ts_image_index >= len(images) - 1:
        _fail(
            tool_name,
            message=f"ts_image_index must be between 1 and {len(images) - 2}, got {ts_image_index}.",
            error_code="invalid_ts_index",
        )

    prev_image = images[ts_image_index - 1]
    next_image = images[ts_image_index + 1]
    displacement = next_image.positions - prev_image.positions
    if params.use_mic and np.any(next_image.pbc):
        from ase.geometry import find_mic

        displacement, _ = find_mic(displacement, next_image.cell, pbc=next_image.pbc)
    raw_mode = np.asarray(displacement, dtype=float)
    normalized_mode = _mass_normalize_mode(raw_mode, images[ts_image_index].get_masses())

    raw_path = output_root / "dimer_mode_raw.txt"
    normalized_path = output_root / "dimer_mode_mass_normalized.txt"
    summary_path = output_root / "summary.json"
    _write_mode_vectors(raw_path, raw_mode)
    _write_mode_vectors(normalized_path, normalized_mode)
    _write_json(
        summary_path,
        {
            "images_root_rel": workspace_relpath(images_root),
            "image_sources": labels,
            "ts_image_index": ts_image_index,
            "neighbor_indices": [ts_image_index - 1, ts_image_index + 1],
            "raw_mode_rel": workspace_relpath(raw_path),
            "mass_normalized_mode_rel": workspace_relpath(normalized_path),
            "use_mic": params.use_mic,
        },
    )
    data = {
        "images_root_rel": workspace_relpath(images_root),
        "output_root_rel": workspace_relpath(output_root),
        "ts_image_index": ts_image_index,
        "raw_mode_rel": workspace_relpath(raw_path),
        "mass_normalized_mode_rel": workspace_relpath(normalized_path),
        "summary_rel": workspace_relpath(summary_path),
    }
    lines = [
        "make_dimer_mode_from_neb completed.",
        f"ts_image_index={ts_image_index}",
        f"raw_mode_rel={data['raw_mode_rel']}",
        f"mass_normalized_mode_rel={data['mass_normalized_mode_rel']}",
        f"summary_rel={data['summary_rel']}",
    ]
    return _success(tool_name, content="\n".join(lines), data=data)


def make_dimer_mode_from_mace(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ts/dimer] Approximate a dimer direction from ASE/MACE finite-difference vibrations."""
    params = DimerModeFromMaceInput(**payload)
    tool_name = "make_dimer_mode_from_mace"
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    if not input_path.is_file():
        _fail(tool_name, message=f"input_path is not a file: {workspace_relpath(input_path)}", error_code="invalid_input")
    _prepare_output_root(tool_name, output_root=output_root, overwrite=params.overwrite)

    try:
        atoms = ase_read(str(input_path))
        active_indices, active_source = _resolve_active_indices_and_source(
            len(atoms),
            params.active_atom_indices,
            input_path=input_path,
            atoms=atoms,
        )
        records, modes, calc_info = _compute_mace_vibrational_modes(
            atoms=atoms,
            output_root=output_root,
            model=params.model,
            head=(str(params.head).strip() or None) if params.head is not None else None,
            dispersion=params.dispersion,
            device_preference=params.device_preference,
            default_dtype=params.default_dtype,
            active_indices=active_indices,
            method="auto",
            delta=params.delta,
            nfree=params.nfree,
        )
    except Exception as exc:
        _fail(
            tool_name,
            message=f"Failed to compute MACE vibrational modes: {exc}",
            data={"input_path_rel": workspace_relpath(input_path), "output_root_rel": workspace_relpath(output_root)},
            error_code="mace_vibration_failed",
        )

    if not records:
        _fail(tool_name, message="No vibrational modes were produced.", error_code="no_modes")

    if params.mode_index is None:
        selected_index = min(records, key=lambda item: float(item["frequency_cm1"]))["mode_index"]
    else:
        selected_index = int(params.mode_index)
        if selected_index < 0 or selected_index >= len(records):
            _fail(
                tool_name,
                message=f"mode_index must be between 0 and {len(records) - 1}, got {selected_index}.",
                error_code="invalid_mode_index",
            )

    raw_mode = _validate_mode_shape(modes[selected_index], len(atoms))
    normalized_mode = _mass_normalize_mode(raw_mode, atoms.get_masses())
    raw_path = output_root / "dimer_mode_raw.txt"
    normalized_path = output_root / "dimer_mode_mass_normalized.txt"
    summary_path = output_root / "summary.json"
    _write_mode_vectors(raw_path, raw_mode)
    _write_mode_vectors(normalized_path, normalized_mode)
    imaginary_count = sum(1 for record in records if bool(record["imaginary"]))
    _write_json(
        summary_path,
        {
            "input_path_rel": workspace_relpath(input_path),
            "output_root_rel": workspace_relpath(output_root),
            "raw_mode_rel": workspace_relpath(raw_path),
            "mass_normalized_mode_rel": workspace_relpath(normalized_path),
            "selected_mode_index": selected_index,
            "selected_frequency_cm1": float(records[selected_index]["frequency_cm1"]),
            "imaginary_mode_count": imaginary_count,
            "active_atom_indices": active_indices,
            "active_atom_indices_source": active_source,
            "calculator": calc_info["calculator"],
            "modes": records,
        },
    )
    warnings: list[str] = []
    if imaginary_count > 1:
        warnings.append(
            "Multiple imaginary modes were detected; the exported mode may need manual chemical validation before dimer use."
        )
    elif imaginary_count == 0:
        warnings.append(
            "No imaginary mode was detected; the exported lowest-frequency mode is only a heuristic dimer direction guess."
        )

    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "raw_mode_rel": workspace_relpath(raw_path),
        "mass_normalized_mode_rel": workspace_relpath(normalized_path),
        "summary_rel": workspace_relpath(summary_path),
        "selected_mode_index": selected_index,
        "selected_frequency_cm1": float(records[selected_index]["frequency_cm1"]),
        "imaginary_mode_count": imaginary_count,
        "active_atom_indices": active_indices,
        "active_atom_indices_source": active_source,
        "method_used": calc_info.get("method_used"),
    }
    lines = [
        "make_dimer_mode_from_mace completed.",
        f"selected_mode_index={selected_index} selected_frequency_cm1={data['selected_frequency_cm1']:.6f}",
        f"raw_mode_rel={data['raw_mode_rel']}",
        f"mass_normalized_mode_rel={data['mass_normalized_mode_rel']}",
        f"summary_rel={data['summary_rel']}",
    ]
    return _success(tool_name, content="\n".join(lines), data=data, warnings=warnings)


def mace_analyze_frequencies(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[mace/analysis] Compute and export all MACE vibrational frequencies and mode vectors."""
    params = MaceAnalyzeFrequenciesInput(**payload)
    tool_name = "mace_analyze_frequencies"
    input_path = resolve_workspace_path(params.input_path, must_exist=True)
    output_root = resolve_workspace_path(params.output_root)
    if not input_path.is_file():
        _fail(tool_name, message=f"input_path is not a file: {workspace_relpath(input_path)}", error_code="invalid_input")
    _prepare_output_root(tool_name, output_root=output_root, overwrite=params.overwrite)

    try:
        atoms = ase_read(str(input_path))
        active_indices, active_source = _resolve_active_indices_and_source(
            len(atoms),
            params.active_atom_indices,
            input_path=input_path,
            atoms=atoms,
        )
        records, modes, calc_info = _compute_mace_vibrational_modes(
            atoms=atoms,
            output_root=output_root,
            model=params.model,
            head=(str(params.head).strip() or None) if params.head is not None else None,
            dispersion=params.dispersion,
            device_preference=params.device_preference,
            default_dtype=params.default_dtype,
            active_indices=active_indices,
            method=params.method,
            delta=params.delta,
            nfree=params.nfree,
        )
    except Exception as exc:
        _fail(
            tool_name,
            message=f"Failed to compute MACE frequencies: {exc}",
            data={"input_path_rel": workspace_relpath(input_path), "output_root_rel": workspace_relpath(output_root)},
            error_code="mace_frequency_failed",
        )

    if not records:
        _fail(tool_name, message="No vibrational modes were produced.", error_code="no_modes")

    modes_dir = output_root / "modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    summary_modes: list[dict[str, Any]] = []
    for record, mode in zip(records, modes):
        mode_index = int(record["mode_index"])
        mode_path = modes_dir / f"mode_{mode_index:04d}.txt"
        _write_mode_vectors(mode_path, mode)
        row = {
            "mode_index": mode_index,
            "frequency_cm1": float(record["frequency_cm1"]),
            "imaginary": bool(record["imaginary"]),
            "mode_vector_rel": workspace_relpath(mode_path),
        }
        csv_rows.append(row)
        summary_modes.append(row)

    frequencies_csv = output_root / "frequencies.csv"
    _write_csv(
        frequencies_csv,
        csv_rows,
        fieldnames=["mode_index", "frequency_cm1", "imaginary", "mode_vector_rel"],
    )
    summary_path = output_root / "summary.json"
    imaginary_count = sum(1 for record in records if bool(record["imaginary"]))
    lowest_frequency_cm1 = min(float(record["frequency_cm1"]) for record in records)
    highest_frequency_cm1 = max(float(record["frequency_cm1"]) for record in records)
    _write_json(
        summary_path,
        {
            "input_path_rel": workspace_relpath(input_path),
            "output_root_rel": workspace_relpath(output_root),
            "frequencies_csv_rel": workspace_relpath(frequencies_csv),
            "modes_dir_rel": workspace_relpath(modes_dir),
            "method_requested": params.method,
            "method_used": calc_info.get("method_used"),
            "active_atom_indices": active_indices,
            "active_atom_indices_source": active_source,
            "calculator": calc_info["calculator"],
            "mode_count": len(records),
            "imaginary_mode_count": imaginary_count,
            "lowest_frequency_cm1": lowest_frequency_cm1,
            "highest_frequency_cm1": highest_frequency_cm1,
            "modes": summary_modes,
            "fallback_notes": calc_info.get("fallback_notes") or [],
        },
    )

    warnings: list[str] = []
    if imaginary_count > 1:
        warnings.append("Multiple imaginary modes were detected; mode assignment may need manual chemical validation.")
    data = {
        "input_path_rel": workspace_relpath(input_path),
        "output_root_rel": workspace_relpath(output_root),
        "summary_rel": workspace_relpath(summary_path),
        "frequencies_csv_rel": workspace_relpath(frequencies_csv),
        "modes_dir_rel": workspace_relpath(modes_dir),
        "method_requested": params.method,
        "method_used": calc_info.get("method_used"),
        "active_atom_indices": active_indices,
        "active_atom_indices_source": active_source,
        "mode_count": len(records),
        "imaginary_mode_count": imaginary_count,
        "lowest_frequency_cm1": lowest_frequency_cm1,
        "highest_frequency_cm1": highest_frequency_cm1,
        "modes": summary_modes,
    }
    lines = [
        "mace_analyze_frequencies completed.",
        f"method_used={data['method_used']} mode_count={data['mode_count']} imaginary_mode_count={data['imaginary_mode_count']}",
        f"lowest_frequency_cm1={data['lowest_frequency_cm1']:.6f} highest_frequency_cm1={data['highest_frequency_cm1']:.6f}",
        f"frequencies_csv_rel={data['frequencies_csv_rel']}",
        f"summary_rel={data['summary_rel']}",
    ]
    return _success(tool_name, content="\n".join(lines), data=data, warnings=warnings)


__all__ = [
    "MaceAnalyzeFrequenciesInput",
    "DimerModeFromMaceInput",
    "DimerModeFromNebInput",
    "VaspDimerPrepareInput",
    "mace_analyze_frequencies",
    "make_dimer_mode_from_mace",
    "make_dimer_mode_from_neb",
    "vasp_dimer_prepare",
]
