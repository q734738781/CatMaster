from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLOAT_PATTERN = r"[-+]?\d+(?:\.\d+)?"
_OUTCAR_REAL_FREQ_PATTERN = re.compile(
    rf"f\s*=\s*{_FLOAT_PATTERN}\s*THz\s*{_FLOAT_PATTERN}\s*2PiTHz\s*"
    rf"(?P<cm1>{_FLOAT_PATTERN})\s*cm-1\s*(?P<mev>{_FLOAT_PATTERN})\s*meV"
)
_VASP_ATM_IN_PA = 101325.0
_CM1_TO_EV = 1.0 / 8065.54429
_ADSORBATE_FREQ_FLOOR_EV = 50.0 * _CM1_TO_EV
_THERMO_PATTERNS = {
    "temperature_k": re.compile(rf"Temperature \(T\)\s*:?\s+(?P<value>{_FLOAT_PATTERN})\s+K"),
    "pressure_pa": re.compile(rf"Pressure \(P\)\s*:?\s+(?P<value>{_FLOAT_PATTERN})\s+Pa"),
    "spin_multiplicity": re.compile(rf"Spin-Multiplicity \(S\)\s*:?\s+(?P<value>{_FLOAT_PATTERN})"),
    "e_zpe_ev": re.compile(
        rf"Zero-point energy E_ZPE\s*:\s*{_FLOAT_PATTERN}\s+kcal/mol\s+(?P<value>{_FLOAT_PATTERN})\s+eV"
    ),
    "thermal_correction_u_ev": re.compile(
        rf"Thermal correction to U\(T\)\s*:\s*{_FLOAT_PATTERN}\s+kcal/mol\s+(?P<value>{_FLOAT_PATTERN})\s+eV"
    ),
    "thermal_correction_h_ev": re.compile(
        rf"Thermal correction to H\(T\)\s*:\s*{_FLOAT_PATTERN}\s+kcal/mol\s+(?P<value>{_FLOAT_PATTERN})\s+eV"
    ),
    "thermal_correction_g_ev": re.compile(
        rf"Thermal correction to G\(T\)\s*:\s*{_FLOAT_PATTERN}\s+kcal/mol\s+(?P<value>{_FLOAT_PATTERN})\s+eV"
    ),
    "entropy_ev_per_k": re.compile(
        rf"Entropy S\s*:\s*{_FLOAT_PATTERN}\s+J/\(mol\*K\)\s+(?P<value>{_FLOAT_PATTERN})\s+eV/K"
    ),
    "entropy_contribution_ev": re.compile(
        rf"Entropy contribution T\*S\s*:\s*{_FLOAT_PATTERN}\s+J/\(mol\)\s+(?P<value>{_FLOAT_PATTERN})\s+eV"
    ),
}


class VaspkitAdsorbateThermoCorrectionInput(BaseModel):
    """Run VASPKIT task 501 for adsorbate thermochemistry and return parsed corrections."""

    calculation_dir: str = Field(
        ...,
        description="Workspace-relative path or repo-local test path to a directory containing VASP outputs for task 501.",
    )
    temperature_k: float = Field(
        298.15,
        gt=0.0,
        description="Temperature in Kelvin for the thermochemistry run. Defaults to the standard-state value 298.15 K.",
    )

    @field_validator("calculation_dir")
    @classmethod
    def _validate_calculation_dir(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("calculation_dir must not be empty")
        return text


class VaspkitGasThermoCorrectionInput(BaseModel):
    """Run VASPKIT task 502 for gas-phase thermochemistry and return parsed corrections."""

    calculation_dir: str = Field(
        ...,
        description="Workspace-relative path or repo-local test path to a directory containing VASP outputs for task 502.",
    )
    temperature_k: float = Field(
        298.15,
        gt=0.0,
        description="Temperature in Kelvin for the thermochemistry run. Defaults to the standard-state value 298.15 K.",
    )
    pressure_atm: float = Field(
        1.0,
        gt=0.0,
        description="Pressure in atm for the thermochemistry run. Defaults to the standard-state value 1 atm.",
    )
    spin_multiplicity: int = Field(
        1,
        ge=1,
        description="Spin multiplicity, i.e. number of unpaired electrons plus one. Defaults to singlet (1).",
    )

    @field_validator("calculation_dir")
    @classmethod
    def _validate_calculation_dir(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("calculation_dir must not be empty")
        return text


@dataclass(frozen=True)
class _ResolvedCalculationDir:
    path: Path
    scope: Literal["workspace", "repo"]
    display_path: str


@dataclass(frozen=True)
class VaspkitThermoResult:
    task_id: int
    backend: Literal["vaspkit", "ase"]
    calculation_dir: Path
    calculation_dir_scope: Literal["workspace", "repo"]
    calculation_dir_display: str
    temperature_k: float
    e_zpe_ev: float
    thermal_correction_u_ev: float
    thermal_correction_h_ev: float
    thermal_correction_g_ev: float
    entropy_ev_per_k: float
    entropy_contribution_ev: float
    pressure_pa: float | None = None
    spin_multiplicity: int | None = None
    pressure_atm_input: float | None = None
    approximation: str | None = None
    stdout: str = ""

    def as_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "backend": self.backend,
            "calculation_dir": self.calculation_dir_display,
            "calculation_dir_scope": self.calculation_dir_scope,
            "temperature_k": self.temperature_k,
            "e_zpe_ev": self.e_zpe_ev,
            "thermal_correction_u_ev": self.thermal_correction_u_ev,
            "thermal_correction_h_ev": self.thermal_correction_h_ev,
            "thermal_correction_g_ev": self.thermal_correction_g_ev,
            "entropy_ev_per_k": self.entropy_ev_per_k,
            "entropy_contribution_ev": self.entropy_contribution_ev,
        }
        if self.pressure_pa is not None:
            data["pressure_pa"] = self.pressure_pa
        if self.pressure_atm_input is not None:
            data["pressure_atm_input"] = self.pressure_atm_input
        if self.spin_multiplicity is not None:
            data["spin_multiplicity"] = self.spin_multiplicity
        if self.approximation:
            data["approximation"] = self.approximation
        return data


def _fail(
    tool_name: str,
    *,
    message: str,
    data: dict[str, Any] | None = None,
    error_code: str = "",
) -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_repo_local_path(raw_path: str) -> Path | None:
    candidate = Path(str(raw_path or "").strip()).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (_REPO_ROOT / candidate).resolve()
    if not _is_relative_to(resolved, _REPO_ROOT):
        return None
    if not resolved.exists():
        return None
    return resolved


def _resolve_calculation_dir(raw_path: str, *, tool_name: str) -> _ResolvedCalculationDir:
    try:
        workspace_path = resolve_workspace_path(raw_path, must_exist=True)
    except Exception:
        workspace_path = None
    if workspace_path is not None:
        if not workspace_path.is_dir():
            _fail(
                tool_name,
                message=f"calculation_dir is not a directory: {workspace_path}",
                data={"calculation_dir": workspace_relpath(workspace_path)},
                error_code="invalid_calculation_dir",
            )
        return _ResolvedCalculationDir(
            path=workspace_path,
            scope="workspace",
            display_path=workspace_relpath(workspace_path),
        )

    repo_path = _resolve_repo_local_path(raw_path)
    if repo_path is None:
        _fail(
            tool_name,
            message=f"calculation_dir not found under workspace files or repository: {raw_path}",
            data={"calculation_dir": raw_path},
            error_code="missing_calculation_dir",
        )
    if not repo_path.is_dir():
        _fail(
            tool_name,
            message=f"calculation_dir is not a directory: {repo_path}",
            data={"calculation_dir": str(repo_path.relative_to(_REPO_ROOT))},
            error_code="invalid_calculation_dir",
        )
    return _ResolvedCalculationDir(
        path=repo_path,
        scope="repo",
        display_path=str(repo_path.relative_to(_REPO_ROOT)),
    )


def _resolve_vaspkit_executable() -> Path | None:
    resolved = shutil.which("vaspkit")
    if resolved:
        candidate = Path(resolved).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = str(raw_entry or "").strip()
        if not entry:
            continue
        expanded = Path(os.path.expandvars(entry)).expanduser()
        candidate = expanded / "vaspkit"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    fallback = Path("~/vaspkit/bin/vaspkit").expanduser()
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return fallback.resolve()
    return None


def _require_vaspkit_executable(*, tool_name: str, calculation_dir: str) -> Path:
    executable = _resolve_vaspkit_executable()
    if executable is None:
        _fail(
            tool_name,
            message="vaspkit executable was not found. Ensure it is installed and reachable from PATH.",
            data={"calculation_dir": calculation_dir},
            error_code="missing_vaspkit",
        )
    return executable


def _extract_float(output_text: str, key: str, *, tool_name: str, task_id: int) -> float:
    pattern = _THERMO_PATTERNS[key]
    match = pattern.search(output_text)
    if match is None:
        preview = output_text[-1200:] if output_text else ""
        _fail(
            tool_name,
            message=f"Failed to parse `{key}` from VASPKIT {task_id} output.",
            data={"task_id": task_id, "output_tail": preview},
            error_code="parse_failure",
        )
    return float(match.group("value"))


def _read_required_outcar(calculation_dir: Path, *, tool_name: str) -> Path:
    outcar_path = calculation_dir / "OUTCAR"
    if not outcar_path.is_file():
        _fail(
            tool_name,
            message=f"Required OUTCAR not found in {calculation_dir}",
            data={"calculation_dir": str(calculation_dir)},
            error_code="missing_outcar",
        )
    return outcar_path


def _extract_outcar_vib_energies_ev(outcar_path: Path, *, tool_name: str) -> list[float]:
    text = outcar_path.read_text(encoding="utf-8", errors="ignore")
    vib_energies = [float(match.group("mev")) / 1000.0 for match in _OUTCAR_REAL_FREQ_PATTERN.finditer(text)]
    if not vib_energies:
        _fail(
            tool_name,
            message=f"Failed to parse real vibrational frequencies from {outcar_path}",
            data={"outcar": str(outcar_path)},
            error_code="missing_frequencies",
        )
    return vib_energies


def _infer_gas_geometry(atoms: Any) -> Literal["monatomic", "linear", "nonlinear"]:
    if len(atoms) <= 1:
        return "monatomic"
    inertias = sorted(float(value) for value in atoms.get_moments_of_inertia())
    largest = max(inertias[-1], 1.0)
    if inertias[0] / largest < 1.0e-3 and inertias[1] / largest > 1.0e-3:
        return "linear"
    return "nonlinear"


def _infer_symmetry_number(atoms: Any) -> tuple[int, str | None]:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        molecule = AseAtomsAdaptor.get_molecule(atoms)
        analyzer = PointGroupAnalyzer(molecule)
        symmetry_number = int(analyzer.get_rotational_symmetry_number())
        point_group = str(analyzer.sch_symbol)
        if symmetry_number >= 1:
            return symmetry_number, point_group
    except Exception:
        pass
    return 1, None


def _run_ase_adsorbate_thermo(
    *,
    calculation_dir: Path,
    calculation_dir_scope: Literal["workspace", "repo"],
    calculation_dir_display: str,
    temperature_k: float,
) -> VaspkitThermoResult:
    from ase.thermochemistry import HarmonicThermo

    tool_name = "vaspkit_adsorbate_thermo_correction"
    outcar_path = _read_required_outcar(calculation_dir, tool_name=tool_name)
    vib_energies = _extract_outcar_vib_energies_ev(outcar_path, tool_name=tool_name)
    floored_vib_energies = [max(value, _ADSORBATE_FREQ_FLOOR_EV) for value in vib_energies]

    thermo = HarmonicThermo(vib_energies=floored_vib_energies, potentialenergy=0.0, ignore_imag_modes=False)
    zpe = float(thermo.get_ZPE_correction())
    internal_energy = float(thermo.get_internal_energy(temperature_k, verbose=False))
    entropy = float(thermo.get_entropy(temperature_k, verbose=False))
    free_energy = float(thermo.get_helmholtz_energy(temperature_k, verbose=False))
    return VaspkitThermoResult(
        task_id=501,
        backend="ase",
        calculation_dir=calculation_dir,
        calculation_dir_scope=calculation_dir_scope,
        calculation_dir_display=calculation_dir_display,
        temperature_k=float(temperature_k),
        e_zpe_ev=zpe,
        thermal_correction_u_ev=internal_energy,
        thermal_correction_h_ev=internal_energy,
        thermal_correction_g_ev=free_energy,
        entropy_ev_per_k=entropy,
        entropy_contribution_ev=float(temperature_k) * entropy,
        approximation=(
            "ASE HarmonicThermo fallback; frequencies below 50 cm^-1 are floored to 50 cm^-1, "
            "and only harmonic vibrational contributions are retained, matching VASPKIT 501 semantics."
        ),
    )


def _run_ase_gas_thermo(
    *,
    calculation_dir: Path,
    calculation_dir_scope: Literal["workspace", "repo"],
    calculation_dir_display: str,
    temperature_k: float,
    pressure_atm: float,
    spin_multiplicity: int,
) -> VaspkitThermoResult:
    from ase import units
    from ase.io.vasp import read_vasp_out
    from ase.thermochemistry import IdealGasThermo

    tool_name = "vaspkit_gas_thermo_correction"
    outcar_path = _read_required_outcar(calculation_dir, tool_name=tool_name)
    vib_energies = _extract_outcar_vib_energies_ev(outcar_path, tool_name=tool_name)
    atoms = read_vasp_out(str(outcar_path), index=-1)
    geometry = _infer_gas_geometry(atoms)
    symmetry_number, point_group = _infer_symmetry_number(atoms)
    pressure_pa = float(pressure_atm) * _VASP_ATM_IN_PA
    spin_quantum_number = max(0.0, (float(spin_multiplicity) - 1.0) / 2.0)

    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=0.0,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry_number,
        spin=spin_quantum_number,
        ignore_imag_modes=False,
    )
    zpe = float(thermo.get_ZPE_correction())
    enthalpy = float(thermo.get_enthalpy(temperature_k, verbose=False))
    entropy = float(thermo.get_entropy(temperature_k, pressure_pa, verbose=False))
    gibbs_energy = float(thermo.get_gibbs_energy(temperature_k, pressure_pa, verbose=False))
    internal_energy = enthalpy - float(units.kB) * float(temperature_k)
    approximation = (
        "ASE IdealGasThermo fallback; uses ideal-gas translational/rotational/vibrational contributions "
        f"with geometry={geometry} and symmetrynumber={symmetry_number}"
    )
    if point_group:
        approximation += f" inferred from point group {point_group}"
    approximation += "."
    return VaspkitThermoResult(
        task_id=502,
        backend="ase",
        calculation_dir=calculation_dir,
        calculation_dir_scope=calculation_dir_scope,
        calculation_dir_display=calculation_dir_display,
        temperature_k=float(temperature_k),
        pressure_pa=pressure_pa,
        pressure_atm_input=float(pressure_atm),
        spin_multiplicity=int(spin_multiplicity),
        e_zpe_ev=zpe,
        thermal_correction_u_ev=internal_energy,
        thermal_correction_h_ev=enthalpy,
        thermal_correction_g_ev=gibbs_energy,
        entropy_ev_per_k=entropy,
        entropy_contribution_ev=float(temperature_k) * entropy,
        approximation=approximation,
    )


def _run_vaspkit_task(
    *,
    task_id: int,
    calculation_dir: Path,
    stdin_payload: str,
    tool_name: str,
) -> str:
    executable = _require_vaspkit_executable(tool_name=tool_name, calculation_dir=str(calculation_dir))
    try:
        completed = subprocess.run(
            [str(executable)],
            input=stdin_payload,
            cwd=calculation_dir,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _fail(
            tool_name,
            message=f"VASPKIT {task_id} timed out after {exc.timeout} seconds.",
            data={"task_id": task_id, "calculation_dir": str(calculation_dir)},
            error_code="vaspkit_timeout",
        )
    except OSError as exc:
        _fail(
            tool_name,
            message=f"Failed to start VASPKIT {task_id}: {exc}",
            data={"task_id": task_id, "calculation_dir": str(calculation_dir)},
            error_code="vaspkit_launch_failed",
        )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        _fail(
            tool_name,
            message=f"VASPKIT {task_id} exited with code {completed.returncode}.",
            data={
                "task_id": task_id,
                "calculation_dir": str(calculation_dir),
                "stdout_tail": stdout[-1200:],
                "stderr_tail": stderr[-1200:],
            },
            error_code="vaspkit_nonzero_exit",
        )
    return stdout


def _parse_vaspkit_thermo_output(
    *,
    task_id: int,
    calculation_dir: Path,
    calculation_dir_scope: Literal["workspace", "repo"],
    calculation_dir_display: str,
    output_text: str,
    tool_name: str,
    pressure_atm_input: float | None = None,
) -> VaspkitThermoResult:
    pressure_pa = None
    spin_multiplicity = None
    if task_id == 502:
        pressure_pa = _extract_float(output_text, "pressure_pa", tool_name=tool_name, task_id=task_id)
        spin_multiplicity = int(round(_extract_float(output_text, "spin_multiplicity", tool_name=tool_name, task_id=task_id)))
    return VaspkitThermoResult(
        task_id=task_id,
        backend="vaspkit",
        calculation_dir=calculation_dir,
        calculation_dir_scope=calculation_dir_scope,
        calculation_dir_display=calculation_dir_display,
        temperature_k=_extract_float(output_text, "temperature_k", tool_name=tool_name, task_id=task_id),
        pressure_pa=pressure_pa,
        pressure_atm_input=pressure_atm_input,
        spin_multiplicity=spin_multiplicity,
        e_zpe_ev=_extract_float(output_text, "e_zpe_ev", tool_name=tool_name, task_id=task_id),
        thermal_correction_u_ev=_extract_float(output_text, "thermal_correction_u_ev", tool_name=tool_name, task_id=task_id),
        thermal_correction_h_ev=_extract_float(output_text, "thermal_correction_h_ev", tool_name=tool_name, task_id=task_id),
        thermal_correction_g_ev=_extract_float(output_text, "thermal_correction_g_ev", tool_name=tool_name, task_id=task_id),
        entropy_ev_per_k=_extract_float(output_text, "entropy_ev_per_k", tool_name=tool_name, task_id=task_id),
        entropy_contribution_ev=_extract_float(output_text, "entropy_contribution_ev", tool_name=tool_name, task_id=task_id),
        stdout=output_text,
    )


def run_vaspkit_adsorbate_thermo(
    *,
    calculation_dir: Path,
    calculation_dir_scope: Literal["workspace", "repo"],
    calculation_dir_display: str,
    temperature_k: float,
) -> VaspkitThermoResult:
    tool_name = "vaspkit_adsorbate_thermo_correction"
    executable = _resolve_vaspkit_executable()
    if executable is None:
        return _run_ase_adsorbate_thermo(
            calculation_dir=calculation_dir,
            calculation_dir_scope=calculation_dir_scope,
            calculation_dir_display=calculation_dir_display,
            temperature_k=temperature_k,
        )
    stdout = _run_vaspkit_task(
        task_id=501,
        calculation_dir=calculation_dir,
        stdin_payload=f"501\n{temperature_k}\n",
        tool_name=tool_name,
    )
    return _parse_vaspkit_thermo_output(
        task_id=501,
        calculation_dir=calculation_dir,
        calculation_dir_scope=calculation_dir_scope,
        calculation_dir_display=calculation_dir_display,
        output_text=stdout,
        tool_name=tool_name,
    )


def run_vaspkit_gas_thermo(
    *,
    calculation_dir: Path,
    calculation_dir_scope: Literal["workspace", "repo"],
    calculation_dir_display: str,
    temperature_k: float,
    pressure_atm: float,
    spin_multiplicity: int,
) -> VaspkitThermoResult:
    tool_name = "vaspkit_gas_thermo_correction"
    executable = _resolve_vaspkit_executable()
    if executable is None:
        return _run_ase_gas_thermo(
            calculation_dir=calculation_dir,
            calculation_dir_scope=calculation_dir_scope,
            calculation_dir_display=calculation_dir_display,
            temperature_k=temperature_k,
            pressure_atm=pressure_atm,
            spin_multiplicity=spin_multiplicity,
        )
    stdout = _run_vaspkit_task(
        task_id=502,
        calculation_dir=calculation_dir,
        stdin_payload=f"502\n{temperature_k}\n{pressure_atm}\n{spin_multiplicity}\n",
        tool_name=tool_name,
    )
    return _parse_vaspkit_thermo_output(
        task_id=502,
        calculation_dir=calculation_dir,
        calculation_dir_scope=calculation_dir_scope,
        calculation_dir_display=calculation_dir_display,
        output_text=stdout,
        tool_name=tool_name,
        pressure_atm_input=pressure_atm,
    )


def vaspkit_adsorbate_thermo_correction(payload: dict[str, object]) -> tuple[str, dict[str, Any]]:
    params = VaspkitAdsorbateThermoCorrectionInput(**payload)
    resolved = _resolve_calculation_dir(params.calculation_dir, tool_name="vaspkit_adsorbate_thermo_correction")
    result = run_vaspkit_adsorbate_thermo(
        calculation_dir=resolved.path,
        calculation_dir_scope=resolved.scope,
        calculation_dir_display=resolved.display_path,
        temperature_k=params.temperature_k,
    )
    data = result.as_data()
    content = (
        f"Thermochemistry backend={result.backend} task=501 dir={resolved.display_path} "
        f"T={result.temperature_k:.1f} K "
        f"E_ZPE={result.e_zpe_ev:.6f} eV "
        f"Delta_U={result.thermal_correction_u_ev:.6f} eV "
        f"Delta_H={result.thermal_correction_h_ev:.6f} eV "
        f"Delta_G={result.thermal_correction_g_ev:.6f} eV "
        f"S={result.entropy_ev_per_k:.6f} eV/K "
        f"T*S={result.entropy_contribution_ev:.6f} eV."
    )
    if result.approximation:
        content += f" Approximation: {result.approximation}"
    return content, {"tool_name": "vaspkit_adsorbate_thermo_correction", "data": data}


def vaspkit_gas_thermo_correction(payload: dict[str, object]) -> tuple[str, dict[str, Any]]:
    params = VaspkitGasThermoCorrectionInput(**payload)
    resolved = _resolve_calculation_dir(params.calculation_dir, tool_name="vaspkit_gas_thermo_correction")
    result = run_vaspkit_gas_thermo(
        calculation_dir=resolved.path,
        calculation_dir_scope=resolved.scope,
        calculation_dir_display=resolved.display_path,
        temperature_k=params.temperature_k,
        pressure_atm=params.pressure_atm,
        spin_multiplicity=params.spin_multiplicity,
    )
    data = result.as_data()
    content = (
        f"Thermochemistry backend={result.backend} task=502 dir={resolved.display_path} "
        f"T={result.temperature_k:.1f} K "
        f"P={float(result.pressure_atm_input or 0.0):.6f} atm "
        f"spin_multiplicity={int(result.spin_multiplicity or 0)} "
        f"E_ZPE={result.e_zpe_ev:.6f} eV "
        f"Delta_U={result.thermal_correction_u_ev:.6f} eV "
        f"Delta_H={result.thermal_correction_h_ev:.6f} eV "
        f"Delta_G={result.thermal_correction_g_ev:.6f} eV "
        f"S={result.entropy_ev_per_k:.6f} eV/K "
        f"T*S={result.entropy_contribution_ev:.6f} eV."
    )
    if result.approximation:
        content += f" Approximation: {result.approximation}"
    return content, {"tool_name": "vaspkit_gas_thermo_correction", "data": data}


__all__ = [
    "VaspkitAdsorbateThermoCorrectionInput",
    "VaspkitGasThermoCorrectionInput",
    "VaspkitThermoResult",
    "_resolve_vaspkit_executable",
    "run_vaspkit_adsorbate_thermo",
    "run_vaspkit_gas_thermo",
    "vaspkit_adsorbate_thermo_correction",
    "vaspkit_gas_thermo_correction",
]
