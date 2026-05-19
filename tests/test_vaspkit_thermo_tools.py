from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.tools.analysis.vaspkit_thermo import (
    _resolve_vaspkit_executable,
    VaspkitAdsorbateThermoCorrectionInput,
    VaspkitGasThermoCorrectionInput,
    vaspkit_adsorbate_thermo_correction,
    vaspkit_gas_thermo_correction,
)


_VASP_ADSORBATE_REFERENCE = "demos/reference_scripts/vaspkit_501_502/501_Z5_PX"
_VASP_GAS_REFERENCE = "demos/reference_scripts/vaspkit_501_502/502_PX"


def test_resolve_vaspkit_executable_expands_tilde_in_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "vaspkit"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "~/bin")

    resolved = _resolve_vaspkit_executable()

    assert resolved == executable.resolve()


def _require_vaspkit() -> None:
    if _resolve_vaspkit_executable() is None:
        pytest.skip("vaspkit executable is not available in this environment")


def test_thermo_input_models_default_to_standard_state() -> None:
    ads = VaspkitAdsorbateThermoCorrectionInput(calculation_dir="foo")
    gas = VaspkitGasThermoCorrectionInput(calculation_dir="bar")

    assert ads.temperature_k == pytest.approx(298.15)
    assert gas.temperature_k == pytest.approx(298.15)
    assert gas.pressure_atm == pytest.approx(1.0)
    assert gas.spin_multiplicity == 1


def test_vaspkit_adsorbate_thermo_correction_matches_reference_sample() -> None:
    _require_vaspkit()

    content, artifact = vaspkit_adsorbate_thermo_correction(
        {
            "calculation_dir": _VASP_ADSORBATE_REFERENCE,
            "temperature_k": 623.0,
        }
    )

    data = artifact["data"]
    assert artifact["tool_name"] == "vaspkit_adsorbate_thermo_correction"
    assert data["task_id"] == 501
    assert data["backend"] == "vaspkit"
    assert data["calculation_dir_scope"] == "repo"
    assert data["calculation_dir"] == _VASP_ADSORBATE_REFERENCE
    assert data["temperature_k"] == pytest.approx(623.0)
    assert data["e_zpe_ev"] == pytest.approx(4.151605)
    assert data["thermal_correction_u_ev"] == pytest.approx(5.081731)
    assert data["thermal_correction_h_ev"] == pytest.approx(5.081731)
    assert data["thermal_correction_g_ev"] == pytest.approx(3.099589)
    assert data["entropy_ev_per_k"] == pytest.approx(0.003182)
    assert data["entropy_contribution_ev"] == pytest.approx(1.982141)
    assert "E_ZPE=4.151605 eV" in content
    assert "backend=vaspkit" in content
    assert "task=501" in content


def test_vaspkit_gas_thermo_correction_matches_reference_sample() -> None:
    _require_vaspkit()

    content, artifact = vaspkit_gas_thermo_correction(
        {
            "calculation_dir": _VASP_GAS_REFERENCE,
            "temperature_k": 623.0,
            "pressure_atm": 1.0,
            "spin_multiplicity": 1,
        }
    )

    data = artifact["data"]
    assert artifact["tool_name"] == "vaspkit_gas_thermo_correction"
    assert data["task_id"] == 502
    assert data["backend"] == "vaspkit"
    assert data["calculation_dir_scope"] == "repo"
    assert data["calculation_dir"] == _VASP_GAS_REFERENCE
    assert data["temperature_k"] == pytest.approx(623.0)
    assert data["pressure_atm_input"] == pytest.approx(1.0)
    assert data["pressure_pa"] == pytest.approx(101325.0)
    assert data["spin_multiplicity"] == 1
    assert data["e_zpe_ev"] == pytest.approx(4.105209)
    assert data["thermal_correction_u_ev"] == pytest.approx(4.959939)
    assert data["thermal_correction_h_ev"] == pytest.approx(5.013624)
    assert data["thermal_correction_g_ev"] == pytest.approx(1.756929)
    assert data["entropy_ev_per_k"] == pytest.approx(0.005227)
    assert data["entropy_contribution_ev"] == pytest.approx(3.256695)
    assert "E_ZPE=4.105209 eV" in content
    assert "backend=vaspkit" in content
    assert "task=502" in content


def test_vaspkit_adsorbate_thermo_correction_falls_back_to_ase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catmaster.tools.analysis.vaspkit_thermo._resolve_vaspkit_executable", lambda: None)

    content, artifact = vaspkit_adsorbate_thermo_correction(
        {
            "calculation_dir": _VASP_ADSORBATE_REFERENCE,
            "temperature_k": 623.0,
        }
    )

    data = artifact["data"]
    assert data["backend"] == "ase"
    assert data["task_id"] == 501
    assert data["e_zpe_ev"] == pytest.approx(4.153255, abs=5e-6)
    assert data["thermal_correction_u_ev"] == pytest.approx(5.081759, abs=5e-6)
    assert data["thermal_correction_h_ev"] == pytest.approx(5.081759, abs=5e-6)
    assert data["thermal_correction_g_ev"] == pytest.approx(3.099624, abs=5e-6)
    assert data["entropy_ev_per_k"] == pytest.approx(0.0031816, abs=5e-7)
    assert "HarmonicThermo" in data["approximation"]
    assert "backend=ase" in content
    assert "task=501" in content
    assert "Approximation:" in content


def test_vaspkit_gas_thermo_correction_falls_back_to_ase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catmaster.tools.analysis.vaspkit_thermo._resolve_vaspkit_executable", lambda: None)

    content, artifact = vaspkit_gas_thermo_correction(
        {
            "calculation_dir": _VASP_GAS_REFERENCE,
            "temperature_k": 623.0,
            "pressure_atm": 1.0,
            "spin_multiplicity": 1,
        }
    )

    data = artifact["data"]
    assert data["backend"] == "ase"
    assert data["task_id"] == 502
    assert data["pressure_pa"] == pytest.approx(101325.0)
    assert data["spin_multiplicity"] == 1
    assert data["e_zpe_ev"] == pytest.approx(4.105312, abs=5e-6)
    assert data["thermal_correction_u_ev"] == pytest.approx(4.960062, abs=1e-5)
    assert data["thermal_correction_h_ev"] == pytest.approx(5.013748, abs=5e-6)
    assert data["thermal_correction_g_ev"] == pytest.approx(1.756973, abs=5e-6)
    assert data["entropy_ev_per_k"] == pytest.approx(0.00522757, abs=5e-7)
    assert "IdealGasThermo" in data["approximation"]
    assert "point group C2v" in data["approximation"]
    assert "backend=ase" in content
    assert "task=502" in content
    assert "Approximation:" in content
