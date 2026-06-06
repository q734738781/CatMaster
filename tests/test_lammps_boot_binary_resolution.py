from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.remote.cpu import lammps_boot


def _exe(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_lammps_boot_auto_prefers_cpu_mpi_binary(tmp_path: Path, monkeypatch) -> None:
    _exe(tmp_path / "lmp")
    expected = _exe(tmp_path / "lmp_mpi")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CATMASTER_LAMMPS_BIN", raising=False)
    monkeypatch.delenv("CATMASTER_LAMMPS_BIN_CANDIDATES", raising=False)

    resolved, candidates = lammps_boot._resolve_lammps_binary("auto", mode="auto", gpu_count=0)

    assert resolved == str(expected)
    assert candidates.index("lmp_mpi") < candidates.index("lmp")


def test_lammps_boot_auto_prefers_gpu_binary_when_gpu_visible(tmp_path: Path, monkeypatch) -> None:
    _exe(tmp_path / "lmp_mpi")
    expected = _exe(tmp_path / "lmp_kokkos_cuda_mpi")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CATMASTER_LAMMPS_BIN", raising=False)
    monkeypatch.delenv("CATMASTER_LAMMPS_BIN_CANDIDATES", raising=False)

    resolved, candidates = lammps_boot._resolve_lammps_binary("auto", mode="auto", gpu_count=1)

    assert resolved == str(expected)
    assert candidates.index("lmp_kokkos_cuda_mpi") < candidates.index("lmp_mpi")


def test_lammps_boot_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    expected = _exe(tmp_path / "custom_lammps")
    _exe(tmp_path / "lmp_mpi")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("CATMASTER_LAMMPS_BIN", "custom_lammps")
    monkeypatch.delenv("CATMASTER_LAMMPS_BIN_CANDIDATES", raising=False)

    resolved, candidates = lammps_boot._resolve_lammps_binary("auto", mode="auto", gpu_count=0)

    assert resolved == str(expected)
    assert candidates[0] == "custom_lammps"


def test_lammps_boot_reports_all_candidates_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("CATMASTER_LAMMPS_BIN_CANDIDATES", "missing_a,missing_b")
    monkeypatch.delenv("CATMASTER_LAMMPS_BIN", raising=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        lammps_boot._resolve_lammps_binary("auto", mode="auto", gpu_count=0)

    message = str(excinfo.value)
    assert "Unable to resolve LAMMPS executable" in message
    assert "missing_a" in message
    assert "missing_b" in message
    assert "lmp_mpi" in message
