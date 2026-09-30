from __future__ import annotations

from pathlib import Path
import subprocess

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


def test_lammps_boot_cpu_mode_never_enables_visible_gpu() -> None:
    prefix, acceleration = lammps_boot._gpu_command_prefix(
        "/opt/lammps/lmp",
        "off",
        1,
        "Installed packages: KOKKOS GPU",
    )

    assert prefix == ["/opt/lammps/lmp"]
    assert acceleration == "cpu"


def test_lammps_boot_explicit_kokkos_requires_visible_gpu() -> None:
    with pytest.raises(RuntimeError, match="requires at least one visible GPU"):
        lammps_boot._gpu_command_prefix(
            "/opt/lammps/lmp",
            "kokkos",
            0,
            "Installed packages: KOKKOS",
        )


def test_lammps_boot_explicit_kokkos_requires_package() -> None:
    with pytest.raises(RuntimeError, match="lacks KOKKOS"):
        lammps_boot._gpu_command_prefix(
            "/opt/lammps/lmp",
            "kokkos",
            1,
            "Installed packages: MOLECULE",
        )


def test_lammps_boot_explicit_kokkos_builds_strict_gpu_prefix() -> None:
    prefix, acceleration = lammps_boot._gpu_command_prefix(
        "/opt/lammps/lmp",
        "kokkos",
        1,
        "Installed packages: KOKKOS",
    )

    assert prefix == ["/opt/lammps/lmp", "-k", "on", "g", "1", "-sf", "kk"]
    assert acceleration == "kokkos"


def test_lammps_boot_nprocs_follow_slurm_slots(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "16")

    assert lammps_boot._resolve_nprocs(None) == 16
    assert lammps_boot._resolve_nprocs(4) == 4


def test_lammps_boot_nprocs_default_to_one_without_scheduler(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_NTASKS", raising=False)

    assert lammps_boot._resolve_nprocs(None) == 1


@pytest.mark.parametrize("value", ["0", "-2", "not-an-int"])
def test_lammps_boot_rejects_invalid_slurm_slots(value: str, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_NTASKS", value)

    with pytest.raises(ValueError, match="SLURM_NTASKS"):
        lammps_boot._resolve_nprocs(None)


def test_lammps_boot_detects_real_and_stub_mpi_builds() -> None:
    enabled, report = lammps_boot._lammps_mpi_build("LAMMPS\nMPI v4.1: Open MPI v5.0.8\n")
    stubbed, stub_report = lammps_boot._lammps_mpi_build("MPI v1.0: LAMMPS MPI STUBS\n")

    assert enabled is True
    assert report == "MPI v4.1: Open MPI v5.0.8"
    assert stubbed is False
    assert "STUBS" in stub_report


def test_lammps_boot_multi_rank_launcher_uses_resolved_binary(tmp_path: Path, monkeypatch) -> None:
    launcher = _exe(tmp_path / "mpirun")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CATMASTER_LAMMPS_MPI_LAUNCHER", raising=False)

    prefix, resolved = lammps_boot._resolve_mpi_launcher("auto", nprocs=16)

    assert resolved == str(launcher)
    assert prefix == [str(launcher), "-n", "16"]


def test_lammps_boot_single_rank_does_not_require_launcher(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")

    prefix, resolved = lammps_boot._resolve_mpi_launcher("auto", nprocs=1)

    assert prefix == []
    assert resolved == "direct"


def test_lammps_boot_intel_mpi_uses_fork_on_single_node_without_srun(tmp_path: Path) -> None:
    original = {
        "PATH": str(tmp_path),
        "SLURM_JOB_ID": "4366",
        "SLURM_NNODES": "1",
    }

    configured, report = lammps_boot._configure_mpi_environment(
        original,
        mpi_report="MPI v4.1: Intel(R) MPI Library 2021.16",
        nprocs=16,
    )

    assert "I_MPI_HYDRA_BOOTSTRAP" not in original
    assert configured["I_MPI_HYDRA_BOOTSTRAP"] == "fork"
    assert report == {
        "intel_mpi": True,
        "slurm_nodes": 1,
        "srun_available": False,
        "hydra_bootstrap": "fork",
        "hydra_bootstrap_source": "auto_single_node_without_srun",
    }


def test_lammps_boot_preserves_explicit_intel_mpi_bootstrap(tmp_path: Path) -> None:
    configured, report = lammps_boot._configure_mpi_environment(
        {
            "PATH": str(tmp_path),
            "SLURM_NNODES": "1",
            "I_MPI_HYDRA_BOOTSTRAP": "ssh",
        },
        mpi_report="MPI v4.1: Intel(R) MPI Library 2021.16",
        nprocs=16,
    )

    assert configured["I_MPI_HYDRA_BOOTSTRAP"] == "ssh"
    assert report["hydra_bootstrap_source"] == "environment"


def test_lammps_boot_replaces_unusable_intel_slurm_bootstrap(tmp_path: Path) -> None:
    configured, report = lammps_boot._configure_mpi_environment(
        {
            "PATH": str(tmp_path),
            "SLURM_NNODES": "1",
            "I_MPI_HYDRA_BOOTSTRAP": "slurm",
        },
        mpi_report="MPI v4.1: Intel(R) MPI Library 2021.16",
        nprocs=16,
    )

    assert configured["I_MPI_HYDRA_BOOTSTRAP"] == "fork"
    assert report["hydra_bootstrap_source"] == "auto_single_node_without_srun"


def test_lammps_boot_does_not_apply_intel_fork_to_openmpi(tmp_path: Path) -> None:
    configured, report = lammps_boot._configure_mpi_environment(
        {"PATH": str(tmp_path), "SLURM_NNODES": "1"},
        mpi_report="MPI v3.1: Open MPI v5.0.8",
        nprocs=16,
    )

    assert "I_MPI_HYDRA_BOOTSTRAP" not in configured
    assert report["intel_mpi"] is False
    assert report["hydra_bootstrap_source"] == "unchanged"


def test_lammps_boot_mpi_probe_requires_exact_process_count(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="\n".join(["CATMASTER_LAMMPS_MPI_PROBE"] * 16) + "\n",
        )

    monkeypatch.setattr(lammps_boot.subprocess, "run", _fake_run)
    result = lammps_boot._probe_mpi_launcher(
        ["/usr/bin/mpirun", "-n", "16"],
        expected_ranks=16,
        env={},
    )

    assert result["status"] == "passed"
    assert result["observed_processes"] == 16


def test_lammps_boot_mpi_probe_rejects_slot_mismatch(monkeypatch) -> None:
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="\n".join(["CATMASTER_LAMMPS_MPI_PROBE"] * 15) + "\n",
        )

    monkeypatch.setattr(lammps_boot.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="expected=16, observed=15"):
        lammps_boot._probe_mpi_launcher(
            ["/usr/bin/mpirun", "-n", "16"],
            expected_ranks=16,
            env={},
        )
