from __future__ import annotations

from catmaster.remote.cpu import xtb_boot


def test_xtb_boot_prefers_explicit_thread_count(monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_XTB_THREADS", "12")
    monkeypatch.setenv("SLURM_NTASKS", "32")

    assert xtb_boot._resolve_threads() == 12


def test_xtb_boot_reads_slurm_thread_count(monkeypatch) -> None:
    monkeypatch.delenv("CATMASTER_XTB_THREADS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "16")

    assert xtb_boot._resolve_threads() == 16


def test_xtb_boot_parses_slurm_job_cpus_per_node(monkeypatch) -> None:
    monkeypatch.delenv("CATMASTER_XTB_THREADS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32(x1)")

    assert xtb_boot._resolve_threads() == 32


def test_xtb_boot_sets_openmp_environment(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    env = xtb_boot._xtb_env(8)

    assert env["OMP_NUM_THREADS"] == "8,1"
    assert env["MKL_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == "8"
    assert env["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert env["OMP_STACKSIZE"] == "4G"
