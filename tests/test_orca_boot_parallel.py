from __future__ import annotations

from catmaster.remote.cpu import orca_boot


def test_resolve_nprocs_prefers_slurm_ntasks(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "32")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "64(x1)")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert orca_boot._resolve_nprocs() == 32


def test_resolve_nprocs_falls_back_to_slurm_cpu_shape(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "48")
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "48(x1)")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert orca_boot._resolve_nprocs() == 48


def test_runtime_pal_reconciliation_inserts_block_without_mutating_canonical_text() -> None:
    canonical = "! B3LYP def2-SVP Opt\n%maxcore 512\n* xyzfile 0 1 input.xyz\n"
    runtime = orca_boot._runtime_input_text(canonical, 24)

    assert canonical == "! B3LYP def2-SVP Opt\n%maxcore 512\n* xyzfile 0 1 input.xyz\n"
    assert "! B3LYP def2-SVP Opt\n%pal\n  nprocs 24\nend\n%maxcore 512" in runtime


def test_runtime_pal_reconciliation_replaces_only_total_process_count() -> None:
    canonical = (
        "! XTB2 def2-SVP TightSCF Opt\n"
        "%pal\n  nprocs 8\nend\n"
        "%maxcore 1000\n"
        "* xyzfile 0 3 input.xyz\n"
    )
    runtime = orca_boot._runtime_input_text(canonical, 32)

    assert "nprocs 32" in runtime
    assert "nprocs 8" not in runtime
