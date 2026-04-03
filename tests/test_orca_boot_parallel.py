from __future__ import annotations

from pathlib import Path

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


def test_upsert_pal_nprocs_inserts_block_when_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "job.inp"
    input_path.write_text("! B3LYP def2-SVP Opt\n%maxcore 512\n* xyzfile 0 1 input.xyz\n", encoding="utf-8")
    status = orca_boot._upsert_pal_nprocs(input_path, 24)
    assert status == "inserted"
    text = input_path.read_text(encoding="utf-8")
    assert "! B3LYP def2-SVP Opt\n%pal\n  nprocs 24\nend\n%maxcore 512" in text


def test_upsert_pal_nprocs_replaces_existing_block(tmp_path: Path) -> None:
    input_path = tmp_path / "job.inp"
    input_path.write_text(
        "! XTB2 def2-SVP TightSCF Opt\n%pal\n  nprocs 8\nend\n%maxcore 1000\n* xyzfile 0 3 input.xyz\n",
        encoding="utf-8",
    )
    status = orca_boot._upsert_pal_nprocs(input_path, 32)
    assert status == "replaced"
    text = input_path.read_text(encoding="utf-8")
    assert "nprocs 32" in text
    assert "nprocs 8" not in text

