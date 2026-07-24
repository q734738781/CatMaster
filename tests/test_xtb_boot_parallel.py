from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _write_manifest(stage_dir: Path, **overrides) -> Path:
    payload = {
        "schema_version": 1,
        "program": "xtb",
        "coordinate_file": "coord.xyz",
        "xcontrol_file": "",
        "mode": "sp",
        "gfn": "gfn2",
        "solvent_model": "none",
        "solvent": "",
        "charge": 0,
        "uhf": 0,
        "opt_level": "normal",
    }
    payload.update(overrides)
    (stage_dir / "coord.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
    manifest_path = stage_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_xtb_boot_builds_command_from_prepared_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "xtb.inp").write_text("$constrain\n  distance: 1, 2, 0.8\n$end\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        xcontrol_file="xtb.inp",
        mode="opt",
        gfn="gfn1",
        solvent_model="alpb",
        solvent="water",
        charge=-1,
        uhf=1,
        opt_level="tight",
    )

    config = xtb_boot._load_manifest(manifest_path)
    command = xtb_boot._build_xtb_command(config, xtb_bin="/opt/xtb")

    assert command == [
        "/opt/xtb",
        "coord.xyz",
        "--gfn",
        "1",
        "--chrg",
        "-1",
        "--uhf",
        "1",
        "--alpb",
        "water",
        "--opt",
        "tight",
        "--input",
        "xtb.inp",
    ]


def test_xtb_boot_uses_default_single_point_without_runtime_science_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = xtb_boot._load_manifest(_write_manifest(tmp_path))

    command = xtb_boot._build_xtb_command(config, xtb_bin="xtb")

    assert command == ["xtb", "coord.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0"]


def test_xtb_boot_rejects_nested_manifest_file_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "controls").mkdir()
    (tmp_path / "controls" / "xtb.inp").write_text("$end\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, xcontrol_file="controls/xtb.inp")

    with pytest.raises(ValueError, match="direct stage filename"):
        xtb_boot._load_manifest(manifest_path)


def test_xtb_boot_main_executes_prepared_manifest_and_writes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["xtb_boot.py", "--manifest", "manifest.json", "--xtb_bin", "/bin/true"],
    )

    assert xtb_boot.main() == 0

    summary = json.loads((tmp_path / "xtb_summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["returncode"] == 0
    assert summary["manifest"] == "manifest.json"
    assert summary["coordinate_file"] == "coord.xyz"
    assert summary["command"][0] == "/bin/true"
