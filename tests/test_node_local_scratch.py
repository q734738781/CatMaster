from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from catmaster.remote.cpu import cp2k_boot, node_local_scratch, orca_boot


def _enable_test_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_local_scratch, "_same_filesystem", lambda _left, _right: False)
    monkeypatch.setattr(node_local_scratch.shutil, "which", lambda _name: None)


def test_single_node_uses_scratch_and_preserves_shared_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    scratch_root = tmp_path / "node-tmp"
    shared.mkdir()
    scratch_root.mkdir()
    (shared / "job.inp").write_text("input\n", encoding="utf-8")
    (shared / "stdout.log").write_text("active wrapper log\n", encoding="utf-8")
    monkeypatch.chdir(shared)
    _enable_test_scratch(monkeypatch)

    env = {
        "SLURM_NNODES": "1",
        "SLURM_JOB_ID": "123",
        "CATMASTER_SCRATCH_ROOT": str(scratch_root),
    }
    with node_local_scratch.node_local_workdir("orca", env=env) as workdir:
        assert workdir.node_local is True
        assert Path.cwd() == workdir.run_dir
        assert (Path.cwd() / "job.inp").read_text(encoding="utf-8") == "input\n"
        (Path.cwd() / "job.out").write_text("result\n", encoding="utf-8")
        (Path.cwd() / "stdout.log").write_text("stale scratch log\n", encoding="utf-8")

    assert Path.cwd() == shared
    assert (shared / "job.out").read_text(encoding="utf-8") == "result\n"
    assert (shared / "stdout.log").read_text(encoding="utf-8") == "active wrapper log\n"
    assert list(scratch_root.iterdir()) == []


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
def test_rsync_copy_back_does_not_overwrite_shared_bookkeeping(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    shared = tmp_path / "shared"
    scratch.mkdir()
    shared.mkdir()
    (scratch / "job.out").write_text("result\n", encoding="utf-8")
    (scratch / "stdout.log").write_text("stale scratch log\n", encoding="utf-8")
    (shared / "stdout.log").write_text("active wrapper log\n", encoding="utf-8")

    node_local_scratch._sync_tree(scratch, shared)

    assert (shared / "job.out").read_text(encoding="utf-8") == "result\n"
    assert (shared / "stdout.log").read_text(encoding="utf-8") == "active wrapper log\n"


@pytest.mark.parametrize("env", [{"SLURM_NNODES": "2"}, {}])
def test_multi_node_or_unknown_allocation_stays_in_shared_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env: dict[str, str],
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.chdir(shared)

    with node_local_scratch.node_local_workdir("cp2k", env=env) as workdir:
        assert workdir.node_local is False
        assert workdir.run_dir == shared
        assert Path.cwd() == shared


def test_scientific_exception_still_copies_partial_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    scratch_root = tmp_path / "node-tmp"
    shared.mkdir()
    scratch_root.mkdir()
    monkeypatch.chdir(shared)
    _enable_test_scratch(monkeypatch)
    env = {"SLURM_NNODES": "1", "CATMASTER_SCRATCH_ROOT": str(scratch_root)}

    with pytest.raises(RuntimeError, match="scientific failure"):
        with node_local_scratch.node_local_workdir("orca", env=env):
            Path("partial.gbw").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("scientific failure")

    assert (shared / "partial.gbw").read_text(encoding="utf-8") == "partial\n"
    assert list(scratch_root.iterdir()) == []


def test_copy_back_failure_is_reported_and_retains_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    scratch_root = tmp_path / "node-tmp"
    shared.mkdir()
    scratch_root.mkdir()
    (shared / "job.inp").write_text("input\n", encoding="utf-8")
    monkeypatch.chdir(shared)
    _enable_test_scratch(monkeypatch)
    original_sync = node_local_scratch._sync_tree
    calls = 0

    def _fail_on_copy_back(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("shared filesystem unavailable")
        original_sync(source, destination)

    monkeypatch.setattr(node_local_scratch, "_sync_tree", _fail_on_copy_back)
    env = {"SLURM_NNODES": "1", "CATMASTER_SCRATCH_ROOT": str(scratch_root)}
    retained: Path | None = None
    with pytest.raises(node_local_scratch.ScratchStageOutError, match="shared filesystem unavailable"):
        with node_local_scratch.node_local_workdir("orca", env=env) as workdir:
            retained = workdir.run_dir
            Path("job.out").write_text("result\n", encoding="utf-8")

    assert retained is not None and retained.is_dir()
    assert (retained / "job.out").read_text(encoding="utf-8") == "result\n"
    assert not (shared / "job.out").exists()


def test_orca_boot_runs_in_single_node_scratch_and_returns_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "orca-stage"
    scratch_root = tmp_path / "node-tmp"
    shared.mkdir()
    scratch_root.mkdir()
    canonical = "! HF def2-SVP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n"
    (shared / "job.inp").write_text(canonical, encoding="utf-8")
    (shared / "job.out").write_text("stale output from an earlier attempt\n", encoding="utf-8")
    monkeypatch.chdir(shared)
    _enable_test_scratch(monkeypatch)
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setenv("CATMASTER_SCRATCH_ROOT", str(scratch_root))
    observed_cwd: list[Path] = []

    def _fake_run(argv, *, stdout, stderr, check, **kwargs):
        _ = (stderr, check, kwargs)
        observed_cwd.append(Path.cwd())
        assert Path(stdout.name) == shared / "job.out"
        assert not (Path.cwd() / "job.out").exists()
        stdout.write("ORCA TERMINATED NORMALLY\n")
        stdout.flush()
        assert (shared / "job.out").read_text(encoding="utf-8") == "ORCA TERMINATED NORMALLY\n"
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(orca_boot.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", [orca_boot.__file__, "--input", "job.inp", "--orca_bin", "orca"])

    assert orca_boot.main() == 0
    assert observed_cwd and observed_cwd[0].parent == scratch_root
    assert (shared / "job.out").read_text(encoding="utf-8") == "ORCA TERMINATED NORMALLY\n"
    summary = json.loads((shared / "orca_summary.json").read_text(encoding="utf-8"))
    assert summary["normal_termination"] is True
    assert "job.out" in summary["outputs"]
    assert (shared / "job.inp").read_text(encoding="utf-8") == canonical
    assert list(scratch_root.iterdir()) == []


def test_cp2k_boot_maps_summary_paths_back_to_shared_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "cp2k-stage"
    scratch_root = tmp_path / "node-tmp"
    shared.mkdir()
    scratch_root.mkdir()
    (shared / "job.inp").write_text("&GLOBAL\n&END GLOBAL\n", encoding="utf-8")
    monkeypatch.chdir(shared)
    _enable_test_scratch(monkeypatch)
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("CATMASTER_SCRATCH_ROOT", str(scratch_root))
    observed_cwd: list[Path] = []

    def _fake_run(argv, *, stdout, stderr, env, check, **kwargs):
        _ = (stdout, stderr, env, check, kwargs)
        observed_cwd.append(Path.cwd())
        Path("job.out").write_text("PROGRAM ENDED AT\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cp2k_boot.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", [cp2k_boot.__file__, "--cp2k_bin", "cp2k.psmp"])

    assert cp2k_boot.main() == 0
    assert observed_cwd and observed_cwd[0].parent == scratch_root
    summary = json.loads((shared / "cp2k_summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["outputs"]["job.out"] == str((shared / "job.out").resolve())
    assert all(str(scratch_root) not in value for value in summary["outputs"].values())
    assert list(scratch_root.iterdir()) == []


def test_registered_orca_and_cp2k_tasks_stage_the_scratch_helper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tasks = yaml.safe_load((repo_root / "configs" / "dpdispatcher" / "tasks_template.yaml").read_text(encoding="utf-8"))

    for task_name in ("orca_execute", "cp2k_execute"):
        assert "task_script/node_local_scratch.py" in tasks[task_name]["forward_files"]


@pytest.mark.parametrize("boot_name", ["orca_boot.py", "cp2k_boot.py"])
def test_boot_script_imports_scratch_helper_from_staged_task_directory(
    tmp_path: Path,
    boot_name: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "catmaster" / "remote" / "cpu"
    task_script = tmp_path / "task_script"
    task_script.mkdir()
    shutil.copy2(source_dir / boot_name, task_script / boot_name)
    shutil.copy2(source_dir / "node_local_scratch.py", task_script / "node_local_scratch.py")

    completed = subprocess.run(
        [sys.executable, str(task_script / boot_name), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
