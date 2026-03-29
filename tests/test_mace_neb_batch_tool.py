from __future__ import annotations

import json
from pathlib import Path

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.dpdispatcher_runner import DispatchResult
from catmaster.tools.execution.mace_neb import _resolve_local_model, mace_neb_batch
from catmaster.tools.registry import ToolRegistry


def _write_neb_task(task_dir: Path, count: int = 3) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (task_dir / f"{idx:02d}.vasp").write_text("dummy\n", encoding="utf-8")


def test_registry_contains_mace_neb_batch() -> None:
    pytest.importorskip("pymatgen")
    registry = ToolRegistry()
    assert "mace_neb_batch" in registry.list_tools()


def test_resolve_local_model_accepts_file_and_directory(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        model_file = files_root / "models" / "best.model"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text("weights", encoding="utf-8")
        model_dir = files_root / "trained"
        (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints" / "best.model").write_text("weights", encoding="utf-8")

        file_path, file_kind, file_ref = _resolve_local_model("models/best.model")
        dir_path, dir_kind, dir_ref = _resolve_local_model("trained")

    assert file_kind == "local_file"
    assert file_ref == "models/best.model"
    assert file_path.endswith("best.model")
    assert dir_kind == "local_dir"
    assert dir_ref == "trained"
    assert dir_path.endswith("trained/checkpoints/best.model")


def test_mace_neb_batch_dispatches_single_task_via_dpdispatcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_machine(_resources_key: str) -> str:
        return "fake-machine"

    def _fake_dispatch(req) -> DispatchResult:
        captured["machine"] = req.machine
        captured["resources"] = req.resources
        captured["work_base"] = req.work_base
        captured["forward_files"] = list(req.tasks[0].forward_files)
        captured["command"] = req.tasks[0].command

        stage_root = Path(req.local_root) / req.work_base
        stage_input = stage_root / "input"
        stage_output = stage_root / "output"
        task_dirs = sorted(path for path in stage_input.iterdir() if path.is_dir())
        assert [path.name for path in task_dirs] == ["neb_case"]
        assert (task_dirs[0] / "00.vasp").is_file()
        assert (stage_root / "task_script" / "mace_neb.py").is_file()

        task_output = stage_output / "neb_case"
        task_output.mkdir(parents=True, exist_ok=True)
        for idx in range(3):
            (task_output / f"{idx:02d}.vasp").write_text("final\n", encoding="utf-8")
        (task_output / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "results": {"barrier_eV": 0.12, "converged": True},
                    "artifacts": {"final_image_files": [f"{idx:02d}.vasp" for idx in range(3)]},
                }
            ),
            encoding="utf-8",
        )
        (stage_output / "batch_summary.json").write_text(
            json.dumps({"task_count": 1, "tasks": [{"task_rel": "neb_case", "status": "completed"}]}),
            encoding="utf-8",
        )
        for name in ("status.json", "stdout.log", "stderr.log"):
            (stage_root / name).write_text("ok\n", encoding="utf-8")
        return DispatchResult(
            work_base=req.work_base,
            local_root=req.local_root,
            output_dir=str(stage_root / "output"),
            task_states=["finished"],
            submission_dir="/remote/fake",
            duration_s=1.23,
        )

    monkeypatch.setattr("catmaster.tools.execution.mace_neb._resolve_machine_for_resources", _fake_resolve_machine)
    monkeypatch.setattr("catmaster.tools.execution.mace_neb.dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        task_dir = files_root / "neb_case"
        _write_neb_task(task_dir)

        _content, artifact = mace_neb_batch(
            {
                "input_root": "neb_case",
                "output_root": "outputs",
                "model": "mh-1",
            }
        )

        collected = files_root / "outputs" / "neb_case" / "summary.json"
        batch_summary = files_root / "outputs" / "batch_summary.json"

    data = artifact["data"]
    assert captured["machine"] == "fake-machine"
    assert captured["resources"] == "mace_gpu"
    assert captured["forward_files"] == ["input", "task_script/mace_neb.py"]
    assert "task_script/mace_neb.py" in str(captured["command"])
    assert "--climb false" in str(captured["command"])
    assert "--model mh-1" in str(captured["command"])
    assert data["single_task_mode"] is True
    assert data["task_count"] == 1
    assert data["model_source_kind"] == "pretrained"
    assert data["batch_summary_rel"] == "outputs/batch_summary.json"
    assert collected.is_file()
    assert batch_summary.is_file()


def test_mace_neb_batch_stages_local_model_asset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_machine(_resources_key: str) -> str:
        return "fake-machine"

    def _fake_dispatch(req) -> DispatchResult:
        stage_root = Path(req.local_root) / req.work_base
        captured["forward_files"] = list(req.tasks[0].forward_files)
        captured["command"] = req.tasks[0].command
        assert (stage_root / "assets" / "models" / "best.model").is_file()
        (stage_root / "output").mkdir(parents=True, exist_ok=True)
        (stage_root / "output" / "batch_summary.json").write_text(json.dumps({"task_count": 0}), encoding="utf-8")
        for name in ("status.json", "stdout.log", "stderr.log"):
            (stage_root / name).write_text("ok\n", encoding="utf-8")
        return DispatchResult(
            work_base=req.work_base,
            local_root=req.local_root,
            output_dir=str(stage_root / "output"),
            task_states=["finished"],
            submission_dir="/remote/fake",
            duration_s=0.5,
        )

    monkeypatch.setattr("catmaster.tools.execution.mace_neb._resolve_machine_for_resources", _fake_resolve_machine)
    monkeypatch.setattr("catmaster.tools.execution.mace_neb.dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        _write_neb_task(files_root / "neb_case")
        model_file = files_root / "models" / "best.model"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text("weights", encoding="utf-8")

        _content, artifact = mace_neb_batch(
            {
                "input_root": "neb_case",
                "output_root": "outputs",
                "model": "models/best.model",
            }
        )

    data = artifact["data"]
    assert "assets" in captured["forward_files"]
    assert "assets/models/best.model" in str(captured["command"])
    assert data["model_source_kind"] == "local_file"
    assert data["model_source_rel"] == "models/best.model"
    assert data["model_asset_rel"] == "assets/models/best.model"


def test_mace_neb_batch_can_enable_ci_neb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_machine(_resources_key: str) -> str:
        return "fake-machine"

    def _fake_dispatch(req) -> DispatchResult:
        captured["command"] = req.tasks[0].command
        stage_root = Path(req.local_root) / req.work_base
        (stage_root / "output").mkdir(parents=True, exist_ok=True)
        (stage_root / "output" / "batch_summary.json").write_text(json.dumps({"task_count": 0}), encoding="utf-8")
        for name in ("status.json", "stdout.log", "stderr.log"):
            (stage_root / name).write_text("ok\n", encoding="utf-8")
        return DispatchResult(
            work_base=req.work_base,
            local_root=req.local_root,
            output_dir=str(stage_root / "output"),
            task_states=["finished"],
            submission_dir="/remote/fake",
            duration_s=0.5,
        )

    monkeypatch.setattr("catmaster.tools.execution.mace_neb._resolve_machine_for_resources", _fake_resolve_machine)
    monkeypatch.setattr("catmaster.tools.execution.mace_neb.dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        _write_neb_task(files_root / "neb_case")

        mace_neb_batch(
            {
                "input_root": "neb_case",
                "output_root": "outputs",
                "climb": True,
            }
        )

    assert "--climb true" in str(captured["command"])


def test_mace_neb_batch_rejects_nested_task_content(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        task_dir = files_root / "neb_case"
        _write_neb_task(task_dir)
        (task_dir / "nested").mkdir(parents=True, exist_ok=True)

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_neb_batch(
                {
                    "input_root": "neb_case",
                    "output_root": "outputs",
                }
            )

    assert "Invalid MACE NEB input layout" in str(excinfo.value)


def test_mace_neb_batch_rejects_non_task_children_in_batch_root(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        batch_root = files_root / "neb_batch"
        _write_neb_task(batch_root / "task0")
        (batch_root / "notes.txt").write_text("oops\n", encoding="utf-8")

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_neb_batch(
                {
                    "input_root": "neb_batch",
                    "output_root": "outputs",
                }
            )

    assert "Batch MACE NEB input_root must contain only task directories" in str(excinfo.value)
