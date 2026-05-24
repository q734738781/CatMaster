from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.dpdispatcher_runner import DispatchResult
from catmaster.tools.execution.mace_neb import _resolve_local_model, mace_neb_batch
from catmaster.remote.gpu.mace_neb import _discover_task_dirs as remote_discover_task_dirs
from catmaster.remote.gpu.mace_neb import _write_profile_outputs
from catmaster.tools.registry import ToolRegistry


def _write_neb_task(task_dir: Path, count: int = 3) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (task_dir / f"{idx:02d}.vasp").write_text("dummy\n", encoding="utf-8")


def test_registry_replaces_mace_neb_batch_with_generic_remote_submission() -> None:
    pytest.importorskip("pymatgen")
    registry = ToolRegistry()
    assert "mace_neb_batch" not in registry.list_tools()
    assert "remote_submission" in registry.list_tools()
    assert "remote_submission_batch" in registry.list_tools()


def test_mace_neb_batch_input_default_dtype_default_and_override() -> None:
    from catmaster.tools.execution.mace_neb import MaceNebBatchInput

    default_params = MaceNebBatchInput(input_root="inputs", output_root="outputs")
    assert default_params.default_dtype == "float64"

    override_params = MaceNebBatchInput(input_root="inputs", output_root="outputs", default_dtype="float32")
    assert override_params.default_dtype == "float32"


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
                "default_dtype": "float32",
            }
        )

        collected = files_root / "outputs" / "neb_case" / "summary.json"
        batch_summary = files_root / "outputs" / "batch_summary.json"

    data = artifact["data"]
    assert captured["machine"] == "fake-machine"
    assert captured["resources"] == "mace_gpu"
    assert captured["forward_files"] == ["input", "task_script/mace_neb.py"]
    assert "task_script/mace_neb.py" in str(captured["command"])
    assert "--mode plain" in str(captured["command"])
    assert "--autoneb_target_images 0" in str(captured["command"])
    assert "--autoneb_n_simul 0" in str(captured["command"])
    assert "--climb false" in str(captured["command"])
    assert "--model mh-1" in str(captured["command"])
    assert "--default_dtype float32" in str(captured["command"])
    assert data["single_task_mode"] is True
    assert data["task_count"] == 1
    assert data["model_source_kind"] == "pretrained"
    assert data["default_dtype"] == "float32"
    assert data["mode"] == "plain"
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


def test_mace_neb_batch_can_request_autoneb_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

        _content, artifact = mace_neb_batch(
            {
                "input_root": "neb_case",
                "output_root": "outputs",
                "mode": "autoneb",
                "autoneb": {
                    "target_images": 9,
                    "n_simul": 2,
                    "space_energy_ratio": 0.3,
                    "interpolate_method": "linear",
                },
                "climb": True,
            }
        )

    data = artifact["data"]
    assert "--mode autoneb" in str(captured["command"])
    assert "--autoneb_target_images 9" in str(captured["command"])
    assert "--autoneb_n_simul 2" in str(captured["command"])
    assert "--autoneb_space_energy_ratio 0.3" in str(captured["command"])
    assert "--autoneb_interpolate_method linear" in str(captured["command"])
    assert "--climb true" in str(captured["command"])
    assert data["mode"] == "autoneb"
    assert data["autoneb"] == {
        "target_images": 9,
        "n_simul": 2,
        "space_energy_ratio": 0.3,
        "interpolate_method": "linear",
    }


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


def test_mace_neb_batch_ignores_root_level_batch_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_machine(_resources_key: str) -> str:
        return "fake-machine"

    def _fake_dispatch(req) -> DispatchResult:
        captured["command"] = req.tasks[0].command
        stage_root = Path(req.local_root) / req.work_base
        stage_input = stage_root / "input"
        assert (stage_input / "batch_summary.json").is_file()
        assert (stage_input / "notes.txt").is_file()
        task_dirs = sorted(path.name for path in stage_input.iterdir() if path.is_dir())
        assert task_dirs == ["task0"]
        (stage_root / "output").mkdir(parents=True, exist_ok=True)
        (stage_root / "output" / "batch_summary.json").write_text(json.dumps({"task_count": 1}), encoding="utf-8")
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
        batch_root = files_root / "neb_batch"
        _write_neb_task(batch_root / "task0")
        (batch_root / "batch_summary.json").write_text("{}", encoding="utf-8")
        (batch_root / "notes.txt").write_text("ok\n", encoding="utf-8")

        _content, artifact = mace_neb_batch(
            {
                "input_root": "neb_batch",
                "output_root": "outputs",
            }
        )

    assert "task_script/mace_neb.py" in str(captured["command"])
    assert artifact["data"]["task_count"] == 1


def test_remote_mace_neb_discovery_ignores_root_level_files(tmp_path: Path) -> None:
    batch_root = tmp_path / "neb_batch"
    _write_neb_task(batch_root / "task0")
    (batch_root / "batch_summary.json").write_text("{}", encoding="utf-8")
    (batch_root / "notes.txt").write_text("ok\n", encoding="utf-8")

    task_dirs = remote_discover_task_dirs(batch_root)

    assert [path.name for path in task_dirs] == ["task0"]


def test_mace_neb_batch_rejects_mixed_root_images_and_task_dirs(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        batch_root = files_root / "neb_batch"
        _write_neb_task(batch_root / "task0")
        for idx in range(3):
            (batch_root / f"{idx:02d}.vasp").write_text("dummy\n", encoding="utf-8")

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_neb_batch(
                {
                    "input_root": "neb_batch",
                    "output_root": "outputs",
                }
            )

    assert "mixes batch task subdirectories with root-level numbered image files" in str(excinfo.value)


def test_write_profile_outputs_does_not_call_forcefit_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeAxes:
        def plot(self, x, y, *args, **kwargs):
            calls["plot"] = (list(x), list(y), args, kwargs)

        def set_xlabel(self, value):
            calls["xlabel"] = value

        def set_ylabel(self, value):
            calls["ylabel"] = value

        def grid(self, *args, **kwargs):
            calls["grid"] = (args, kwargs)

    class _FakeFigure:
        def tight_layout(self):
            calls["tight_layout"] = True

        def savefig(self, path):
            calls["savefig"] = str(path)

    fake_pyplot = types.SimpleNamespace(
        subplots=lambda **kwargs: (_FakeFigure(), _FakeAxes()),
        close=lambda fig: calls.setdefault("closed", True),
    )
    fake_matplotlib = types.SimpleNamespace(pyplot=fake_pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    class _ForceFit:
        def plot(self, ax=None):
            raise AssertionError("forcefit.plot should not be called")

    rows = [
        {"image_index": 0, "path_A": 0.0, "energy_eV": -1.0, "relative_energy_eV": 0.0, "max_force_eV_per_A": 0.0, "rms_force_eV_per_A": 0.0},
        {"image_index": 1, "path_A": 0.8, "energy_eV": -0.8, "relative_energy_eV": 0.2, "max_force_eV_per_A": 0.1, "rms_force_eV_per_A": 0.1},
        {"image_index": 2, "path_A": 1.6, "energy_eV": -0.9, "relative_energy_eV": 0.1, "max_force_eV_per_A": 0.1, "rms_force_eV_per_A": 0.1},
    ]
    energies_csv = tmp_path / "image_energies.csv"
    profile_png = tmp_path / "profile.png"

    _write_profile_outputs(rows=rows, forcefit=_ForceFit(), energies_csv=energies_csv, profile_png=profile_png)

    assert energies_csv.is_file()
    assert calls["plot"][0] == [0.0, 0.8, 1.6]
    assert calls["plot"][1] == [0.0, 0.2, 0.1]
    assert calls["xlabel"] == "Reaction coordinate (A)"
    assert calls["ylabel"] == "Relative energy (eV)"
