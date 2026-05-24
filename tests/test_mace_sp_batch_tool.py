from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import MaceSPBatchInput, mace_sp_batch
from catmaster.tools.registry import ToolRegistry


def test_registry_replaces_mace_sp_batch_with_generic_remote_submission() -> None:
    pytest.importorskip("pymatgen")
    registry = ToolRegistry()
    assert "mace_sp_batch" not in registry.list_tools()
    assert "remote_submission" in registry.list_tools()
    assert "remote_submission_batch" in registry.list_tools()


def test_mace_sp_batch_input_default_dtype_default_and_override() -> None:
    default_params = MaceSPBatchInput(input_dir="inputs", output_root="outputs")
    assert default_params.default_dtype == "float64"

    override_params = MaceSPBatchInput(input_dir="inputs", output_root="outputs", default_dtype="float32")
    assert override_params.default_dtype == "float32"


def test_mace_sp_batch_rejects_output_inside_input(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "POSCAR").write_text("dummy", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_sp_batch(
                {
                    "input_dir": "inputs",
                    "output_root": "inputs/outputs",
                }
            )
    assert "must not be inside input_dir" in str(excinfo.value)


def test_mace_sp_batch_requires_structure_files(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "empty_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_sp_batch(
                {
                    "input_dir": "empty_inputs",
                    "output_root": "outputs",
                }
            )
    assert "No structure files found" in str(excinfo.value)


def test_mace_sp_batch_stages_local_model_directory_and_resolves_best_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req):
        stage_root = Path(req.local_root) / req.work_base
        staged_model = stage_root / "assets" / "models" / "trained-run" / "checkpoints" / "best.model"
        staged_aux = stage_root / "assets" / "models" / "trained-run" / "config.json"
        captured["command"] = req.tasks[0].command
        captured["forward_files"] = list(req.tasks[0].forward_files)
        captured["staged_model_exists"] = staged_model.is_file()
        captured["staged_aux_exists"] = staged_aux.is_file()
        return SimpleNamespace(
            task_states=["5"],
            submission_dir=str((Path(req.local_root) / "_fake_submission").resolve()),
            work_base=req.work_base,
            duration_s=0.01,
        )

    monkeypatch.setattr("catmaster.tools.execution.mace_dispatch._resolve_machine_for_resources", lambda _: "dummy")
    monkeypatch.setattr("catmaster.tools.execution.mace_dispatch.dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "POSCAR").write_text("dummy", encoding="utf-8")
        model_dir = files_root / "trained-run"
        (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints" / "best.model").write_text("weights", encoding="utf-8")
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        _content, artifact = mace_sp_batch(
            {
                "input_dir": "inputs",
                "output_root": "outputs",
                "model": "trained-run",
                "default_dtype": "float32",
            }
        )

    data = artifact["data"]
    assert captured["staged_model_exists"] is True
    assert captured["staged_aux_exists"] is True
    assert "assets" in captured["forward_files"]
    assert "--model assets/models/trained-run/checkpoints/best.model" in str(captured["command"])
    assert "--default_dtype float32" in str(captured["command"])
    assert data["model_source_kind"] == "local_dir"
    assert data["model_source_rel"] == "trained-run"
    assert data["model_asset_rel"] == "assets/models/trained-run/checkpoints/best.model"
    assert data["default_dtype"] == "float32"
