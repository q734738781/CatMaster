from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import MaceRelaxBatchInput, mace_relax_batch
from catmaster.tools.execution.task_registry import TaskRegistry


def test_mace_relax_batch_input_relax_lattice_default_and_override() -> None:
    default_params = MaceRelaxBatchInput(input_dir="inputs", output_root="outputs")
    assert default_params.relax_lattice is False
    assert default_params.default_dtype == "float64"

    override_params = MaceRelaxBatchInput(
        input_dir="inputs",
        output_root="outputs",
        relax_lattice=True,
        default_dtype="float32",
    )
    assert override_params.relax_lattice is True
    assert override_params.default_dtype == "float32"


def test_mace_relax_batch_accepts_relax_lattice_field(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "POSCAR").write_text("dummy", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_relax_batch(
                {
                    "input_dir": "inputs",
                    "output_root": "inputs/outputs",
                    "relax_lattice": True,
                }
            )
    assert "must not be inside input_dir" in str(excinfo.value)


def test_mace_relax_dir_task_command_has_relax_lattice_placeholder() -> None:
    cfg = TaskRegistry().get("mace_relax_dir")
    assert "--relax_lattice {relax_lattice}" in cfg.command
    assert "--default_dtype {default_dtype}" in cfg.command
    assert cfg.defaults["device"] == "auto"


def test_mace_relax_batch_stages_local_model_file_for_dpdispatcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req):
        stage_root = Path(req.local_root) / req.work_base
        staged_model = stage_root / "assets" / "models" / "my model.pt"
        captured["command"] = req.tasks[0].command
        captured["forward_files"] = list(req.tasks[0].forward_files)
        captured["staged_model_exists"] = staged_model.is_file()
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
        model_path = files_root / "models" / "my model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("weights", encoding="utf-8")

        _content, artifact = mace_relax_batch(
            {
                "input_dir": "inputs",
                "output_root": "outputs",
                "model": "models/my model.pt",
                "default_dtype": "float32",
            }
        )

    data = artifact["data"]
    assert captured["staged_model_exists"] is True
    assert "assets" in captured["forward_files"]
    assert "--model 'assets/models/my model.pt'" in str(captured["command"])
    assert "--default_dtype float32" in str(captured["command"])
    assert data["model_source_kind"] == "local_file"
    assert data["model_source_rel"] == "models/my model.pt"
    assert data["model_asset_rel"] == "assets/models/my model.pt"
    assert data["default_dtype"] == "float32"
