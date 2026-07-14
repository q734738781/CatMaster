from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.remote.gpu import mace_relax as remote_mace_relax
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import MaceRelaxBatchInput, mace_relax_batch
from catmaster.tools.execution.task_registry import TaskRegistry


def _install_fake_mace_calculators(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    mace_module = ModuleType("mace")
    calculators_module = ModuleType("mace.calculators")

    def _fake_mace_mp(**kwargs):
        captured.update(kwargs)
        return object()

    calculators_module.mace_mp = _fake_mace_mp  # type: ignore[attr-defined]
    mace_module.calculators = calculators_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", mace_module)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators_module)


def test_mace_relax_calculator_passes_cueq_to_cuda_mace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_mace_calculators(monkeypatch, captured)
    monkeypatch.setattr(remote_mace_relax, "_resolve_device", lambda _device: "cuda")

    _calculator, device = remote_mace_relax._make_calculator(
        model="mh-1",
        head="omat_pbe",
        dispersion=False,
        device="auto",
        default_dtype="float64",
        enable_cueq=True,
    )

    assert device == "cuda"
    assert captured["device"] == "cuda"
    assert captured["default_dtype"] == "float64"
    assert captured["enable_cueq"] is True


def test_mace_relax_calculator_rejects_cueq_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_mace_calculators(monkeypatch, captured)
    monkeypatch.setattr(remote_mace_relax, "_resolve_device", lambda _device: "cpu")

    with pytest.raises(ValueError, match="enable_cueq requires a CUDA device"):
        remote_mace_relax._make_calculator(
            model="mh-1",
            head="omat_pbe",
            dispersion=False,
            device="auto",
            default_dtype="float64",
            enable_cueq=True,
        )

    assert captured == {}


def test_mace_relax_batch_input_relax_lattice_default_and_override() -> None:
    default_params = MaceRelaxBatchInput(input_dir="inputs", output_root="outputs")
    assert default_params.relax_lattice is False
    assert default_params.default_dtype == "float64"
    assert default_params.enable_cueq is False

    override_params = MaceRelaxBatchInput(
        input_dir="inputs",
        output_root="outputs",
        relax_lattice=True,
        default_dtype="float32",
        enable_cueq=True,
    )
    assert override_params.relax_lattice is True
    assert override_params.default_dtype == "float32"
    assert override_params.enable_cueq is True


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
    assert "--enable_cueq {enable_cueq}" in cfg.command
    assert cfg.defaults["enable_cueq"] is False
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
                "enable_cueq": True,
            }
        )

    data = artifact["data"]
    assert captured["staged_model_exists"] is True
    assert "assets" in captured["forward_files"]
    assert "--model 'assets/models/my model.pt'" in str(captured["command"])
    assert "--default_dtype float32" in str(captured["command"])
    assert "--enable_cueq true" in str(captured["command"])
    assert data["model_source_kind"] == "local_file"
    assert data["model_source_rel"] == "models/my model.pt"
    assert data["model_asset_rel"] == "assets/models/my model.pt"
    assert data["default_dtype"] == "float32"
    assert data["enable_cueq"] is True
