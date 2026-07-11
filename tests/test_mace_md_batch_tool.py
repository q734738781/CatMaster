from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.remote.gpu.mace_md import _config_from_compact_payload, _timing_statistics, _validate_config
from catmaster.specialists.runtime import _DYNAMICS_WORKER_TOOL_ALLOWLIST, _MATERIALS_WORKER_TOOL_ALLOWLIST
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import MaceMDBatchInput, mace_md_batch
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.registry import ToolRegistry


def test_registry_replaces_mace_md_batch_with_generic_remote_submission() -> None:
    pytest.importorskip("pymatgen")
    registry = ToolRegistry()
    assert "mace_md_batch" not in registry.list_tools()
    assert "mace_md_batch" not in _MATERIALS_WORKER_TOOL_ALLOWLIST
    assert "mace_md_batch" not in _DYNAMICS_WORKER_TOOL_ALLOWLIST
    assert "remote_submission" in registry.list_tools()
    assert "remote_submission" in _MATERIALS_WORKER_TOOL_ALLOWLIST
    assert "remote_submission" in _DYNAMICS_WORKER_TOOL_ALLOWLIST


def test_mace_md_batch_input_defaults_are_generic_md() -> None:
    params = MaceMDBatchInput(input_dir="inputs", output_root="outputs")
    assert params.md_config == {}
    assert params.default_dtype == "float32"
    assert "gpu_ids" not in MaceMDBatchInput.model_fields
    assert "scales" not in MaceMDBatchInput.model_fields


def test_mace_md_batch_schema_keeps_md_controls_free_form() -> None:
    schema = MaceMDBatchInput.model_json_schema()
    assert list(schema["properties"]) == [
        "input_dir",
        "output_root",
        "model",
        "head",
        "dispersion",
        "default_dtype",
        "md_config",
        "check_interval",
    ]
    assert "$defs" not in schema
    assert "friction_per_fs" not in json.dumps(schema)
    assert "tchain" not in json.dumps(schema)


def test_mace_md_batch_supports_npt_with_grouped_barostat() -> None:
    params = MaceMDBatchInput(
        input_dir="inputs",
        output_root="outputs",
        md_config={"dynamics": {"ensemble": "npt"}},
    )
    assert params.md_config["dynamics"]["ensemble"] == "npt"


def test_mace_md_acceleration_config_is_validated() -> None:
    config = _validate_config(
        _config_from_compact_payload(
            {
                "md_config": {
                    "calculator": {
                        "enable_cueq": True,
                        "compile_mode": "reduce-overhead",
                    }
                }
            }
        )
    )
    assert config["calculator"]["enable_cueq"] is True
    assert config["calculator"]["compile_mode"] == "reduce-overhead"

    with pytest.raises(ValueError, match="compile_mode"):
        _validate_config(
            _config_from_compact_payload(
                {"md_config": {"calculator": {"compile_mode": "fastest"}}}
            )
        )


def test_mace_md_step_timing_statistics_separate_warmup() -> None:
    stats = _timing_statistics([5.0, 3.0, 1.0, 1.0, 1.0], warmup_steps=2)
    assert stats["all_steps"]["count"] == 5
    assert stats["first_step_s"] == 5.0
    assert stats["warmup_steps_excluded"] == 2
    assert stats["steady_state"]["count"] == 3
    assert stats["steady_state"]["median"] == 1.0


def test_mace_md_batch_requires_berendsen_npt_compressibility() -> None:
    with pytest.raises(ValueError, match="compressibility"):
        _validate_config(
            _config_from_compact_payload(
                {
                    "md_config": {
                        "dynamics": {"ensemble": "npt"},
                        "barostat": {"type": "berendsen"},
                    }
                }
            )
        )


def test_mace_md_dir_task_command_has_no_historical_gpu_or_scale_options() -> None:
    cfg = TaskRegistry().get("mace_md_dir")
    assert cfg.audiences == ["materials_worker", "dynamics_worker"]
    assert "mace_md.py" in cfg.command
    assert "--gpu_ids" not in cfg.command
    assert "--scales" not in cfg.command
    assert "--params {params_path}" in cfg.command
    assert "--device {device}" in cfg.command
    assert "--ensemble" not in cfg.command
    assert "--temperature_K" not in cfg.command
    assert "--steps" not in cfg.command


def test_mace_md_batch_rejects_output_inside_input(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "POSCAR").write_text("dummy", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_md_batch(
                {
                    "input_dir": "inputs",
                    "output_root": "inputs/outputs",
                }
            )
    assert "must not be inside input_dir" in str(excinfo.value)


def test_mace_md_batch_dispatches_generic_md_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req):
        stage_root = Path(req.local_root) / req.work_base
        captured["command"] = req.tasks[0].command
        captured["forward_files"] = list(req.tasks[0].forward_files)
        captured["script_exists"] = (stage_root / "task_script" / "mace_md.py").is_file()
        captured["params_payload"] = json.loads((stage_root / "params" / "md_params.json").read_text(encoding="utf-8"))
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

        _content, artifact = mace_md_batch(
            {
                "input_dir": "inputs",
                "output_root": "outputs",
                "md_config": {"dynamics": {"temperature_K": 500, "steps": 25}},
                "default_dtype": "float32",
            }
        )

    command = str(captured["command"])
    data = artifact["data"]
    params_payload = captured["params_payload"]
    assert captured["script_exists"] is True
    assert "task_script/mace_md.py" in captured["forward_files"]
    assert "params" in captured["forward_files"]
    assert "--params params/md_params.json" in command
    assert "--temperature_K" not in command
    assert "--steps" not in command
    assert "--default_dtype" not in command
    assert "--gpu_ids" not in command
    assert "--scales" not in command
    assert params_payload["schema_version"] == 2
    assert params_payload["md_config"]["dynamics"]["temperature_K"] == 500
    assert params_payload["md_config"]["dynamics"]["steps"] == 25
    assert params_payload["default_dtype"] == "float32"
    assert "dynamics" not in data
    assert data["md_config"]["dynamics"]["temperature_K"] == 500
    assert data["default_dtype"] == "float32"
