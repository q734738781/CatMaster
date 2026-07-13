from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import read
from ase.io.trajectory import Trajectory

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.remote.gpu.mace_md import (
    _collect_structure_files,
    _config_from_compact_payload,
    _prepare_initial_velocities,
    _run_md_single,
    _timing_statistics,
    _validate_config,
)
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


def test_mace_md_defaults_to_preserving_input_velocities() -> None:
    config = _validate_config(_config_from_compact_payload({}))
    assert config["dynamics"]["reinitialize_velocities"] is False
    assert config["dynamics"]["seed"] == 2026

    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.5, 0, 0]], cell=[10, 10, 10], pbc=True)
    momenta = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    atoms.set_momenta(momenta)

    source = _prepare_initial_velocities(
        atoms,
        dyn_cfg=config["dynamics"],
        rng=np.random.default_rng(7),
    )

    assert source == "input_last_frame"
    np.testing.assert_allclose(atoms.get_momenta(), momenta)


def test_mace_md_seed_controls_generated_velocities() -> None:
    config = _validate_config(
        _config_from_compact_payload({"md_config": {"dynamics": {"seed": 17}}})
    )
    assert config["dynamics"]["seed"] == 17

    momenta = []
    for seed in (17, 17, 18):
        atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.5, 0, 0]], cell=[10, 10, 10], pbc=True)
        _prepare_initial_velocities(
            atoms,
            dyn_cfg=config["dynamics"],
            rng=np.random.default_rng(seed),
        )
        momenta.append(atoms.get_momenta().copy())

    np.testing.assert_allclose(momenta[0], momenta[1])
    assert not np.allclose(momenta[0], momenta[2])


def test_mace_md_generates_velocities_only_when_missing_or_explicit() -> None:
    config = _validate_config(_config_from_compact_payload({}))
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.5, 0, 0]], cell=[10, 10, 10], pbc=True)

    source = _prepare_initial_velocities(
        atoms,
        dyn_cfg=config["dynamics"],
        rng=np.random.default_rng(7),
    )
    assert source == "generated_missing_input_velocities"
    assert atoms.has("momenta")

    original = atoms.get_momenta().copy()
    config["dynamics"]["reinitialize_velocities"] = True
    source = _prepare_initial_velocities(
        atoms,
        dyn_cfg=config["dynamics"],
        rng=np.random.default_rng(8),
    )
    assert source == "generated_explicit_reinitialization"
    assert not np.allclose(atoms.get_momenta(), original)


def test_mace_md_accepts_trajectory_and_starts_from_last_frame(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_path = input_dir / "restart.traj"
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.5, 0, 0]], cell=[10, 10, 10], pbc=True)
    with Trajectory(source_path, "w") as trajectory:
        atoms.set_momenta([[1.0, 0, 0], [-1.0, 0, 0]])
        trajectory.write(atoms)
        atoms.positions[1, 0] = 3.6
        atoms.set_momenta([[2.0, 0, 0], [-2.0, 0, 0]])
        trajectory.write(atoms)

    assert _collect_structure_files(input_dir) == [source_path]
    config = _validate_config(
        _config_from_compact_payload(
            {
                "md_config": {
                    "dynamics": {"ensemble": "nve", "steps": 1},
                    "output": {"traj_interval": 1, "log_interval": 1},
                }
            }
        )
    )
    summary = _run_md_single(
        structure_path=source_path,
        output_dir=tmp_path / "output",
        calc=LennardJones(),
        config=config,
        device="cpu",
        seed=2026,
    )
    staged = read(tmp_path / "output" / "start.traj", index=-1)

    assert summary["velocity_source"] == "input_last_frame"
    assert summary["rng_seed"] == 2026
    assert summary["rng_source"] == "configured_seed"
    assert summary["restart_trajectory"] == "restart.traj"
    assert summary["input_frame"] == -1
    assert staged.positions[1, 0] == pytest.approx(3.6)
    np.testing.assert_allclose(staged.get_momenta(), [[2.0, 0, 0], [-2.0, 0, 0]])


@pytest.mark.parametrize(
    ("thermostat", "integrator_state_source"),
    [("bussi", "restored"), ("langevin", "not_required")],
)
def test_mace_md_stochastic_restart_matches_uninterrupted_run(
    tmp_path: Path,
    thermostat: str,
    integrator_state_source: str,
) -> None:
    source = tmp_path / "source.traj"
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.5, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.write(source)

    def config(steps: int) -> dict:
        return _validate_config(
            _config_from_compact_payload(
                {
                    "md_config": {
                        "dynamics": {"ensemble": "nvt", "steps": steps, "seed": 17},
                        "thermostat": {"type": thermostat},
                        "output": {"traj_interval": 5, "log_interval": 20},
                    }
                }
            )
        )

    uninterrupted = _run_md_single(
        structure_path=source,
        output_dir=tmp_path / "uninterrupted",
        calc=LennardJones(),
        config=config(10),
        device="cpu",
        seed=17,
    )
    first = _run_md_single(
        structure_path=source,
        output_dir=tmp_path / "first",
        calc=LennardJones(),
        config=config(5),
        device="cpu",
        seed=17,
    )
    second = _run_md_single(
        structure_path=tmp_path / "first" / "restart.traj",
        output_dir=tmp_path / "second",
        calc=LennardJones(),
        config=config(5),
        device="cpu",
        seed=17,
    )
    second_from_md = _run_md_single(
        structure_path=tmp_path / "first" / "md.traj",
        output_dir=tmp_path / "second_from_md",
        calc=LennardJones(),
        config=config(5),
        device="cpu",
        seed=999,
    )

    uninterrupted_atoms = read(tmp_path / "uninterrupted" / "restart.traj", index=-1)
    restarted_atoms = read(tmp_path / "second" / "restart.traj", index=-1)
    restarted_from_md_atoms = read(tmp_path / "second_from_md" / "restart.traj", index=-1)
    np.testing.assert_allclose(restarted_atoms.positions, uninterrupted_atoms.positions, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        restarted_atoms.get_momenta(),
        uninterrupted_atoms.get_momenta(),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        restarted_from_md_atoms.positions,
        uninterrupted_atoms.positions,
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        restarted_from_md_atoms.get_momenta(),
        uninterrupted_atoms.get_momenta(),
        rtol=0,
        atol=1e-14,
    )
    assert uninterrupted["rng_source"] == "configured_seed"
    assert first["rng_source"] == "configured_seed"
    assert second["rng_source"] == "restart_checkpoint"
    assert second["integrator_state_source"] == integrator_state_source
    assert second_from_md["rng_source"] == "restart_checkpoint"
    assert second_from_md["integrator_state_source"] == integrator_state_source


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


def test_mace_md_rejects_unknown_nested_config_key() -> None:
    with pytest.raises(ValueError, match="Unknown md_config.dynamics key.*temperature"):
        _config_from_compact_payload(
            {"md_config": {"dynamics": {"temperature": 900}}}
        )


def test_mace_md_dir_task_command_has_no_historical_gpu_or_scale_options() -> None:
    cfg = TaskRegistry().get("mace_md_dir")
    assert cfg.audiences == ["materials_worker", "dynamics_worker"]
    assert "mace_md.py" in cfg.command
    assert "--gpu_ids" not in cfg.command
    assert "--scales" not in cfg.command
    assert "--params {params}" in cfg.command
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
