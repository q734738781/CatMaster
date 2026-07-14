from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from catmaster.tools.base import workspace_scope

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "tests" / "assets"
RUN_REMOTE = os.environ.get("CATMASTER_RUN_REMOTE_EXECUTION_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RUN_UMA = os.environ.get("CATMASTER_RUN_REMOTE_UMA_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = pytest.mark.skipif(
    not RUN_REMOTE,
    reason=(
        "remote DPDispatcher smoke tests are opt-in; set "
        "CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 to submit real jobs"
    ),
)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _remote_check_interval(default: int) -> int:
    return _int_env("CATMASTER_REMOTE_CHECK_INTERVAL", default)


def _project_space(tmp_path: Path) -> Path:
    root = tmp_path / "catmaster_remote_execution_project"
    (root / "files").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    return root


def _files_root(project_space: Path) -> Path:
    return project_space / "files"


def _files_rel(project_space: Path, rel: str | None) -> Path:
    assert rel, "expected a files-root-relative path"
    path = Path(rel)
    if path.is_absolute():
        return path
    return _files_root(project_space) / path


def _read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing JSON file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_status_success(status_path: Path) -> dict[str, Any]:
    status = _read_json(status_path)
    assert status.get("returncode") == 0, (
        f"remote command failed rc={status.get('returncode')} "
        f"stderr_tail={status.get('stderr_tail')!r} "
        f"stdout_tail={status.get('stdout_tail')!r}"
    )
    return status


def _materialize_tool_artifact(project_space: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    refs = artifact.get("offload_refs") or []
    if not refs:
        return artifact
    assert len(refs) == 1, f"expected one offloaded artifact, got {refs!r}"
    loaded = _read_json(_files_rel(project_space, str(refs[0])))
    assert isinstance(loaded, dict)
    return loaded


def _stage_mace_o2_sp_inputs(project_space: Path) -> Path:
    stage_dir = _files_root(project_space) / "remote_execution" / "mace_o2_sp_stage"
    input_dir = stage_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / "O2_VASP_inputs" / "POSCAR", input_dir / "O2.vasp")
    return stage_dir


def _stage_vasp_o2_structure(project_space: Path) -> Path:
    input_dir = _files_root(project_space) / "remote_execution" / "vasp_o2_prepare_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    structure_path = input_dir / "O2.vasp"
    shutil.copy2(ASSETS / "O2_VASP_inputs" / "POSCAR", structure_path)
    return structure_path


def _write_o2_xyz(project_space: Path, rel_dir: str) -> Path:
    input_dir = _files_root(project_space) / rel_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    structure_path = input_dir / "O2.xyz"
    structure_path.write_text(
        "2\nO2 smoke\nO 0.000000 0.000000 0.000000\nO 0.000000 0.000000 1.210000\n",
        encoding="utf-8",
    )
    return structure_path


def _write_water_xyz(project_space: Path, rel_dir: str) -> Path:
    input_dir = _files_root(project_space) / rel_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    structure_path = input_dir / "H2O.xyz"
    structure_path.write_text(
        "\n".join(
            [
                "3",
                "H2O UMA smoke",
                "O 0.000000 0.000000 0.000000",
                "H 0.758602 0.000000 0.504284",
                "H -0.758602 0.000000 0.504284",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return structure_path


def _uma_check_interval() -> int:
    return _int_env("CATMASTER_REMOTE_UMA_CHECK_INTERVAL", _remote_check_interval(60))


def _uma_model() -> str:
    return os.environ.get("CATMASTER_REMOTE_UMA_MODEL", "uma-s-1p2").strip() or "uma-s-1p2"


def _uma_device() -> str:
    return os.environ.get("CATMASTER_REMOTE_UMA_DEVICE", "auto").strip() or "auto"


def _uma_relax_fmax() -> float:
    return float(os.environ.get("CATMASTER_REMOTE_UMA_RELAX_FMAX", "0.05"))


def _uma_relax_steps() -> int:
    return int(os.environ.get("CATMASTER_REMOTE_UMA_RELAX_STEPS", "5"))


def _invoke_agent_tool(project_space: Path, tool_name: str, payload: dict[str, Any], *, audience: str = "") -> tuple[Any, dict[str, Any]]:
    from catmaster.tools.registry import ToolRegistry

    run_dir = project_space / "metadata" / "runs" / "remote_execution_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry().as_langchain_tools(
        allowlist=[tool_name],
        workspace=str(project_space),
        run_dir=str(run_dir),
        audience=audience,
    )
    assert len(tools) == 1
    content, artifact = tools[0].func(**payload)
    return content, _materialize_tool_artifact(project_space, artifact)


def test_agent_tool_mace_sp_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    stage_dir = _stage_mace_o2_sp_inputs(project_space)

    payload = {
        "work_dir": "remote_execution/mace_o2_sp_stage",
        "task_name": "mace_sp_dir",
        "template_overrides": {
            "model": os.environ.get("CATMASTER_REMOTE_MACE_MODEL", "mh-1").strip() or "mh-1",
            "head": os.environ.get("CATMASTER_REMOTE_MACE_HEAD", "omat_pbe"),
            "default_dtype": os.environ.get("CATMASTER_REMOTE_MACE_DTYPE", "float32").strip() or "float32",
        },
        "submission_config": {"check_interval": _remote_check_interval(30)},
    }

    _content, artifact = _invoke_agent_tool(project_space, "remote_submission", payload, audience="materials_worker")
    data = artifact.get("data") or {}

    assert artifact.get("tool_name") == "remote_submission"
    assert data.get("task_count") == 1
    assert data.get("task_state_counts"), "DPDispatcher returned no task states"

    _assert_status_success(stage_dir / "status.json")

    batch_summary = _read_json(stage_dir / "output" / "batch_summary.json")
    assert batch_summary.get("errors") == []
    results = batch_summary.get("results") or []
    assert len(results) == 1

    energy = results[0].get("summary", {}).get("energy_eV")
    assert isinstance(energy, (int, float)) and math.isfinite(float(energy))

    output_root = stage_dir / "output"
    assert (output_root / "O2" / "summary.json").is_file()
    assert (output_root / "O2" / "sp.vasp").is_file()


@pytest.mark.skipif(
    not RUN_UMA,
    reason="UMA remote smoke tests need gated model access; set CATMASTER_RUN_REMOTE_UMA_TESTS=1",
)
def test_agent_tool_uma_omol_sp_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    stage_dir = _files_root(project_space) / "remote_execution" / "uma_h2o_sp_stage"
    input_dir = stage_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_water_xyz(project_space, "remote_execution/uma_h2o_sp_stage/input")

    payload = {
        "work_dir": "remote_execution/uma_h2o_sp_stage",
        "task_name": "uma_sp_dir",
        "template_overrides": {
            "model": _uma_model(),
            "uma_task": "omol",
            "charge": 0,
            "spin": int(os.environ.get("CATMASTER_REMOTE_UMA_MOL_SPIN", "1")),
            "device": _uma_device(),
        },
        "submission_config": {"check_interval": _uma_check_interval()},
    }

    _content, artifact = _invoke_agent_tool(project_space, "remote_submission", payload, audience="orca_xtb_worker")
    data = artifact.get("data") or {}

    assert artifact.get("tool_name") == "remote_submission"
    assert data.get("resources") == "uma_gpu"
    assert data.get("task_count") == 1
    assert data.get("task_state_counts"), "DPDispatcher returned no task states"

    _assert_status_success(stage_dir / "status.json")

    batch_summary = _read_json(stage_dir / "output" / "batch_summary.json")
    assert batch_summary.get("errors") == []
    results = batch_summary.get("results") or []
    assert len(results) == 1

    energy = results[0].get("summary", {}).get("energy_eV")
    assert isinstance(energy, (int, float)) and math.isfinite(float(energy))

    output_root = stage_dir / "output"
    assert (output_root / "H2O" / "summary.json").is_file()
    assert (output_root / "H2O" / "sp.xyz").is_file()


@pytest.mark.skipif(
    not RUN_UMA,
    reason="UMA remote smoke tests need gated model access; set CATMASTER_RUN_REMOTE_UMA_TESTS=1",
)
def test_agent_tool_uma_periodic_sp_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    stage_dir = _files_root(project_space) / "remote_execution" / "uma_o2_periodic_sp_stage"
    input_dir = stage_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / "O2_VASP_inputs" / "POSCAR", input_dir / "O2.vasp")

    payload = {
        "work_dir": "remote_execution/uma_o2_periodic_sp_stage",
        "task_name": "uma_sp_dir",
        "template_overrides": {
            "model": _uma_model(),
            "uma_task": os.environ.get("CATMASTER_REMOTE_UMA_TASK", "omat").strip() or "omat",
            "charge": 0,
            "spin": 0,
            "device": _uma_device(),
        },
        "submission_config": {"check_interval": _uma_check_interval()},
    }

    _content, artifact = _invoke_agent_tool(project_space, "remote_submission", payload, audience="materials_worker")
    data = artifact.get("data") or {}

    assert artifact.get("tool_name") == "remote_submission"
    assert data.get("resources") == "uma_gpu"
    assert data.get("task_count") == 1
    assert data.get("task_state_counts"), "DPDispatcher returned no task states"

    _assert_status_success(stage_dir / "status.json")

    batch_summary = _read_json(stage_dir / "output" / "batch_summary.json")
    assert batch_summary.get("errors") == []
    results = batch_summary.get("results") or []
    assert len(results) == 1

    energy = results[0].get("summary", {}).get("energy_eV")
    assert isinstance(energy, (int, float)) and math.isfinite(float(energy))

    output_root = stage_dir / "output"
    assert (output_root / "O2" / "summary.json").is_file()
    assert (output_root / "O2" / "sp.vasp").is_file()


@pytest.mark.skipif(
    not RUN_UMA,
    reason="UMA remote smoke tests need gated model access; set CATMASTER_RUN_REMOTE_UMA_TESTS=1",
)
def test_agent_tool_uma_omol_relax_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    stage_dir = _files_root(project_space) / "remote_execution" / "uma_h2o_relax_stage"
    input_dir = stage_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_water_xyz(project_space, "remote_execution/uma_h2o_relax_stage/input")

    payload = {
        "work_dir": "remote_execution/uma_h2o_relax_stage",
        "task_name": "uma_relax_dir",
        "template_overrides": {
            "model": _uma_model(),
            "uma_task": "omol",
            "charge": 0,
            "spin": int(os.environ.get("CATMASTER_REMOTE_UMA_MOL_SPIN", "1")),
            "device": _uma_device(),
            "fmax": _uma_relax_fmax(),
            "steps": _uma_relax_steps(),
            "optimizer": "FIRE",
            "relax_cell": "false",
        },
        "submission_config": {"check_interval": _uma_check_interval()},
    }

    _content, artifact = _invoke_agent_tool(project_space, "remote_submission", payload, audience="orca_xtb_worker")
    data = artifact.get("data") or {}

    assert artifact.get("tool_name") == "remote_submission"
    assert data.get("resources") == "uma_gpu"
    assert data.get("task_count") == 1
    assert data.get("task_state_counts"), "DPDispatcher returned no task states"

    _assert_status_success(stage_dir / "status.json")

    batch_summary = _read_json(stage_dir / "output" / "batch_summary.json")
    assert batch_summary.get("errors") == []
    results = batch_summary.get("results") or []
    assert len(results) == 1

    summary = results[0].get("summary", {})
    final_energy = summary.get("final_energy_eV")
    max_force = summary.get("max_force_abs_eVA")
    assert isinstance(final_energy, (int, float)) and math.isfinite(float(final_energy))
    assert isinstance(max_force, (int, float)) and math.isfinite(float(max_force))
    assert isinstance(summary.get("converged"), bool)

    output_root = stage_dir / "output"
    assert (output_root / "H2O" / "summary.json").is_file()
    assert (output_root / "H2O" / "opt.xyz").is_file()


@pytest.mark.skipif(
    not RUN_UMA,
    reason="UMA remote smoke tests need gated model access; set CATMASTER_RUN_REMOTE_UMA_TESTS=1",
)
def test_agent_tool_uma_periodic_relax_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    stage_dir = _files_root(project_space) / "remote_execution" / "uma_o2_periodic_relax_stage"
    input_dir = stage_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / "O2_VASP_inputs" / "POSCAR", input_dir / "O2.vasp")

    payload = {
        "work_dir": "remote_execution/uma_o2_periodic_relax_stage",
        "task_name": "uma_relax_dir",
        "template_overrides": {
            "model": _uma_model(),
            "uma_task": os.environ.get("CATMASTER_REMOTE_UMA_TASK", "omat").strip() or "omat",
            "charge": 0,
            "spin": 0,
            "device": _uma_device(),
            "fmax": _uma_relax_fmax(),
            "steps": _uma_relax_steps(),
            "optimizer": "FIRE",
            "relax_cell": "false",
        },
        "submission_config": {"check_interval": _uma_check_interval()},
    }

    _content, artifact = _invoke_agent_tool(project_space, "remote_submission", payload, audience="materials_worker")
    data = artifact.get("data") or {}

    assert artifact.get("tool_name") == "remote_submission"
    assert data.get("resources") == "uma_gpu"
    assert data.get("task_count") == 1
    assert data.get("task_state_counts"), "DPDispatcher returned no task states"

    _assert_status_success(stage_dir / "status.json")

    batch_summary = _read_json(stage_dir / "output" / "batch_summary.json")
    assert batch_summary.get("errors") == []
    results = batch_summary.get("results") or []
    assert len(results) == 1

    summary = results[0].get("summary", {})
    final_energy = summary.get("final_energy_eV")
    max_force = summary.get("max_force_abs_eVA")
    assert isinstance(final_energy, (int, float)) and math.isfinite(float(final_energy))
    assert isinstance(max_force, (int, float)) and math.isfinite(float(max_force))
    assert isinstance(summary.get("converged"), bool)

    output_root = stage_dir / "output"
    assert (output_root / "O2" / "summary.json").is_file()
    assert (output_root / "O2" / "opt.vasp").is_file()


def test_vasp_prepare_then_remote_submission_o2_sp_remote(tmp_path: Path) -> None:
    from catmaster.tools.base import workspace_relpath

    project_space = _project_space(tmp_path)
    with workspace_scope(project_space):
        _stage_vasp_o2_structure(project_space)
        _prepare_content, prepare_artifact = _invoke_agent_tool(
            project_space,
            "vasp_prepare",
            {
                "input_path": "remote_execution/vasp_o2_prepare_input/O2.vasp",
                "output_root": "remote_execution/vasp_o2_sp_prepared/O2",
                "preset": "static",
                "regime": "gas",
                "k_product": 1,
                "user_incar_patch": {
                    "NSW": 0,
                    "ISPIN": 2,
                    "MAGMOM": {"O": 1.0},
                },
                "patch_policy": "force",
            },
        )
        prepare_data = prepare_artifact.get("data") or {}
        prepared_rel = str(
            prepare_data.get("prepared_directory_rel")
            or prepare_data.get("output_root_rel")
            or ""
        )
        input_dir = _files_rel(project_space, prepared_rel)
        for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR"):
            assert (input_dir / name).is_file(), f"vasp_prepare did not write {name}"
        assert "TITEL" in (input_dir / "POTCAR").read_text(encoding="utf-8", errors="ignore")

        task_name = os.environ.get("CATMASTER_REMOTE_VASP_TASK", "vasp_execute").strip() or "vasp_execute"
        _submit_content, submit_artifact = _invoke_agent_tool(
            project_space,
            "remote_submission",
            {
                "work_dir": prepared_rel,
                "task_name": task_name,
                "submission_config": {"check_interval": _int_env("CATMASTER_REMOTE_VASP_CHECK_INTERVAL", _remote_check_interval(60))},
            },
            audience="materials_worker",
        )
        submit_data = submit_artifact.get("data") or {}

        assert submit_data.get("task_state_counts"), "DPDispatcher returned no task states"
        status_path = input_dir / "status.json"
        _assert_status_success(status_path)
        assert any((input_dir / name).is_file() for name in ("OUTCAR", "OSZICAR", "vasprun.xml")), (
            "VASP did not return expected output files; "
            f"stage_dir={workspace_relpath(input_dir)}"
        )


def test_cp2k_prepare_then_remote_submission_o2_sp_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    with workspace_scope(project_space):
        _write_o2_xyz(project_space, "remote_execution/cp2k_o2_sp_input")
        _prepare_content, prepare_artifact = _invoke_agent_tool(
            project_space,
            "cp2k_prepare",
            {
                "input_path": "remote_execution/cp2k_o2_sp_input/O2.xyz",
                "output_root": "remote_execution/cp2k_o2_sp_prepared",
                "recipe": "sp",
                "settings": {"periodic": "none", "cell_abc": [12, 12, 12], "xc": "PBE"},
            },
            audience="materials_worker",
        )
        record = (prepare_artifact.get("data") or {})["records"][0]
        stage_rel = str(record["stage_dir_rel"])
        stage_dir = _files_rel(project_space, stage_rel)
        assert (stage_dir / "job.inp").is_file()

        _submit_content, submit_artifact = _invoke_agent_tool(
            project_space,
            "remote_submission",
            {
                "work_dir": stage_rel,
                "task_name": "cp2k_execute",
                "submission_config": {"check_interval": _int_env("CATMASTER_REMOTE_CP2K_CHECK_INTERVAL", _remote_check_interval(60))},
            },
            audience="materials_worker",
        )
        submit_data = submit_artifact.get("data") or {}

    assert submit_data.get("task_state_counts"), "DPDispatcher returned no task states"
    _assert_status_success(stage_dir / "status.json")
    cp2k_summary = _read_json(stage_dir / "cp2k_summary.json")
    assert cp2k_summary.get("completed") is True
    assert (stage_dir / "job.out").is_file()


def test_cp2k_prepare_then_remote_submission_o2_geo_opt_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    with workspace_scope(project_space):
        _write_o2_xyz(project_space, "remote_execution/cp2k_o2_geo_input")
        _prepare_content, prepare_artifact = _invoke_agent_tool(
            project_space,
            "cp2k_prepare",
            {
                "input_path": "remote_execution/cp2k_o2_geo_input/O2.xyz",
                "output_root": "remote_execution/cp2k_o2_geo_prepared",
                "recipe": "geo_opt",
                "settings": {
                    "periodic": "none",
                    "cell_abc": [12, 12, 12],
                    "xc": "PBE",
                    "max_iter": 3,
                },
            },
            audience="materials_worker",
        )
        record = (prepare_artifact.get("data") or {})["records"][0]
        stage_rel = str(record["stage_dir_rel"])
        stage_dir = _files_rel(project_space, stage_rel)
        assert "RUN_TYPE GEO_OPT" in (stage_dir / "job.inp").read_text(encoding="utf-8")

        _submit_content, submit_artifact = _invoke_agent_tool(
            project_space,
            "remote_submission",
            {
                "work_dir": stage_rel,
                "task_name": "cp2k_execute",
                "submission_config": {"check_interval": _int_env("CATMASTER_REMOTE_CP2K_CHECK_INTERVAL", _remote_check_interval(60))},
            },
            audience="materials_worker",
        )
        submit_data = submit_artifact.get("data") or {}

    assert submit_data.get("task_state_counts"), "DPDispatcher returned no task states"
    _assert_status_success(stage_dir / "status.json")
    assert _read_json(stage_dir / "cp2k_summary.json").get("completed") is True


def test_cp2k_aimd_prepare_then_remote_submission_o2_short_nvt_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    with workspace_scope(project_space):
        _write_o2_xyz(project_space, "remote_execution/cp2k_o2_aimd_input")
        _prepare_content, prepare_artifact = _invoke_agent_tool(
            project_space,
            "cp2k_aimd_prepare",
            {
                "input_path": "remote_execution/cp2k_o2_aimd_input/O2.xyz",
                "output_root": "remote_execution/cp2k_o2_aimd_prepared",
                "recipe": "nvt",
                "settings": {
                    "periodic": "none",
                    "cell_abc": [12, 12, 12],
                    "xc": "PBE",
                    "steps": 3,
                    "temperature": 300,
                    "trajectory_stride": 1,
                    "restart_stride": 1,
                },
            },
            audience="dynamics_worker",
        )
        record = (prepare_artifact.get("data") or {})["records"][0]
        stage_rel = str(record["stage_dir_rel"])
        stage_dir = _files_rel(project_space, stage_rel)
        assert "RUN_TYPE MD" in (stage_dir / "job.inp").read_text(encoding="utf-8")

        _submit_content, submit_artifact = _invoke_agent_tool(
            project_space,
            "remote_submission",
            {
                "work_dir": stage_rel,
                "task_name": "cp2k_execute",
                "submission_config": {"check_interval": _int_env("CATMASTER_REMOTE_CP2K_CHECK_INTERVAL", _remote_check_interval(60))},
            },
            audience="dynamics_worker",
        )
        submit_data = submit_artifact.get("data") or {}

    assert submit_data.get("task_state_counts"), "DPDispatcher returned no task states"
    _assert_status_success(stage_dir / "status.json")
    assert _read_json(stage_dir / "cp2k_summary.json").get("completed") is True


def test_lammps_lj_prepare_then_remote_submission_o2_minimize_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    with workspace_scope(project_space):
        _write_o2_xyz(project_space, "remote_execution/lammps_o2_lj_input")
        _ff_content, ff_artifact = _invoke_agent_tool(
            project_space,
            "lammps_forcefield_validate",
            {
                "forcefield_card": {
                    "units": "metal",
                    "atom_style": "atomic",
                    "pair_style": "lj/cut 8.5",
                    "pair_coeff": ["* * 0.0103 3.0"],
                }
            },
            audience="dynamics_worker",
        )
        _prepare_content, prepare_artifact = _invoke_agent_tool(
            project_space,
            "lammps_prepare",
            {
                "input_path": "remote_execution/lammps_o2_lj_input/O2.xyz",
                "output_root": "remote_execution/lammps_o2_lj_min_prepared",
                "recipe": "minimize",
                "forcefield_card_path": (ff_artifact.get("data") or {})["output_path_rel"],
                "settings": {"cell_abc": [20, 20, 20], "thermo": 1},
            },
            audience="dynamics_worker",
        )
        record = (prepare_artifact.get("data") or {})["records"][0]
        stage_rel = str(record["stage_dir_rel"])
        stage_dir = _files_rel(project_space, stage_rel)
        assert (stage_dir / "in.lammps").is_file()

        _submit_content, submit_artifact = _invoke_agent_tool(
            project_space,
            "remote_submission",
            {
                "work_dir": stage_rel,
                "task_name": "lammps_execute",
                "submission_config": {"check_interval": _int_env("CATMASTER_REMOTE_LAMMPS_CHECK_INTERVAL", _remote_check_interval(30))},
            },
            audience="dynamics_worker",
        )
        submit_data = submit_artifact.get("data") or {}

    assert submit_data.get("task_state_counts"), "DPDispatcher returned no task states"
    _assert_status_success(stage_dir / "status.json")
    lammps_summary = _read_json(stage_dir / "lammps_summary.json")
    assert lammps_summary.get("completed") is True
    assert any((stage_dir / name).is_file() for name in ("log.lammps", "lammps_stdout.out"))


def test_lammps_lj_prepare_then_remote_submission_o2_short_nvt_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    with workspace_scope(project_space):
        _write_o2_xyz(project_space, "remote_execution/lammps_o2_lj_nvt_input")
        _ff_content, ff_artifact = _invoke_agent_tool(
            project_space,
            "lammps_forcefield_validate",
            {
                "forcefield_card": {
                    "units": "metal",
                    "atom_style": "atomic",
                    "pair_style": "lj/cut 8.5",
                    "pair_coeff": ["* * 0.0103 3.0"],
                }
            },
            audience="dynamics_worker",
        )
        _prepare_content, prepare_artifact = _invoke_agent_tool(
            project_space,
            "lammps_prepare",
            {
                "input_path": "remote_execution/lammps_o2_lj_nvt_input/O2.xyz",
                "output_root": "remote_execution/lammps_o2_lj_nvt_prepared",
                "recipe": "nvt",
                "forcefield_card_path": (ff_artifact.get("data") or {})["output_path_rel"],
                "settings": {
                    "cell_abc": [20, 20, 20],
                    "steps": 5,
                    "thermo": 1,
                    "dump_stride": 1,
                    "restart_stride": 5,
                },
            },
            audience="dynamics_worker",
        )
        record = (prepare_artifact.get("data") or {})["records"][0]
        stage_rel = str(record["stage_dir_rel"])
        stage_dir = _files_rel(project_space, stage_rel)
        assert "fix int all nvt" in (stage_dir / "in.lammps").read_text(encoding="utf-8")

        _submit_content, submit_artifact = _invoke_agent_tool(
            project_space,
            "remote_submission",
            {
                "work_dir": stage_rel,
                "task_name": "lammps_execute",
                "submission_config": {"check_interval": _int_env("CATMASTER_REMOTE_LAMMPS_CHECK_INTERVAL", _remote_check_interval(30))},
            },
            audience="dynamics_worker",
        )
        submit_data = submit_artifact.get("data") or {}

    assert submit_data.get("task_state_counts"), "DPDispatcher returned no task states"
    _assert_status_success(stage_dir / "status.json")
    assert _read_json(stage_dir / "lammps_summary.json").get("completed") is True
    assert (stage_dir / "trajectory.lammpstrj").is_file()
