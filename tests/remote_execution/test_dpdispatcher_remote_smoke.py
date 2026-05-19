from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "tests" / "assets"
RUN_REMOTE = os.environ.get("CATMASTER_RUN_REMOTE_EXECUTION_TESTS", "").strip().lower() in {
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
    input_dir = _files_root(project_space) / "remote_execution" / "mace_o2_sp_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / "O2_VASP_inputs" / "POSCAR", input_dir / "O2.vasp")
    return input_dir


def _stage_vasp_o2_structure(project_space: Path) -> Path:
    input_dir = _files_root(project_space) / "remote_execution" / "vasp_o2_prepare_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    structure_path = input_dir / "O2.vasp"
    shutil.copy2(ASSETS / "O2_VASP_inputs" / "POSCAR", structure_path)
    return structure_path


def _invoke_agent_tool(project_space: Path, tool_name: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from catmaster.tools.registry import ToolRegistry

    run_dir = project_space / "metadata" / "runs" / "remote_execution_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry().as_langchain_tools(
        allowlist=[tool_name],
        workspace=str(project_space),
        run_dir=str(run_dir),
    )
    assert len(tools) == 1
    content, artifact = tools[0].func(**payload)
    return content, _materialize_tool_artifact(project_space, artifact)


def test_agent_tool_mace_sp_remote(tmp_path: Path) -> None:
    project_space = _project_space(tmp_path)
    _stage_mace_o2_sp_inputs(project_space)

    payload = {
        "input_dir": "remote_execution/mace_o2_sp_inputs",
        "output_root": "remote_execution/mace_o2_sp_outputs",
        "model": os.environ.get("CATMASTER_REMOTE_MACE_MODEL", "mh-1").strip() or "mh-1",
        "head": os.environ.get("CATMASTER_REMOTE_MACE_HEAD", "omat_pbe"),
        "default_dtype": os.environ.get("CATMASTER_REMOTE_MACE_DTYPE", "float32").strip() or "float32",
        "check_interval": _remote_check_interval(30),
    }

    _content, artifact = _invoke_agent_tool(project_space, "mace_sp_batch", payload)
    data = artifact.get("data") or {}

    assert artifact.get("tool_name") == "mace_sp_batch"
    assert data.get("structures_found") == 1
    assert data.get("task_states"), "DPDispatcher returned no task states"

    _assert_status_success(_files_rel(project_space, data.get("status_file_rel")))

    batch_summary = _read_json(_files_rel(project_space, data.get("batch_summary_rel")))
    assert batch_summary.get("errors") == []
    results = batch_summary.get("results") or []
    assert len(results) == 1

    energy = results[0].get("summary", {}).get("energy_eV")
    assert isinstance(energy, (int, float)) and math.isfinite(float(energy))

    output_root = _files_rel(project_space, data.get("output_root_rel"))
    assert (output_root / "O2" / "summary.json").is_file()
    assert (output_root / "O2" / "sp.vasp").is_file()


def test_vasp_prepare_then_manual_dpdispatcher_o2_sp_remote(tmp_path: Path) -> None:
    from catmaster.tools.base import workspace_relpath, workspace_scope
    from catmaster.tools.execution.dpdispatcher_runner import (
        STATUS_FILE_NAME,
        BatchDispatchRequest,
        TaskSpec,
        dispatch_submission,
        make_work_base,
    )
    from catmaster.tools.execution.machine_registry import MachineRegister
    from catmaster.tools.execution.task_payloads import render_task_fields
    from catmaster.tools.execution.task_registry import TaskRegistry

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

        output_root = _files_root(project_space) / "remote_execution" / "vasp_o2_sp_manual_outputs"
        output_root.mkdir(parents=True, exist_ok=True)

        task_name = os.environ.get("CATMASTER_REMOTE_VASP_TASK", "vasp_execute").strip() or "vasp_execute"
        cfg = TaskRegistry().get(task_name)
        resources_key = cfg.resources
        assert resources_key, f"{task_name} missing resources in task config"

        machine_register = MachineRegister()
        resources_cfg = machine_register.get_resources(resources_key)
        machine = str(resources_cfg.get("machine") or "")
        assert machine, f"{resources_key} missing machine binding"
        machine_register.get_machine(machine)

        work_base = make_work_base("remote_vasp_o2_sp")
        stage_dir = output_root / work_base / "O2"
        shutil.copytree(input_dir, stage_dir)

        script_src = ROOT / "catmaster" / "remote" / "cpu" / "vasp_boot.py"
        assert script_src.is_file(), f"missing VASP boot script: {script_src}"
        script_dst = stage_dir / "task_script" / "vasp_boot.py"
        script_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_src, script_dst)

        rendered = render_task_fields(cfg, {"task_name": task_name}, stage_dir)
        backward_files = list(rendered["backward_files"])
        if "*" not in backward_files and STATUS_FILE_NAME not in backward_files:
            backward_files.append(STATUS_FILE_NAME)

        request = BatchDispatchRequest(
            machine=machine,
            resources=resources_key,
            work_base=work_base,
            local_root=str(output_root),
            tasks=[
                TaskSpec(
                    command=rendered["command"],
                    task_work_path="O2",
                    forward_files=rendered["forward_files"],
                    backward_files=backward_files,
                )
            ],
            clean_remote=False,
            check_interval=_int_env("CATMASTER_REMOTE_VASP_CHECK_INTERVAL", _remote_check_interval(60)),
        )

        result = dispatch_submission(request)

        assert result.task_states, "DPDispatcher returned no task states"
        status_path = stage_dir / STATUS_FILE_NAME
        _assert_status_success(status_path)
        assert any((stage_dir / name).is_file() for name in ("OUTCAR", "OSZICAR", "vasprun.xml")), (
            "VASP did not return expected output files; "
            f"stage_dir={workspace_relpath(stage_dir)}"
        )
