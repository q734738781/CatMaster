from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib
import json

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.remote_submission import (
    get_avail_remote_task,
    get_avail_resources,
    remote_submission,
    remote_submission_batch,
)
from catmaster.tools.execution.dpdispatcher_runner import DPDispatcherDispatchError
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry

remote_submission_mod = importlib.import_module("catmaster.tools.execution.remote_submission")


def test_package_root_preserves_legacy_execution_exports() -> None:
    from catmaster.tools.execution import (
        MaceRelaxInput,
        VaspExecuteInput,
        crest_conformer_search,
        orca_execute_batch,
        xtb_run_batch,
    )

    assert MaceRelaxInput.__name__ == "MaceRelaxInput"
    assert VaspExecuteInput.__name__ == "VaspExecuteInput"
    assert callable(crest_conformer_search)
    assert callable(orca_execute_batch)
    assert callable(xtb_run_batch)


def test_remote_task_catalog_is_filtered_by_worker_audience() -> None:
    with toolcall_context("catalog", audience="materials_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert {"vasp_execute", "mace_sp_dir"}.issubset(task_names)
    assert "mace_md_dir" not in task_names
    assert "orca_execute" not in task_names
    assert "mace_train_dir" not in task_names
    first_with_resource = next(item for item in artifact["data"]["tasks"] if item.get("resources"))
    resource = first_with_resource["resources"]
    assert "machine" not in resource
    assert "batch_type" not in resource
    assert "context_type" not in resource
    assert "queue_name" not in resource
    assert "custom_flags" not in resource
    assert "remote_root" not in resource
    assert "remote_profile" not in resource
    assert "key_filename" not in resource

    with toolcall_context("catalog", audience="materials_worker"):
        _, artifact = get_avail_resources({})
    material_resource_names = {item["resources"] for item in artifact["data"]["resources"]}
    assert material_resource_names == {"general_cpu", "general_gpu"}
    general_cpu = next(item for item in artifact["data"]["resources"] if item["resources"] == "general_cpu")
    assert general_cpu["kind"] == "general_cpu"
    assert general_cpu["default_for_custom_boot"] is True
    assert "machine" not in general_cpu
    assert "custom_flags" not in general_cpu

    with toolcall_context("catalog", audience="orca_xtb_worker"):
        _, artifact = get_avail_resources({})
    resource_names = {item["resources"] for item in artifact["data"]["resources"]}
    assert resource_names == {"general_cpu"}

    with toolcall_context("catalog", audience="dynamics_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    dynamics_task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert "mace_md_dir" in dynamics_task_names
    assert "mace_sp_dir" not in dynamics_task_names
    assert "mace_relax_dir" not in dynamics_task_names


def test_remote_task_catalog_references_existing_boot_scripts_and_layout_sections() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = TaskRegistry()
    register = MachineRegister()
    layout_text = (repo_root / "skills" / "execution" / "remote-stage-layouts" / "SKILL.md").read_text(encoding="utf-8")

    assert registry.tasks
    for task_name, cfg in registry.list_tasks().items():
        assert cfg.resources in register.resources
        if cfg.requires:
            capabilities = set(register.get_resources(str(cfg.resources)).get("capabilities") or [])
            assert set(cfg.requires).issubset(capabilities), task_name
        assert cfg.boot_script, f"{task_name} should declare a boot_script"
        assert (repo_root / str(cfg.boot_script)).is_file(), task_name
        assert cfg.layout_ref, f"{task_name} should declare layout_ref"
        anchor = str(cfg.layout_ref).rsplit("#", 1)[-1]
        assert f"## {anchor}" in layout_text


def test_remote_submission_builds_one_task_from_stage_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        stage_copy = Path(req.local_root) / req.work_base
        (stage_copy / "output_marker.txt").write_text("downloaded", encoding="utf-8")
        captured["work_base"] = req.work_base
        captured["local_root"] = req.local_root
        captured["machine"] = req.machine
        captured["resources"] = req.resources
        captured["command"] = req.tasks[0].command
        captured["task_work_path"] = req.tasks[0].task_work_path
        captured["forward_files"] = list(req.tasks[0].forward_files)
        captured["check_interval"] = req.check_interval
        captured["clean_remote"] = req.clean_remote
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={
                "remote_context_id": "dp_test",
                "submitted_at": "2026-05-20T00:00:00+08:00",
                "submission_hash": "abc123",
                "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_test.json",
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage" / "mace_sp"
        (stage / "input").mkdir(parents=True)
        (stage / "input" / "CO.vasp").write_text("dummy", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            _, artifact = remote_submission(
                {
                    "work_dir": "stage/mace_sp",
                    "task_name": "mace_sp_dir",
                    "params": {"model": "medium-mpa-0", "default_dtype": "float32"},
                    "config": {"check_interval": 7, "clean_remote": True, "cpu_per_node": 8},
                }
            )

    assert str(captured["work_base"]).startswith("remote_submission_stage_mace_sp_")
    assert str(captured["local_root"]).endswith("metadata/dpdispatcher/staging")
    assert captured["task_work_path"] == "."
    assert captured["resources"] == "mace_gpu"
    assert captured["check_interval"] == 7
    assert captured["clean_remote"] is True
    assert "medium-mpa-0" in str(captured["command"])
    assert "float32" in str(captured["command"])
    assert (stage / "task_script" / "mace_sp.py").is_file()
    assert (stage / "output_marker.txt").read_text(encoding="utf-8") == "downloaded"
    assert artifact["data"]["work_base"] == captured["work_base"]
    assert artifact["data"]["remote_context_id"] == "dp_test"
    assert artifact["data"]["submission_hash"] == "abc123"
    assert "jobs" not in artifact["data"]


def test_remote_submission_uses_unique_work_base_for_same_basename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work_bases: list[str] = []

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        work_bases.append(req.work_base)
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": f"dp_{len(work_bases)}", "submission_hash": f"hash_{len(work_bases)}", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        for rel in ("a/stage", "b/stage"):
            (tmp_path / "files" / rel).mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            remote_submission({"work_dir": "a/stage", "task_name": "vasp_execute"})
            remote_submission({"work_dir": "b/stage", "task_name": "vasp_execute"})

    assert len(work_bases) == 2
    assert work_bases[0] != work_bases[1]
    assert work_bases[0].startswith("remote_submission_a_stage_")
    assert work_bases[1].startswith("remote_submission_b_stage_")


def test_remote_submission_quotes_template_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        captured["command"] = req.tasks[0].command
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_quote", "submission_hash": "hash_quote", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        (stage / "input").mkdir(parents=True)
        (stage / "input" / "CO.vasp").write_text("dummy", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            remote_submission(
                {
                    "work_dir": "stage",
                    "task_name": "mace_sp_dir",
                    "params": {"head": "", "model": "model with spaces"},
                }
            )

    assert "--head '' --dispersion false" in captured["command"]
    assert "--model 'model with spaces'" in captured["command"]


def test_remote_submission_batch_maps_first_level_children(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        stage_copy = Path(req.local_root) / req.work_base
        (stage_copy / "a" / "OUTCAR").write_text("done", encoding="utf-8")
        captured["task_work_paths"] = [task.task_work_path for task in req.tasks]
        captured["commands"] = [task.command for task in req.tasks]
        return SimpleNamespace(
            task_states=["finished", "finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_batch", "submission_hash": "hash_batch", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        root = tmp_path / "files" / "vasp_batch"
        for name in ("a", "b"):
            child = root / name
            child.mkdir(parents=True)
            for filename in ("INCAR", "POTCAR", "POSCAR", "KPOINTS"):
                (child / filename).write_text("dummy", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            _, artifact = remote_submission_batch({"work_dir": "vasp_batch", "task_name": "vasp_execute"})

    assert captured["task_work_paths"] == ["a", "b"]
    assert all("vasp_boot.py" in command for command in captured["commands"])
    assert (root / "a" / "task_script" / "vasp_boot.py").is_file()
    assert (root / "b" / "task_script" / "vasp_boot.py").is_file()
    assert (root / "a" / "OUTCAR").read_text(encoding="utf-8") == "done"
    assert artifact["data"]["task_count"] == 2
    assert artifact["data"]["task_state_counts"] == {"finished": 2}


def test_remote_submission_failure_exposes_receipt_context_in_message_and_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        stage_copy = Path(req.local_root) / req.work_base
        (stage_copy / "partial.txt").write_text("partial", encoding="utf-8")
        raise DPDispatcherDispatchError(
            "ConnectionResetError: connection reset by peer",
            remote_context={
                "remote_context_id": "dp_failed",
                "submitted_at": "2026-05-20T12:00:00+08:00",
                "updated_at": "2026-05-20T12:01:00+08:00",
                "submission_hash": "hash_failed",
                "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_failed.json",
                "jobs": [{"job_hash": "jhash", "job_id": "12345", "status_code": 2, "status": "running"}],
                "job_status_counts": {"running": 1},
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission({"work_dir": "stage", "task_name": "vasp_execute"})

    message = str(excinfo.value)
    assert "remote_context_id=dp_failed" in message
    assert "submission_hash=hash_failed" in message
    assert "submitted_at=2026-05-20T12:00:00+08:00" in message
    assert "jobs=1" in message
    assert '"running": 1' in message
    data = excinfo.value.artifact["data"]
    assert data["receipt_rel"] == ".deepagents/dpdispatcher/receipts/dp_failed.json"
    assert data["jobs"][0]["job_id"] == "12345"
    assert (stage / "partial.txt").read_text(encoding="utf-8") == "partial"


def test_remote_submission_pre_dispatch_failure_writes_attempt_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (req, register, config_path)
        raise TimeoutError("timed out")

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission({"work_dir": "stage", "task_name": "vasp_execute"})

        message = str(excinfo.value)
        data = excinfo.value.artifact["data"]
        assert "remote_context_id=" in message
        assert "receipt_rel=" in message
        assert data["submission_hash"] == ""
        assert data["jobs"] == []

        receipt_path = tmp_path / "files" / data["receipt_rel"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["context_id"] == data["remote_context_id"]
        assert receipt["submission_hash"] == ""
        assert receipt["jobs"] == []
        assert receipt["task_name"] == "vasp_execute"
        assert receipt["work_dir_rel"] == "stage"
        assert receipt["resources"] == "vasp_cpu"
        assert "timed out" in receipt["dispatch_error"]


def test_remote_submission_parses_boolean_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        captured["clean_remote"] = req.clean_remote
        captured["check_interval"] = req.check_interval
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_bool", "submission_hash": "hash_bool", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            remote_submission(
                {
                    "work_dir": "stage",
                    "task_name": "vasp_execute",
                    "config": {"clean_remote": "false", "check_interval": "9"},
                }
            )

    assert captured["clean_remote"] is False
    assert captured["check_interval"] == 9


def test_custom_boot_script_can_build_resource_from_machine_without_worker_audience(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = config_path
        captured["machine"] = req.machine
        captured["resources"] = req.resources
        captured["command"] = req.tasks[0].command
        captured["resource_cfg"] = dict(register.get_resources(req.resources))
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_custom", "submission_hash": "hash_custom", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        stage = files_root / "stage"
        stage.mkdir(parents=True)
        script = files_root / "run_custom.sh"
        script.write_text("echo custom\n", encoding="utf-8")
        with toolcall_context("submit"):
            remote_submission(
                {
                    "work_dir": "stage",
                    "boot_script": "run_custom.sh",
                    "config": {
                        "machine": "cpu_server_2",
                        "cpu_per_node": 90,
                        "queue_name": "batch",
                        "group_size": 1,
                    },
                }
            )

    assert captured["machine"] == "cpu_server_2"
    assert captured["resources"] == "custom_cpu_server_2"
    assert captured["command"] == "bash task_script/run_custom.sh"
    assert captured["resource_cfg"]["machine"] == "cpu_server_2"
    assert captured["resource_cfg"]["cpu_per_node"] == 90
    assert captured["resource_cfg"]["queue_name"] == "batch"
    assert (stage / "task_script" / "run_custom.sh").is_file()


def test_task_submission_can_override_machine_resource_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = config_path
        captured["machine"] = req.machine
        captured["resources"] = req.resources
        captured["resource_cfg"] = dict(register.get_resources(req.resources))
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_neb", "submission_hash": "hash_neb", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "neb_stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit"):
            remote_submission(
                {
                    "work_dir": "neb_stage",
                    "task_name": "vasp_execute_neb",
                    "config": {"machine": "cpu_server_2", "cpu_per_node": 90, "group_size": 5},
                }
            )

    assert captured["machine"] == "cpu_server_2"
    assert captured["resources"] == "vasp_cpu_neb"
    assert captured["resource_cfg"]["machine"] == "cpu_server_2"
    assert captured["resource_cfg"]["cpu_per_node"] == 90
    assert captured["resource_cfg"]["group_size"] == 5


def test_worker_submission_rejects_machine_override(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        (files_root / "stage").mkdir(parents=True)
        (files_root / "run_custom.sh").write_text("echo custom\n", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission(
                    {
                        "work_dir": "stage",
                        "boot_script": "run_custom.sh",
                        "config": {"machine": "cpu_server_2", "cpu_per_node": 4},
                    }
                )
    assert "config.machine is not available to worker tools" in str(excinfo.value)


def test_worker_registered_task_rejects_resource_card_swap(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission(
                    {
                        "work_dir": "stage",
                        "task_name": "vasp_execute",
                        "config": {"resources": "general_cpu"},
                    }
                )
    assert "task-bound resource card" in str(excinfo.value)


def test_worker_custom_boot_rejects_domain_resource_card(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        (files_root / "stage").mkdir(parents=True)
        (files_root / "run_custom.sh").write_text("echo custom\n", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission(
                    {
                        "work_dir": "stage",
                        "boot_script": "run_custom.sh",
                        "config": {"resources": "vasp_cpu"},
                    }
                )
    assert "not available for custom boot_script" in str(excinfo.value)


def test_custom_boot_script_uses_visible_default_resource(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        captured["machine"] = req.machine
        captured["resources"] = req.resources
        captured["command"] = req.tasks[0].command
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={
                "remote_context_id": "dp_default_custom",
                "submission_hash": "hash_default_custom",
                "receipt_rel": "receipt.json",
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        stage = files_root / "stage"
        stage.mkdir(parents=True)
        (files_root / "run_custom.sh").write_text("echo custom\n", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            _, artifact = remote_submission({"work_dir": "stage", "boot_script": "run_custom.sh"})

    assert captured["machine"] == "cpu_server_2"
    assert captured["resources"] == "general_cpu"
    assert captured["command"] == "bash task_script/run_custom.sh"
    assert artifact["data"]["resources"] == "general_cpu"
    assert (stage / "task_script" / "run_custom.sh").is_file()


def test_custom_boot_script_can_select_general_gpu_resource_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = config_path
        captured["machine"] = req.machine
        captured["resources"] = req.resources
        captured["resource_cfg"] = dict(register.get_resources(req.resources))
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={
                "remote_context_id": "dp_general_gpu",
                "submission_hash": "hash_general_gpu",
                "receipt_rel": "receipt.json",
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        stage = files_root / "stage"
        stage.mkdir(parents=True)
        (files_root / "run_custom.py").write_text("print('custom gpu')\n", encoding="utf-8")
        with toolcall_context("submit", audience="materials_worker"):
            _, artifact = remote_submission(
                {
                    "work_dir": "stage",
                    "boot_script": "run_custom.py",
                    "config": {"resources": "general_gpu"},
                }
            )

    assert captured["machine"] == "gpu_server"
    assert captured["resources"] == "general_gpu"
    assert captured["resource_cfg"]["gpu_per_node"] == 1
    assert artifact["data"]["resources"] == "general_gpu"


def test_remote_submission_rejects_forbidden_resource_override(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission(
                    {
                        "work_dir": "stage",
                        "task_name": "vasp_execute",
                        "config": {"remote_root": "/tmp/unsafe"},
                    }
                )
    assert "Forbidden remote config field" in str(excinfo.value)


def test_remote_submission_rejects_non_positive_check_interval(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission(
                    {
                        "work_dir": "stage",
                        "task_name": "vasp_execute",
                        "config": {"check_interval": 0},
                    }
                )
    assert "config.check_interval must be a positive integer" in str(excinfo.value)


def test_remote_submission_rejects_cross_audience_task(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError) as excinfo:
                remote_submission({"work_dir": "stage", "task_name": "orca_execute"})
    assert "not visible to audience" in str(excinfo.value)


def test_remote_submission_skills_use_stage_layout_schema() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vasp_text = (repo_root / "skills" / "materials" / "vasp-batch-execution" / "SKILL.md").read_text(encoding="utf-8")
    mace_text = (repo_root / "skills" / "materials" / "mace-screening-and-relaxation" / "SKILL.md").read_text(encoding="utf-8")

    for forbidden in ("input_dir", "output_dir", "_BATCH_STATE", "batch_state"):
        assert forbidden not in vasp_text
    assert "work_dir" in vasp_text
    assert "first-level child" in vasp_text
    assert "status.json" in vasp_text

    for forbidden in ("input_dir", "output_root", "_BATCH_STATE", "batch_state", "md_config"):
        assert forbidden not in mace_text
    assert "work_dir" in mace_text
    assert "input/" in mace_text
    assert "mace_md_dir" in mace_text

    mace_md_text = (repo_root / "skills" / "dynamics" / "mace-md-sampling" / "SKILL.md").read_text(encoding="utf-8")
    assert "work_dir" in mace_md_text
    assert "input/" in mace_md_text
    assert "params/md_params.json" in mace_md_text
    assert "md_config" in mace_md_text
