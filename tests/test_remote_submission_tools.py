from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib
import json

import pytest
from ase.build import bulk, molecule
from ase.io import write
from langchain_core.messages import ToolMessage

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.remote_submission import (
    RemoteSubmissionInput,
    get_avail_remote_task,
    get_avail_resources,
    get_remote_task_spec,
    remote_submission,
    remote_submission_batch,
)
from catmaster.tools.execution.dpdispatcher_runner import DPDispatcherDispatchError
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.registry import ToolRegistry

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
        content, artifact = get_avail_remote_task({"return_resource": True})
    assert "One prepared stage: use remote_submission" in content
    assert "use one remote_submission_batch call" in content
    assert "block until every submitted task is terminal" in content
    assert "prefer one remote_submission_batch" not in content
    assert "get_remote_task_spec" in content
    assert "execution_binding=configured as sufficient infrastructure provenance" in content
    assert "registered domain task's resource card is intentionally absent" in content
    assert "Block only on a concrete catalog/spec/submission error" in content
    assert "submission_guidance" in artifact["data"]
    assert "remote_submission_batch" in artifact["data"]["submission_guidance"]
    assert "template_overrides" in artifact["data"]["submission_guidance"]
    assert "blocks until all are terminal" in artifact["data"]["submission_guidance"]["remote_submission_batch"]
    task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert {"vasp_execute", "mlff_sp", "mlff_relax", "mlff_md", "mlff_neb"}.issubset(task_names)
    assert all(item["execution_binding"]["status"] == "configured" for item in artifact["data"]["tasks"])
    assert all(item["execution_binding"]["platform_preflight"] == "passed" for item in artifact["data"]["tasks"])
    vasp_item = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "vasp_execute")
    assert vasp_item["resources"]["resources"] == "vasp_cpu"
    assert vasp_item["execution_binding"]["runtime_health"] == "determined by submission result"
    mlff_item = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "mlff_sp")
    assert "input/" in mlff_item["submission_hint"]
    assert mlff_item["template_override_keys"] == ["backend", "backend_config", "task_config"]
    assert mlff_item["default_backend"] == "mace"
    assert {"mace", "fairchem_uma"}.issubset(mlff_item["available_backends"])
    assert "orca_execute" not in task_names
    assert "mace_train" not in task_names
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
        resource_content, artifact = get_avail_resources({})
    assert "Available general remote resources" in resource_content
    assert "custom boot_script only" in resource_content
    assert "their absence is not a missing binding or submission blocker" in resource_content
    assert "vasp_cpu" not in resource_content
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
    with toolcall_context("catalog", audience="orca_xtb_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    qchem_task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert {"xtb_run", "orca_execute", "mlff_sp", "mlff_relax"}.issubset(qchem_task_names)
    mlff_qchem = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "mlff_sp")
    assert "fairchem_uma" in mlff_qchem["available_backends"]

    with toolcall_context("catalog", audience="dynamics_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    dynamics_task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert "mlff_md" in dynamics_task_names
    assert "mlff_sp" not in dynamics_task_names
    assert "mlff_relax" not in dynamics_task_names


def test_registered_vasp_spec_reports_configured_platform_binding_without_admin_internals() -> None:
    with toolcall_context("spec", audience="materials_worker"):
        content, artifact = get_remote_task_spec({"task_name": "vasp_execute"})

    assert "registered_execution_binding=configured" in content
    assert "hidden administrator fields are not user prerequisites" in content
    assert "runtime health is determined by the submission result" in content
    binding = artifact["data"]["execution_binding"]
    assert binding == {
        "status": "configured",
        "authority": "deployment",
        "platform_preflight": "passed",
        "scope": "registered task/backend binding only; stage inputs and user approval remain separate",
        "runtime_health": "determined by submission result",
    }
    for hidden in ("machine", "queue_name", "account", "module", "license", "revision"):
        assert hidden not in binding


def test_remote_task_catalog_references_existing_boot_scripts_and_layout_sections() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = TaskRegistry()
    register = MachineRegister()
    layout_text = (repo_root / "skills" / "execution" / "remote-stage-layouts" / "SKILL.md").read_text(encoding="utf-8")

    assert registry.tasks
    for task_name, cfg in registry.list_tasks().items():
        if cfg.operation:
            assert cfg.resources is None
            assert cfg.layout_ref == ""
            expected_heading = "mlff_sp and mlff_relax" if task_name in {"mlff_sp", "mlff_relax"} else task_name
            assert f"#### {expected_heading}" in layout_text
            assert cfg.boot_script
            assert (repo_root / str(cfg.boot_script)).is_file(), task_name
            continue
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
        write(stage / "input" / "CO.vasp", bulk("Cu", cubic=True))
        with toolcall_context("submit", audience="materials_worker"):
            content, artifact = remote_submission(
                {
                    "work_dir": "stage/mace_sp",
                    "task_name": "mlff_sp",
                    "template_overrides": {"backend": "mace", "backend_config": {"default_dtype": "float32"}},
                    "submission_config": {"check_interval": 7, "clean_remote": True, "cpu_per_node": 8},
                }
    )

    assert str(captured["work_base"]).startswith("remote_submission_stage_mace_sp_")
    assert Path(str(captured["local_root"])).parent.name == "staging"
    assert Path(str(captured["local_root"])).name == captured["work_base"]
    assert captured["task_work_path"] == "."
    assert captured["resources"] == "mace_gpu"
    assert captured["check_interval"] == 7
    assert captured["clean_remote"] is True
    assert captured["command"] == "python task_script/mlff_sp.py --run_config .catmaster/generated/run_config.json"
    assert (stage / "task_script" / "mlff_sp.py").is_file()
    assert (stage / "task_script" / "mlff_common.py").is_file()
    run_config = json.loads((stage / ".catmaster" / "generated" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["backend_config"]["default_dtype"] == "float32"
    assert (stage / "output_marker.txt").read_text(encoding="utf-8") == "downloaded"
    assert artifact["data"]["work_base"] == captured["work_base"]
    assert artifact["data"]["remote_context_id"] == "dp_test"
    assert artifact["data"]["submission_hash"] == "abc123"
    assert artifact["data"]["duration_s"] == 0.1
    assert "remote_context_id=dp_test" in content
    assert "submission_hash=abc123" in content
    assert "receipt_rel=.deepagents/dpdispatcher/receipts/dp_test.json" in content
    assert "duration_s=0.1" in content
    assert "jobs" not in artifact["data"]


def test_remote_submission_copies_common_mlff_helper_and_materializes_uma_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        captured["resources"] = req.resources
        captured["command"] = req.tasks[0].command
        captured["forward_files"] = list(req.tasks[0].forward_files)
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_uma", "submission_hash": "hash_uma", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage" / "uma_sp"
        (stage / "input").mkdir(parents=True)
        (stage / "input" / "H2O.xyz").write_text("3\nH2O\nO 0 0 0\nH 0 0 1\nH 1 0 0\n", encoding="utf-8")
        with toolcall_context("submit", audience="orca_xtb_worker"):
            remote_submission(
                {
                    "work_dir": "stage/uma_sp",
                    "task_name": "mlff_sp",
                    "template_overrides": {
                        "backend": "fairchem_uma",
                        "backend_config": {"defaults": {"uma_task": "omol", "charge": 0, "spin": 1}},
                    },
                }
            )

    assert captured["resources"] == "uma_gpu"
    assert captured["command"] == "python task_script/mlff_sp.py --run_config .catmaster/generated/run_config.json"
    assert "task_script/mlff_sp.py" in captured["forward_files"]
    assert "task_script/mlff_common.py" in captured["forward_files"]
    assert (stage / "task_script" / "mlff_sp.py").is_file()
    assert (stage / "task_script" / "mlff_common.py").is_file()
    run_config = json.loads((stage / ".catmaster" / "generated" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["items"]["H2O.xyz"]["uma_task"] == "omol"
    assert run_config["items"]["H2O.xyz"]["spin"] == 1


def test_registered_task_stages_declared_helper_before_wildcard_forward_collapse(tmp_path: Path) -> None:
    script_dir = tmp_path / "script_source"
    script_dir.mkdir()
    main_script = script_dir / "main.py"
    main_script.write_text("print('main')\n", encoding="utf-8")
    (script_dir / "helper.py").write_text("print('helper')\n", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    cfg = TaskRegistry().get("vasp_execute").model_copy(
        update={
            "command": "python task_script/main.py",
            "boot_script": str(main_script),
            "forward_files": ["*", "task_script/main.py", "task_script/helper.py"],
        }
    )

    task = remote_submission_mod._build_task_spec(
        cfg=cfg,
        task_name="helper_probe",
        boot_script_src=main_script,
        stage_dir=stage,
        stage_name=None,
        template_overrides=None,
    )

    assert task.forward_files == ["*"]
    assert (stage / "task_script" / "main.py").is_file()
    assert (stage / "task_script" / "helper.py").is_file()


def test_remote_submission_uses_unique_work_base_for_same_basename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work_bases: list[str] = []
    local_roots: list[str] = []

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        work_bases.append(req.work_base)
        local_roots.append(req.local_root)
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
    assert local_roots[0] != local_roots[1]
    assert Path(local_roots[0]).name == work_bases[0]
    assert Path(local_roots[1]).name == work_bases[1]


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
        stage.mkdir(parents=True)
        (stage / "input.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
        with toolcall_context("submit", audience="orca_xtb_worker"):
            remote_submission(
                {
                    "work_dir": "stage",
                    "task_name": "xtb_run",
                    "template_overrides": {"solvent_model": "alpb", "solvent": "water model"},
                }
            )

    assert "--solvent_model alpb" in captured["command"]
    assert "--solvent 'water model'" in captured["command"]


def test_remote_submission_template_overrides_render_command(
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
            remote_context={"remote_context_id": "dp_override", "submission_hash": "hash_override", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    schema = RemoteSubmissionInput.model_json_schema()["properties"]
    assert "template_overrides" in schema
    assert schema["template_overrides"]["type"] == "object"
    assert "anyOf" not in schema["template_overrides"]
    assert "params" not in schema
    assert schema["submission_config"]["type"] == "object"
    assert "anyOf" not in schema["submission_config"]
    assert "With task_name, do not pass resources or machine" in schema["submission_config"]["description"]
    assert "config" not in schema
    assert schema["task_name"]["type"] == "string"
    assert "anyOf" not in schema["task_name"]
    assert schema["boot_script"]["type"] == "string"
    assert "anyOf" not in schema["boot_script"]

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        (stage / "input.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
        with toolcall_context("submit", audience="orca_xtb_worker"):
            remote_submission(
                {
                    "work_dir": "stage",
                    "task_name": "xtb_run",
                    "template_overrides": {"mode": "sp", "charge": -1, "uhf": 1},
                }
            )

    assert "--mode sp" in captured["command"]
    assert "--charge -1" in captured["command"]
    assert "--uhf 1" in captured["command"]


def test_remote_submission_accepts_legacy_config_and_null_object_fields() -> None:
    parsed = RemoteSubmissionInput(
        work_dir="stage",
        task_name="mlff_relax",
        boot_script=None,
        template_overrides=None,
        config=None,
    )

    assert parsed.task_name == "mlff_relax"
    assert parsed.boot_script == ""
    assert parsed.template_overrides == {}
    assert parsed.submission_config == {}

    legacy = RemoteSubmissionInput(
        work_dir="stage",
        boot_script="run.sh",
        config={"resources": "general_gpu"},
    )
    assert legacy.submission_config == {"resources": "general_gpu"}


def test_remote_submission_rejects_unknown_template_override_key(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        (stage / "input").mkdir(parents=True)
        write(stage / "input" / "Cu.vasp", bulk("Cu", cubic=True))
        with toolcall_context("submit", audience="materials_worker"):
            with pytest.raises(CatMasterToolExecutionError, match="Unknown MLFF template_overrides key.*maxsteps"):
                remote_submission(
                    {
                        "work_dir": "stage",
                        "task_name": "mlff_relax",
                        "template_overrides": {"maxsteps": 100},
                    }
                )


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
    assert "duration_s=" in message
    assert "jobs=1" in message
    assert '"running": 1' in message
    data = excinfo.value.artifact["data"]
    assert data["receipt_rel"] == ".deepagents/dpdispatcher/receipts/dp_failed.json"
    assert data["jobs"][0]["job_id"] == "12345"
    assert data["duration_s"] >= 0
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
        assert "duration_s=" in message
        assert data["submission_hash"] == ""
        assert data["jobs"] == []
        assert data["duration_s"] >= 0

        receipt_path = tmp_path / "files" / data["receipt_rel"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["context_id"] == data["remote_context_id"]
        assert receipt["submission_hash"] == ""
        assert receipt["jobs"] == []
        assert receipt["duration_s"] == data["duration_s"]
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
                    "submission_config": {"clean_remote": "false", "check_interval": "9"},
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
                    "submission_config": {
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
                    "submission_config": {"machine": "cpu_server_2", "cpu_per_node": 90, "group_size": 5},
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
                        "submission_config": {"machine": "cpu_server_2", "cpu_per_node": 4},
                    }
                )
    assert "submission_config.machine is not available to worker tools" in str(excinfo.value)


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
                        "submission_config": {"resources": "general_cpu"},
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
                        "submission_config": {"resources": "vasp_cpu"},
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
                    "submission_config": {"resources": "general_gpu"},
                }
            )

    assert captured["machine"] == "gpu_server"
    assert captured["resources"] == "general_gpu"
    assert captured["resource_cfg"]["gpu_per_node"] == 1
    assert artifact["data"]["resources"] == "general_gpu"


def test_langchain_tool_surface_preserves_custom_gpu_submission_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = config_path
        captured["resources"] = req.resources
        captured["resource_cfg"] = dict(register.get_resources(req.resources))
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={
                "remote_context_id": "dp_langchain_general_gpu",
                "submission_hash": "hash_langchain_general_gpu",
                "receipt_rel": "receipt.json",
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)
    files_root = tmp_path / "files"
    (files_root / "stage").mkdir(parents=True)
    (files_root / "run_custom.py").write_text("print('custom gpu')\n", encoding="utf-8")
    tool = next(
        tool
        for tool in ToolRegistry().as_langchain_tools(
            allowlist=["remote_submission"],
            workspace=str(tmp_path),
            audience="materials_worker",
        )
        if tool.name == "remote_submission"
    )

    properties = tool.args_schema["properties"]
    assert "submission_config" in properties
    assert "config" not in properties
    assert "With task_name, do not pass resources or machine" in properties["submission_config"]["description"]
    result = tool.invoke(
        {
            "name": "remote_submission",
            "args": {
                "work_dir": "stage",
                "boot_script": "run_custom.py",
                "submission_config": {"resources": "general_gpu"},
            },
            "id": "call_remote_submission_receipt",
            "type": "tool_call",
        }
    )

    assert captured["resources"] == "general_gpu"
    assert captured["resource_cfg"]["gpu_per_node"] == 1
    assert isinstance(result, ToolMessage)
    assert "remote_context_id=dp_langchain_general_gpu" in result.content
    assert "submission_hash=hash_langchain_general_gpu" in result.content
    assert "receipt_rel=receipt.json" in result.content


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
                        "submission_config": {"remote_root": "/tmp/unsafe"},
                    }
                )
    assert "Forbidden remote submission_config field" in str(excinfo.value)


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
                        "submission_config": {"check_interval": 0},
                    }
                )
    assert "submission_config.check_interval must be a positive integer" in str(excinfo.value)


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
    vasp_text = (repo_root / "skills" / "materials_worker" / "vasp-batch-execution" / "SKILL.md").read_text(encoding="utf-8")
    mace_text = (repo_root / "skills" / "materials_worker" / "mlff-screening-and-relaxation" / "SKILL.md").read_text(encoding="utf-8")

    for forbidden in ("input_dir", "output_dir", "_BATCH_STATE", "batch_state"):
        assert forbidden not in vasp_text
    assert "work_dir" in vasp_text
    assert "first-level child" in vasp_text
    assert "status.json" in vasp_text
    assert "execution_binding.status=configured" in vasp_text
    assert "gamma-binary confirmation" in vasp_text
    assert "absence of `vasp_cpu` there is expected" in vasp_text

    for forbidden in ("input_dir", "output_root", "_BATCH_STATE", "batch_state", "md_config"):
        assert forbidden not in mace_text
    assert "work_dir" in mace_text
    assert "input/" in mace_text
    assert "mlff_relax" in mace_text
    assert '"backend": "mattersim"' in mace_text
    assert "MatterSim-v1.0.0-1M" in mace_text
    assert 'detail="full"' in mace_text

    mace_md_text = (repo_root / "skills" / "dynamics_worker" / "mlff-md-sampling" / "SKILL.md").read_text(encoding="utf-8")
    assert "work_dir" in mace_md_text
    assert "input/" in mace_md_text
    assert "task_config" in mace_md_text
    assert "get_remote_task_spec" in mace_md_text
