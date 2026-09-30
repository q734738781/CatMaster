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
from catmaster.tools.execution.task_registry import TaskConfig, TaskRegistry
from catmaster.tools.registry import ToolRegistry

remote_submission_mod = importlib.import_module("catmaster.tools.execution.remote_submission")


def test_package_roots_expose_retained_execution_and_xtb_prepare_tools() -> None:
    from catmaster.tools.execution import (
        MaceRelaxInput,
        VaspExecuteInput,
        orca_execute_batch,
    )
    from catmaster.tools.geometry_inputs import crest_prepare, xtb_prepare

    assert MaceRelaxInput.__name__ == "MaceRelaxInput"
    assert VaspExecuteInput.__name__ == "VaspExecuteInput"
    assert callable(crest_prepare)
    assert callable(orca_execute_batch)
    assert callable(xtb_prepare)


def test_remote_task_catalog_is_filtered_by_worker_audience() -> None:
    with toolcall_context("catalog", audience="materials_worker"):
        content, artifact = get_avail_remote_task({"return_resource": True})
    assert "One prepared stage: use remote_submission" in content
    assert "use one remote_submission_batch call" in content
    assert "block until every submitted task is terminal" in content
    assert "prefer one remote_submission_batch" not in content
    assert "get_remote_task_spec" in content
    assert "execution_binding=configured as sufficient infrastructure provenance" in content
    assert "Block only on a concrete catalog/spec/submission error" in content
    assert "submission_guidance" in artifact["data"]
    assert "remote_submission_batch" in artifact["data"]["submission_guidance"]
    assert "template_overrides" in artifact["data"]["submission_guidance"]
    assert "blocks until all are terminal" in artifact["data"]["submission_guidance"]["remote_submission_batch"]
    task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert {
        "vasp_execute",
        "mlff_sp",
        "mlff_relax",
        "mlff_md",
        "mlff_neb",
        "mlff_vib",
        "mlff_ts",
    }.issubset(task_names)
    assert all(item["execution_binding"]["status"] == "configured" for item in artifact["data"]["tasks"]
               if item["task_name"] != "general_execute")
    assert all(item["execution_binding"]["platform_preflight"] == "passed" for item in artifact["data"]["tasks"]
               if item["task_name"] != "general_execute")
    vasp_item = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "vasp_execute")
    assert vasp_item["resources"]["resources"] == "vasp_cpu"
    assert vasp_item["execution_binding"]["runtime_health"] == "determined by submission result"
    mlff_item = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "mlff_sp")
    for item in (vasp_item, mlff_item):
        ref = item["layout_ref"]
        assert ref in content
        assert ref.startswith("/.deepagents/")
        assert (Path(__file__).resolve().parents[1] / ref.removeprefix("/.deepagents/")).is_file()
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
    assert "vasp_cpu" not in resource_content
    material_resource_names = {item["resources"] for item in artifact["data"]["resources"]}
    assert {"general_cpu", "general_gpu", "uma_gpu", "orb_gpu"} <= material_resource_names
    general_cpu = next(item for item in artifact["data"]["resources"] if item["resources"] == "general_cpu")
    assert general_cpu["description"]
    assert "machine" not in general_cpu
    assert "custom_flags" not in general_cpu

    with toolcall_context("catalog", audience="orca_xtb_worker"):
        _, artifact = get_avail_resources({})
    resource_names = {item["resources"] for item in artifact["data"]["resources"]}
    assert resource_names == {"general_cpu", "uma_gpu"}
    with toolcall_context("catalog", audience="orca_xtb_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    qchem_task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert {
        "xtb_execute",
        "crest_execute",
        "orca_execute",
        "mlff_sp",
        "mlff_relax",
        "mlff_vib",
        "mlff_ts",
    }.issubset(qchem_task_names)
    xtb_qchem = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "xtb_execute")
    assert xtb_qchem["template_override_keys"] == []
    mlff_qchem = next(item for item in artifact["data"]["tasks"] if item["task_name"] == "mlff_sp")
    assert "fairchem_uma" in mlff_qchem["available_backends"]

    with toolcall_context("catalog", audience="dynamics_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    dynamics_task_names = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert "mlff_md" in dynamics_task_names
    assert "mlff_sp" not in dynamics_task_names
    assert "mlff_relax" not in dynamics_task_names
    assert "mlff_ts" not in dynamics_task_names


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


@pytest.mark.parametrize(
    ("task_name", "prepare_tool"),
    [("xtb_execute", "xtb_prepare"), ("crest_execute", "crest_prepare")],
)
def test_native_argv_task_spec_has_no_scientific_or_internal_execution_fields(
    task_name: str,
    prepare_tool: str,
) -> None:
    with toolcall_context("spec", audience="orca_xtb_worker"):
        content, artifact = get_remote_task_spec({"task_name": task_name})

    data = artifact["data"]
    assert data["prepare_tool"] == prepare_tool
    assert data["single_submission"] is True
    assert data["batch_submission"] is True
    assert data["template_override_keys"] == []
    assert data["template_defaults"] == {}
    assert data["fields"] == []
    assert not ({"parameter_file", "program", "forward_files", "backward_files"} & set(data))
    assert "validation=ok" in content

    with toolcall_context("spec", audience="orca_xtb_worker"):
        _, invalid_artifact = get_remote_task_spec(
            {"task_name": task_name, "template_overrides": {"mode": "sp"}}
        )
    errors = invalid_artifact["data"]["errors"]
    assert errors
    assert "Accepted keys: none" in errors[0]["message"]


def test_remote_task_catalog_references_existing_boot_scripts_and_layout_documents() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = TaskRegistry()
    register = MachineRegister()

    assert registry.tasks
    for task_name, cfg in registry.list_tasks().items():
        if task_name == "general_execute":
            assert cfg.resources is None
            assert cfg.boot_script is None
            assert (repo_root / "skills/execution/general-remote-scripts/SKILL.md").is_file()
            continue
        ref = registry.layout_reference(task_name)
        assert ref.startswith("/.deepagents/")
        assert (repo_root / ref.removeprefix("/.deepagents/")).is_file(), task_name
        if cfg.operation:
            assert cfg.resources is None
            assert cfg.boot_script
            assert (repo_root / str(cfg.boot_script)).is_file(), task_name
            continue
        assert cfg.resources in register.resources
        if cfg.requires:
            capabilities = set(register.get_resources(str(cfg.resources)).get("capabilities") or [])
            assert set(cfg.requires).issubset(capabilities), task_name
        assert cfg.boot_script, f"{task_name} should declare a boot_script"
        assert (repo_root / str(cfg.boot_script)).is_file(), task_name


def test_layout_navigation_resolves_old_cards_without_changing_custom_refs(tmp_path, monkeypatch) -> None:
    module = importlib.import_module("catmaster.tools.execution.task_registry")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "catmaster/tools/execution/task_registry.py"))
    prefix = "/.deepagents/skills/execution/remote-stage-layouts/"
    reference = tmp_path / "skills/execution/remote-stage-layouts/references/demo.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("Internal layout instructions.\n")
    registry = TaskRegistry()
    original = prefix + "SKILL.md#demo"
    registry.tasks = {
        "legacy": TaskConfig(command="true", layout_ref=original),
        "demo": TaskConfig(command="true", operation="sp"),
        "custom": TaskConfig(command="true", layout_ref="/workspace/custom.md#input"),
        "missing": TaskConfig(command="true", layout_ref=prefix + "SKILL.md#missing"),
    }
    assert registry.layout_reference("legacy") == prefix + "references/demo.md"
    assert registry.layout_reference("demo") == prefix + "references/demo.md"
    assert registry.get("legacy").layout_ref == original
    assert registry.layout_reference("custom") == "/workspace/custom.md#input"
    assert registry.layout_reference("missing") == prefix + "SKILL.md#missing"


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


def test_crest_execute_rejects_scientific_template_overrides(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        (stage / "input.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
        (stage / "manifest.json").write_text(
            json.dumps({"argv": ["input.xyz"], "input_assets": ["input.xyz"]}),
            encoding="utf-8",
        )
        with toolcall_context("submit", audience="orca_xtb_worker"):
            with pytest.raises(CatMasterToolExecutionError, match="does not accept template_overrides"):
                remote_submission(
                    {
                        "work_dir": "stage",
                        "task_name": "crest_execute",
                        "template_overrides": {"solvent": "water"},
                    }
                )


def test_remote_submission_runs_xtb_execute_without_runtime_science_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        captured["command"] = req.tasks[0].command
        captured["forward_files"] = list(req.tasks[0].forward_files)
        dispatch_stage = Path(req.local_root) / req.work_base
        captured["coordinate_bytes"] = (dispatch_stage / "coord.xyz").read_bytes()
        captured["charge_bytes"] = (dispatch_stage / ".CHRG").read_bytes()
        captured["ambient_uploaded"] = (dispatch_stage / ".ambient").exists()
        (dispatch_stage / ".hidden-result").write_text("reachable", encoding="utf-8")
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(Path(req.local_root) / req.work_base),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={
                "remote_context_id": "dp_xtb_execute",
                "submission_hash": "hash_xtb_execute",
                "receipt_rel": "receipt.json",
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        stage.mkdir(parents=True)
        (stage / "coord.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
        (stage / ".CHRG").write_text("-1\n", encoding="utf-8")
        (stage / ".ambient").write_text("must not upload\n", encoding="utf-8")
        (stage / "manifest.json").write_text(
            json.dumps({"argv": ["coord.xyz", "--gfn", "2"], "input_assets": ["coord.xyz", ".CHRG"]}),
            encoding="utf-8",
        )
        coordinate_before = (stage / "coord.xyz").read_bytes()
        manifest_before = (stage / "manifest.json").read_bytes()
        with toolcall_context("submit", audience="orca_xtb_worker"):
            _, artifact = remote_submission({"work_dir": "stage", "task_name": "xtb_execute"})

        with toolcall_context("submit", audience="orca_xtb_worker"):
            with pytest.raises(CatMasterToolExecutionError, match="does not accept template_overrides"):
                remote_submission(
                    {
                        "work_dir": "stage",
                        "task_name": "xtb_execute",
                        "template_overrides": {"mode": "opt"},
                    }
                )

    assert captured["command"] == "python task_script/xtb_boot.py --manifest manifest.json --program xtb"
    assert captured["forward_files"] == [
        "manifest.json",
        "coord.xyz",
        ".CHRG",
        "task_script/xtb_boot.py",
        "task_script/native_argv_runtime.py",
    ]
    assert "*" not in captured["forward_files"]
    assert captured["coordinate_bytes"] == coordinate_before
    assert captured["charge_bytes"] == b"-1\n"
    assert captured["ambient_uploaded"] is False
    assert (stage / "coord.xyz").read_bytes() == coordinate_before
    assert (stage / "manifest.json").read_bytes() == manifest_before
    attempt = tmp_path / "files" / artifact["data"]["attempt_paths"][0]
    assert (attempt / ".hidden-result").read_text(encoding="utf-8") == "reachable"


def test_remote_submission_schema_keeps_optional_controls_non_nullable() -> None:
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


def test_native_batch_preflights_every_child_before_dispatch_or_stage_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dispatched = False

    def _unexpected_dispatch(req, *, register=None, config_path=None):
        nonlocal dispatched
        _ = (req, register, config_path)
        dispatched = True
        raise AssertionError("dispatch must not start before all native manifests and assets pass preflight")

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _unexpected_dispatch)
    with workspace_scope(tmp_path):
        root = tmp_path / "files" / "xtb_batch"
        for name in ("a", "b"):
            child = root / name
            child.mkdir(parents=True)
            if name == "a":
                (child / "coord.xyz").write_text("1\na\nH 0 0 0\n", encoding="utf-8")
            (child / "manifest.json").write_text(
                json.dumps({"argv": ["coord.xyz"], "input_assets": ["coord.xyz"]}),
                encoding="utf-8",
            )
        with toolcall_context("submit", audience="orca_xtb_worker"):
            with pytest.raises(CatMasterToolExecutionError, match="Missing declared native input asset"):
                remote_submission_batch({"work_dir": "xtb_batch", "task_name": "xtb_execute"})

        assert dispatched is False
        assert not (root / "a" / "task_script").exists()
        assert not (root / "b" / "task_script").exists()
        assert not (root / "a" / "attempts").exists()
        assert not (root / "b" / "attempts").exists()


def test_native_batch_rejects_accidental_nonstage_child_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dispatched = False

    def _unexpected_dispatch(req, *, register=None, config_path=None):
        nonlocal dispatched
        _ = (req, register, config_path)
        dispatched = True
        raise AssertionError("dispatch must not start with a nonstage batch child")

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _unexpected_dispatch)
    with workspace_scope(tmp_path):
        root = tmp_path / "files" / "xtb_batch"
        stage = root / "case_a"
        stage.mkdir(parents=True)
        (stage / "coord.xyz").write_text("1\na\nH 0 0 0\n", encoding="utf-8")
        (stage / "manifest.json").write_text(
            json.dumps({"argv": ["coord.xyz"], "input_assets": ["coord.xyz"]}),
            encoding="utf-8",
        )
        (root / "common_assets").mkdir()
        with toolcall_context("submit", audience="orca_xtb_worker"):
            with pytest.raises(CatMasterToolExecutionError, match="Missing prepared native parameter file"):
                remote_submission_batch({"work_dir": "xtb_batch", "task_name": "xtb_execute"})

    assert dispatched is False
    assert not (stage / "task_script").exists()
    assert not (stage / "attempts").exists()


def test_native_batch_isolates_same_named_assets_and_writes_one_attempt_per_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, bytes] = {}

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        dispatch_root = Path(req.local_root) / req.work_base
        for task in req.tasks:
            stage = dispatch_root / task.task_work_path
            captured[task.task_work_path] = (stage / "coord.xyz").read_bytes()
            (stage / ".native-result").write_text(task.task_work_path, encoding="utf-8")
        return SimpleNamespace(
            task_states=["finished", "finished"],
            submission_dir=str(dispatch_root),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={"remote_context_id": "dp_native_batch", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)
    with workspace_scope(tmp_path):
        root = tmp_path / "files" / "xtb_batch"
        for name, coordinate in (("a", "H 0 0 0"), ("b", "H 0 0 1")):
            child = root / name
            child.mkdir(parents=True)
            (child / "coord.xyz").write_text(f"1\n{name}\n{coordinate}\n", encoding="utf-8")
            (child / "manifest.json").write_text(
                json.dumps({"argv": ["coord.xyz"], "input_assets": ["coord.xyz"]}),
                encoding="utf-8",
            )
        with toolcall_context("submit", audience="orca_xtb_worker"):
            _, artifact = remote_submission_batch({"work_dir": "xtb_batch", "task_name": "xtb_execute"})

        assert captured["a"] != captured["b"]
        assert len(artifact["data"]["attempt_paths"]) == 2
        for name, attempt_rel in zip(("a", "b"), artifact["data"]["attempt_paths"]):
            attempt = tmp_path / "files" / attempt_rel
            assert (attempt / ".native-result").read_text(encoding="utf-8") == name
            assert (root / name / "manifest.json").is_file()


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


def test_orca_submission_keeps_runtime_input_in_dispatch_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = "! HF def2-SVP PAL8\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n"

    def _fake_dispatch(req, *, register=None, config_path=None):
        _ = (register, config_path)
        dispatch_stage = Path(req.local_root) / req.work_base
        assert (dispatch_stage / "job.inp").read_text(encoding="utf-8") == canonical
        assert (dispatch_stage / "task_script" / "node_local_scratch.py").is_file()
        (dispatch_stage / "job.runtime.inp").write_text(
            canonical.replace("PAL8", "PAL4"),
            encoding="utf-8",
        )
        (dispatch_stage / "job.out").write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(dispatch_stage),
            work_base=req.work_base,
            duration_s=0.1,
            remote_context={
                "remote_context_id": "dp_orca_runtime_copy",
                "receipt_rel": "receipt.json",
            },
        )

    monkeypatch.setattr(remote_submission_mod, "dispatch_submission", _fake_dispatch)
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "orca_stage"
        stage.mkdir(parents=True)
        (stage / "job.inp").write_text(canonical, encoding="utf-8")
        with toolcall_context("submit", audience="orca_xtb_worker"):
            remote_submission({"work_dir": "orca_stage", "task_name": "orca_execute"})

    assert (stage / "job.inp").read_text(encoding="utf-8") == canonical
    assert (stage / "job.out").read_text(encoding="utf-8") == "ORCA TERMINATED NORMALLY\n"
    assert not (stage / "job.runtime.inp").exists()


@pytest.mark.parametrize("tool_name", ["remote_submission", "remote_submission_batch"])
def test_remote_submission_final_schema_uses_stage_paths(tool_name: str) -> None:
    registry = ToolRegistry()
    schema = next(t for t in registry.as_openai_tools() if t["name"] == tool_name)["parameters"]
    properties = schema["properties"]
    assert "work_dir" in schema["required"]
    assert properties["work_dir"]["type"] == "string"
    assert not ({"input_dir", "output_dir", "batch_state"} & set(properties))
    if tool_name == "remote_submission_batch":
        assert properties["stage_paths"]["type"] == "array"
        assert properties["stage_paths"]["items"]["type"] == "string"
        assert "anyOf" not in properties["stage_paths"]
        assert "stage_paths" not in schema["required"]
        params = remote_submission_mod.RemoteSubmissionBatchInput(work_dir="batch", task_name="orca_execute")
        assert params.stage_paths == []
    else:
        assert "stage_paths" not in properties
