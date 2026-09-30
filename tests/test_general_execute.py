from __future__ import annotations

import copy
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.execution.dpdispatcher_runner import _build_resources
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.registry import ToolRegistry

remote = importlib.import_module("catmaster.tools.execution.remote_submission")


@pytest.fixture
def deployment(monkeypatch):
    machines = object.__new__(MachineRegister)
    machines.machines = {"cluster": {"enabled": True}, "offline": {"enabled": False}}
    card = {"machine": "cluster", "kind": "domain", "allow_custom_boot": True,
            "audiences": ["materials_worker", "orca_xtb_worker"],
            "description": "Python, ASE and FairChem UMA with omol molecules",
            "number_node": 1, "cpu_per_node": 1, "gpu_per_node": 0, "group_size": 1,
            "source_list": ["/opt/science/uma env.sh"]}
    machines.resources = {
        "uma": card,
        "denied": {**card, "allow_custom_boot": False},
        "hidden": {**card, "audiences": ["ml_worker"]},
        "disabled": {**card, "enabled": False},
        "offline": {**card, "machine": "offline"},
    }
    tasks = TaskRegistry()
    tasks.tasks = {"general_execute": tasks.get("general_execute")}
    monkeypatch.setattr(remote, "MachineRegister", lambda: copy.deepcopy(machines))
    monkeypatch.setattr(remote, "TaskRegistry", lambda: copy.deepcopy(tasks))
    return machines


def test_general_catalog_final_tool_message_and_schema(deployment):
    registry = ToolRegistry()
    tool = next(t for t in registry.as_langchain_tools(allowlist=["get_remote_task_spec"], audience="orca_xtb_worker")
                if t.name == "get_remote_task_spec")
    with toolcall_context("spec", audience="orca_xtb_worker"):
        message = tool.invoke({"name": tool.name, "args": {"task_name": "general_execute", "detail": "full"},
                               "id": "spec-call", "type": "tool_call"})
        assert "FairChem UMA" in message.content
        assert "entrypoint" in message.content and "argv" in message.content
        data = message.artifact["data"]
        assert data["environments"] == [{"name": "uma", "description": deployment.resources["uma"]["description"]}]
        schema = data["template_schema"]
        assert schema["required"] == ["environment", "entrypoint"]
        for name, expected in [("environment", "string"), ("entrypoint", "string"), ("argv", "array")]:
            assert schema["properties"][name]["type"] == expected
            assert "anyOf" not in schema["properties"][name]
        assert not data["errors"]
        assert "general_execute" in remote.get_avail_remote_task({})[0]
        resources = remote.get_avail_resources({})[1]["data"]["resources"]
        assert resources == [{"resources": "uma", "description": deployment.resources["uma"]["description"]}]
    for secret in ["/opt/science", "source_list", "cluster", "allow_custom_boot"]:
        assert secret not in message.content


@pytest.mark.parametrize("environment", ["denied", "hidden", "disabled", "offline", "unknown"])
def test_unavailable_environment_rejected_before_dispatch(deployment, monkeypatch, tmp_path, environment):
    monkeypatch.setattr(remote, "dispatch_submission", lambda *a, **k: pytest.fail("must not submit"))
    with workspace_scope(tmp_path), toolcall_context("submit", audience="materials_worker"):
        workspace_root().joinpath("run.py").write_text("pass\n")
        spec = remote.get_remote_task_spec({"task_name": "general_execute", "template_overrides": {"environment": environment}})[1]["data"]
        assert spec["errors"]
        with pytest.raises(CatMasterToolExecutionError):
            remote.remote_submission({"work_dir": ".", "task_name": "general_execute",
                                      "template_overrides": {"environment": environment, "entrypoint": "run.py"}})


@pytest.mark.parametrize("batch", [False, True])
def test_general_submission_preserves_argv_environment_and_outputs(deployment, monkeypatch, tmp_path, batch):
    captured = []
    argv = ["space value", "$(touch SHOULD_NOT_EXIST)", "a'b", "", "; echo wrong"]

    def dispatch(req, *, register):
        assert req.resources == "uma"
        assert register.get_resources(req.resources)["source_list"] == ["/opt/science/uma env.sh"]
        root = Path(req.local_root) / req.work_base
        for task in req.tasks:
            captured.append(task)
            command = shlex.split(task.command)
            assert command[0] == "python"
            assert command[2:] == argv
            # Execute the generated command, including shell quoting, without
            # remote transport or scientific model inference.
            command_text = shlex.quote(sys.executable) + task.command[len("python"):]
            subprocess.run(command_text, shell=True, cwd=root / task.task_work_path, check=True)
        return SimpleNamespace(task_states=["finished"] * len(req.tasks), work_base=req.work_base,
                               submission_dir=str(root), duration_s=0.1)

    monkeypatch.setattr(remote, "dispatch_submission", dispatch)
    with workspace_scope(tmp_path), toolcall_context("submit", audience="materials_worker"):
        paths = [workspace_root() / "batch" / name for name in ("a", "b")] if batch else [workspace_root() / "stage"]
        for stage in paths:
            (stage / "scripts").mkdir(parents=True)
            (stage / "scripts/run.py").write_text("import json,sys,pathlib\npathlib.Path('out.json').write_text(json.dumps(sys.argv[1:]))\n")
        submit = remote.remote_submission_batch if batch else remote.remote_submission
        result = submit({"work_dir": "batch" if batch else "stage", "task_name": "general_execute",
                         "template_overrides": {"environment": "uma", "entrypoint": "scripts/run.py", "argv": argv}})
        assert result[1]["data"]["task_count"] == len(paths)
        for stage in paths:
            assert json.loads((stage / "out.json").read_text()) == argv
            assert not (stage / "SHOULD_NOT_EXIST").exists()
    assert len(captured) == (2 if batch else 1)


def test_batch_preflight_and_resource_override_rejection(deployment, monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "dispatch_submission", lambda *a, **k: pytest.fail("must not submit"))
    with workspace_scope(tmp_path), toolcall_context("submit", audience="materials_worker"):
        root = workspace_root() / "batch"
        for name in ["a", "b"]:
            (root / name).mkdir(parents=True)
        (root / "a/run.py").write_text("pass\n")
        args = {"work_dir": "batch", "task_name": "general_execute",
                "template_overrides": {"environment": "uma", "entrypoint": "run.py"}}
        with pytest.raises(CatMasterToolExecutionError, match="Missing general_execute entrypoint"):
            remote.remote_submission_batch(args)
        assert not list(root.rglob("task_script"))
        with pytest.raises(CatMasterToolExecutionError, match="omit resources and machine"):
            remote.remote_submission_batch({**args, "submission_config": {"resources": "uma"}})


def test_resource_activation_uses_native_dpdispatcher_prepend(deployment):
    resource = _build_resources(deployment.resources["uma"], env_setup=None)
    assert "source '/opt/science/uma env.sh'" in "\n".join(resource.prepend_script)


def test_shell_entrypoint_and_missing_required_controls(tmp_path):
    (tmp_path / "run.sh").write_text("exit 0\n")
    cfg = TaskRegistry().get("general_execute")
    task = remote._build_task_spec(cfg=cfg, task_name="general_execute", boot_script_src=None,
                                  stage_dir=tmp_path, stage_name=None,
                                  template_overrides={"environment": "uma", "entrypoint": "run.sh"})
    assert shlex.split(task.command) == ["bash", "./run.sh"]
    for kwargs in [{}, {"environment": "uma"}, {"environment": "uma", "entrypoint": "../run.py"},
                   {"environment": "uma", "entrypoint": "run.py", "argv": None}]:
        with pytest.raises(ValueError):
            remote.GeneralExecuteOverrides(**kwargs)
