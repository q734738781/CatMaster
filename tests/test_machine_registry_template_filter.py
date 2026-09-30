from __future__ import annotations

from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry


def test_machine_registry_ignores_template_files(tmp_path) -> None:
    (tmp_path / "machines.yaml").write_text(
        "\n".join(
            [
                "zz_template_filter_case:",
                "  batch_type: Shell",
                "  context_type: SSHContext",
                "  local_root: /tmp/local",
                "  remote_root: /tmp/remote",
                "  remote_profile:",
                "    hostname: 1.2.3.4",
                "    port: 22",
                "    username: real_user",
                "    key_filename: /tmp/key",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "machines_template.yaml").write_text(
        "\n".join(
            [
                "zz_template_filter_case:",
                "  batch_type: Shell",
                "  context_type: SSHContext",
                "  local_root: <LOCAL>",
                "  remote_root: <REMOTE>",
                "  remote_profile:",
                "    hostname: <HOST>",
                "    port: 22",
                "    username: <USER>",
                "    key_filename: <KEY>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    reg = MachineRegister(extra_paths=[tmp_path])
    machine = reg.get_machine("zz_template_filter_case")
    assert machine["remote_profile"]["hostname"] == "1.2.3.4"
    assert machine["remote_profile"]["username"] == "real_user"


def test_task_registry_ignores_template_files(tmp_path) -> None:
    (tmp_path / "tasks.yaml").write_text(
        "\n".join(
            [
                "zz_real_task:",
                "  command: echo real",
                "  resources: real_cpu",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "tasks_template.yaml").write_text(
        "\n".join(
            [
                "zz_template_task:",
                "  command: echo template",
                "  resources: template_cpu",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = TaskRegistry(extra_paths=[tmp_path])
    assert "zz_real_task" in registry.tasks
    assert "zz_template_task" not in registry.tasks
