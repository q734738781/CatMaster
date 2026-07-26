from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from catmaster.tools.execution.task_registry import TaskConfig, TaskRegistry


_PLACEHOLDER_RE = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_CONFIG_NAMES = [
    name
    for name in ("tasks_template.yaml", "tasks.yaml")
    if (_REPO_ROOT / "configs" / "dpdispatcher" / name).is_file()
]


def _argparse_destinations(script_path: Path) -> dict[str, str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    destinations: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        flag = first.value
        if not flag.startswith("--"):
            continue
        explicit_dest = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "dest"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            None,
        )
        destinations[flag] = explicit_dest or flag[2:].replace("-", "_")
    return destinations


@pytest.mark.parametrize("config_name", _TASK_CONFIG_NAMES)
def test_registered_template_keys_match_cli_flags_and_script_variables(config_name: str) -> None:
    tasks = yaml.safe_load((_REPO_ROOT / "configs" / "dpdispatcher" / config_name).read_text(encoding="utf-8"))

    for task_name, task in tasks.items():
        command_tokens = shlex.split(str(task["command"]))
        placeholders: set[str] = set()
        destinations = _argparse_destinations(_REPO_ROOT / str(task["boot_script"]))
        for flag, destination in destinations.items():
            assert flag == f"--{destination}", f"{task_name}: {flag} populates args.{destination}"
        for index, token in enumerate(command_tokens):
            match = _PLACEHOLDER_RE.fullmatch(token)
            if match is None:
                continue
            key = match.group(1)
            placeholders.add(key)
            assert index > 0, task_name
            flag = command_tokens[index - 1]
            assert flag == f"--{key}", f"{task_name}: {key} is passed through {flag}"
            assert destinations.get(flag) == key, f"{task_name}: {flag} does not populate args.{key}"

        assert set(task.get("defaults") or {}) == placeholders, task_name


def test_task_config_rejects_command_default_mismatch() -> None:
    with pytest.raises(ValidationError, match="missing defaults: steps"):
        TaskConfig(command="runner --steps {steps}", defaults={})
    with pytest.raises(ValidationError, match="unused defaults: maxsteps"):
        TaskConfig(command="runner --steps {steps}", defaults={"steps": 10, "maxsteps": 10})


def test_disabled_tasks_are_hidden_from_agent_visible_catalogs() -> None:
    registry = TaskRegistry()
    registry.tasks = {
        "enabled_task": TaskConfig(command="echo enabled", enabled=True),
        "disabled_task": TaskConfig(command="echo disabled", enabled=False),
    }

    assert set(registry.list_tasks()) == {"enabled_task"}
    assert set(registry.list_tasks(audience="materials_worker")) == {"enabled_task"}
    assert registry.task_visible_to("enabled_task", audience="materials_worker") is True
    assert registry.task_visible_to("disabled_task", audience="materials_worker") is False
    assert "disabled_task" not in registry.describe_for_llm()


def test_lammps_template_has_explicit_cpu_and_strict_kokkos_tasks() -> None:
    config_root = _REPO_ROOT / "configs" / "dpdispatcher"
    tasks = yaml.safe_load((config_root / "tasks_template.yaml").read_text(encoding="utf-8"))
    resources = yaml.safe_load((config_root / "resources_template.yaml").read_text(encoding="utf-8"))

    cpu_task = tasks["lammps_execute"]
    kokkos_task = tasks["lammps_execute_kokkos"]
    assert cpu_task["resources"] == "lammps_cpu"
    assert "--gpu off" in cpu_task["command"]
    assert "--mpi_launcher auto" in cpu_task["command"]
    assert kokkos_task["resources"] == "lammps_gpu"
    assert "--gpu kokkos" in kokkos_task["command"]
    assert "--no-allow_cpu_fallback" in kokkos_task["command"]
    assert "--mpi_launcher auto" in kokkos_task["command"]
    assert set(kokkos_task["requires"]) == {"lammps", "gpu", "kokkos"}
    assert resources["lammps_cpu"]["capabilities"] == ["lammps"]
    assert set(resources["lammps_gpu"]["capabilities"]) == {"lammps", "gpu", "kokkos"}
    assert resources["lammps_gpu"]["gpu_per_node"] == 1
