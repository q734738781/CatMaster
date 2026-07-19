from __future__ import annotations

"""Task metadata loader for DPDispatcher submissions.

- Loads task definitions from repo `configs/dpdispatcher/*`.
- Supports either a dedicated tasks file (tasks.yaml/json) or a top-level `tasks` section
  inside a combined dpdispatcher config file.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_file(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(path.read_text()) or {}
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.warning("Failed to load task config %s: %s", path, exc)
    return {}


def _iter_config_files(base: Path) -> Iterable[Path]:
    if not base or not str(base):
        return []
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    files: List[Path] = []
    for ext in ("*.yaml", "*.yml", "*.json"):
        files.extend(
            sorted(
                p for p in base.glob(ext)
                # Ignore example/template files to avoid exposing them as active task cards.
                if "template" not in p.stem.lower()
            )
        )
    return files


DEFAULT_TASK_PATHS = [
    Path(__file__).resolve().parents[3] / "configs" / "dpdispatcher",
]


class TaskConfig(BaseModel):
    """Single task template loaded from YAML/JSON."""

    model_config = ConfigDict(extra="forbid")

    command: str
    enabled: bool = True
    resources: str | None = None
    audiences: List[str] = Field(default_factory=list)
    description: str = ""
    boot_script: str | None = None
    operation: str = ""
    layout_ref: str = ""
    defaults: Dict[str, object] = Field(default_factory=dict)
    requires: List[str] = Field(default_factory=list)
    task_work_path: str = "."
    forward_files: List[str] = Field(default_factory=list)
    backward_files: List[str] = Field(default_factory=list)
    forward_common_files: List[str] = Field(default_factory=list)
    backward_common_files: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _defaults_match_command_placeholders(self) -> "TaskConfig":
        placeholders = set(_PLACEHOLDER_RE.findall(self.command or ""))
        defaults = set(self.defaults)
        if placeholders != defaults:
            missing = sorted(placeholders - defaults)
            unused = sorted(defaults - placeholders)
            details: list[str] = []
            if missing:
                details.append("missing defaults: " + ", ".join(missing))
            if unused:
                details.append("unused defaults: " + ", ".join(unused))
            raise ValueError("Task command/default mismatch (" + "; ".join(details) + ")")
        return self


class TaskRegistry:
    """Aggregates task templates from known config locations."""

    def __init__(self, extra_paths: Optional[Iterable[Path]] = None):
        paths = [p for p in DEFAULT_TASK_PATHS if str(p)]
        if extra_paths:
            paths = list(extra_paths) + paths
        self.tasks: Dict[str, TaskConfig] = {}
        self._load(paths)

    def _load(self, paths: Iterable[Path]) -> None:
        for base in paths:
            for path in _iter_config_files(base):
                data = _load_file(path)
                if not data or not isinstance(data, dict):
                    continue

                tasks_section = None
                if path.name.startswith("tasks"):
                    # Accept both top-level tasks mapping or nested under 'tasks'
                    tasks_section = data.get("tasks", data)
                elif "tasks" in data:
                    tasks_section = data.get("tasks")

                if not tasks_section:
                    continue

                for name, cfg in tasks_section.items():
                    if not isinstance(cfg, Mapping):
                        logger.debug("Skip task %s in %s (not a mapping)", name, path)
                        continue
                    try:
                        self.tasks[name] = TaskConfig(**cfg)
                    except Exception as exc:
                        logger.warning("Invalid task config %s in %s: %s", name, path, exc)

    def get(self, name: str) -> TaskConfig:
        if name not in self.tasks:
            raise KeyError(f"Task '{name}' not found in task configs")
        return self.tasks[name]

    def list_tasks(self, *, audience: str | None = None) -> Dict[str, TaskConfig]:
        audience_name = str(audience or "").strip()
        if not audience_name:
            return {name: cfg for name, cfg in self.tasks.items() if cfg.enabled}
        return {
            name: cfg
            for name, cfg in self.tasks.items()
            if cfg.enabled and (not cfg.audiences or audience_name in cfg.audiences)
        }

    def task_visible_to(self, name: str, *, audience: str | None = None) -> bool:
        cfg = self.get(name)
        audience_name = str(audience or "").strip()
        return cfg.enabled and (not audience_name or not cfg.audiences or audience_name in cfg.audiences)

    def describe_for_llm(self) -> str:
        visible_tasks = self.list_tasks()
        if not visible_tasks:
            return "No DPDispatcher task templates found."
        lines = ["Available DPDispatcher tasks:"]
        for name, cfg in sorted(visible_tasks.items()):
            lines.append(f"- {name}: command={cfg.command}")
        return "\n".join(lines)


def format_template(template: str, values: Mapping[str, object]) -> str:
    """Safely format a template string using values; missing keys are left intact."""

    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Safe({k: v for k, v in values.items()}))


def format_list(items: Iterable[str], values: Mapping[str, object]) -> List[str]:
    return [format_template(item, values) for item in items]
