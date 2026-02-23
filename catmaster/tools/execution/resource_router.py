from __future__ import annotations

"""Resource routing derived from DPDispatcher task definitions."""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml
from pydantic import BaseModel, Field

from catmaster.tools.execution.machine_registry import MachineRegister

DEFAULT_PATHS = [
    Path(__file__).resolve().parents[3] / "configs" / "dpdispatcher",
]


class Route(BaseModel):
    resources: str = Field(..., description="Resources key (machine inferred)")
    machine: str = Field(..., description="Machine key resolved from resources")
    defaults: Dict[str, object] = Field(default_factory=dict)


class ResourceRouter:
    def __init__(self, extra_paths: Optional[Iterable[Path]] = None):
        paths = [p for p in DEFAULT_PATHS if str(p)]
        if extra_paths:
            paths = list(extra_paths) + paths
        self.routes: Dict[str, Route] = {}
        self.register = MachineRegister()
        self._load(paths)

    def _load(self, paths: Iterable[Path]) -> None:
        for base in paths:
            if not base:
                continue
            if not base.exists():
                continue
            if base.is_dir():
                candidates = []
                for ext in ("*.yaml", "*.yml", "*.json"):
                    candidates.extend(sorted(base.glob(ext)))
            else:
                candidates = [base]
            for path in candidates:
                if not path.exists():
                    continue
                data = self._read(path)
                if not isinstance(data, dict) or not data:
                    continue

                tasks: Dict[str, object] | object = {}
                if path.name.startswith("tasks"):
                    tasks = data.get("tasks", data)
                elif "tasks" in data:
                    tasks = data.get("tasks", {})
                if not isinstance(tasks, dict):
                    continue

                for name, cfg in tasks.items():
                    if not isinstance(cfg, dict):
                        continue
                    try:
                        route = self._build_route(cfg)
                        if route:
                            self.routes[name] = route
                    except Exception:
                        continue

    def _build_route(self, cfg: Dict[str, object]) -> Optional[Route]:
        if "resources" not in cfg:
            return None
        res_key = cfg["resources"]
        res_cfg = self.register.get_resources(res_key)
        machine = res_cfg.get("machine")
        if not machine:
            raise KeyError(f"Resource '{res_key}' missing machine binding")
        defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
        return Route(resources=res_key, machine=machine, defaults=defaults)

    @staticmethod
    def _read(path: Path) -> Dict:
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(path.read_text()) or {}
            if path.suffix.lower() == ".json":
                return json.loads(path.read_text())
        except Exception:
            return {}
        return {}

    def route(self, task_name: str) -> Route:
        if task_name not in self.routes:
            raise KeyError(f"No route configured for task '{task_name}'")
        return self.routes[task_name]


__all__ = ["ResourceRouter", "Route"]
