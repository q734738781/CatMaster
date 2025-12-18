from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

DEFAULT_DIRS = []
if os.environ.get("CATMASTER_DP_CONFIG"):
    DEFAULT_DIRS.append(Path(os.environ["CATMASTER_DP_CONFIG"]))
DEFAULT_DIRS.extend([
    Path.home() / ".catmaster" / "dpdispatcher.yaml",
    Path.home() / ".catmaster" / "dpdispatcher.d",
    Path(__file__).resolve().parents[3] / "configs" / "dpdispatcher",
])


def _load_file(path: Path) -> Dict:
    if not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text()) or {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return {}


def _iter_config_files(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    files = []
    for ext in ("*.yaml", "*.yml", "*.json"):
        files.extend(sorted(base.glob(ext)))
    return files


class MachineRegister:
    """Aggregates machine/resources definitions from YAML/JSON files.

    - search order: env path, ~/.catmaster/dpdispatcher.yaml, ~/.catmaster/dpdispatcher.d/*, repo configs/dpdispatcher/*
    - supports combined schema with top-level `machines` / `resources` as well as split files named
      `machines*.json|yaml`, `resources*.json|yaml`.
    """

    def __init__(self, extra_paths: Optional[Iterable[Path]] = None):
        paths = list(DEFAULT_DIRS)
        if extra_paths:
            paths.extend(extra_paths)
        self.machines: Dict[str, Dict] = {}
        self.resources: Dict[str, Dict] = {}
        self._load(paths)

    def _load(self, paths: Iterable[Path]) -> None:
        for base in paths:
            for path in _iter_config_files(base):
                data = _load_file(path)
                if not data:
                    continue
                # combined schema
                if "machines" in data or "resources" in data:
                    self.machines.update(data.get("machines", {}))
                    self.resources.update(data.get("resources", {}))
                # split schema
                if path.name.startswith("machines"):
                    self.machines.update(data)
                if path.name.startswith("resources"):
                    self.resources.update(data)

    def get_machine(self, name: str) -> Dict:
        if name not in self.machines:
            raise KeyError(f"Machine '{name}' not found")
        return self.machines[name]

    def get_resources(self, name: str) -> Dict:
        if name not in self.resources:
            raise KeyError(f"Resources '{name}' not found")
        return self.resources[name]

    def list_machines(self) -> Dict[str, Dict]:
        return dict(self.machines)

    def list_resources(self) -> Dict[str, Dict]:
        return dict(self.resources)

    def describe_for_llm(self) -> str:
        lines = ["Available machines (DPDispatcher):"]
        for name, cfg in sorted(self.machines.items()):
            bt = cfg.get("batch_type", "?")
            ctx = cfg.get("context_type", "?")
            rroot = cfg.get("remote_root", "?")
            lines.append(f"- {name}: batch={bt}, context={ctx}, remote_root={rroot}")
        lines.append("Available resources (bind to machines):")
        for name, cfg in sorted(self.resources.items()):
            mch = cfg.get("machine", "?")
            cpu = cfg.get("cpu_per_node", "?")
            gpu = cfg.get("gpu_per_node", "?")
            lines.append(f"- {name}: machine={mch}, cpu/node={cpu}, gpu/node={gpu}")
        return "\n".join(lines)


__all__ = ["MachineRegister"]
