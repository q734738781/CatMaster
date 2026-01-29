from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ToolPolicy:
    allowed_tools: set[str] | None = None
    denied_tools: set[str] | None = None
    builtin_tools: list[dict] = field(default_factory=list)
    parallel_tool_calls: bool = False
    max_tool_calls_per_task: int = 100
    use_skill_allowlist: bool = False

    def filter_function_tools(self, function_tools: list[dict]) -> list[dict]:
        allowed = self.allowed_tools
        denied = self.denied_tools or set()
        filtered: list[dict] = []
        for tool in function_tools:
            name = tool.get("name")
            if allowed is not None and name not in allowed:
                continue
            if name in denied:
                continue
            filtered.append(tool)
        return filtered

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPolicy":
        allowed_raw = data.get("allowed_tools", None)
        denied_raw = data.get("denied_tools", None)
        allowed: set[str] | None
        if allowed_raw is None or allowed_raw == "all" or allowed_raw == "*":
            allowed = None
        else:
            allowed = set(allowed_raw or [])
        denied = set(denied_raw or []) if denied_raw is not None else None
        return cls(
            allowed_tools=allowed,
            denied_tools=denied,
            builtin_tools=list(data.get("builtin_tools", []) or []),
            parallel_tool_calls=bool(data.get("parallel_tool_calls", False)),
            max_tool_calls_per_task=int(data.get("max_tool_calls_per_task", 100)),
            use_skill_allowlist=bool(data.get("use_skill_allowlist", False)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ToolPolicy":
        path = Path(path)
        if not path.exists():
            return cls()
        text = path.read_text(encoding="utf-8")
        data: dict[str, Any]
        if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"ToolPolicy config must be a mapping: {path}")
        return cls.from_dict(data)


__all__ = ["ToolPolicy"]
