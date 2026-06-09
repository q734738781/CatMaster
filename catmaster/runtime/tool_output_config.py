from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class ToolOutputConfig:
    inline_data_enabled: bool = True
    preview_chars: int = 3_000
    offload_chars: int = 20_000
    offload_dir_rel: str = "_tool_outputs"
    include_tool_args: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ToolOutputConfig":
        payload = dict(data or {})
        return cls(
            inline_data_enabled=bool(payload.get("inline_data_enabled", cls.inline_data_enabled)),
            preview_chars=max(256, int(payload.get("preview_chars", cls.preview_chars))),
            offload_chars=max(128, int(payload.get("offload_chars", cls.offload_chars))),
            offload_dir_rel=str(payload.get("offload_dir_rel", cls.offload_dir_rel)).strip() or cls.offload_dir_rel,
            include_tool_args=bool(payload.get("include_tool_args", cls.include_tool_args)),
        )


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            return {}
        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    data = json.loads(text or "{}")
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=4)
def get_tool_output_config(path: str | None = None) -> ToolOutputConfig:
    env_path = (os.getenv("CATMASTER_TOOL_OUTPUT_CONFIG", "") or "").strip()
    config_path = Path(path or env_path or "configs/tool_output.yaml")
    raw = _load_config_file(config_path)
    offload = raw.get("offload") if isinstance(raw, dict) else {}
    if not isinstance(offload, dict):
        offload = {}
    return ToolOutputConfig.from_dict(offload)


__all__ = ["ToolOutputConfig", "get_tool_output_config"]
