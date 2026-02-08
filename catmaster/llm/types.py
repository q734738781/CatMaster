from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str
    arguments: str
    raw: dict


@dataclass(frozen=True)
class LLMTokenUsage:
    input_tokens: Optional[int] = None
    input_cached_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    source: str = "missing"
    raw: Optional[dict[str, Any]] = None

    def to_dict(self, *, include_raw: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "input_cached_tokens": self.input_cached_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }
        if include_raw:
            payload["raw"] = self.raw
        return payload


@dataclass(frozen=True)
class TurnResult:
    output_text: str
    tool_calls: list[ToolCall]
    output_items_raw: list[dict]
    usage: Optional[LLMTokenUsage] = None
