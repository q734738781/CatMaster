from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str
    arguments: str
    raw: dict


@dataclass(frozen=True)
class TurnResult:
    output_text: str
    tool_calls: list[ToolCall]
    output_items_raw: list[dict]
