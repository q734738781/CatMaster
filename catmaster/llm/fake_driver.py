from __future__ import annotations

from typing import Any, Iterable

from catmaster.llm.driver import ToolCallingDriver
from catmaster.llm.types import ToolCall, TurnResult


def _parse_output_items(output_items: list[dict]) -> TurnResult:
    tool_calls: list[ToolCall] = []
    output_text_parts: list[str] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "function_call":
            tool_calls.append(ToolCall(
                name=item.get("name", ""),
                call_id=item.get("call_id", ""),
                arguments=item.get("arguments", ""),
                raw=item,
            ))
        elif item_type == "message":
            content = item.get("content") or []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("output_text", "input_text"):
                        text = part.get("text")
                        if text:
                            output_text_parts.append(text)
        elif item_type == "output_text":
            text = item.get("text")
            if text:
                output_text_parts.append(text)
    return TurnResult(
        output_text="".join(output_text_parts),
        tool_calls=tool_calls,
        output_items_raw=output_items,
    )


class FakeDriver(ToolCallingDriver):
    def __init__(self, script: Iterable[Any]):
        self._script = list(script)
        self._cursor = 0

    def create_turn(
        self,
        *,
        input_items: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> TurnResult:
        if self._cursor >= len(self._script):
            raise RuntimeError("FakeDriver script exhausted")
        item = self._script[self._cursor]
        self._cursor += 1
        if isinstance(item, TurnResult):
            return item
        if isinstance(item, list):
            return _parse_output_items(item)
        if isinstance(item, dict):
            return _parse_output_items([item])
        raise TypeError(f"Unsupported FakeDriver script item: {type(item).__name__}")
