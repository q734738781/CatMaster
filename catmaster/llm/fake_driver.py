from __future__ import annotations

from typing import Any, Iterable

from catmaster.llm.driver import ToolCallingDriver
from catmaster.llm.types import LLMTokenUsage, ToolCall, TurnResult


def _parse_output_items(output_items: list[dict], *, usage: LLMTokenUsage | None = None) -> TurnResult:
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
        usage=usage,
    )


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _parse_usage(item: Any) -> LLMTokenUsage | None:
    if item is None:
        return None
    if isinstance(item, LLMTokenUsage):
        return item
    if not isinstance(item, dict):
        return None
    source = str(item.get("source") or "provider")
    return LLMTokenUsage(
        input_tokens=_to_int(item.get("input_tokens")),
        input_cached_tokens=_to_int(item.get("input_cached_tokens")),
        output_tokens=_to_int(item.get("output_tokens")),
        total_tokens=_to_int(item.get("total_tokens")),
        source=source,
        raw=item.get("raw") if isinstance(item.get("raw"), dict) else None,
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
            if "output_items" in item:
                output_items = item.get("output_items")
                if not isinstance(output_items, list):
                    raise TypeError("FakeDriver script item output_items must be a list")
                return _parse_output_items(output_items, usage=_parse_usage(item.get("usage")))
            return _parse_output_items([item])
        raise TypeError(f"Unsupported FakeDriver script item: {type(item).__name__}")
