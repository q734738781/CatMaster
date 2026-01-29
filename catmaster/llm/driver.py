from __future__ import annotations

from typing import Any, Protocol

from catmaster.llm.types import TurnResult


class ToolCallingDriver(Protocol):
    def create_turn(
        self,
        *,
        input_items: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> TurnResult:
        ...
