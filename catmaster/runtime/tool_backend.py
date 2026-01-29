from __future__ import annotations

from typing import Protocol


class ToolBackend(Protocol):
    def list_function_tools(self) -> list[dict]:
        ...

    def call(
        self,
        name: str,
        arguments_json: str,
        *,
        toolcall_key: str,
        call_id: str | None = None,
    ) -> dict:
        ...
