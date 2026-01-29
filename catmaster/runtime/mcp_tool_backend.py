from __future__ import annotations

from catmaster.runtime.tool_backend import ToolBackend


class MCPToolBackend(ToolBackend):
    def list_function_tools(self) -> list[dict]:
        return []

    def call(
        self,
        name: str,
        arguments_json: str,
        *,
        toolcall_key: str,
        call_id: str | None = None,
    ) -> dict:
        raise NotImplementedError("MCPToolBackend is a placeholder. Implement MCP connectivity in PR4+.")


__all__ = ["MCPToolBackend"]
