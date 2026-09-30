from __future__ import annotations

from langchain_core.messages import ToolMessage

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
    ) -> ToolMessage:
        raise NotImplementedError("MCPToolBackend is a placeholder. Implement MCP connectivity in PR4+.")

    def cancel_active_call(self, toolcall_key: str) -> bool:
        _ = toolcall_key
        return False


__all__ = ["MCPToolBackend"]
