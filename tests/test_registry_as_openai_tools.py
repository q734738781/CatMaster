from __future__ import annotations

from pydantic import BaseModel, Field

from catmaster.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    """Dummy tool input."""

    text: str = Field(..., description="Text to echo")


def dummy_tool(payload: dict) -> dict:
    return {
        "status": "success",
        "tool_name": "dummy_tool",
        "data": {"text": payload.get("text")},
    }


def test_registry_as_openai_tools() -> None:
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("dummy_tool", dummy_tool, DummyInput)

    tools = registry.as_openai_tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "dummy_tool"
    assert "parameters" in tool
    assert tool["parameters"]["type"] == "object"
    assert tool["parameters"].get("additionalProperties") is False
