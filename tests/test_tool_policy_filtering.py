from __future__ import annotations

from catmaster.runtime.tool_policy import ToolPolicy


def test_tool_policy_filtering() -> None:
    tools = [
        {"name": "a", "type": "function"},
        {"name": "b", "type": "function"},
        {"name": "c", "type": "function"},
    ]
    policy = ToolPolicy(allowed_tools={"a", "c"})
    filtered = policy.filter_function_tools(tools)
    assert {tool["name"] for tool in filtered} == {"a", "c"}

    policy = ToolPolicy(allowed_tools=None, denied_tools={"b"})
    filtered = policy.filter_function_tools(tools)
    assert {tool["name"] for tool in filtered} == {"a", "c"}
