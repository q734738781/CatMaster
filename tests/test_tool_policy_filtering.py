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


def test_tool_policy_ignores_legacy_skill_flag() -> None:
    policy = ToolPolicy.from_dict({
        "allowed_tools": ["a", "b"],
        "denied_tools": ["b"],
        "use_skill_allowlist": True,
    })
    filtered = policy.filter_function_tools([
        {"name": "a", "type": "function"},
        {"name": "b", "type": "function"},
        {"name": "c", "type": "function"},
    ])
    assert {tool["name"] for tool in filtered} == {"a"}


def test_tool_policy_default_max_tool_calls_per_task_is_100() -> None:
    policy = ToolPolicy.from_dict({})
    assert policy.max_tool_calls_per_task == 100


def test_tool_policy_normalizes_legacy_bash_exec_alias() -> None:
    tools = [
        {"name": "bash", "type": "function"},
        {"name": "write_note", "type": "function"},
    ]

    policy = ToolPolicy(allowed_tools={"bash_exec"})
    filtered = policy.filter_function_tools(tools)
    assert [tool["name"] for tool in filtered] == ["bash"]

    policy = ToolPolicy(denied_tools={"bash_exec"})
    filtered = policy.filter_function_tools(tools)
    assert [tool["name"] for tool in filtered] == ["write_note"]
