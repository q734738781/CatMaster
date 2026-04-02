from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import StructuredTool

from catmaster.agents import graph


def test_memory_scoped_apply_tool_forces_allowed_paths() -> None:
    captured: dict[str, object] = {}

    def _base_tool(runtime=None, **kwargs):
        _ = runtime
        captured.update(kwargs)
        return ("ok", {"data": {}})

    base = StructuredTool.from_function(
        func=_base_tool,
        name="apply_aider_edits",
        description="test",
        args_schema={
            "type": "object",
            "properties": {
                "edits_text": {"type": "string"},
                "allowed_paths": {"type": "array", "items": {"type": "string"}},
                "emit_diff": {"type": "boolean"},
            },
            "required": ["edits_text"],
            "additionalProperties": False,
        },
        infer_schema=False,
        response_format="content_and_artifact",
    )

    scoped = graph._make_memory_scoped_apply_tool(base)
    result = scoped.invoke(
        {
            "edits_text": "MEMORY/topics/FACTS.md\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n",
            "allowed_paths": ["notes/"],
            "emit_diff": True,
        }
    )

    assert result == "ok"
    assert captured.get("allowed_paths") == ["MEMORY/"]
    schema = scoped.args_schema
    assert isinstance(schema, dict)
    assert "allowed_paths" not in (schema.get("properties") or {})


def test_memory_scoped_apply_tool_is_noop_for_other_tools() -> None:
    def _base_tool(runtime=None, **kwargs):
        _ = (runtime, kwargs)
        return ("ok", {"data": {}})

    base = StructuredTool.from_function(
        func=_base_tool,
        name="execute",
        description="test",
        args_schema={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
            "additionalProperties": False,
        },
        infer_schema=False,
        response_format="content_and_artifact",
    )

    same = graph._make_memory_scoped_apply_tool(base)
    assert same is base
