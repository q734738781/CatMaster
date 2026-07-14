from __future__ import annotations

from typing import Any

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


class NullableOptionalInput(BaseModel):
    """Tool input with optional fields that should remain agent-legible."""

    query: str = Field(..., description="Required query")
    optional_text: str | None = Field(None, description="Optional text")
    optional_items: list[str] | None = Field(None, description="Optional items")
    optional_mapping: dict[str, Any] | None = Field(None, description="Optional mapping")
    optional_default: str | None = Field("auto", description="Optional text with default")


def nullable_optional_tool(payload: dict) -> dict:
    return {
        "status": "success",
        "tool_name": "nullable_optional_tool",
        "data": payload,
    }


def _schema_null_markers(schema: Any, path: str = "$") -> list[str]:
    markers: list[str] = []
    if isinstance(schema, dict):
        schema_type = schema.get("type")
        if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
            markers.append(path)
        for key in ("anyOf", "oneOf"):
            variants = schema.get(key)
            if isinstance(variants, list) and any(
                isinstance(item, dict) and item.get("type") == "null" for item in variants
            ):
                markers.append(f"{path}.{key}")
        if schema.get("default") is None and "default" in schema:
            markers.append(f"{path}.default")
        for key, value in schema.items():
            markers.extend(_schema_null_markers(value, f"{path}.{key}"))
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            markers.extend(_schema_null_markers(value, f"{path}[{index}]"))
    return markers


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


def test_optional_nullable_fields_are_agent_legible_in_exported_schemas() -> None:
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("nullable_optional_tool", nullable_optional_tool, NullableOptionalInput)

    openai_schema = registry.as_openai_tools()[0]["parameters"]
    properties = openai_schema["properties"]
    assert properties["optional_text"]["type"] == "string"
    assert properties["optional_items"]["type"] == "array"
    assert properties["optional_mapping"]["type"] == "object"
    assert properties["optional_default"]["type"] == "string"
    assert properties["optional_default"]["default"] == "auto"
    assert "optional_text" not in openai_schema.get("required", [])
    assert not _schema_null_markers(openai_schema)

    langchain_tool = registry.as_langchain_tools()[0]
    langchain_schema = langchain_tool.args_schema
    if hasattr(langchain_schema, "model_json_schema"):
        langchain_schema = langchain_schema.model_json_schema()
    assert isinstance(langchain_schema, dict)
    assert not _schema_null_markers(langchain_schema)


def test_registered_tools_export_valid_openai_tool_shapes() -> None:
    registry = ToolRegistry()
    tools = registry.as_openai_tools()

    assert {tool["name"] for tool in tools} == set(registry.tools)
    for tool in tools:
        assert tool["type"] == "function"
        assert isinstance(tool["name"], str)
        assert tool["name"]
        assert isinstance(tool["description"], str)
        assert tool["description"].strip()
        assert tool["parameters"]["type"] == "object"
        assert tool["parameters"].get("additionalProperties") is False


def test_registered_tools_do_not_export_langchain_reserved_argument_names() -> None:
    registry = ToolRegistry()

    for tool in registry.as_openai_tools():
        properties = tool["parameters"].get("properties", {})
        assert not ({"config", "runtime"} & set(properties)), tool["name"]


def test_registered_tools_do_not_export_optional_null_schema_markers() -> None:
    registry = ToolRegistry()

    openai_markers = {
        tool["name"]: _schema_null_markers(tool["parameters"])
        for tool in registry.as_openai_tools()
    }
    assert {name: markers for name, markers in openai_markers.items() if markers} == {}

    langchain_markers = {}
    for tool in registry.as_langchain_tools():
        args_schema = tool.args_schema
        if hasattr(args_schema, "model_json_schema"):
            args_schema = args_schema.model_json_schema()
        langchain_markers[tool.name] = _schema_null_markers(args_schema)
    assert {name: markers for name, markers in langchain_markers.items() if markers} == {}

    listed_markers = {
        name: _schema_null_markers(info["parameters"])
        for name, info in registry.list_tools().items()
    }
    assert {name: markers for name, markers in listed_markers.items() if markers} == {}
