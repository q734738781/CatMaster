from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage

from catmaster.runtime.safe_tool_selector import SafeLLMToolSelectorMiddleware


@dataclass
class _Tool:
    name: str
    description: str = "tool"


@dataclass
class _Request:
    tools: list[object]
    messages: list[object]
    model: object

    def override(self, **changes):
        return replace(self, **changes)


class _BrokenStructuredModel:
    async def ainvoke(self, _messages):
        raise RuntimeError("selector failure")


class _BrokenModel:
    def with_structured_output(self, _schema):
        return _BrokenStructuredModel()


def test_safe_selector_prepare_filters_missing_always_include_without_error() -> None:
    middleware = SafeLLMToolSelectorMiddleware(
        model=None,
        max_tools=20,
        always_include=["read_text_file", "create_directory", "write_file"],
    )
    request = _Request(
        tools=[_Tool("read_text_file"), _Tool("bash")],
        messages=[HumanMessage(content="hello")],
        model=SimpleNamespace(),
    )

    prepared = middleware._prepare_selection_request(request)

    assert prepared is not None
    assert prepared.valid_tool_names == ["bash"]
    assert "read_text_file" in prepared.system_message
    assert "write_file" not in prepared.system_message


def test_safe_selector_invalid_selection_falls_back_to_original_request() -> None:
    middleware = SafeLLMToolSelectorMiddleware(
        model=None,
        max_tools=20,
        always_include=["read_text_file", "create_directory"],
    )
    request = _Request(
        tools=[_Tool("read_text_file"), _Tool("create_directory"), _Tool("bash")],
        messages=[HumanMessage(content="make a folder")],
        model=SimpleNamespace(),
    )
    available_tools = [_Tool("bash")]

    out = middleware._process_selection_response(
        {"tools": ["make_directory"]},
        available_tools,
        ["bash"],
        request,
    )

    assert out is request


def test_safe_selector_only_always_included_selection_falls_back_to_original_request() -> None:
    middleware = SafeLLMToolSelectorMiddleware(
        model=None,
        max_tools=20,
        always_include=["read_text_file"],
    )
    request = _Request(
        tools=[_Tool("read_text_file"), _Tool("bash")],
        messages=[HumanMessage(content="inspect file")],
        model=SimpleNamespace(),
    )

    out = middleware._process_selection_response(
        {"tools": ["read_text_file"]},
        [_Tool("bash")],
        ["bash"],
        request,
    )

    assert out is request


def test_safe_selector_async_exception_falls_back_to_original_request() -> None:
    request = _Request(
        tools=[_Tool("read_text_file"), _Tool("bash")],
        messages=[HumanMessage(content="inspect file")],
        model=_BrokenModel(),
    )
    middleware = SafeLLMToolSelectorMiddleware(
        model=None,
        max_tools=20,
        always_include=["read_text_file"],
    )

    async def _handler(req):
        return req

    out = asyncio.run(middleware.awrap_model_call(request, _handler))

    assert out is request
