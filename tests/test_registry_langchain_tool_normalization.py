from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("langchain_core")

from langchain_core.utils.function_calling import convert_to_openai_tool

import catmaster.tools.registry as registry_module
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.runtime.tool_runtime import current_run_dir, current_toolcall_key
from catmaster.tools.registry import _make_langchain_tool


class _DummyInput(BaseModel):
    text: str = Field(..., description="dummy")


class _DefaultFactoryInput(BaseModel):
    text: str = Field(..., description="dummy")
    options: dict = Field(default_factory=dict, description="Optional options map.")


def test_wrapper_rejects_invalid_return_type() -> None:
    def _tool(payload: dict) -> int:
        return 123

    tool = _make_langchain_tool("dummy_tool", _tool, _DummyInput)
    with pytest.raises(CatMasterToolExecutionError):
        tool.invoke({"text": "x"})


def test_wrapper_success_returns_content_and_artifact_tuple() -> None:
    def _tool(payload: dict) -> tuple[str, dict]:
        return (
            "done",
            {
                "tool_name": "dummy_tool",
                "data": {"summary": "done", "value": payload.get("text")},
            },
        )

    tool = _make_langchain_tool("dummy_tool", _tool, _DummyInput)
    out = tool.invoke({"text": "x"})
    content, artifact = tool.func(text="x")
    assert "done" in str(content)
    assert isinstance(artifact, dict)


def test_wrapper_schema_preserves_optional_default_factory_fields() -> None:
    def _tool(payload: dict) -> tuple[str, dict]:
        return ("ok", {"tool_name": "dummy_tool", "data": payload})

    tool = _make_langchain_tool("dummy_tool", _tool, _DefaultFactoryInput)
    params = convert_to_openai_tool(tool)["function"]["parameters"]
    required = set(params.get("required", []))

    assert "text" in required
    assert "options" not in required
    assert tool.response_format == "content_and_artifact"


def test_wrapper_passes_workspace_files_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def _tool(payload: dict) -> tuple[str, dict]:
        return ("ok", {"tool_name": "dummy_tool", "data": payload})

    captured: dict[str, object] = {}

    def _fake_adapt_tool_return(
        *,
        tool_name,
        raw_result,
        workspace_files_root=None,
        output_config=None,
    ):
        captured["workspace_files_root"] = workspace_files_root
        return "ok", {"tool_name": tool_name}

    monkeypatch.setattr(registry_module, "adapt_tool_return", _fake_adapt_tool_return)
    tool = _make_langchain_tool("dummy_tool", _tool, _DummyInput)

    with workspace_scope(tmp_path):
        out = tool.invoke({"text": "x"})

    assert out == "ok"
    assert captured["workspace_files_root"] == workspace_root(tmp_path)


def test_wrapper_sets_toolcall_context_from_runtime(tmp_path) -> None:
    observed: dict[str, str] = {}
    run_dir = tmp_path / "metadata" / "runs" / "run_01"

    def _tool(payload: dict) -> tuple[str, dict]:
        observed["toolcall_key"] = current_toolcall_key()
        observed["run_dir"] = current_run_dir()
        return (
            payload.get("text", ""),
            {
                "tool_name": "dummy_tool",
                "data": {"summary": payload.get("text", "")},
            },
        )

    tool = _make_langchain_tool(
        "dummy_tool",
        _tool,
        _DummyInput,
        run_dir=str(run_dir),
        workspace=str(tmp_path),
    )

    runtime = SimpleNamespace(tool_call_id="call_ctx_001", context={})
    out = tool.func(text="hello", runtime=runtime)

    assert out[0]
    assert observed["toolcall_key"] == "call_ctx_001"
    assert observed["run_dir"] == str(run_dir)


def test_async_only_wrapper_uses_coroutine_and_runtime_context(tmp_path) -> None:
    observed: dict[str, str] = {}
    run_dir = tmp_path / "metadata" / "runs" / "run_01"

    async def _tool(payload: dict) -> tuple[str, dict]:
        observed["toolcall_key"] = current_toolcall_key()
        observed["run_dir"] = current_run_dir()
        return (
            payload.get("text", ""),
            {
                "tool_name": "dummy_tool_async",
                "data": {"summary": payload.get("text", "")},
            },
        )

    tool = _make_langchain_tool(
        "dummy_tool_async",
        None,
        _DummyInput,
        coroutine=_tool,
        run_dir=str(run_dir),
        workspace=str(tmp_path),
    )

    runtime = SimpleNamespace(tool_call_id="call_async_001", context={})
    out = asyncio.run(tool.coroutine(text="hello", runtime=runtime))

    assert out[0]
    assert observed["toolcall_key"] == "call_async_001"
    assert observed["run_dir"] == str(run_dir)
    with pytest.raises(NotImplementedError):
        tool.invoke({"text": "x"})
