from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("langchain_core")
from langchain_core.messages import ToolMessage

from catmaster.runtime import ArtifactStore, TraceStore, ToolExecutor
import catmaster.runtime.local_tool_backend as local_backend_module
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.tools.base import ensure_project_space_layout, workspace_root
from catmaster.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    """Dummy tool input."""

    text: str = Field(..., description="Text to echo")


def dummy_tool(payload: dict) -> tuple[str, dict]:
    return (
        "dummy_tool completed.",
        {
            "tool_name": "dummy_tool",
            "data": {"text": payload.get("text")},
        },
    )


def test_local_tool_backend_call(tmp_path) -> None:
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("dummy_tool", dummy_tool, DummyInput)

    tool_executor = ToolExecutor(registry)
    artifact_store = ArtifactStore(tmp_path)
    trace_store = TraceStore(tmp_path)
    backend = LocalToolBackend(
        registry=registry,
        tool_executor=tool_executor,
        artifact_store=artifact_store,
        trace_store=trace_store,
    )

    output = backend.call(
        "dummy_tool",
        json.dumps({"text": "hello"}),
        toolcall_key="call-1",
        call_id="call-1",
    )

    assert isinstance(output, ToolMessage)
    assert output.status == "success"
    input_path = tmp_path / "toolcalls" / "call-1" / "input.json"
    output_path = tmp_path / "toolcalls" / "call-1" / "output.json"
    assert input_path.exists()
    assert output_path.exists()


def test_local_tool_backend_passes_workspace_files_root(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)

    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("dummy_tool", dummy_tool, DummyInput)

    tool_executor = ToolExecutor(registry)
    run_dir = workspace / "metadata" / "runs" / "run_01"
    artifact_store = ArtifactStore(run_dir)
    trace_store = TraceStore(run_dir)
    backend = LocalToolBackend(
        registry=registry,
        tool_executor=tool_executor,
        artifact_store=artifact_store,
        trace_store=trace_store,
        workspace=workspace,
    )

    captured: dict[str, object] = {}

    def _fake_adapt_tool_return(
        *,
        tool_name,
        raw_result,
        tool_args=None,
        workspace_files_root=None,
        output_config=None,
    ):
        captured["workspace_files_root"] = workspace_files_root
        return "done", {"tool_name": tool_name}

    monkeypatch.setattr(local_backend_module, "adapt_tool_return", _fake_adapt_tool_return)
    output = backend.call(
        "dummy_tool",
        json.dumps({"text": "hello"}),
        toolcall_key="call-2",
        call_id="call-2",
    )

    assert output.status == "success"
    assert captured["workspace_files_root"] == workspace_root(workspace)


def test_local_tool_backend_returns_error_for_async_tool_callable(tmp_path) -> None:
    async def _async_tool(payload: dict) -> tuple[str, dict]:
        return (
            "async_tool completed.",
            {"tool_name": "async_tool", "data": {"text": payload.get("text")}},
        )

    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("async_tool", _async_tool, DummyInput)

    tool_executor = ToolExecutor(registry)
    artifact_store = ArtifactStore(tmp_path)
    trace_store = TraceStore(tmp_path)
    backend = LocalToolBackend(
        registry=registry,
        tool_executor=tool_executor,
        artifact_store=artifact_store,
        trace_store=trace_store,
    )

    output = backend.call(
        "async_tool",
        json.dumps({"text": "hello"}),
        toolcall_key="call-async-1",
        call_id="call-async-1",
    )

    assert output.status == "error"
    assert "sync-only" in str(output.content)
