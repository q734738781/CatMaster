from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, adapt_tool_return
from catmaster.runtime.tool_output_config import ToolOutputConfig
from catmaster.tools.base import compact_list_for_artifact, ensure_project_space_layout
from catmaster.tools.registry import ToolRegistry


class _LargeRegisteredOutputInput(BaseModel):
    value: str = Field("x")


def _large_registered_output(_payload: dict):
    return (
        "registered large output",
        {
            "tool_name": "large_registered_output",
            "suppress_content_offload_ref": True,
            "data": {
                "stdout": "stdout-value-" * 100,
                "records": [{"index": index, "value": "record-value" * 10} for index in range(20)],
            },
        },
    )


def test_adapt_tool_return_offload_refs_are_unique_per_call(tmp_path) -> None:
    config = ToolOutputConfig(offload_chars=1)
    raw_result = ("done", {"tool_name": "dummy_tool", "data": {"summary": "done", "value": "x"}})

    _, artifact_1 = adapt_tool_return(
        tool_name="dummy_tool",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )
    _, artifact_2 = adapt_tool_return(
        tool_name="dummy_tool",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )

    offload_1 = artifact_1["offload_refs"][0]
    offload_2 = artifact_2["offload_refs"][0]

    assert offload_1 != offload_2
    assert (tmp_path / offload_1).exists()
    assert (tmp_path / offload_2).exists()


def test_adapt_tool_return_preserves_tool_content(tmp_path) -> None:
    config = ToolOutputConfig(
        offload_chars=20_000,
        preview_chars=200,
    )
    raw_result = (
        "mp_search_materials completed.\nreturned=3 output_csv_rel=retrieval/mp.csv",
        {
            "tool_name": "mp_search_materials",
            "data": {
                "count": 3,
                "returned": 3,
                "output_csv_rel": "retrieval/mp.csv",
                "preview_rows": [{"material_id": "mp-149", "band_gap": 1.1}],
            },
        },
    )
    content, artifact = adapt_tool_return(
        tool_name="mp_search_materials",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )

    assert "returned=3" in str(content)
    assert "output_csv_rel" in str(content)
    assert isinstance(artifact.get("data"), dict)


def test_adapt_tool_return_offloads_large_fields_by_hard_limit(tmp_path) -> None:
    config = ToolOutputConfig(
        offload_chars=256,
        preview_chars=128,
    )
    raw_result = (
        "bash completed.",
        {
            "tool_name": "bash",
            "data": {
                "stdout": "x" * 800,
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "cwd": ".",
            },
        },
    )

    content, artifact = adapt_tool_return(
        tool_name="bash",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )

    data = artifact.get("data") or {}
    if isinstance(data.get("stdout"), dict):
        stdout_field = data.get("stdout")
        assert "offload_ref" in stdout_field
        assert (tmp_path / stdout_field["offload_ref"]).exists()
        assert str(content) == "bash completed."
    else:
        refs = artifact.get("offload_refs") or []
        assert refs
        assert (tmp_path / refs[0]).exists()
        assert "Offload:" in str(content)


def test_adapt_tool_return_offload_preserves_observability_metadata(tmp_path) -> None:
    config = ToolOutputConfig(offload_chars=1)
    raw_result = (
        "done",
        {
            "tool_name": "dummy_tool",
            "warnings": ["keep-warning"],
            "data": {
                "summary": "done",
                "value": "x" * 600,
            },
        },
    )

    content, artifact = adapt_tool_return(
        tool_name="dummy_tool",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )

    assert "Offload:" in str(content)
    assert artifact.get("warnings") == ["keep-warning"]
    assert "tool_args" not in artifact
    refs = artifact.get("offload_refs") or []
    assert len(refs) == 1
    offload_path = tmp_path / refs[0]
    assert offload_path.exists()
    payload = json.loads(offload_path.read_text(encoding="utf-8"))
    assert payload.get("tool_name") == "dummy_tool"
    assert "tool_args" not in payload


def test_adapt_tool_return_never_copies_tool_args_into_output_artifact(tmp_path) -> None:
    config = ToolOutputConfig(offload_chars=20_000)
    raw_result = (
        "done",
        {
            "tool_name": "dummy_tool",
            "tool_args": {"text": "top-level"},
            "raw_params": {"text": "top-level-raw"},
            "validated_params": {"text": "top-level-validated"},
            "data": {
                "summary": "done",
                "tool_args": {"text": "nested"},
                "raw_params": {"text": "nested-raw"},
                "validated_params": {"text": "nested-validated"},
            },
        },
    )

    _content, artifact = adapt_tool_return(
        tool_name="dummy_tool",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )

    for key in ("tool_args", "raw_params", "validated_params"):
        assert key not in artifact
        assert key not in artifact["data"]


def test_adapt_tool_return_can_suppress_content_offload_ref(tmp_path) -> None:
    config = ToolOutputConfig(offload_chars=1)
    raw_result = (
        "literature summary with inline refs",
        {
            "tool_name": "query_literature_corpus",
            "suppress_content_offload_ref": True,
            "data": {
                "summary": "x" * 600,
                "key_papers": [{"title": "Paper A", "year": 2020}],
            },
        },
    )

    content, artifact = adapt_tool_return(
        tool_name="query_literature_corpus",
        raw_result=raw_result,
        workspace_files_root=tmp_path,
        output_config=config,
    )

    assert str(content).startswith("literature summary with inline refs")
    assert "Field offload refs:" in str(content)
    for item in artifact.get("field_offload_refs") or []:
        assert item["field"] in str(content)
        assert item["offload_ref"] in str(content)
    refs = artifact.get("offload_refs") or []
    assert len(refs) == 1
    assert (tmp_path / refs[0]).exists()


def test_registered_toolmessage_content_exposes_every_field_offload_ref(
    tmp_path,
    monkeypatch,
) -> None:
    layout = ensure_project_space_layout(tmp_path, create=True)
    monkeypatch.setattr(
        "catmaster.runtime.tool_output_adapter.get_tool_output_config",
        lambda: ToolOutputConfig(offload_chars=256, preview_chars=128),
    )
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool(
        "large_registered_output",
        _large_registered_output,
        _LargeRegisteredOutputInput,
    )
    tool = registry.as_langchain_tools(
        allowlist=["large_registered_output"],
        workspace=str(tmp_path),
    )[0]

    message = tool.invoke(
        {
            "name": "large_registered_output",
            "args": {"value": "x"},
            "id": "registered-large-output",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    refs = list((message.artifact or {}).get("field_offload_refs") or [])
    assert {item["field"] for item in refs} == {"stdout", "records"}
    for item in refs:
        assert item["field"] in str(message.content)
        assert item["offload_ref"] in str(message.content)
        payload = json.loads(
            (layout["files_root"] / item["offload_ref"]).read_text(encoding="utf-8")
        )
        assert payload["field"] == item["field"]


def test_compact_list_requires_and_preserves_exact_manifest_ref(tmp_path) -> None:
    items = [{"index": index, "value": f"item-{index}"} for index in range(7)]
    manifest = tmp_path / "batch_state.json"
    manifest.write_text(json.dumps(items), encoding="utf-8")

    with pytest.raises(ValueError, match="exact manifest"):
        compact_list_for_artifact(
            items,
            count_key="items_count",
            inline_key="items",
            preview_key="items_preview",
            truncated_key="items_truncated",
            max_inline=3,
        )

    compact = compact_list_for_artifact(
        items,
        count_key="items_count",
        inline_key="items",
        preview_key="items_preview",
        truncated_key="items_truncated",
        full_rel_key="items_full_rel",
        full_rel=manifest.name,
        max_inline=3,
    )

    assert compact == {
        "items_count": 7,
        "items_full_rel": "batch_state.json",
        "items_preview": items[:3],
        "items_truncated": 4,
    }
    assert json.loads((tmp_path / compact["items_full_rel"]).read_text()) == items


@pytest.mark.parametrize(
    "raw_result",
    [
        {"status": "success", "tool_name": "dummy", "data": {}},
        "plain text",
        [{"type": "text", "text": "line one"}],
        None,
    ],
)
def test_adapt_tool_return_rejects_non_tuple_returns(tmp_path, raw_result) -> None:
    with pytest.raises(CatMasterToolExecutionError):
        adapt_tool_return(
            tool_name="dummy_tool",
            raw_result=raw_result,
            workspace_files_root=tmp_path,
        )


def test_adapt_tool_return_accepts_toolmessage_success(tmp_path) -> None:
    message = ToolMessage(
        content=[{"type": "text", "text": "native block"}],
        artifact={"raw_kind": "tool_message"},
        tool_call_id="call_001",
        name="dummy_tool",
    )
    content, artifact = adapt_tool_return(
        tool_name="dummy_tool",
        raw_result=message,
        workspace_files_root=tmp_path,
    )

    assert isinstance(content, list)
    assert content[0]["text"] == "native block"
    assert artifact.get("raw_kind") == "tool_message"


def test_adapt_tool_return_rejects_toolmessage_error(tmp_path) -> None:
    message = ToolMessage(
        content="boom",
        artifact={"error": "boom"},
        tool_call_id="call_002",
        name="dummy_tool",
        status="error",
    )
    with pytest.raises(CatMasterToolExecutionError):
        adapt_tool_return(
            tool_name="dummy_tool",
            raw_result=message,
            workspace_files_root=tmp_path,
        )
