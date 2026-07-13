from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import ToolMessage

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, adapt_tool_return
from catmaster.runtime.tool_output_config import ToolOutputConfig


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

    assert str(content) == "literature summary with inline refs"
    refs = artifact.get("offload_refs") or []
    assert len(refs) == 1
    assert (tmp_path / refs[0]).exists()


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
