from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from catmaster.tools.registry import ToolRegistry
from catmaster.tools.retrieval import matdb
from catmaster.tools.retrieval.matdb import MPSearchMaterialsInput, mp_search_materials


class _FakeSummaryClient:
    @staticmethod
    def count(_criteria: dict[str, object]) -> int:
        return 100

    @staticmethod
    def search(**_kwargs):
        for index in range(100):
            yield {
                "material_id": f"mp-{index}",
                "formula_pretty": f"X{index}",
            }


class _FakeMPRester:
    def __init__(self) -> None:
        self.materials = SimpleNamespace(summary=_FakeSummaryClient())

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def _registered_tool(workspace: Path):
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool(
        "mp_search_materials",
        mp_search_materials,
        MPSearchMaterialsInput,
    )
    return registry.as_langchain_tools(
        allowlist=["mp_search_materials"],
        workspace=str(workspace),
    )[0]


def _invoke(tool, *, tool_call_id: str, output_csv: str, limit: int | None):
    args: dict[str, object] = {
        "criteria": {"elements": ["Si"]},
        "fields": ["material_id", "formula_pretty"],
        "output_csv": output_csv,
    }
    if limit is not None:
        args["limit"] = limit
    result = tool.invoke(
        {
            "name": "mp_search_materials",
            "args": args,
            "id": tool_call_id,
            "type": "tool_call",
        }
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    return result


def test_final_mp_tool_writes_more_than_fifty_agent_selected_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(matdb, "_mpr", lambda **_kwargs: _FakeMPRester())
    tool = _registered_tool(tmp_path)

    result = _invoke(
        tool,
        tool_call_id="mp-explicit-75",
        output_csv="retrieval/explicit_75.csv",
        limit=75,
    )

    data = result.artifact["data"]
    assert data["returned"] == 75
    assert data["requested_limit"] == 75
    assert data["limit_source"] == "agent_selected"
    assert data["source_completeness"] == "explicit_subset"
    with (tmp_path / "files" / data["output_csv_rel"]).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 75


def test_final_mp_tool_labels_omitted_limit_as_visible_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(matdb, "_mpr", lambda **_kwargs: _FakeMPRester())
    tool = _registered_tool(tmp_path)

    result = _invoke(
        tool,
        tool_call_id="mp-visible-default",
        output_csv="retrieval/default.csv",
        limit=None,
    )

    data = result.artifact["data"]
    assert data["returned"] == 50
    assert data["requested_limit"] == 50
    assert data["limit_source"] == "visible_default"
    assert data["source_completeness"] == "visible_default_subset"
    assert "limit_source=visible_default" in str(result.content)
