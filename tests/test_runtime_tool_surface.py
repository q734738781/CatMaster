from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_core.tools import StructuredTool

pytest.importorskip("langchain_core")

from catmaster.runtime.run_context import RunContext
from catmaster.runtime.tool_surface import build_runtime_tool_surface


@dataclass
class _Tool:
    name: str


class _DummyInputModel:
    __doc__ = "Dummy local tool input model."


def _fake_literature_impl(payload):
    return (
        f"role={payload.get('role')}",
        {"tool_name": "run_literature_research", "data": dict(payload)},
    )


class _Registry:
    def as_langchain_tools(self, *, run_dir=None, workspace=None):
        _ = (run_dir, workspace)
        return [
            _Tool("write_note"),
            _Tool("read_file"),
            _Tool("list_directory_with_sizes"),
            _Tool("get_file_info"),
            StructuredTool.from_function(
                func=lambda runtime=None, **kwargs: _fake_literature_impl(kwargs),
                name="run_literature_research",
                description="Run literature research.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "depth": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                infer_schema=False,
                response_format="content_and_artifact",
            ),
            _Tool("create_molecule_from_smiles"),
            _Tool("apply_aider_edits"),
        ]

    def get_tool_info(self, name: str):
        _ = name
        return {"input_model": _DummyInputModel}


def test_build_runtime_tool_surface_role_split(tmp_path) -> None:
    run_ctx = RunContext.create(
        workspace=tmp_path,
        model_name="dummy-model",
    )
    surface = build_runtime_tool_surface(
        registry=_Registry(),
        run_context=run_ctx,
        run_dir=run_ctx.run_dir,
        task_runner_denylist=set(),
    )

    proposal_names = [tool.name for tool in surface.proposal_tools]
    director_names = [tool.name for tool in surface.director_tools]
    fast_director_names = [tool.name for tool in surface.fast_director_tools]
    task_names = [tool.name for tool in surface.task_tools]

    assert "run_literature_research" in proposal_names
    assert "read_file" not in proposal_names
    assert "list_directory_with_sizes" not in proposal_names
    assert "get_file_info" not in proposal_names
    assert "apply_aider_edits" not in proposal_names
    assert "write_note" not in proposal_names

    assert "run_literature_research" in director_names
    assert "apply_aider_edits" in director_names
    assert "write_note" not in director_names

    assert "run_literature_research" in fast_director_names
    assert "apply_aider_edits" not in fast_director_names
    assert "write_note" not in fast_director_names

    assert "apply_aider_edits" in task_names
    assert "create_molecule_from_smiles" in task_names
    assert "run_literature_research" in task_names
    assert "write_note" not in task_names
    assert "read_file" not in task_names
    assert "list_directory_with_sizes" not in task_names
    assert "get_file_info" not in task_names

    assert "Task runner standard capabilities" in surface.task_runner_capability_guide_full
    assert "Task runner capabilities (short)" in surface.task_runner_capability_guide_short

    proposal_lit = next(tool for tool in surface.proposal_tools if tool.name == "run_literature_research")
    director_lit = next(tool for tool in surface.director_tools if tool.name == "run_literature_research")
    fast_director_lit = next(tool for tool in surface.fast_director_tools if tool.name == "run_literature_research")
    task_lit = next(tool for tool in surface.task_tools if tool.name == "run_literature_research")

    proposal_result = proposal_lit.func(query="papers please")
    director_result = director_lit.func(query="papers please")
    fast_director_result = fast_director_lit.func(query="papers please")
    task_result = task_lit.func(query="papers please")

    assert proposal_result[1]["data"]["role"] == "proposal"
    assert director_result[1]["data"]["role"] == "director"
    assert fast_director_result[1]["data"]["role"] == "fast_director"
    assert task_result[1]["data"]["role"] == "task_runner"
    assert "role" not in ((proposal_lit.args_schema or {}).get("properties") or {})
    assert "role" not in ((task_lit.args_schema or {}).get("properties") or {})
