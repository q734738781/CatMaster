from __future__ import annotations

import pytest
from dataclasses import dataclass

pytest.importorskip("langchain_core")
from langchain_core.tools import StructuredTool

from catmaster.runtime.run_context import RunContext
from catmaster.runtime.tool_surface import build_runtime_tool_surface


@dataclass
class _Tool:
    name: str


class _DummyInputModel:
    __doc__ = "Dummy local tool input model."


def _fake_literature_impl(payload):
    return ("ok", {"tool_name": "run_literature_research", "data": dict(payload)})


class _Registry:
    def as_langchain_tools(self, *, run_dir=None, workspace=None):
        _ = (run_dir, workspace)
        return [
            StructuredTool.from_function(
                func=lambda runtime=None, **kwargs: _fake_literature_impl(kwargs),
                name="run_literature_research",
                description="Run literature research.",
                args_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "role": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                infer_schema=False,
                response_format="content_and_artifact",
            ),
            _Tool("create_molecule_from_smiles"),
        ]

    def get_tool_info(self, name: str):
        _ = name
        return {"input_model": _DummyInputModel}


def test_build_runtime_tool_surface_can_disable_literature_tool(tmp_path) -> None:
    run_ctx = RunContext.create(workspace=tmp_path, model_name="dummy-model")
    surface = build_runtime_tool_surface(
        registry=_Registry(),
        run_context=run_ctx,
        run_dir=run_ctx.run_dir,
        include_literature_tool=False,
    )

    assert "run_literature_research" not in [tool.name for tool in surface.proposal_tools]
    assert "run_literature_research" not in [tool.name for tool in surface.director_tools]
    assert "run_literature_research" not in [tool.name for tool in surface.fast_director_tools]
    assert "run_literature_research" not in [tool.name for tool in surface.task_tools]
