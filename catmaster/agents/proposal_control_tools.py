from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from catmaster.tools.registry import sanitize_json_schema


class ProposalFinishInput(BaseModel):
    """Return the finalized proposal and work packages."""

    proposal_md: str = Field(..., description="Full proposal in markdown format.")
    work_packages: list[str] = Field(..., description="Ordered list of work packages.")


class ProposalFailInput(BaseModel):
    """Signal that proposal creation failed and needs human intervention."""

    error: str = Field(..., description="Summary of the proposal failure.")
    needs_human: bool = Field(default=True, description="Whether a human must intervene.")


PROPOSAL_CONTROL_TOOL_NAMES = {"proposal_finish", "proposal_fail"}


def _schema_for(name: str, model: type[BaseModel], *, strict: bool) -> dict[str, Any]:
    description = (model.__doc__ or f"Input for {name}").strip()
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": sanitize_json_schema(model.model_json_schema()),
        "strict": strict,
    }


def get_proposal_control_tool_schemas(*, strict: bool = False) -> list[dict]:
    return [
        _schema_for("proposal_finish", ProposalFinishInput, strict=strict),
        _schema_for("proposal_fail", ProposalFailInput, strict=strict),
    ]


def _make_control_tool(name: str, model: type[BaseModel]) -> StructuredTool:
    description = (model.__doc__ or f"Input for {name}").strip()

    def _tool(**kwargs: Any) -> str:
        payload = model.model_validate(kwargs).model_dump(mode="json", exclude_none=True)
        return json.dumps(
            {
                "status": "control",
                "tool_name": name,
                "payload": payload,
            },
            ensure_ascii=False,
        )

    _tool.__name__ = name
    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
        args_schema=model,
        return_direct=True,
    )


def as_langchain_control_tools() -> list[StructuredTool]:
    return [
        _make_control_tool("proposal_finish", ProposalFinishInput),
        _make_control_tool("proposal_fail", ProposalFailInput),
    ]


__all__ = [
    "ProposalFinishInput",
    "ProposalFailInput",
    "PROPOSAL_CONTROL_TOOL_NAMES",
    "get_proposal_control_tool_schemas",
    "as_langchain_control_tools",
]
