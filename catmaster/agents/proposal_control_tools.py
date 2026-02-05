from __future__ import annotations

from typing import Any

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


__all__ = [
    "ProposalFinishInput",
    "ProposalFailInput",
    "PROPOSAL_CONTROL_TOOL_NAMES",
    "get_proposal_control_tool_schemas",
]
