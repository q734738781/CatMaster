from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.specialists.schemas import ResearchKernel, ResearchRunCard
from catmaster.tools.base import resolve_workspace_path, workspace_root


class ResearchRunCardInput(BaseModel):
    """Compact subagent result card for the Research Kernel."""

    source: str = Field("", description="Subagent or source that produced the card.")
    summary: str = Field("", description="Short execution summary.")
    facts: list[str] = Field(default_factory=list, description="Decision-relevant facts only.")
    artifacts: list[str] = Field(default_factory=list, description="Relevant artifact paths only.")


class UpdateResearchKernelInput(BaseModel):
    """Update the lightweight Research Kernel JSON used by ResearchSpecialist."""

    kernel_path: str = Field(..., description="Workspace-relative path to the kernel JSON file.")
    question: str | None = Field(None, description="Replace the current research question.")
    hypotheses: list[str] | None = Field(
        None,
        description="Replace active hypotheses with a short 3-5 item list.",
    )
    append_run_card: ResearchRunCardInput | None = Field(
        None,
        description="Append one compact run card returned from a subagent.",
    )
    replace_run_cards: list[ResearchRunCardInput] | None = Field(
        None,
        description="Replace the full run card list.",
    )
    frontier: list[str] | None = Field(
        None,
        description="Replace the current frontier list of next questions/actions.",
    )
    conclusion_draft: str | None = Field(
        None,
        description="Replace the current conclusion draft.",
    )
    max_run_cards: int = Field(
        12,
        ge=1,
        le=50,
        description="Maximum number of run cards to retain after update.",
    )


def _normalize_text_list(values: list[str] | None, *, max_items: int | None = None) -> list[str]:
    out: list[str] = []
    for item in list(values or []):
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _kernel_file(path_text: str) -> Path:
    target = resolve_workspace_path(path_text, must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _load_kernel(path: Path) -> ResearchKernel:
    if not path.exists():
        return ResearchKernel()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="update_research_kernel",
            public_message=f"update_research_kernel failed: invalid existing kernel JSON: {exc}",
            artifact={"path": str(path)},
            error_code="invalid_research_kernel",
        ) from exc
    try:
        return ResearchKernel.model_validate(payload)
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="update_research_kernel",
            public_message=f"update_research_kernel failed: existing kernel does not match schema: {exc}",
            artifact={"path": str(path)},
            error_code="invalid_research_kernel_schema",
        ) from exc


def update_research_kernel(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "update_research_kernel"
    try:
        params = UpdateResearchKernelInput(**payload)
        path = _kernel_file(params.kernel_path)
        kernel = _load_kernel(path)

        if params.question is not None:
            kernel.question = str(params.question or "").strip()
        if params.hypotheses is not None:
            kernel.hypotheses = _normalize_text_list(params.hypotheses, max_items=5)
        if params.replace_run_cards is not None:
            kernel.run_cards = [
                ResearchRunCard.model_validate(item.model_dump())
                for item in params.replace_run_cards[: params.max_run_cards]
            ]
        if params.append_run_card is not None:
            kernel.run_cards.append(ResearchRunCard.model_validate(params.append_run_card.model_dump()))
            kernel.run_cards = kernel.run_cards[-int(params.max_run_cards) :]
        if params.frontier is not None:
            kernel.frontier = _normalize_text_list(params.frontier)
        if params.conclusion_draft is not None:
            kernel.conclusion_draft = str(params.conclusion_draft or "").strip()

        kernel.hypotheses = _normalize_text_list(kernel.hypotheses, max_items=5)
        kernel.run_cards = kernel.run_cards[-int(params.max_run_cards) :]

        path.write_text(kernel.model_dump_json(indent=2), encoding="utf-8")
        relpath = str(path.relative_to(workspace_root())).replace("\\", "/")
        content = (
            f"Updated research kernel: {relpath}\n"
            f"question: {kernel.question or '(empty)'}\n"
            f"hypotheses: {len(kernel.hypotheses)}\n"
            f"run_cards: {len(kernel.run_cards)}\n"
            f"frontier: {len(kernel.frontier)}"
        )
        artifact = {
            "tool_name": tool_name,
            "data": {
                "kernel_path": relpath,
                "kernel": kernel.model_dump(),
            },
            "suppress_content_offload_ref": True,
        }
        return content, artifact
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"kernel_path": payload.get("kernel_path")}},
            error_code="update_research_kernel_failed",
        ) from exc


__all__ = [
    "ResearchRunCardInput",
    "UpdateResearchKernelInput",
    "update_research_kernel",
]
