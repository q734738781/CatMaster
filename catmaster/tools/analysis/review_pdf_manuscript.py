from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class ReviewPdfManuscriptInput(BaseModel):
    """[writing/review] Review a local manuscript PDF directly with a multimodal model and return comment-only suggestions."""

    pdf_path: str = Field(..., description="Workspace-relative path to the manuscript PDF.")
    focus: str | None = Field(
        None,
        description="Optional concise review focus, such as publication readiness, figure logic, or title/abstract alignment.",
    )
    context_text: str | None = Field(
        None,
        description="Optional concise context about venue, audience, or the author's revision goals.",
    )
    model: str | None = Field(
        None,
        description=(
            "Optional model override. Accepts either a configured llm.yaml model label or a raw model id; "
            "default uses the academic_polisher model config."
        ),
    )


def _resolve_model_config(model_override: str | None):
    profile = LLMProfile.from_env_or_file()
    base_cfg = profile.config_for_role("academic_polisher")
    override = str(model_override or "").strip()
    if not override:
        return base_cfg
    if override in profile.models:
        return profile.models[override]
    return replace(base_cfg, model=override)


def _pdf_data_block(path: Path) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "source_type": "base64",
        "mime_type": "application/pdf",
        "data": encoded,
        "filename": path.name,
    }


def _build_query_text(*, pdf_ref: str, focus: str | None, context_text: str | None) -> str:
    lines = [
        "Review the provided manuscript PDF directly.",
        "Treat the PDF pages, figures, captions, ordering, and visible layout as primary evidence.",
        "Return comment-only publication-readiness suggestions. Do not rewrite the manuscript unless explicitly asked.",
        "Focus on whole-manuscript issues: title/abstract/conclusion alignment, novelty clarity, Results versus Discussion separation, figure logic, weak openings/closings, self-undermining filler, and places where the strongest defensible claim is not stated clearly enough.",
        "Keep limitation comments brief and selective. Mention them only when they materially affect interpretation, scope, or reviewer confidence.",
        "Respond in natural language with a compact prioritized diagnostic.",
        "",
        f"PDF: {pdf_ref}",
    ]
    if focus:
        lines.extend(["", f"Review focus: {focus.strip()}"])
    if context_text:
        lines.extend(["", "Context:", str(context_text).strip()])
    return "\n".join(lines).strip()


def review_pdf_manuscript(payload: dict) -> tuple[str, dict]:
    """[writing/review] Review a local manuscript PDF directly with a multimodal model and return comment-only suggestions."""
    tool_name = "review_pdf_manuscript"
    try:
        params = ReviewPdfManuscriptInput(**payload)
        pdf_path = resolve_workspace_path(params.pdf_path, must_exist=True)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("pdf_path must point to a .pdf file")
        pdf_ref = workspace_relpath(pdf_path)
        query_text = _build_query_text(
            pdf_ref=pdf_ref,
            focus=params.focus,
            context_text=params.context_text,
        )
        cfg = _resolve_model_config(params.model)
        model = build_chat_model(cfg)
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a manuscript reviewer. "
                        "Review the provided paper PDF directly, including figures and visible page content. "
                        "Return concise comment-only publication-readiness suggestions in natural language. "
                        "Do not rewrite the manuscript text unless explicitly asked."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_text},
                        _pdf_data_block(pdf_path),
                    ],
                },
            ]
        )
        response_text = content_to_text(getattr(response, "content", response)).strip()
        answer = response_text or "No usable manuscript review returned."
        data = {
            "answer": answer,
            "review_text": answer,
            "model_name": str(getattr(cfg, "model", "") or ""),
            "pdf_path": pdf_ref,
            "focus": str(params.focus or "").strip(),
        }
        return answer, {"tool_name": tool_name, "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"pdf_path": payload.get("pdf_path")}},
            error_code="review_pdf_manuscript_failed",
        ) from exc


__all__ = ["ReviewPdfManuscriptInput", "review_pdf_manuscript"]
