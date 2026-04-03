from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_DEFAULT_GEMINI_PEER_REVIEW_MODEL = "google/gemini-3.1-pro"


class PeerReviewPdfManuscriptInput(BaseModel):
    """[writing/review] Run a lightweight external-model peer review on one local manuscript PDF."""

    pdf_path: str = Field(..., description="Workspace-relative path to the manuscript PDF.")
    user_request_context: str = Field(..., description="Compact original user request or author intent.")
    round_index: int = Field(1, ge=1, description="Current peer-review round number, starting from 1.")
    max_rounds: int = Field(1, ge=1, description="Maximum peer-review rounds requested for this run.")
    prior_review_summary: str | None = Field(
        None,
        description="Optional concise summary of prior review concerns to avoid repeating the same points.",
    )
    model: str | None = Field(
        None,
        description=(
            "Optional model override. Accepts either a configured llm.yaml model label or a raw model id; "
            f"default uses `{_DEFAULT_GEMINI_PEER_REVIEW_MODEL}`."
        ),
    )


def _resolve_model_config(model_override: str | None):
    profile = LLMProfile.from_env_or_file()
    base_cfg = profile.config_for_role("academic_polisher")
    override = str(model_override or "").strip() or _DEFAULT_GEMINI_PEER_REVIEW_MODEL
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


def _build_query_text(
    *,
    pdf_ref: str,
    user_request_context: str,
    round_index: int,
    max_rounds: int,
    prior_review_summary: str | None,
) -> str:
    lines = [
        "Review the provided manuscript PDF as an external peer reviewer.",
        "Treat the PDF pages, figures, tables, captions, ordering, and visible layout as primary evidence.",
        "You only know the user request context below and the attached PDF. Do not assume hidden workspace context.",
        "Focus on scientific soundness, claim-evidence fit, missing controls or validation logic, overclaiming, weak comparisons, figure logic, and whether the manuscript would concern a serious reviewer.",
        "Be concise and prioritized.",
        "Return plain text with exactly four short sections:",
        "1. Verdict",
        "2. Major concerns",
        "3. Minor concerns",
        "4. Recommended revision focus",
        "",
        f"Review round: {int(round_index)}/{int(max_rounds)}",
        f"User request context: {str(user_request_context or '').strip()}",
        f"PDF: {pdf_ref}",
    ]
    prior = str(prior_review_summary or "").strip()
    if prior:
        lines.extend(["", f"Prior review summary: {prior}"])
    return "\n".join(lines).strip()


def peer_review_pdf_manuscript(payload: dict) -> tuple[str, dict]:
    """[writing/review] Run a lightweight external-model peer review on one local manuscript PDF."""
    tool_name = "peer_review_pdf_manuscript"
    try:
        params = PeerReviewPdfManuscriptInput(**payload)
        pdf_path = resolve_workspace_path(params.pdf_path, must_exist=True)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("pdf_path must point to a .pdf file")
        pdf_ref = workspace_relpath(pdf_path)
        query_text = _build_query_text(
            pdf_ref=pdf_ref,
            user_request_context=params.user_request_context,
            round_index=params.round_index,
            max_rounds=params.max_rounds,
            prior_review_summary=params.prior_review_summary,
        )
        cfg = _resolve_model_config(params.model)
        model = build_chat_model(cfg)
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an external scientific peer reviewer. "
                        "Review the attached paper PDF directly. "
                        "Prioritize scientific issues over prose polish. "
                        "Do not rewrite the manuscript. "
                        "Return concise review notes only."
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
        review_text = content_to_text(getattr(response, "content", response)).strip()
        answer = review_text or "No usable peer review returned."
        data = {
            "review_text": answer,
            "answer": answer,
            "model_name": str(getattr(cfg, "model", "") or ""),
            "pdf_path": pdf_ref,
            "round_index": int(params.round_index),
            "max_rounds": int(params.max_rounds),
        }
        return answer, {"tool_name": tool_name, "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"pdf_path": payload.get("pdf_path")}},
            error_code="peer_review_pdf_manuscript_failed",
        ) from exc


__all__ = ["PeerReviewPdfManuscriptInput", "peer_review_pdf_manuscript"]
