from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

class PeerReviewRequestInput(BaseModel):
    """[writing/review] Send one manuscript PDF to all configured peer-review models and collect raw reviews."""

    pdf_path: str = Field(..., description="Workspace-relative path to the manuscript PDF.")
    review_request: str = Field(
        "",
        description="Compact editor-facing review request, including user constraints or evaluation focus.",
    )


def _resolve_model_configs(model_labels: list[str]) -> list[tuple[str, LLMConfig]]:
    profile = LLMProfile.from_env_or_file()
    requested = [str(item or "").strip() for item in list(model_labels or []) if str(item or "").strip()]
    if not requested:
        requested = list(profile.peer_review_models or [])
    if not requested:
        requested = [profile.label_for_role("academic_polisher")]
    return [(label, profile.models[label]) for label in requested]


def _pdf_data_block(path: Path) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "source_type": "base64",
        "mime_type": "application/pdf",
        "data": encoded,
        "filename": path.name,
    }


def _review_prompt(*, pdf_ref: str, review_request: str) -> str:
    request = str(review_request or "").strip()
    lines = [
        "Review the attached manuscript PDF as an external scientific reviewer for an ACS-style journal.",
        "Judge scientific soundness, evidence-claim fit, novelty positioning, controls, validation sufficiency, comparison quality, figure logic, and whether the work meets a serious reviewer threshold.",
        "Be tough but fair. Prioritize scientific weaknesses over prose polish.",
        "Return plain text only with exactly these section headings:",
        "General Comments",
        "Major Comments",
        "Minor Comments",
        "",
        f"PDF: {pdf_ref}",
    ]
    if request:
        lines.extend(["", f"Editor request: {request}"])
    return "\n".join(lines).strip()


def peer_review_request(payload: dict) -> tuple[str, dict]:
    """[writing/review] Send one manuscript PDF to all configured peer-review models and collect raw reviews."""
    tool_name = "peer_review_request"
    try:
        params = PeerReviewRequestInput(**payload)
        profile = LLMProfile.from_env_or_file()
        pdf_path = resolve_workspace_path(params.pdf_path, must_exist=True)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("pdf_path must point to a .pdf file")
        pdf_ref = workspace_relpath(pdf_path)
        prompt_text = _review_prompt(pdf_ref=pdf_ref, review_request=params.review_request)
        reviews: list[dict[str, str]] = []
        rendered_blocks: list[str] = []
        for idx, (model_label, cfg) in enumerate(_resolve_model_configs(profile.peer_review_models), start=1):
            model = build_chat_model(cfg)
            response = model.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an external peer reviewer. "
                            "Review the attached paper PDF directly. "
                            "Return plain text only. "
                            "Do not output JSON or any structured schema."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            _pdf_data_block(pdf_path),
                        ],
                    },
                ]
            )
            review_text = content_to_text(getattr(response, "content", response)).strip() or "No usable review returned."
            concrete_model_name = str(getattr(cfg, "model", "") or "").strip() or model_label
            reviews.append(
                {
                    "model_label": model_label,
                    "model_name": concrete_model_name,
                    "review_text": review_text,
                }
            )
            rendered_blocks.append(f"Reviewer {idx} ({concrete_model_name})\n{review_text}")
        answer = "\n\n".join(rendered_blocks).strip() or "No peer-review reports returned."
        return answer, {
            "tool_name": tool_name,
            "data": {
                "pdf_path": pdf_ref,
                "review_request": str(params.review_request or "").strip(),
                "model_labels": [item["model_label"] for item in reviews],
                "models": [item["model_name"] for item in reviews],
                "reviews": reviews,
                "answer": answer,
            },
        }
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"pdf_path": payload.get("pdf_path")}},
            error_code="peer_review_request_failed",
        ) from exc


__all__ = ["PeerReviewRequestInput", "peer_review_request"]
