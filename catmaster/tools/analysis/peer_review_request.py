from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

class PeerReviewRequestInput(BaseModel):
    """[writing/review] Send one manuscript PDF to selected configured peer-review models; collect each review or failure independently. Empty model_labels uses configured reviewers; retry only failed labels."""

    model_labels: list[str] = Field(default_factory=list, description="Configured model labels to call; empty uses configured peer-review defaults. Labels must exist in the model profile.")
    output_dir: str = Field("", description="Optional review output parent; each invocation creates its own result directory. Empty uses notes/peer_reviews.")
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
        requested = [profile.label_for_role("write_reviewer")]
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
        "Review the attached manuscript PDF as an external scientific reviewer for the venue and scope specified by the editor.",
        "Judge scientific soundness, evidence-claim fit, novelty positioning, controls, validation sufficiency, comparison quality, figure logic, and whether the work meets a serious reviewer threshold.",
        "Be tough but fair. Prioritize scientific weaknesses over prose polish.",
        "Return plain text in the format requested by the editor. If unspecified, organize by:",
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
    """[writing/review] Send one manuscript PDF to selected configured peer-review models; collect each review or failure independently. Empty model_labels uses configured reviewers; retry only failed labels."""
    tool_name = "peer_review_request"
    try:
        params = PeerReviewRequestInput(**payload)
        pdf_path = resolve_workspace_path(params.pdf_path, must_exist=True)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("pdf_path must point to a .pdf file")
        pdf_ref = workspace_relpath(pdf_path)
        prompt_text = _review_prompt(pdf_ref=pdf_ref, review_request=params.review_request)
        reviews: list[dict[str, str]] = []
        rendered_blocks: list[str] = []
        output_root = resolve_workspace_path(params.output_dir or "notes/peer_reviews")
        output_root.mkdir(parents=True, exist_ok=True)
        review_dir = Path(tempfile.mkdtemp(prefix="review-", dir=output_root))
        results_path = review_dir / "reviews.json"
        failures = []
        def persist():
            results_path.write_text(json.dumps({"pdf_path": pdf_ref, "reviews": reviews, "failures": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for idx, (model_label, cfg) in enumerate(_resolve_model_configs(params.model_labels), start=1):
            try:
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
            except Exception as exc:
                failures.append({"model_label": model_label, "error": f"{type(exc).__name__}: {exc}"})
                rendered_blocks.append(f"Reviewer {idx} ({model_label}) failed: {exc}")
            persist()
        answer = "\n\n".join(rendered_blocks).strip() or "No peer-review reports returned."
        status = "partial" if failures and reviews else "failed" if failures else "completed"
        answer = f"status={status} results_path={workspace_relpath(results_path)}\n\n" + answer
        return answer, {
            "tool_name": tool_name,
            "data": {
                "pdf_path": pdf_ref,
                "review_request": str(params.review_request or "").strip(),
                "model_labels": [item["model_label"] for item in reviews],
                "models": [item["model_name"] for item in reviews],
                "reviews": reviews,
                "failures": failures,
                "status": status,
                "results_path": workspace_relpath(results_path),
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
