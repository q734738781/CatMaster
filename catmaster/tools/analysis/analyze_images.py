from __future__ import annotations

"""Deprecated image-analysis workaround.

Prefer the current agent's built-in multimodal path for images.
This module remains only for backward-compatible direct imports.
"""

import base64
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class AnalyzeImagesInput(BaseModel):
    """[vision/analysis] Analyze one or more local images with a multimodal model and return structured findings."""

    query: str = Field(..., description="Precise question to ask about the image set.")
    image_paths: list[str] = Field(..., min_length=1, description="Workspace-relative local image paths.")
    context_text: str | None = Field(
        None,
        description="Optional concise textual context about the structures or task.",
    )
    structured_hints: dict[str, Any] | None = Field(
        None,
        description="Optional structured hints that should guide the analysis without overriding visual evidence.",
    )
    model: str | None = Field(
        None,
        description=(
            "Optional model override. Accepts either a configured llm.yaml model label or a raw model id; "
            "default uses the task_runner model config."
        ),
    )


def _image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{_image_mime(path)};base64,{encoded}"


def _resolve_model_config(model_override: str | None):
    profile = LLMProfile.from_env_or_file()
    base_cfg = profile.config_for_role("image_analyzer")
    override = str(model_override or "").strip()
    if not override:
        return base_cfg
    if override in profile.models:
        return profile.models[override]
    return replace(base_cfg, model=override)


def _build_query_text(
    *,
    query: str,
    image_paths: list[str],
    context_text: str | None,
    structured_hints: dict[str, Any] | None,
) -> str:
    lines = [
        "Analyze the provided images as a general visual analyzer and answer the question directly.",
        "Respond in natural language, not JSON.",
        "Include three parts in your response: the direct answer, what is high-confidence from the image, and what is lower-confidence or tentative.",
        "Use the image itself, any visible labels or legends, and the query/context together.",
        "When the image supports it, answer concretely instead of being generically cautious.",
        "",
        f"Question: {query.strip()}",
    ]
    if context_text:
        lines.extend(["", "Context:", str(context_text).strip()])
    if structured_hints:
        lines.extend(["", "Structured hints:", str(structured_hints)])
    lines.extend(
        [
            "",
            "Image order:",
            *[f"{idx + 1}. {path}" for idx, path in enumerate(image_paths)],
        ]
    )
    return "\n".join(lines)


def analyze_images(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[vision/analysis] Analyze local images with a multimodal model and return grounded findings."""
    tool_name = "analyze_images"
    try:
        params = AnalyzeImagesInput(**payload)
        image_paths = [resolve_workspace_path(item, must_exist=True) for item in params.image_paths]
        image_refs = [workspace_relpath(path) for path in image_paths]
        query_text = _build_query_text(
            query=params.query,
            image_paths=image_refs,
            context_text=params.context_text,
            structured_hints=params.structured_hints,
        )
        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": query_text}]
        for path in image_paths:
            content_blocks.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})

        cfg = _resolve_model_config(params.model)
        model = build_chat_model(cfg)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a general visual analyzer. "
                        "Answer the user's query from the images and context in concise natural language. "
                        "Use the image itself, visible labels, and visible legends when present. "
                        "State what is high-confidence from the image and what is lower-confidence or tentative."
                    )
                ),
                HumanMessage(content=content_blocks),
            ]
        )
        response_text = content_to_text(getattr(response, "content", response))
        data = {
            "answer": response_text.strip() or "No usable visual answer returned.",
            "analysis_text": response_text.strip() or "No usable visual answer returned.",
            "model_name": str(getattr(cfg, "model", "") or ""),
            "image_paths": image_refs,
            "evidence_refs": [{"image": ref, "region": "global"} for ref in image_refs],
        }
        content = data["analysis_text"]
        return content, {"tool_name": tool_name, "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"image_paths": payload.get("image_paths") or []}},
            error_code="analyze_images_failed",
        ) from exc


__all__ = ["AnalyzeImagesInput", "analyze_images"]
