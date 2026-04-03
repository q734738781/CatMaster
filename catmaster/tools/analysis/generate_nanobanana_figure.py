from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MIME_TO_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class GenerateNanoBananaFigureInput(BaseModel):
    """[figure/viz] Generate a Nano Banana conceptual/manuscript figure from text and save it under the writable workspace."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., description="Detailed text prompt for the schematic or conceptual figure.")
    output_path: str = Field(
        ...,
        description="Workspace-relative output image path under files/. If no suffix is given, one is inferred from the returned MIME type.",
    )


def _endpoint(base_url: str | None = None) -> str:
    base = str(base_url or os.getenv("OPENROUTER_BASE_URL") or _DEFAULT_OPENROUTER_BASE_URL).strip().rstrip("/")
    if base.endswith("/api/v1"):
        return f"{base}/chat/completions"
    return f"{base}/api/v1/chat/completions"


def _headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = str(os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    title = str(os.getenv("OPENROUTER_APP_TITLE") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _parse_data_url(url: str) -> tuple[str, bytes]:
    text = str(url or "").strip()
    if not text.startswith("data:"):
        raise ValueError("OpenRouter image response did not contain a data URL.")
    header, encoded = text.split(",", 1)
    if ";base64" not in header:
        raise ValueError("OpenRouter image data URL was not base64 encoded.")
    mime = header[len("data:") :].split(";", 1)[0].strip() or "image/png"
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"Could not decode image data URL: {exc}") from exc
    return mime, raw


def _resolve_output_path(path_str: str, *, mime: str) -> Path:
    requested = resolve_workspace_path(path_str, must_exist=False)
    if requested.suffix:
        return requested
    return requested.with_suffix(_MIME_TO_SUFFIX.get(mime, ".png"))


def _provider_extra_body(cfg: LLMConfig) -> dict[str, Any]:
    provider_options = cfg.provider_options.get("openrouter") if isinstance(cfg.provider_options, dict) else None
    extra_body = provider_options.get("extra_body") if isinstance(provider_options, dict) else None
    if not isinstance(extra_body, dict):
        return {}
    return {
        str(key): value
        for key, value in extra_body.items()
        if str(key) not in {"model", "messages", "modalities", "image_config"}
    }


def _extract_image_choice(payload: dict[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter image response contained no choices.")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter image response missing assistant message.")
    images = message.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("OpenRouter image response contained no images.")
    first_image = images[0] if isinstance(images[0], dict) else {}
    image_url = first_image.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url, str) or not url.strip():
        raise ValueError("OpenRouter image response missing image_url.url.")
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = [
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        text = "\n".join(parts).strip()
    else:
        text = ""
    return url, text


def _resolve_generation_profile() -> tuple[LLMConfig, dict[str, Any]]:
    profile = LLMProfile.from_env_or_file()
    cfg = profile.config_for_image_generation()
    if str(cfg.provider or "").strip().lower() != "openrouter":
        raise ValueError("image_generation must use an OpenRouter-backed model config.")
    image_config = dict(profile.image_generation.image_config or {})
    return cfg, image_config


def _build_generation_prompt(user_prompt: str) -> str:
    prompt = str(user_prompt or "").strip()
    return "\n".join(
        [
            "Generate a clean scientific schematic or conceptual manuscript figure.",
            "Target publication-quality scientific communication suitable for a journal-style manuscript.",
            "Prioritize visual clarity, sparse composition, publication-style labeling, and legible panel logic.",
            "Use a clean white or very light background with strong contrast and a colorblind-safe palette.",
            "Prefer simple arrows, blocks, motifs, pathway elements, or process flow over decorative illustration.",
            "Keep labels short, precise, and readable; avoid dense paragraphs, tiny text, or screenshot-like UI fragments.",
            "Use consistent visual grammar across the figure: matching arrows, spacing, fonts, and annotation style.",
            "Lay out the figure with a clear reading order, typically left-to-right or top-to-bottom.",
            "When the request implies a workflow, mechanism, architecture, comparison, or pathway, make that structure explicit.",
            "If the figure is conceptual, emphasize the scientific idea rather than photorealistic rendering.",
            "Do not add decorative clutter, faux screenshots, unnecessary background textures, watermarks, or ornamental borders.",
            "Prefer a figure a chemistry/materials paper could plausibly include as a conceptual, mechanistic, or workflow schematic.",
            "",
            "User figure request:",
            prompt,
        ]
    ).strip()


def generate_nanobanana_figure(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[figure/viz] Generate and save a Nano Banana figure from a text prompt."""
    tool_name = "generate_nanobanana_figure"
    try:
        params = GenerateNanoBananaFigureInput(**payload)
        cfg, image_config = _resolve_generation_profile()
        api_key_env = str(cfg.api_key_env or "OPENROUTER_API_KEY").strip() or "OPENROUTER_API_KEY"
        api_key = str(cfg.api_key or os.getenv(api_key_env) or "").strip()
        if not api_key:
            raise ValueError(f"{api_key_env} is required.")

        request_body: dict[str, Any] = dict(_provider_extra_body(cfg))
        request_body.update(
            {
                "model": cfg.model,
                "modalities": ["image", "text"],
                "messages": [{"role": "user", "content": _build_generation_prompt(params.prompt)}],
            }
        )
        if image_config:
            request_body["image_config"] = image_config

        with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
            response = client.post(
                _endpoint(cfg.base_url),
                headers={**cfg.default_headers, **_headers(api_key)},
                json=request_body,
            )
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}: {response.text[:800]}")

        response_json = response.json()
        image_url, response_text = _extract_image_choice(response_json)
        mime, raw = _parse_data_url(image_url)
        output_path = _resolve_output_path(params.output_path, mime=mime)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw)
        output_ref = workspace_relpath(output_path)

        content_lines = [
            f"Generated Nano Banana figure: {output_ref}",
            f"Model: {cfg.model}",
            f"MIME: {mime}",
        ]
        if response_text:
            content_lines.append(f"Model note: {response_text}")
        artifact = {
            "tool_name": tool_name,
            "data": {
                "model_name": cfg.model,
                "prompt": str(params.prompt).strip(),
                "output_path": output_ref,
                "mime_type": mime,
                "image_config": image_config,
                "provider_response_text": response_text,
            },
        }
        return "\n".join(content_lines).strip(), artifact
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={
                "tool_name": tool_name,
                "data": {
                    "output_path": payload.get("output_path"),
                },
            },
            error_code="generate_nanobanana_figure_failed",
        ) from exc


__all__ = ["GenerateNanoBananaFigureInput", "generate_nanobanana_figure"]
