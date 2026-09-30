from __future__ import annotations

import base64
import io
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
}
_OUTPUT_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp", ".svg": "svg"}
_RESERVED_OPTIONS = {"model", "prompt", "n", "input_references", "stream", "messages", "modalities"}


class GenerateFigureInput(BaseModel):
    """[figure/viz] Generate or edit one image asset through OpenRouter and save it in the workspace. Supports selectable image models and reference images for iterative edits. Editable slide text, tables and layout belong in native PPTX objects; quantitative evidence belongs in data-based plots."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, description="Complete generation or editing instruction, including subject, composition, style, labels and what to preserve. Passed to the image model without added style instructions.")
    output_path: str = Field(..., min_length=1, description="Workspace-relative output image path, for example writing/figures/schematic.png. Omit the suffix to use the returned image format.")
    model: str = Field("", description="Configured model label or OpenRouter image-model ID, for example openai/gpt-image-2.5-sunburst or openai/gpt-image-2.5-flare. Omit or leave empty to use image_generation.model_label. A raw ID uses the default image model's connection and credentials.")
    reference_images: list[str] = Field(default_factory=list, description="Workspace-relative paths of source images, template references or previously generated images to edit. Omit or pass [] for text-only generation. Images are attached to the actual request in this order.")
    image_options: dict[str, Any] = Field(default_factory=dict, description='OpenRouter Image API options overriding image_generation.image_config. Omit or pass {} for configured defaults. Examples: {"aspect_ratio": "16:9", "quality": "high"}, {"size": "1024x1024", "background": "transparent", "output_format": "png"}. Other supported keys include resolution, output_compression, seed and provider; model-specific support follows OpenRouter. This tool requests one image; use model and reference_images for those inputs.')

    @model_validator(mode="before")
    @classmethod
    def _legacy_empty_values(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            for name, default in (("model", ""), ("reference_images", []), ("image_options", {})):
                if value.get(name) is None:
                    value[name] = default
        return value


def _generation_config(model: str) -> tuple[LLMConfig, dict[str, Any]]:
    profile = LLMProfile.from_env_or_file()
    default = profile.config_for_image_generation()
    requested = model.strip()
    if requested in profile.models:
        cfg = profile.models[requested]
    elif requested and requested != default.model:
        # A raw model override shares the connection, not another model's
        # provider routing or model-specific request defaults.
        cfg = replace(default, model=requested, provider_options={})
    else:
        cfg = default
    if str(cfg.provider).lower() != "openrouter":
        raise ValueError("generate_figure requires an OpenRouter image-model configuration.")
    return cfg, dict(profile.image_generation.image_config)


def _image_endpoint(cfg: LLMConfig) -> str:
    base = str(cfg.base_url or os.getenv("OPENROUTER_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/images"


def _request_headers(cfg: LLMConfig, api_key: str) -> dict[str, str]:
    headers = dict(cfg.default_headers)
    headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for env_name, header in (("OPENROUTER_HTTP_REFERER", "HTTP-Referer"), ("OPENROUTER_APP_TITLE", "X-Title")):
        if value := os.getenv(env_name, "").strip():
            headers[header] = value
    return headers


def _reference_image(path: str) -> dict[str, Any]:
    source = resolve_workspace_path(path, must_exist=True)
    raw = source.read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        mime = Image.MIME.get(img.format)
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported reference image format: {path}")
    encoded = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _request_body(params: GenerateFigureInput, cfg: LLMConfig, defaults: dict[str, Any], output: Path) -> dict[str, Any]:
    provider_options = cfg.provider_options.get("openrouter", {})
    extra = provider_options.get("extra_body", {}) if isinstance(provider_options, dict) else {}
    body = dict(extra) if isinstance(extra, dict) else {}
    body.update(defaults)
    body.update(params.image_options)
    conflicts = _RESERVED_OPTIONS.intersection(body)
    if conflicts:
        raise ValueError(f"Use the declared tool inputs instead of these image options: {', '.join(sorted(conflicts))}")
    # OpenRouter's dedicated Image API uses top-level options and
    # input_references, unlike the older chat-completions image transport.
    # https://openrouter.ai/docs/guides/overview/multimodal/image-generation
    if "image_size" in body:
        body.setdefault("size", body.pop("image_size"))
    # An explicit size replaces inherited geometry defaults. Explicitly
    # supplied conflicting controls still reach the API's own validation.
    if "size" in params.image_options or "image_size" in params.image_options:
        for key in ("resolution", "aspect_ratio"):
            if key not in params.image_options:
                body.pop(key, None)
    if fmt := _OUTPUT_FORMATS.get(output.suffix.lower()):
        if body.get("output_format", fmt) != fmt:
            raise ValueError("output_path suffix and output_format disagree.")
        body.setdefault("output_format", fmt)
    body.update({"model": cfg.model, "prompt": params.prompt, "n": 1, "stream": False})
    if params.reference_images:
        body["input_references"] = [_reference_image(path) for path in params.reference_images]
    return body


def _decode_image(payload: dict[str, Any]) -> tuple[str, bytes]:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("OpenRouter did not return the single requested image.")
    encoded = data[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("OpenRouter image response is missing b64_json.")
    raw = base64.b64decode(encoded, validate=True)
    mime = data[0].get("media_type")
    if not mime:
        with Image.open(io.BytesIO(raw)) as img:
            mime = Image.MIME.get(img.format)
    if mime not in _MIME_SUFFIXES:
        raise ValueError(f"Unsupported returned image format: {mime}")
    return mime, raw


def generate_figure(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Generate or edit one figure using the selected OpenRouter image model."""
    api_key = ""
    try:
        params = GenerateFigureInput(**payload)
        output = resolve_workspace_path(params.output_path, must_exist=False)
        cfg, defaults = _generation_config(params.model)
        key_env = cfg.api_key_env or "OPENROUTER_API_KEY"
        api_key = str(cfg.api_key or os.getenv(key_env) or "").strip()
        if not api_key:
            raise ValueError(f"{key_env} is required.")
        body = _request_body(params, cfg, defaults, output)
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            response = client.post(_image_endpoint(cfg), headers=_request_headers(cfg, api_key), json=body)
        if response.status_code >= 400:
            # Avoid dumping provider bodies containing reference-image data.
            try:
                error = response.json().get("error", {})
                detail = error.get("message", "Image request failed.") if isinstance(error, dict) else "Image request failed."
            except (ValueError, AttributeError):
                detail = "Image request failed."
            raise ValueError(f"HTTP {response.status_code}: {detail}")
        mime, raw = _decode_image(response.json())
        suffix = _MIME_SUFFIXES[mime]
        if not output.suffix or _OUTPUT_FORMATS.get(output.suffix.lower()) != _OUTPUT_FORMATS.get(suffix):
            output = output.with_suffix(suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
        output_ref = workspace_relpath(output)
        return f"Generated figure: {output_ref}", {
            "tool_name": "generate_figure",
            "data": {"output_path": output_ref, "mime_type": mime, "model_name": cfg.model},
        }
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        message = str(exc)
        if api_key:
            message = message.replace(api_key, "[redacted]")
        raise CatMasterToolExecutionError(
            tool_name="generate_figure",
            public_message=f"generate_figure failed: {message}",
            artifact={"tool_name": "generate_figure", "data": {"output_path": payload.get("output_path")}},
            error_code="generate_figure_failed",
        ) from None


__all__ = ["GenerateFigureInput", "generate_figure"]
