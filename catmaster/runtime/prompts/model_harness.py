"""Automatic model guidance through DeepAgents' native harness profiles."""
from __future__ import annotations

from typing import Any

from .renderer import render_prompt_bundle


def model_prompt_bundle(model_name: str) -> str | None:
    """Match a resolved model identifier, never a user-chosen YAML label."""
    name = model_name.strip().casefold().rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return "catmaster.model.mimo" if name.startswith("mimo-") else None


def register_model_harness(model: Any) -> None:
    from deepagents import HarnessProfile, register_harness_profile
    from langchain_core.language_models import BaseChatModel
    # Use the same identity resolution as native pre-built-model profile lookup.
    # DeepAgents 0.7.x exposes these helpers in _models, not its public exports;
    # using them avoids divergent provider aliases (e.g. compatible OpenAI APIs).
    from deepagents._models import get_model_identifier, get_model_provider

    if not isinstance(model, BaseChatModel):
        return
    identifier = get_model_identifier(model) or ""
    bundle_id = model_prompt_bundle(identifier)
    if bundle_id is None:
        return
    provider = get_model_provider(model)
    if ":" in identifier:
        key = identifier
    elif provider:
        key = f"{provider}:{identifier}"
    else:
        raise ValueError("MiMo harness profile requires the model to report its provider")
    # Package-owned, stable guidance is identical across workspaces. Register only
    # the exact model, never the whole provider; native additive merging preserves
    # tools, middleware and child settings. Re-registration replaces the same text.
    register_harness_profile(key, HarnessProfile(system_prompt_suffix=render_prompt_bundle(bundle_id)))
