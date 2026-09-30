from __future__ import annotations

"""Provider-aware construction of the single model-visible web-search surface."""

from pathlib import Path
from typing import Any

from catmaster.llm.config import LLMProfile
from catmaster.tools.registry import ToolRegistry, get_tool_registry


NATIVE_WEB_SEARCH_PROVIDERS = frozenset({"codex_oauth", "openai"})
NATIVE_WEB_SEARCH_TOOL: dict[str, str] = {"type": "web_search"}


def search_tools_for_role(
    profile: LLMProfile,
    model_role: str,
    *,
    registry: ToolRegistry | None = None,
    workspace: Path | str | None = None,
    run_dir: Path | str | None = None,
    audience: str = "",
    runtime_context: dict[str, Any] | None = None,
) -> list[Any]:
    """Return exactly one search implementation for a role's actual provider.

    OpenAI Responses and Codex OAuth receive the provider-native hosted tool.
    Other providers receive CatMaster's one ``web_search`` function, which owns
    Tavily failure classification, its run-scoped circuit, and scholarly-index
    fallback. Keeping this resolver outside any one agent factory prevents a
    role from accidentally bypassing provider routing.
    """

    cfg = profile.config_for_role(model_role)
    provider = str(cfg.provider or "").strip().lower()
    if provider in NATIVE_WEB_SEARCH_PROVIDERS:
        return [dict(NATIVE_WEB_SEARCH_TOOL)]

    tool_registry = registry or get_tool_registry()
    return tool_registry.as_langchain_tools(
        allowlist=["web_search"],
        run_dir=str(run_dir or "").strip() or None,
        workspace=str(workspace or "").strip() or None,
        audience=str(audience or "").strip() or None,
        runtime_context=dict(runtime_context or {}),
    )


__all__ = [
    "NATIVE_WEB_SEARCH_PROVIDERS",
    "NATIVE_WEB_SEARCH_TOOL",
    "search_tools_for_role",
]
