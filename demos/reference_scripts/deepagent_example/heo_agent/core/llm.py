from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Literal

from deepagents.middleware.subagents import SubAgent
from tavily import TavilyClient

from .config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SEARCH_MODEL,
    DEFAULT_SEARCH_REASONING_EFFORT,
)


if find_spec("langchain_openrouter") is None:
    raise ImportError(
        "Missing dependency 'langchain-openrouter'. "
        "Install it with: pip install -U langchain-openrouter"
    )

from langchain_openrouter import ChatOpenRouter


def build_llm(model_name: str = DEFAULT_MODEL, reasoning_effort: str | None = DEFAULT_REASONING_EFFORT) -> ChatOpenRouter:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to run the DeepAgent stages.")
    reasoning = None
    if reasoning_effort:
        reasoning = {"effort": reasoning_effort}
    return ChatOpenRouter(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_retries=2,
        reasoning=reasoning,
    )


def build_search_subagent() -> list[SubAgent]:
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        return []

    tavily_client = TavilyClient(api_key=tavily_key)

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Search the web for chemistry background facts, synthesis notes, and screening context."""

        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    search_prompt = (
        "You are a chemistry research subagent. Use web search only when the main agent "
        "needs external background facts or chemistry insight that is not already in the "
        "workspace. Treat the literature pool as an important reference source for screening "
        "decisions, especially for synthesis risks, mixed-valence concerns, ionic-radius context, "
        "electronegativity trends, polyanion stability notes, transport clues, or literature precedent "
        "for specific dopants in NFPP-like frameworks. Do not search for local structure files "
        "or implementation details that already exist in the repository. Return concise "
        "bullet findings with source URLs and separate factual findings from inference. "
        "Make clear when a conclusion is supported directly by literature versus inferred from it."
    )
    return [
        SubAgent(
            name="researcher",
            description="Collect targeted web evidence for chemistry background facts, literature precedent, and screening context.",
            system_prompt=search_prompt,
            model=build_llm(DEFAULT_SEARCH_MODEL, reasoning_effort=DEFAULT_SEARCH_REASONING_EFFORT),
            tools=[internet_search],
        )
    ]
