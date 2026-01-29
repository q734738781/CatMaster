from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os

from catmaster.llm.config import LLMProfile, LLMConfig


@dataclass
class LLMBundle:
    llm: Any
    summary_llm: Any
    tool_driver: Any
    model_name: str
    provider: str


def _require_api_key(cfg: LLMConfig) -> str:
    if cfg.api_key:
        return cfg.api_key
    if cfg.api_key_env:
        key = os.getenv(cfg.api_key_env, "")
        if key:
            return key
    raise ValueError(f"Missing API key. Set env {cfg.api_key_env!r} or provide api_key in config.")


def _build_langchain_chat_model(cfg: LLMConfig) -> Any:
    if cfg.provider in ("openai", "openrouter", "oai_compatible", "deepseek"):
        from langchain_openai import ChatOpenAI

        api_key = _require_api_key(cfg)
        kwargs: dict[str, Any] = {}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.default_headers:
            kwargs["default_headers"] = cfg.default_headers
        if cfg.timeout_s is not None:
            kwargs["timeout"] = cfg.timeout_s
        if cfg.max_retries is not None:
            kwargs["max_retries"] = cfg.max_retries

        model_kwargs: dict[str, Any] = {}
        if cfg.top_p is not None:
            model_kwargs["top_p"] = cfg.top_p
        if cfg.frequency_penalty is not None:
            model_kwargs["frequency_penalty"] = cfg.frequency_penalty
        if cfg.presence_penalty is not None:
            model_kwargs["presence_penalty"] = cfg.presence_penalty
        max_tokens = cfg.max_tokens
        if max_tokens is None and cfg.max_output_tokens is not None:
            max_tokens = cfg.max_output_tokens
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        if cfg.extra:
            model_kwargs.update(cfg.extra)

        return ChatOpenAI(
            model=cfg.model,
            api_key=api_key,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    if cfg.provider in ("gemini", "langchain"):
        if not cfg.langchain_class:
            raise ValueError("For provider=gemini/langchain, langchain_class is required")
        mod, cls = cfg.langchain_class.rsplit(".", 1)
        module = __import__(mod, fromlist=[cls])
        klass = getattr(module, cls)
        return klass(**cfg.langchain_kwargs)

    raise ValueError(f"Unsupported provider: {cfg.provider}")


def build_llm_bundle(profile: LLMProfile) -> LLMBundle:
    main = profile.main
    summary_cfg = profile.summary or profile.main

    llm = _build_langchain_chat_model(main)
    summary_llm = _build_langchain_chat_model(summary_cfg)

    driver_kind = main.tool_calling.driver
    if driver_kind == "openai_responses":
        from catmaster.llm.openai_responses_driver import OpenAIResponsesDriver

        tool_driver = OpenAIResponsesDriver(
            model=main.model,
            api_key=_require_api_key(main),
            base_url=main.base_url,
            default_headers=main.default_headers or None,
        )
    elif driver_kind == "openai_chat_completions":
        from catmaster.llm.openai_chat_completions_driver import OpenAIChatCompletionsDriver

        tool_driver = OpenAIChatCompletionsDriver(
            model=main.model,
            api_key=_require_api_key(main),
            base_url=main.base_url,
            default_headers=main.default_headers or None,
        )
    elif driver_kind == "langchain_bind_tools":
        raise NotImplementedError("langchain_bind_tools driver is reserved for future providers.")
    else:
        raise ValueError(f"Unknown driver kind: {driver_kind}")

    return LLMBundle(
        llm=llm,
        summary_llm=summary_llm,
        tool_driver=tool_driver,
        model_name=main.model,
        provider=main.provider,
    )


__all__ = ["LLMBundle", "build_llm_bundle"]
