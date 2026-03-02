from __future__ import annotations

from typing import Any, Dict
import os
import logging
import json

from catmaster.llm.config import LLMConfig

_logger = logging.getLogger(__name__)


def _request_content_text(request: Any) -> str:
    """Best-effort extraction of HTTP request body for logging."""
    try:
        content = getattr(request, "content", b"")
    except Exception as exc:
        return f"<unavailable: {exc}>"
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _build_http_debug_clients(cfg: LLMConfig) -> tuple[Any, Any]:
    """Create httpx clients that log raw POST payloads."""
    try:
        import httpx  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        _logger.warning("print_http_raw_post enabled but httpx is unavailable: %s", exc)
        return None, None

    def _log_request(request: Any) -> None:
        method = str(getattr(request, "method", "") or "").upper()
        if method != "POST":
            return
        url = str(getattr(request, "url", "") or "")
        body = _request_content_text(request)
        _logger.info(
            "[llm.http.raw_post] provider=%s model=%s method=%s url=%s body=%s",
            cfg.provider,
            cfg.model,
            method,
            url,
            body,
        )

    async def _alog_request(request: Any) -> None:
        _log_request(request)

    sync_client = httpx.Client(event_hooks={"request": [_log_request]})
    async_client = httpx.AsyncClient(event_hooks={"request": [_alog_request]})
    return sync_client, async_client


def _require_api_key(cfg: LLMConfig) -> str:
    if cfg.api_key:
        return cfg.api_key
    if cfg.api_key_env:
        key = os.getenv(cfg.api_key_env, "")
        if key:
            return key
    raise ValueError(f"Missing API key. Set env {cfg.api_key_env!r} or provide api_key in config.")


def _provider_options_for(cfg: LLMConfig, provider: str | None = None) -> Dict[str, Any]:
    key = str(provider or cfg.provider or "").strip().lower()
    if not key:
        return {}
    options = cfg.provider_options.get(key)
    return dict(options) if isinstance(options, dict) else {}


def _resolve_extra_body(cfg: LLMConfig) -> Dict[str, Any]:
    provider_extra = _provider_options_for(cfg).get("extra_body")
    if isinstance(provider_extra, dict) and provider_extra:
        return dict(provider_extra)
    return {}


def _resolve_reasoning_effort(cfg: LLMConfig) -> str | None:
    reasoning = cfg.reasoning if isinstance(cfg.reasoning, dict) else {}
    effort = reasoning.get("effort")
    if effort is None:
        return None
    text = str(effort).strip()
    return text or None


def _resolve_model_kwargs(cfg: LLMConfig) -> dict[str, Any]:
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

    extra_model_kwargs = dict(cfg.extra) if isinstance(cfg.extra, dict) else {}
    if "extra_body" in extra_model_kwargs:
        raise ValueError(
            "models.*.extra.extra_body is not supported. "
            "Use models.*.provider_options.<provider>.extra_body."
        )
    if "reasoning_effort" in extra_model_kwargs:
        raise ValueError(
            "models.*.extra.reasoning_effort is not supported. "
            "Use models.*.reasoning.effort instead."
        )
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    return model_kwargs


def _apply_openai_request_options(cfg: LLMConfig, kwargs: Dict[str, Any]) -> None:
    if str(cfg.provider or "").strip().lower() != "openai":
        return

    request_options = _provider_options_for(cfg, "openai").get("request_options")
    if request_options is None:
        return
    if not isinstance(request_options, dict):
        raise ValueError("models.*.provider_options.openai.request_options must be a mapping")

    allowed = {"timeout", "max_retries", "default_headers", "default_query"}
    unknown = sorted(set(request_options.keys()) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported models.*.provider_options.openai.request_options keys: "
            f"{', '.join(unknown)}"
        )

    if "default_headers" in request_options:
        raw_headers = request_options.get("default_headers")
        if raw_headers is not None and not isinstance(raw_headers, dict):
            raise ValueError("models.*.provider_options.openai.request_options.default_headers must be a mapping")
        merged_headers = dict(kwargs.get("default_headers") or {})
        if isinstance(raw_headers, dict):
            merged_headers.update(raw_headers)
        if merged_headers:
            kwargs["default_headers"] = merged_headers

    if "default_query" in request_options:
        raw_query = request_options.get("default_query")
        if raw_query is not None and not isinstance(raw_query, dict):
            raise ValueError("models.*.provider_options.openai.request_options.default_query must be a mapping")
        if isinstance(raw_query, dict) and raw_query:
            kwargs["default_query"] = dict(raw_query)

    if request_options.get("timeout") is not None:
        kwargs["timeout"] = request_options.get("timeout")

    if request_options.get("max_retries") is not None:
        kwargs["max_retries"] = request_options.get("max_retries")


def build_chat_model(cfg: LLMConfig) -> Any:
    """Build a LangChain ChatModel from an LLMConfig."""
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
        _apply_openai_request_options(cfg, kwargs)

        model_kwargs = _resolve_model_kwargs(cfg)

        merged_extra_body = _resolve_extra_body(cfg)
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
            _logger.debug(
                "Using extra_body for ChatOpenAI (provider=%s keys=%s)",
                cfg.provider,
                sorted(merged_extra_body.keys()),
            )

        reasoning_effort = _resolve_reasoning_effort(cfg)

        if cfg.print_http_raw_post:
            http_client, http_async_client = _build_http_debug_clients(cfg)
            if http_client is not None:
                kwargs["http_client"] = http_client
            if http_async_client is not None:
                kwargs["http_async_client"] = http_async_client
            _logger.info(
                "Enabled raw HTTP POST logging for provider=%s model=%s",
                cfg.provider,
                cfg.model,
            )

        return ChatOpenAI(
            model=cfg.model,
            api_key=api_key,
            temperature=cfg.temperature,
            reasoning_effort=reasoning_effort,
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


__all__ = [
    "build_chat_model",
]
