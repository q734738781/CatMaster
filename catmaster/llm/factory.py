from __future__ import annotations

from typing import Any, Dict
import os
import logging
import json
import re
import warnings

from catmaster.llm.config import LLMConfig

_logger = logging.getLogger(__name__)


def _patch_langchain_openrouter_file_wrapper() -> None:
    """Patch langchain_openrouter so file-block messages retain `role`.

    langchain_openrouter currently wraps file-block messages with SDK Pydantic
    models via `model_construct(**fields)` after removing the `role` key. The
    resulting serialized payload omits `role`, which OpenRouter rejects with
    "Could not find discriminator field role". Keep the full dict so the SDK
    models still dump `role` correctly.
    """
    try:
        from langchain_openrouter import chat_models as chat_models_mod  # type: ignore
    except Exception:
        return

    if getattr(chat_models_mod, "_catmaster_file_wrapper_patched", False):
        return

    original = getattr(chat_models_mod, "_wrap_messages_for_sdk", None)
    has_file_blocks = getattr(chat_models_mod, "_has_file_content_blocks", None)
    if not callable(original) or not callable(has_file_blocks):
        return

    def _fixed_wrap_messages_for_sdk(message_dicts: list[dict[str, Any]]) -> list[dict[str, Any]] | list[Any]:
        if not has_file_blocks(message_dicts):
            return message_dicts

        try:
            from openrouter import components  # type: ignore  # noqa: PLC0415
        except Exception:
            return message_dicts

        role_to_model: dict[str, Any] = {
            "user": components.UserMessage,
            "system": components.SystemMessage,
            "assistant": components.AssistantMessage,
            "tool": components.ToolResponseMessage,
            "developer": components.DeveloperMessage,
        }

        wrapped: list[Any] = []
        for msg in message_dicts:
            model_cls = role_to_model.get(str(msg.get("role", "") or ""))
            if model_cls is None:
                wrapped.append(msg)
                continue
            wrapped.append(model_cls.model_construct(**dict(msg)))
        return wrapped

    chat_models_mod._wrap_messages_for_sdk = _fixed_wrap_messages_for_sdk
    chat_models_mod._catmaster_file_wrapper_patched = True


def _suppress_openrouter_file_block_serializer_warnings() -> None:
    """Silence known SDK/Pydantic warnings for OpenRouter file-block payloads.

    The OpenRouter API accepts file content blocks, but the current SDK type
    declarations lag behind and emit noisy `Pydantic serializer warnings`
    during request serialization. These are not actionable for callers once the
    payload is patched to include `role` correctly, so suppress this specific
    warning family while leaving real API failures untouched.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings:.*",
        category=UserWarning,
        module=r"pydantic\.functional_validators|openrouter\.components\.chatgenerationparams",
    )


def _normalize_http_log_body(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"\n[ \t]*\n+", "\n", cleaned)
    return cleaned.replace("\n", "\\n")


def _request_content_text(request: Any) -> str:
    """Best-effort extraction of HTTP request body for logging."""
    try:
        content = getattr(request, "content", b"")
    except Exception as exc:
        return f"<unavailable: {exc}>"
    if isinstance(content, bytes):
        return _normalize_http_log_body(content.decode("utf-8", errors="replace"))
    if isinstance(content, str):
        return _normalize_http_log_body(content)
    try:
        return _normalize_http_log_body(json.dumps(content, ensure_ascii=False, default=str))
    except Exception:
        return _normalize_http_log_body(str(content))


def _response_content_text(response: Any) -> str:
    """Best-effort extraction of HTTP response body for logging."""
    try:
        content = getattr(response, "content", b"")
    except Exception as exc:
        return f"<unavailable: {exc}>"
    if isinstance(content, bytes):
        return _normalize_http_log_body(content.decode("utf-8", errors="replace"))
    if isinstance(content, str):
        return _normalize_http_log_body(content)
    try:
        return _normalize_http_log_body(json.dumps(content, ensure_ascii=False, default=str))
    except Exception:
        return _normalize_http_log_body(str(content))


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

    def _log_response(response: Any) -> None:
        request = getattr(response, "request", None)
        method = str(getattr(request, "method", "") or "").upper()
        if method != "POST":
            return
        url = str(getattr(request, "url", "") or "")
        status_code = getattr(response, "status_code", None)
        body = _response_content_text(response)
        _logger.info(
            "[llm.http.raw_response] provider=%s model=%s method=%s url=%s status=%s body=%s",
            cfg.provider,
            cfg.model,
            method,
            url,
            status_code,
            body,
        )

    async def _alog_request(request: Any) -> None:
        _log_request(request)

    async def _alog_response(response: Any) -> None:
        try:
            await response.aread()
        except Exception as exc:
            _logger.info(
                "[llm.http.raw_response] provider=%s model=%s method=%s url=%s status=%s body=<read failed: %s>",
                cfg.provider,
                cfg.model,
                str(getattr(getattr(response, "request", None), "method", "") or "").upper(),
                str(getattr(getattr(response, "request", None), "url", "") or ""),
                getattr(response, "status_code", None),
                exc,
            )
            return
        _log_response(response)

    def _sync_response_hook(response: Any) -> None:
        try:
            response.read()
        except Exception as exc:
            _logger.info(
                "[llm.http.raw_response] provider=%s model=%s method=%s url=%s status=%s body=<read failed: %s>",
                cfg.provider,
                cfg.model,
                str(getattr(getattr(response, "request", None), "method", "") or "").upper(),
                str(getattr(getattr(response, "request", None), "url", "") or ""),
                getattr(response, "status_code", None),
                exc,
            )
            return
        _log_response(response)

    sync_client = httpx.Client(event_hooks={"request": [_log_request], "response": [_sync_response_hook]})
    async_client = httpx.AsyncClient(event_hooks={"request": [_alog_request], "response": [_alog_response]})
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


def _resolve_reasoning_config(cfg: LLMConfig) -> Dict[str, Any] | None:
    reasoning = cfg.reasoning if isinstance(cfg.reasoning, dict) else {}
    cleaned: Dict[str, Any] = {}
    for key, value in reasoning.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                cleaned[str(key)] = text
        else:
            cleaned[str(key)] = value
    return cleaned or None


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


def _resolve_openrouter_header_fields(cfg: LLMConfig) -> dict[str, str]:
    headers = dict(cfg.default_headers) if isinstance(cfg.default_headers, dict) else {}
    out: dict[str, str] = {}
    referer = str(headers.pop("HTTP-Referer", "") or "").strip()
    title = str(headers.pop("X-Title", "") or "").strip()
    if referer:
        out["app_url"] = referer
    if title:
        out["app_title"] = title
    return out


def _resolve_openrouter_request_kwargs(cfg: LLMConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    kwargs.update(_resolve_openrouter_header_fields(cfg))

    model_kwargs = _resolve_model_kwargs(cfg)
    extra_body = _resolve_extra_body(cfg)
    provider_config = extra_body.pop("provider", None)
    route = extra_body.pop("route", None)
    plugins = extra_body.pop("plugins", None)
    prompt_cache_retention = extra_body.pop("prompt_cache_retention", None)

    if isinstance(provider_config, dict) and provider_config:
        kwargs["openrouter_provider"] = provider_config
    if route is not None:
        kwargs["route"] = route
    if isinstance(plugins, list) and plugins:
        kwargs["plugins"] = plugins
    if prompt_cache_retention is not None:
        _logger.warning(
            "Ignoring unsupported OpenRouter request option prompt_cache_retention=%r for ChatOpenRouter.",
            prompt_cache_retention,
        )
    if extra_body:
        model_kwargs.update(extra_body)
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return kwargs


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
    if cfg.provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        _patch_langchain_openrouter_file_wrapper()
        _suppress_openrouter_file_block_serializer_warnings()
        api_key = _require_api_key(cfg)
        kwargs = _resolve_openrouter_request_kwargs(cfg)
        reasoning_config = _resolve_reasoning_config(cfg)

        if cfg.timeout_s is not None:
            kwargs["timeout"] = int(cfg.timeout_s)
        if cfg.max_retries is not None:
            kwargs["max_retries"] = cfg.max_retries
        if cfg.print_http_raw_post:
            _logger.warning(
                "print_http_raw_post is not supported for provider=openrouter with ChatOpenRouter; ignoring."
            )

        max_tokens = cfg.max_tokens
        if max_tokens is None and cfg.max_output_tokens is not None:
            max_tokens = cfg.max_output_tokens

        return ChatOpenRouter(
            model=cfg.model,
            api_key=api_key,
            temperature=cfg.temperature,
            reasoning=reasoning_config,
            streaming=True,
            max_tokens=max_tokens,
            **kwargs,
        )

    if cfg.provider in ("openai", "oai_compatible", "deepseek"):
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

        reasoning_config = _resolve_reasoning_config(cfg)

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
            reasoning=reasoning_config,
            model_kwargs=model_kwargs,
            streaming=True,
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
