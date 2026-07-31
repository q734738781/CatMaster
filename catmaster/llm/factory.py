from __future__ import annotations

from typing import Any, Dict
import asyncio
from datetime import datetime, timezone
import os
import logging
import json
import re
import warnings
from pathlib import Path

from catmaster.llm.config import LLMConfig

_logger = logging.getLogger(__name__)


def _openrouter_text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": str(text or "").strip() or "[non-text content omitted]"}


def _unsupported_openrouter_block_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "content").strip() or "content"
    details: list[str] = []
    for key in ("id", "mime_type", "filename", "name"):
        value = str(block.get(key) or "").strip()
        if value:
            details.append(f"{key}={value}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"[{block_type} block omitted from replayed message{suffix}]"


def _openrouter_data_url(*, mime_type: str, base64_data: str) -> str:
    mime = str(mime_type or "application/octet-stream").strip() or "application/octet-stream"
    return f"data:{mime};base64,{base64_data}"


def _sanitize_openrouter_content(content: Any, *, role: str | None = None) -> Any:
    """Normalize replayed message blocks to the current OpenRouter SDK schema."""
    _ = role
    if not isinstance(content, list):
        return content

    sanitized: list[Any] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                sanitized.append(_openrouter_text_block(item))
            continue
        if not isinstance(item, dict):
            sanitized.append(_openrouter_text_block(f"[{type(item).__name__} content omitted from replayed message]"))
            continue

        block = dict(item)
        block_type = str(block.get("type") or "").strip().lower()
        if block_type in {"text", "file", "image_url", "input_audio", "input_video", "video_url"}:
            if block_type == "file" and "file" not in block and block.get("base64"):
                sanitized.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": str(block.get("filename") or block.get("name") or "attachment"),
                            "file_data": _openrouter_data_url(
                                mime_type=str(block.get("mime_type") or "application/octet-stream"),
                                base64_data=str(block.get("base64") or ""),
                            ),
                        },
                    }
                )
                continue
            sanitized.append(block)
            continue
        if block_type == "image":
            image_url = block.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                block["type"] = "image_url"
                sanitized.append(block)
                continue
            url = str(block.get("url") or "").strip()
            if url:
                sanitized.append({"type": "image_url", "image_url": {"url": url}})
                continue
            encoded = str(block.get("base64") or "").strip()
            if encoded:
                sanitized.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _openrouter_data_url(
                                mime_type=str(block.get("mime_type") or "image/png"),
                                base64_data=encoded,
                            )
                        },
                    }
                )
                continue
            sanitized.append(_openrouter_text_block(_unsupported_openrouter_block_text(block)))
            continue

        text = str(block.get("text") or "").strip()
        if text:
            sanitized.append(_openrouter_text_block(text))
        else:
            sanitized.append(_openrouter_text_block(_unsupported_openrouter_block_text(block)))
    return sanitized


def _sanitize_openrouter_message_dicts(message_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in message_dicts:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        content = message.get("content")
        role = str(message.get("role") or "").strip().lower() or None
        new_content = _sanitize_openrouter_content(content, role=role)
        if new_content is content:
            sanitized.append(message)
        else:
            sanitized.append({**message, "content": new_content})
    return sanitized


def _suppress_openrouter_file_block_serializer_warnings() -> None:
    """Silence known SDK/Pydantic warnings for OpenRouter file-block payloads.

    The OpenRouter API accepts file content blocks, but the current SDK type
    declarations lag behind and emit noisy `Pydantic serializer warnings`
    during request serialization. These are not actionable for callers once the
    payload is produced by the current langchain-openrouter/openrouter stack, so
    suppress this specific warning family while leaving real API failures
    untouched.
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


def _provider_chat_kwargs_for(cfg: LLMConfig, provider: str | None = None) -> Dict[str, Any]:
    key = str(provider or cfg.provider or "").strip().lower()
    options = _provider_options_for(cfg, key)
    kwargs: Dict[str, Any] = {}
    for field in ("kwargs", "chat_kwargs"):
        raw = options.get(field)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"models.*.provider_options.{key}.{field} must be a mapping")
        kwargs.update(raw)
    return kwargs


def _resolve_extra_body(cfg: LLMConfig) -> Dict[str, Any]:
    provider_extra = _provider_options_for(cfg).get("extra_body")
    if isinstance(provider_extra, dict) and provider_extra:
        return dict(provider_extra)
    return {}


def _resolve_openrouter_cache_control(cfg: LLMConfig) -> Dict[str, Any] | None:
    extra_body = _resolve_extra_body(cfg)
    cache_control = extra_body.get("cache_control")
    return dict(cache_control) if isinstance(cache_control, dict) and cache_control else None


def _attach_openrouter_cache_control(
    message_dicts: list[dict[str, Any]],
    cache_control: Dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(cache_control, dict) or not cache_control:
        return message_dicts

    def _cacheify_content(content: Any) -> Any:
        if isinstance(content, str):
            if not content.strip():
                return content
            return [{"type": "text", "text": content, "cache_control": dict(cache_control)}]
        if not isinstance(content, list):
            return content
        last_text_idx: int | None = None
        updated: list[Any] = []
        for idx, block in enumerate(content):
            if isinstance(block, dict):
                cloned = dict(block)
                if str(cloned.get("type") or "").strip().lower() == "text" and isinstance(cloned.get("text"), str):
                    last_text_idx = idx
                updated.append(cloned)
            else:
                updated.append(block)
        if last_text_idx is None:
            return content
        block = updated[last_text_idx]
        if isinstance(block, dict):
            block["cache_control"] = dict(cache_control)
        return updated

    updated_messages = list(message_dicts)

    def _annotate_first(role_names: set[str]) -> None:
        for idx, message in enumerate(updated_messages):
            role = str(message.get("role") or "").strip().lower()
            if role not in role_names:
                continue
            new_content = _cacheify_content(message.get("content"))
            if new_content is message.get("content"):
                return
            updated_messages[idx] = {**message, "content": new_content}
            return

    _annotate_first({"system", "developer"})
    _annotate_first({"user"})
    return updated_messages


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
    extra_body.pop("cache_control", None)

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
    provider = str(cfg.provider or "").strip().lower()
    if provider not in {"openai", "deepseek"}:
        return

    request_options = _provider_options_for(cfg, provider).get("request_options")
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


def _legacy_codex_oauth_auth_path() -> Path:
    explicit = os.getenv("LANGCHAIN_CODEX_OAUTH_AUTH_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.getenv("LANGCHAIN_CODEX_OAUTH_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "auth" / "openai.json"
    return Path.home() / ".langchain-codex-oauth" / "auth" / "openai.json"


class _CatMasterCodexOAuthTokenProvider:
    """Use LangChain's ChatGPT OAuth store, with read-only legacy migration.

    CatMaster previously documented `langchain-codex-oauth` for local login.
    The LangChain OpenAI Codex model uses `~/.langchain/chatgpt-auth.json`
    instead. To avoid breaking already-authenticated local deploys, this
    provider falls back to the old JSON file only when the LangChain store is
    missing. It does not depend on, import, or call the old adapter package.
    """

    def __init__(self, provider: Any, token_cls: Any) -> None:
        self._provider = provider
        self._token_cls = token_cls
        self._legacy_token: Any | None = None
        self._legacy_warned = False

    def _load_legacy_token(self) -> Any | None:
        cached = self._legacy_token
        expires_at = getattr(cached, "expires_at", None)
        if cached is not None and isinstance(expires_at, datetime) and datetime.now(timezone.utc) < expires_at:
            return cached

        path = _legacy_codex_oauth_auth_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read legacy Codex OAuth token store at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Legacy Codex OAuth token store is not a JSON object: {path}")

        access = str(data.get("access") or "")
        refresh = str(data.get("refresh") or "")
        account_id = str(data.get("account_id") or "")
        try:
            expires_ms = int(data.get("expires") or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Legacy Codex OAuth token store has invalid expires field: {path}") from exc
        if not (access and refresh and account_id and expires_ms):
            raise RuntimeError(f"Legacy Codex OAuth token store is incomplete: {path}")

        expires_at = datetime.fromtimestamp(expires_ms / 1000, tz=timezone.utc)
        if datetime.now(timezone.utc) >= expires_at:
            raise FileNotFoundError(
                "Legacy langchain-codex-oauth credentials are expired. "
                "Run `python -c \"from langchain_openai.chatgpt_oauth import "
                "login_chatgpt_device; login_chatgpt_device()\"` to create "
                "LangChain's ChatGPT OAuth token store."
            )
        if not self._legacy_warned:
            _logger.warning(
                "Using legacy langchain-codex-oauth credentials from %s. "
                "Re-login with langchain_openai.chatgpt_oauth.login_chatgpt_device() "
                "to create LangChain's refreshable token store.",
                path,
            )
            self._legacy_warned = True
        self._legacy_token = self._token_cls(
            access_token=access,
            refresh_token=refresh,
            expires_at=expires_at,
            account_id=account_id,
        )
        return self._legacy_token

    def get_token(self) -> Any:
        try:
            return self._provider.get_token()
        except FileNotFoundError:
            token = self._load_legacy_token()
            if token is not None:
                return token
            raise

    async def aget_token(self) -> Any:
        return await asyncio.to_thread(self.get_token)

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        token = await self.aget_token()
        return token.access_token


def build_chat_model(cfg: LLMConfig) -> Any:
    """Build a LangChain ChatModel from an LLMConfig."""
    if cfg.provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        from langchain_core.messages import BaseMessage

        _suppress_openrouter_file_block_serializer_warnings()
        api_key = _require_api_key(cfg)
        kwargs = _resolve_openrouter_request_kwargs(cfg)
        cache_control_config = _resolve_openrouter_cache_control(cfg)
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

        class CatMasterChatOpenRouter(ChatOpenRouter):
            content_cache_control: dict[str, Any] | None = None

            def _create_message_dicts(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                message_dicts, params = super()._create_message_dicts(messages, stop)
                message_dicts = _attach_openrouter_cache_control(
                    message_dicts,
                    self.content_cache_control,
                )
                message_dicts = _sanitize_openrouter_message_dicts(message_dicts)
                return message_dicts, params

        return CatMasterChatOpenRouter(
            model=cfg.model,
            api_key=api_key,
            temperature=cfg.temperature,
            reasoning=reasoning_config,
            streaming=False,
            max_tokens=max_tokens,
            content_cache_control=cache_control_config,
            **kwargs,
        )

    if cfg.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError(
                "provider=anthropic requires langchain-anthropic. "
                "Install it with `python -m pip install langchain-anthropic`."
            ) from exc

        kwargs = _provider_chat_kwargs_for(cfg, "anthropic")
        api_key = _require_api_key(cfg)
        kwargs.setdefault("api_key", api_key)
        if cfg.base_url:
            kwargs.setdefault("base_url", cfg.base_url)
        if cfg.default_headers:
            kwargs.setdefault("default_headers", cfg.default_headers)
        if cfg.timeout_s is not None:
            kwargs.setdefault("timeout", cfg.timeout_s)
        if cfg.max_retries is not None:
            kwargs.setdefault("max_retries", cfg.max_retries)
        if cfg.top_p is not None:
            kwargs.setdefault("top_p", cfg.top_p)
        max_tokens = cfg.max_tokens
        if max_tokens is None and cfg.max_output_tokens is not None:
            max_tokens = cfg.max_output_tokens
        if max_tokens is not None:
            kwargs.setdefault("max_tokens", max_tokens)
        kwargs.setdefault("model", cfg.model)
        if cfg.temperature is not None:
            kwargs.setdefault("temperature", cfg.temperature)
        kwargs.setdefault("streaming", False)

        return ChatAnthropic(**kwargs)

    if cfg.provider == "codex_oauth":
        try:
            from langchain_openai.chat_models.codex import _ChatOpenAICodex
            from langchain_openai.chatgpt_oauth import (
                _ChatGPTToken,
                _FileChatGPTOAuthTokenProvider,
            )
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError(
                "provider=codex_oauth requires langchain-openai with Codex OAuth support. "
                "Install the pinned CatMaster environment or `python -m pip install langchain-openai==1.4.1`."
            ) from exc

        kwargs = _provider_chat_kwargs_for(cfg, "codex_oauth")
        system_prompt_mode = kwargs.pop("system_prompt_mode", None)
        if system_prompt_mode is not None:
            _logger.warning(
                "Ignoring codex_oauth.chat_kwargs.system_prompt_mode=%r; "
                "_ChatOpenAICodex lifts SystemMessage content into Responses API instructions.",
                system_prompt_mode,
            )
        text_verbosity = kwargs.pop("text_verbosity", None)
        if text_verbosity is not None and "verbosity" not in kwargs:
            kwargs["verbosity"] = text_verbosity
        if cfg.base_url:
            kwargs.setdefault("base_url", cfg.base_url)
        if cfg.timeout_s is not None:
            kwargs.setdefault("timeout", cfg.timeout_s)
        if cfg.max_retries is not None:
            kwargs.setdefault("max_retries", cfg.max_retries)
        if cfg.temperature is not None:
            kwargs.setdefault("temperature", cfg.temperature)
        max_tokens = cfg.max_tokens
        if max_tokens is None and cfg.max_output_tokens is not None:
            max_tokens = cfg.max_output_tokens
        if max_tokens is not None:
            kwargs.setdefault("max_completion_tokens", max_tokens)
        reasoning_config = _resolve_reasoning_config(cfg)
        if "reasoning" not in kwargs and "reasoning_effort" not in kwargs:
            if isinstance(reasoning_config, dict):
                kwargs["reasoning"] = reasoning_config
            elif cfg.reasoning_effort:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
        kwargs.setdefault(
            "instructions",
            "You are CatMaster's Codex OAuth runtime. Use bound tools and DeepAgents "
            "filesystem/subagent tools when they are needed to complete the user task.",
        )
        kwargs.setdefault(
            "token_provider",
            _CatMasterCodexOAuthTokenProvider(
                _FileChatGPTOAuthTokenProvider.from_default_store(),
                _ChatGPTToken,
            ),
        )
        kwargs.setdefault("model", cfg.model)

        return _ChatOpenAICodex(**kwargs)

    if cfg.provider in ("openai", "oai_compatible", "deepseek"):
        if cfg.provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek as ChatModel
            from langchain_core.messages import AIMessage

            class CatMasterChatDeepSeek(ChatModel):
                @staticmethod
                def _thinking_enabled_for_extra_body(*extra_bodies: Any) -> bool:
                    extra_body: dict[str, Any] = {}
                    for candidate in extra_bodies:
                        if isinstance(candidate, dict):
                            extra_body.update(candidate)
                    thinking = extra_body.get("thinking")
                    if isinstance(thinking, dict):
                        return str(thinking.get("type") or "").strip().lower() != "disabled"
                    return True

                def _get_request_payload(
                    self,
                    input_: Any,
                    *,
                    stop: list[str] | None = None,
                    **kwargs: Any,
                ) -> dict[str, Any]:
                    payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                    message_dicts = payload.get("messages")
                    if not isinstance(message_dicts, list):
                        return payload

                    messages = self._convert_input(input_).to_messages()
                    thinking_enabled = self._thinking_enabled_for_extra_body(
                        _resolve_extra_body(cfg),
                        kwargs.get("extra_body"),
                        payload.get("extra_body"),
                    )
                    for message, message_dict in zip(messages, message_dicts):
                        if not isinstance(message, AIMessage) or not isinstance(message_dict, dict):
                            continue
                        if message_dict.get("role") != "assistant" or not message_dict.get("tool_calls"):
                            continue
                        if not thinking_enabled:
                            continue
                        # Backport langchain-ai/langchain PRs #34438/#35094:
                        # DeepSeek thinking-mode tool loops require only
                        # assistant tool-call messages to replay
                        # reasoning_content at the top level. Keep this as an
                        # outbound payload adaptation; do not mutate LangChain
                        # messages or add reasoning to normal assistant turns.
                        reasoning_content = message.additional_kwargs.get("reasoning_content")
                        message_dict["reasoning_content"] = reasoning_content or ""
                    return payload

            ChatModel = CatMasterChatDeepSeek
        else:
            from langchain_openai import ChatOpenAI as ChatModel

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

        provider = str(cfg.provider or "").strip().lower()
        use_responses_api = provider == "openai"
        init_kwargs: dict[str, Any] = {}
        if use_responses_api:
            init_kwargs["reasoning"] = reasoning_config
        elif cfg.reasoning_effort:
            init_kwargs["reasoning_effort"] = cfg.reasoning_effort

        return ChatModel(
            model=cfg.model,
            api_key=api_key,
            temperature=cfg.temperature,
            model_kwargs=model_kwargs,
            streaming=False,
            disable_streaming=True,
            use_responses_api=use_responses_api,
            **init_kwargs,
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
