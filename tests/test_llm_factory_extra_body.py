from __future__ import annotations

import json
import sys
import time
import types

import pytest

from catmaster.llm.config import LLMConfig
from catmaster.llm.factory import build_chat_model


def test_build_chat_model_passes_provider_bound_extra_body(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider_options={
            "openrouter": {"extra_body": {"prompt_cache_retention": "24h"}},
        },
    )

    build_chat_model(cfg)

    assert captured.get("model_kwargs") in (None, {})
    assert captured.get("streaming") is False
    assert captured.get("content_cache_control") is None


def test_build_chat_model_passes_provider_openrouter_extra_body(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider_options={
            "openrouter": {
                "extra_body": {"provider": {"order": ["openai"]}, "x": 2},
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("openrouter_provider") == {"order": ["openai"]}
    assert captured.get("model_kwargs") == {"x": 2}


def test_build_chat_model_maps_openrouter_cache_control_to_content_breakpoints(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider_options={
            "openrouter": {
                "extra_body": {
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("content_cache_control") == {"type": "ephemeral", "ttl": "1h"}
    assert captured.get("model_kwargs") in (None, {})


def test_build_chat_model_passes_reasoning_object(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        reasoning={"effort": "high"},
    )

    build_chat_model(cfg)

    assert captured.get("reasoning") == {"effort": "high"}


def test_build_chat_model_passes_reasoning_summary_config(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    cfg = LLMConfig(
        provider="openai",
        model="gpt-5",
        api_key="test-key",
        reasoning={"effort": "medium", "summary": "auto"},
    )

    build_chat_model(cfg)

    assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
    assert captured.get("streaming") is False
    assert captured.get("disable_streaming") is True
    assert captured.get("use_responses_api") is True


def test_build_chat_model_passes_oai_compatible_top_level_reasoning_effort(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    cfg = LLMConfig(
        provider="oai_compatible",
        model="compatible-model",
        api_key="test-key",
        base_url="https://compatible.example/v1",
        reasoning={"effort": "high"},
        reasoning_effort="medium",
    )

    build_chat_model(cfg)

    assert captured.get("base_url") == "https://compatible.example/v1"
    assert captured.get("reasoning_effort") == "medium"
    assert captured.get("reasoning") is None
    assert captured.get("use_responses_api") is False


def test_build_chat_model_passes_anthropic_common_and_chat_kwargs(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatAnthropic:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_anthropic", types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic))

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=4096,
        timeout_s=30,
        max_retries=4,
        default_headers={"x-test": "1"},
        provider_options={
            "anthropic": {
                "chat_kwargs": {
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "betas": ["token-efficient-tools-2025-02-19"],
                }
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("model") == "claude-sonnet-4-5-20250929"
    assert captured.get("api_key") == "test-key"
    assert captured.get("base_url") == "https://api.anthropic.com"
    assert captured.get("temperature") == 0.2
    assert captured.get("top_p") == 0.9
    assert captured.get("max_tokens") == 4096
    assert captured.get("timeout") == 30
    assert captured.get("max_retries") == 4
    assert captured.get("default_headers") == {"x-test": "1"}
    assert captured.get("streaming") is False
    assert captured.get("thinking") == {"type": "enabled", "budget_tokens": 1024}
    assert captured.get("betas") == ["token-efficient-tools-2025-02-19"]


def test_build_chat_model_does_not_map_anthropic_reasoning(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatAnthropic:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_anthropic", types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic))

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key="test-key",
        reasoning={"effort": "high"},
    )

    build_chat_model(cfg)

    assert "reasoning" not in captured
    assert "thinking" not in captured


def test_build_chat_model_passes_codex_oauth_kwargs_without_api_key(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenAICodex:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    class FakeChatGPTToken:
        pass

    class FakeFileChatGPTOAuthTokenProvider:
        @classmethod
        def from_default_store(cls):
            return cls()

        def get_token(self):
            raise FileNotFoundError("missing official store")

        async def aget_token(self):
            return self.get_token()

        def get_access_token(self):
            return self.get_token().access_token

        async def aget_access_token(self):
            return (await self.aget_token()).access_token

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai.chat_models.codex",
        types.SimpleNamespace(_ChatOpenAICodex=FakeChatOpenAICodex),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai.chatgpt_oauth",
        types.SimpleNamespace(
            _ChatGPTToken=FakeChatGPTToken,
            _FileChatGPTOAuthTokenProvider=FakeFileChatGPTOAuthTokenProvider,
        ),
    )

    cfg = LLMConfig(
        provider="codex_oauth",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        temperature=0.1,
        max_tokens=2048,
        timeout_s=60,
        max_retries=2,
        provider_options={
            "codex_oauth": {
                "chat_kwargs": {
                    "system_prompt_mode": "strict",
                    "text_verbosity": "medium",
                    "reasoning_effort": "high",
                }
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("model") == "gpt-5.5"
    assert captured.get("base_url") == "https://chatgpt.com/backend-api/codex"
    assert captured.get("temperature") == 0.1
    assert captured.get("max_completion_tokens") == 2048
    assert captured.get("timeout") == 60
    assert captured.get("max_retries") == 2
    assert captured.get("reasoning_effort") == "high"
    assert captured.get("verbosity") == "medium"
    assert captured.get("instructions")
    assert captured.get("token_provider") is not None
    assert "system_prompt_mode" not in captured
    assert "text_verbosity" not in captured
    assert "api_key" not in captured


def test_codex_oauth_token_provider_reads_legacy_store(monkeypatch, tmp_path) -> None:
    from catmaster.llm.factory import _CatMasterCodexOAuthTokenProvider

    class MissingOfficialProvider:
        def get_token(self):
            raise FileNotFoundError("missing official store")

    class FakeToken:
        def __init__(self, *, access_token, refresh_token, expires_at, account_id):
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.expires_at = expires_at
            self.account_id = account_id

    legacy_path = tmp_path / "openai.json"
    legacy_path.write_text(
        json.dumps(
            {
                "type": "oauth",
                "access": "legacy-access",
                "refresh": "legacy-refresh",
                "expires": int((time.time() + 3600) * 1000),
                "account_id": "legacy-account",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_AUTH_PATH", str(legacy_path))

    provider = _CatMasterCodexOAuthTokenProvider(MissingOfficialProvider(), FakeToken)
    token = provider.get_token()

    assert token.access_token == "legacy-access"
    assert token.refresh_token == "legacy-refresh"
    assert token.account_id == "legacy-account"


def test_build_chat_model_passes_deepseek_official_reasoning_effort(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatDeepSeek:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_deepseek", types.SimpleNamespace(ChatDeepSeek=FakeChatDeepSeek))

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
        provider_options={
            "deepseek": {
                "extra_body": {"thinking": {"type": "enabled"}},
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("base_url") == "https://api.deepseek.com"
    assert captured.get("reasoning_effort") == "high"
    assert captured.get("extra_body") == {"thinking": {"type": "enabled"}}
    assert captured.get("reasoning") is None
    assert captured.get("use_responses_api") is False


def test_deepseek_payload_replays_reasoning_content() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
    )

    model = build_chat_model(cfg)
    payload = model._get_request_payload(
        [
            HumanMessage(content="weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={"reasoning_content": "need to call weather tool"},
            ),
            ToolMessage(content="sunny", tool_call_id="call_1"),
        ]
    )

    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["reasoning_content"] == "need to call weather tool"


def test_deepseek_payload_adds_empty_reasoning_content_for_tool_calls() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
    )

    model = build_chat_model(cfg)
    payload = model._get_request_payload(
        [
            HumanMessage(content="weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="sunny", tool_call_id="call_1"),
        ]
    )

    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["reasoning_content"] == ""


def test_deepseek_payload_does_not_replay_reasoning_content_when_thinking_disabled() -> None:
    from langchain_core.messages import AIMessage

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
        provider_options={
            "deepseek": {
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        },
    )

    model = build_chat_model(cfg)
    payload = model._get_request_payload(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={"reasoning_content": "hidden"},
            ),
        ]
    )

    assert "reasoning_content" not in payload["messages"][0]


def test_deepseek_payload_does_not_replay_reasoning_content_without_tool_calls() -> None:
    from langchain_core.messages import AIMessage

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
    )

    model = build_chat_model(cfg)
    payload = model._get_request_payload(
        [
            AIMessage(
                content="plain answer",
                additional_kwargs={"reasoning_content": "hidden"},
            ),
        ]
    )

    assert payload["messages"][0]["role"] == "assistant"
    assert payload["messages"][0]["content"] == "plain answer"
    assert "reasoning_content" not in payload["messages"][0]


def test_deepseek_payload_honors_runtime_thinking_disabled() -> None:
    from langchain_core.messages import AIMessage

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
    )

    model = build_chat_model(cfg)
    payload = model._get_request_payload(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={"reasoning_content": "hidden"},
            ),
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    assert "reasoning_content" not in payload["messages"][0]


def test_deepseek_payload_does_not_mutate_original_ai_message() -> None:
    from langchain_core.messages import AIMessage

    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
    )
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "get_weather",
                "args": {"city": "Paris"},
                "type": "tool_call",
            }
        ],
    )

    model = build_chat_model(cfg)
    payload = model._get_request_payload([message])

    assert payload["messages"][0]["reasoning_content"] == ""
    assert "reasoning_content" not in message.additional_kwargs


def test_build_chat_model_maps_openai_request_options(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    cfg = LLMConfig(
        provider="openai",
        model="gpt-5",
        api_key="test-key",
        default_headers={"a": "1"},
        timeout_s=10,
        max_retries=2,
        provider_options={
            "openai": {
                "request_options": {
                    "timeout": 30,
                    "max_retries": 5,
                    "default_headers": {"b": "2"},
                    "default_query": {"x": "y"},
                }
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("timeout") == 30
    assert captured.get("max_retries") == 5
    assert captured.get("default_query") == {"x": "y"}
    assert captured.get("default_headers") == {"a": "1", "b": "2"}


def test_build_chat_model_rejects_unknown_openai_request_options(monkeypatch) -> None:
    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    cfg = LLMConfig(
        provider="openai",
        model="gpt-5",
        api_key="test-key",
        provider_options={"openai": {"request_options": {"metadata": {"run": "x"}}}},
    )

    with pytest.raises(ValueError, match="request_options"):
        build_chat_model(cfg)


def test_build_chat_model_rejects_extra_extra_body(monkeypatch) -> None:
    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        extra={"extra_body": {"provider": {"order": ["openai"]}}},
    )

    with pytest.raises(ValueError, match=r"extra\.extra_body"):
        build_chat_model(cfg)


def test_build_chat_model_enables_http_raw_post_clients(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        print_http_raw_post=True,
    )

    build_chat_model(cfg)

    assert captured.get("http_client") is None
    assert captured.get("http_async_client") is None


def test_build_chat_model_maps_openrouter_headers_and_route(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://catmaster.local", "X-Title": "CatMaster"},
        provider_options={
            "openrouter": {
                "extra_body": {
                    "route": "fallback",
                    "plugins": [{"id": "web"}],
                    "prompt_cache_retention": "24h",
                }
            }
        },
    )

    build_chat_model(cfg)

    assert captured.get("app_url") == "https://catmaster.local"
    assert captured.get("app_title") == "CatMaster"
    assert captured.get("route") == "fallback"
    assert captured.get("plugins") == [{"id": "web"}]
    assert captured.get("model_kwargs") in (None, {})
