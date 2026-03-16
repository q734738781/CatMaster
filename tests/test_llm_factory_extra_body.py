from __future__ import annotations

import sys
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
        model="openai/gpt-5.2:online",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider_options={
            "openrouter": {"extra_body": {"prompt_cache_retention": "24h"}},
        },
    )

    build_chat_model(cfg)

    assert captured.get("model_kwargs") in (None, {})
    assert captured.get("streaming") is True


def test_build_chat_model_passes_provider_openrouter_extra_body(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2:online",
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


def test_build_chat_model_passes_reasoning_object(monkeypatch) -> None:
    captured: dict = {}

    class FakeChatOpenRouter:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openrouter", types.SimpleNamespace(ChatOpenRouter=FakeChatOpenRouter))

    cfg = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5.2:online",
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
        model="openai/gpt-5.2:online",
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
        model="openai/gpt-5.2:online",
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
        model="openai/gpt-5.2:online",
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
