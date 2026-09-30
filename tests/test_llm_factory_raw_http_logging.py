from __future__ import annotations

from types import SimpleNamespace
import importlib
import logging
import sys

from catmaster.llm.config import LLMConfig


def test_http_debug_clients_log_raw_post_and_response(monkeypatch, caplog) -> None:
    factory = importlib.import_module("catmaster.llm.factory")

    class _FakeClient:
        def __init__(self, *, event_hooks):
            self.event_hooks = event_hooks

    class _FakeAsyncClient:
        def __init__(self, *, event_hooks):
            self.event_hooks = event_hooks

    fake_httpx = SimpleNamespace(Client=_FakeClient, AsyncClient=_FakeAsyncClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    cfg = LLMConfig(provider="openrouter", model="openai/gpt-5.4")

    caplog.set_level(logging.INFO, logger="catmaster.llm.factory")
    sync_client, _ = factory._build_http_debug_clients(cfg)

    request = SimpleNamespace(method="POST", url="https://example.test/v1/chat/completions", content=b'{"hello":"world"}')
    response = SimpleNamespace(
        request=request,
        status_code=500,
        content=b'{"message":"Internal Server Error","code":500}',
        read=lambda: b'{"message":"Internal Server Error","code":500}',
    )

    sync_client.event_hooks["request"][0](request)
    sync_client.event_hooks["response"][0](response)

    text = caplog.text
    assert "[llm.http.raw_post]" in text
    assert '{"hello":"world"}' in text
    assert "[llm.http.raw_response]" in text
    assert '"code":500' in text


def test_http_debug_logging_compacts_blank_lines(monkeypatch, caplog) -> None:
    factory = importlib.import_module("catmaster.llm.factory")

    class _FakeClient:
        def __init__(self, *, event_hooks):
            self.event_hooks = event_hooks

    class _FakeAsyncClient:
        def __init__(self, *, event_hooks):
            self.event_hooks = event_hooks

    fake_httpx = SimpleNamespace(Client=_FakeClient, AsyncClient=_FakeAsyncClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    cfg = LLMConfig(provider="openrouter", model="openai/gpt-5.4")

    caplog.set_level(logging.INFO, logger="catmaster.llm.factory")
    sync_client, _ = factory._build_http_debug_clients(cfg)

    request = SimpleNamespace(
        method="POST",
        url="https://example.test/v1/chat/completions",
        content=b'\n\n{"hello":"world"}\n\n',
    )
    response = SimpleNamespace(
        request=request,
        status_code=200,
        content=b'\n\n{"ok":true}\n\n',
        read=lambda: b'\n\n{"ok":true}\n\n',
    )

    sync_client.event_hooks["request"][0](request)
    sync_client.event_hooks["response"][0](response)

    text = caplog.text
    assert 'body={"hello":"world"}' in text
    assert 'body={"ok":true}' in text
    assert "\n\n{" not in text
