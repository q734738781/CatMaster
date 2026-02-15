from __future__ import annotations

from types import SimpleNamespace

from catmaster.llm.openai_chat_completions_driver import OpenAIChatCompletionsDriver


def _make_chat_response(*, usage=None):
    message = {
        "content": "pong",
        "tool_calls": [],
    }
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_openai_chat_completions_driver_maps_usage(monkeypatch) -> None:
    monkeypatch.setattr("catmaster.llm.openai_chat_completions_driver.OpenAI", object())
    fake_response = _make_chat_response(usage={
        "prompt_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 60},
        "completion_tokens": 50,
        "total_tokens": 250,
    })
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: fake_response),
        )
    )
    driver = OpenAIChatCompletionsDriver(
        model="gpt-test",
        api_key="dummy",
        client=client,
    )

    turn = driver.create_turn(input_items=[])

    assert turn.usage is not None
    assert turn.usage.input_tokens == 200
    assert turn.usage.input_cached_tokens == 60
    assert turn.usage.output_tokens == 50
    assert turn.usage.total_tokens == 250
    assert turn.usage.source == "provider"


def test_openai_chat_completions_driver_usage_missing(monkeypatch) -> None:
    monkeypatch.setattr("catmaster.llm.openai_chat_completions_driver.OpenAI", object())
    fake_response = _make_chat_response(usage=None)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: fake_response),
        )
    )
    driver = OpenAIChatCompletionsDriver(
        model="gpt-test",
        api_key="dummy",
        client=client,
    )

    turn = driver.create_turn(input_items=[])

    assert turn.usage is not None
    assert turn.usage.source == "missing"
    assert turn.usage.output_tokens is None


def test_openai_chat_completions_driver_passes_prompt_cache_retention(monkeypatch) -> None:
    monkeypatch.setattr("catmaster.llm.openai_chat_completions_driver.OpenAI", object())
    fake_response = _make_chat_response(usage=None)
    captured: dict = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return fake_response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        )
    )
    driver = OpenAIChatCompletionsDriver(
        model="gpt-test",
        api_key="dummy",
        client=client,
    )

    driver.create_turn(
        input_items=[],
        extra_body={"provider": {"order": ["openai"]}},
        prompt_cache_retention="24h",
    )

    extra_body = captured.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body.get("prompt_cache_retention") == "24h"
    assert extra_body.get("provider") == {"order": ["openai"]}
