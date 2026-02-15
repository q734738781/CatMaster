from __future__ import annotations

from types import SimpleNamespace

from catmaster.llm.openai_responses_driver import OpenAIResponsesDriver


def _build_driver_response(*, usage=None):
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "pong"}],
        }
    ]
    return SimpleNamespace(output=output, usage=usage)


def test_openai_responses_driver_maps_usage(monkeypatch) -> None:
    monkeypatch.setattr("catmaster.llm.openai_responses_driver.OpenAI", object())
    fake_response = _build_driver_response(usage={
        "input_tokens": 123,
        "input_tokens_details": {"cached_tokens": 45},
        "output_tokens": 67,
        "total_tokens": 190,
    })
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: fake_response),
    )
    driver = OpenAIResponsesDriver(client=client, model="gpt-test")

    turn = driver.create_turn(input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ping"}]}])

    assert turn.usage is not None
    assert turn.usage.input_tokens == 123
    assert turn.usage.input_cached_tokens == 45
    assert turn.usage.output_tokens == 67
    assert turn.usage.total_tokens == 190
    assert turn.usage.source == "provider"


def test_openai_responses_driver_usage_missing(monkeypatch) -> None:
    monkeypatch.setattr("catmaster.llm.openai_responses_driver.OpenAI", object())
    fake_response = _build_driver_response(usage=None)
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: fake_response),
    )
    driver = OpenAIResponsesDriver(client=client, model="gpt-test")

    turn = driver.create_turn(input_items=[])

    assert turn.usage is not None
    assert turn.usage.source == "missing"
    assert turn.usage.input_tokens is None


def test_openai_responses_driver_passes_prompt_cache_retention(monkeypatch) -> None:
    monkeypatch.setattr("catmaster.llm.openai_responses_driver.OpenAI", object())
    fake_response = _build_driver_response(usage=None)
    captured: dict = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return fake_response

    client = SimpleNamespace(
        responses=SimpleNamespace(create=_create),
    )
    driver = OpenAIResponsesDriver(client=client, model="gpt-test")

    driver.create_turn(input_items=[], prompt_cache_retention="24h")

    assert captured.get("prompt_cache_retention") == "24h"
