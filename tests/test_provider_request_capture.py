import asyncio
import gzip
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace
from threading import Event

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from catmaster.llm.config import LLMConfig
from catmaster.llm.factory import build_chat_model
from catmaster.llm.request_capture import _active_run
from catmaster.runtime.artifact_callback import ObservabilityCallbackHandler


class TokenProvider:
    def get_token(self):
        return SimpleNamespace(
            access_token="test-secret", account_id="test-account",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    async def aget_token(self):
        return self.get_token()

    def get_access_token(self):
        return self.get_token().access_token

    async def aget_access_token(self):
        return self.get_access_token()


def response_stream():
    message = {
        "id": "msg_test", "type": "message", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
    }
    response = {
        "id": "resp_test", "object": "response", "created_at": 1,
        "model": "gpt-6-astra", "status": "completed", "output": [message],
        "prompt_cache_key": "returned-key", "prompt_cache_retention": "24h",
        "usage": {"input_tokens": 100, "output_tokens": 2, "total_tokens": 102,
                  "input_tokens_details": {"cached_tokens": 64},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }
    events = [
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**message, "content": []}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
         "item_id": "msg_test", "delta": "OK"},
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": response},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def make_model(handler, *, max_retries=0):
    transport = httpx.MockTransport(handler)
    return build_chat_model(LLMConfig(
        provider="codex_oauth", model="gpt-6-astra", max_retries=max_retries,
        provider_options={"codex_oauth": {"chat_kwargs": {
            "token_provider": TokenProvider(),
            "http_client": httpx.Client(transport=transport),
            "http_async_client": httpx.AsyncClient(transport=transport),
            "reasoning": {"effort": "high"},
            "extra_body": {"prompt_cache_key": "test-key"},
        }}},
    ))


def observations(path):
    with sqlite3.connect(path / "observability.sqlite") as connection:
        return [(name, callback, json.loads(payload)) for name, callback, payload in connection.execute(
            "SELECT name,callback_run_id,payload_json FROM observation_events ORDER BY id"
        )]


def test_capture_includes_explicit_thread_identity(monkeypatch, tmp_path):
    monkeypatch.setenv('CATMASTER_CAPTURE_REQUEST_RUN_IDS', 'selected-run')
    model = make_model(lambda request: httpx.Response(200, text=response_stream()))
    callback = ObservabilityCallbackHandler(tmp_path, run_id='selected-run')
    model.invoke('OK', {'callbacks': [callback]}, extra_headers={'thread-id': 'thread-a'})
    captured = next(row[2] for row in observations(tmp_path) if row[0] == 'LLM_PROVIDER_REQUEST')
    assert captured['cache_headers'] == {'session-id': 'test-key', 'thread-id': 'thread-a'}
    model.http_client.close()
    asyncio.run(model.http_async_client.aclose())


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('compressed', [False, True])
def test_fragmented_sse_without_content_type_preserves_stream(monkeypatch, tmp_path, asynchronous, compressed):
    monkeypatch.setenv('CATMASTER_CAPTURE_REQUEST_RUN_IDS', 'selected-run')
    raw = response_stream().encode()
    wire = gzip.compress(raw) if compressed else raw
    closed = []

    class Chunks(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            for offset in range(0, len(wire), 7):
                yield wire[offset:offset + 7]

        async def __aiter__(self):
            for chunk in self:
                yield chunk

        def close(self):
            closed.append(True)

        async def aclose(self):
            self.close()

    def transport(request):
        return httpx.Response(200, headers={'content-encoding': 'gzip'} if compressed else {}, stream=Chunks())
    model = make_model(transport)
    callback = ObservabilityCallbackHandler(tmp_path, run_id='selected-run')
    config = {'callbacks': [callback]}
    if asynchronous:
        reply = asyncio.run(model.ainvoke('OK', config))
    else:
        reply = model.invoke('OK', config)
    assert reply.text == 'OK'
    assert closed
    rows = observations(tmp_path)
    assert len([r for r in rows if r[0] == 'LLM_PROVIDER_RESPONSE']) == 1
    model.http_client.close()
    asyncio.run(model.http_async_client.aclose())


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "events", "aevents"])
def test_prepared_body_matches_transport_and_joins_usage(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_RUN_IDS", "selected-run")
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_AGENTS", "writing")
    sent = []

    def handler(request):
        sent.append(request.content.decode())
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_stream())

    model = make_model(handler)
    callback = ObservabilityCallbackHandler(tmp_path, run_id="selected-run")
    completed = Event()
    record_end = callback.on_llm_end

    def on_end(*args, **kwargs):
        record_end(*args, **kwargs)
        completed.set()

    monkeypatch.setattr(callback, "on_llm_end", on_end)
    config = {"callbacks": [callback], "metadata": {"lc_agent_name": "writing"}}
    messages = [SystemMessage("Stable instructions"), HumanMessage(content=[
        {"type": "text", "text": "Inspect the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
    ])]
    bound = model.bind_tools([{"type": "function", "function": {
        "name": "inspect_page", "parameters": {"type": "object", "properties": {"page": {"type": "integer"}}},
    }}])

    async def run_async():
        if mode == "ainvoke":
            await bound.ainvoke(messages, config)
        else:
            stream = await bound.astream_events(messages, config, version="v3")
            async for _ in stream:
                assert _active_run.get() is None
            # The native event queue closes before its background producer
            # finishes on_llm_end. Wait for the public callback, not a sleep.
            assert await asyncio.to_thread(completed.wait, 5)
        await model.http_async_client.aclose()

    try:
        if mode == "invoke":
            bound.invoke(messages, config)
        elif mode == "events":
            for _ in bound.stream_events(messages, config, version="v3"):
                assert _active_run.get() is None
        else:
            asyncio.run(run_async())
        assert _active_run.get() is None
        rows = observations(tmp_path)
        captures = [row for row in rows if row[0] == "LLM_PROVIDER_REQUEST"]
        ends = [row for row in rows if row[0] == "LLM_CALL_END"]
        assert len(captures) == len(ends) == 1
        _, callback_id, captured = captures[0]
        assert callback_id == ends[0][1]
        assert captured["body"] == sent[0]
        body = json.loads(captured["body"])
        assert body["instructions"] == "Stable instructions"
        assert body["tools"][0]["name"] == "inspect_page"
        assert body["prompt_cache_key"] == "test-key"
        assert body["input"][0]["content"][1]["type"] == "input_image"
        assert "test-secret" not in json.dumps(captured)
        assert "test-account" not in json.dumps(captured)
        assert ends[0][2]["usage"]["input_cached_tokens"] == 64
        responses = [row for row in rows if row[0] == "LLM_PROVIDER_RESPONSE"]
        assert len(responses) == 1
        assert responses[0][1] == callback_id
        raw = responses[0][2]["event"]["response"]
        assert raw["prompt_cache_key"] == "returned-key"
        assert raw["prompt_cache_retention"] == "24h"
        assert raw["usage"]["input_tokens_details"]["cached_tokens"] == 64
        assert captured["cache_headers"] == {"session-id": "test-key", "thread-id": "test-key"}
    finally:
        model.http_client.close()


def test_selection_budget_and_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_RUN_IDS", "selected-run")
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_AGENTS", "research")
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_LIMIT", "2")
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(500, json={"error": {"message": "retry", "type": "server_error"}},
                                  headers={"retry-after-ms": "1"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_stream())

    callback = ObservabilityCallbackHandler(tmp_path, run_id="selected-run")
    model = make_model(handler, max_retries=1)
    try:
        model.invoke("test", {"callbacks": [callback], "metadata": {"lc_agent_name": "research"}})
        model.invoke("over budget", {"callbacks": [callback], "metadata": {"lc_agent_name": "research"}})
        captures = [r for r in observations(tmp_path) if r[0] == "LLM_PROVIDER_REQUEST"]
        assert len(captures) == 2
        assert captures[0][1] == captures[1][1]
        assert [r[2]["attempt"] for r in captures] == [1, 2]
        assert len([r for r in observations(tmp_path) if r[0] == "LLM_CALL_END"]) == 2
        assert _active_run.get() is None
    finally:
        model.http_client.close()


@pytest.mark.parametrize("selected_run,agent", [("another-run", "research"), ("", "research"), ("selected-run", "writing")])
def test_unselected_calls_are_not_captured(monkeypatch, tmp_path, selected_run, agent):
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_RUN_IDS", selected_run)
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_AGENTS", "research")
    model = make_model(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, text=response_stream()))
    try:
        model.invoke("test", {"callbacks": [ObservabilityCallbackHandler(tmp_path, run_id="selected-run")],
                              "metadata": {"lc_agent_name": agent}})
        assert not [r for r in observations(tmp_path) if r[0] == "LLM_PROVIDER_REQUEST"]
    finally:
        model.http_client.close()


def test_concurrent_async_calls_keep_their_callback_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_RUN_IDS", "selected-run")
    callback = ObservabilityCallbackHandler(tmp_path, run_id="selected-run")
    sent = []

    async def run():
        both_arrived = asyncio.Event()

        async def handler(request):
            sent.append(request.content.decode())
            if len(sent) == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_stream())

        model = make_model(handler)
        try:
            await asyncio.gather(*[
                model.ainvoke(role, {"callbacks": [callback], "metadata": {"lc_agent_name": role}})
                for role in ("research", "writing")
            ])
            assert _active_run.get() is None
        finally:
            model.http_client.close()
            await model.http_async_client.aclose()

    asyncio.run(run())
    rows = observations(tmp_path)
    captures = [r for r in rows if r[0] == "LLM_PROVIDER_REQUEST"]
    ends = {r[1]: r[2] for r in rows if r[0] == "LLM_CALL_END"}
    assert len(captures) == len(ends) == 2
    assert len({r[1] for r in captures}) == 2
    for _, callback_id, payload in captures:
        assert ends[callback_id]["agent_name"] == payload["agent_name"]
        assert json.loads(payload["body"])["input"][0]["content"] == payload["agent_name"]
        assert payload["body"] in sent


def test_provider_failure_and_observation_failure_do_not_leak_or_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("CATMASTER_CAPTURE_REQUEST_RUN_IDS", "selected-run")
    callback = ObservabilityCallbackHandler(tmp_path, run_id="selected-run")
    config = {"callbacks": [callback], "metadata": {"lc_agent_name": "research"}}
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("offline transport failure", request=request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_stream())

    model = make_model(handler)
    try:
        import openai

        with pytest.raises(openai.APIConnectionError):
            model.invoke("failure", config)
        assert _active_run.get() is None
        assert len(requests) == 1
        rows = observations(tmp_path)
        captured = next(r for r in rows if r[0] == "LLM_PROVIDER_REQUEST")
        failed = next(r for r in rows if r[0] == "LLM_ERROR")
        assert captured[1] == failed[1]

        def fail_recording(*args):
            raise OSError("observation unavailable")

        monkeypatch.setattr(callback, "capture_provider_request", fail_recording)
        assert model.invoke("success", config).text == "OK"
        assert len(requests) == 2
        assert _active_run.get() is None
    finally:
        model.http_client.close()
