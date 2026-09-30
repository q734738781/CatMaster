import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from test_provider_request_capture import make_model, response_stream


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "events", "aevents"])
def test_wire_key_stable_across_steps_models_and_compaction(mode):
    sent = []

    def transport(request):
        sent.append((json.loads(request.content), request.headers.get("session-id")))
        assert request.headers["thread-id"] == request.headers["session-id"]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_stream())

    async def run():
        # Reconstructing models simulates a new process/resume. Model node IDs
        # and run IDs change; logical conversation identity does not.
        for index, (thread, namespace) in enumerate([
            ("thread-a", "model:step1"), ("thread-a", "model:step2"),
            ("thread-a", "model:step3"), ("thread-b", "model:step4"),
            ("thread-a", "tools:child-a|model:step1"),
            ("thread-a", "tools:child-a|model:step2"),
            ("thread-a", "tools:child-b|model:step1"),
        ]):
            model = make_model(transport)
            model.extra_body = None
            config = {"metadata": {"thread_id": thread, "catmaster_run_id": f"run{index}",
                                   "langgraph_checkpoint_ns": namespace, "lc_agent_name": "writing"}}
            messages = [HumanMessage("Compacted summary" if index == 2 else "Original context")]
            if mode == "invoke":
                model.invoke(messages, config)
            elif mode == "ainvoke":
                await model.ainvoke(messages, config)
            elif mode == "events":
                list(model.stream_events(messages, config, version="v3"))
            else:
                stream = await model.astream_events(messages, config, version="v3")
                async for _ in stream:
                    pass
            model.http_client.close()
            await model.http_async_client.aclose()

    asyncio.run(run())
    keys = [body["prompt_cache_key"] for body, _ in sent]
    assert keys[0] == keys[1] == keys[2]
    assert keys[4] == keys[5]
    assert len({keys[0], keys[3], keys[4], keys[6]}) == 4
    for body, header in sent:
        assert body["prompt_cache_key"] == header
        assert "extra_headers" not in body
        assert len(header) <= 64


def test_explicit_key_precedence_and_direct_model_fallback():
    sent = []
    def transport(request):
        sent.append((json.loads(request.content), request.headers.get("session-id")))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_stream())
    model = make_model(transport)
    model.invoke("OK", prompt_cache_key="top-level")
    assert sent[-1][0]["prompt_cache_key"] == sent[-1][1] == "test-key"
    model.extra_body = None
    model.invoke("OK", prompt_cache_key="explicit")
    assert sent[-1][0]["prompt_cache_key"] == sent[-1][1] == "explicit"
    model.invoke("OK")
    model.invoke("OK again")
    assert sent[-1][1] == sent[-2][1]
    model.invoke("OK", prompt_cache_key="explicit", extra_headers={"session-id": "custom-session"})
    assert sent[-1][1] == "custom-session"
    model.http_client.close()
    asyncio.run(model.http_async_client.aclose())


def test_actual_graph_metadata_keeps_affinity_across_user_turns():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    sent = []
    def transport(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, text=response_stream())
    model = make_model(transport)
    model.extra_body = None
    native_search = {"type": "web_search", "search_context_size": "low"}
    graph = create_agent(model, tools=[native_search], checkpointer=InMemorySaver())
    for thread in ['a', 'a', 'b']:
        graph.invoke({'messages': [HumanMessage('Continue')]}, {'configurable': {'thread_id': thread}})
    assert sent[0]['prompt_cache_key'] == sent[1]['prompt_cache_key']
    assert sent[2]['prompt_cache_key'] != sent[0]['prompt_cache_key']
    assert all(body["tools"] == [native_search] for body in sent)
    model.http_client.close()
    asyncio.run(model.http_async_client.aclose())


@pytest.mark.parametrize("headers, expected", [
    ({}, ("test-key", "test-key")),
    ({"Session-Id": "custom-session"}, ("custom-session", "custom-session")),
    ({"Thread-Id": "custom-thread"}, ("test-key", "custom-thread")),
    ({"Session-Id": "s", "Thread-Id": "t"}, ("s", "t")),
])
def test_wire_identity_defaults_respect_case_insensitive_overrides(headers, expected):
    def transport(request):
        assert (request.headers["session-id"], request.headers["thread-id"]) == expected
        assert len(request.headers.get_list("session-id")) == 1
        assert len(request.headers.get_list("thread-id")) == 1
        return httpx.Response(200, text=response_stream())
    model = make_model(transport)
    try:
        model.invoke("OK", extra_headers=headers)
    finally:
        model.http_client.close()
        asyncio.run(model.http_async_client.aclose())
