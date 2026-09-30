"""Verify content survives the real model, event-reader and SDK boundaries."""
import asyncio
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from catmaster.llm.config import LLMConfig
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.observed_invocation import observed_invocation
from catmaster.runtime.self_evolution.query import EvolutionTraceScope
from catmaster.runtime.self_evolution.trace import TurnTrace
from catmaster.runtime.usage_stats import load_usage_summary


class BrokenStream(FakeMessagesListChatModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="保留失败前的回答"))
        raise RuntimeError("stream disconnected")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            yield chunk


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_stream_retains_partial_output_without_success_accounting(tmp_path, asynchronous):
    with pytest.raises(RuntimeError, match="stream disconnected"):
        with observed_invocation(workspace=tmp_path, entrypoint="self_evolution", stage="reviewer",
                                 model_label="test", context={}) as (config, cb):
            model = BrokenStream(responses=[])
            if asynchronous:
                async def consume():
                    return [chunk async for chunk in model.astream("question", config=config)]
                asyncio.run(consume())
            else:
                list(model.stream("question", config=config))
    raw, = cb.store.read_events_page(names=["LLM_RAW_RESPONSE"], limit=100)["events"]
    assert raw["payload"]["partial"] is True
    assert raw["payload"]["status"] == "error"
    assert raw["payload"]["generations"][0]["response_text"] == "保留失败前的回答"
    assert not cb.store.read_events_page(names=["LLM_CALL_END"], limit=100)["events"]
    assert load_usage_summary(cb.run_dir)["calls"] == 0
    assert load_usage_summary(cb.run_dir)["failed_calls"] == 1
    scope = EvolutionTraceScope({cb.run_id: (cb.run_dir, TurnTrace(
        run_id=cb.run_id, thread_id="", entrypoint="self_evolution", status="error",
        user_prompt="question", final_answer="", summary="", task_outcome="", outcome_ref="",
    ))})
    data = json.loads(scope.execute("SELECT payload_json FROM trajectory_events WHERE name='MODEL_RESPONSE'")["rows"][0]["payload_json"])
    assert data["partial"] and data["status"] == "error"
    assert data["generations"][0]["assistant_text"] == "保留失败前的回答"


def test_reasoning_annotations_and_response_details_reach_evolution_reader(tmp_path):
    annotations = [{"type": "citation", "url": "https://example.org/paper", "title": "原始证据"}]
    blocks = [
        {"type": "reasoning", "reasoning": "先核对实验条件，再比较催化剂活性。"},
        {"type": "reasoning", "reasoning": "甲催化剂活性高，乙催化剂活性低。"},
        {"type": "reasoning", "reasoning": "甲催化剂活性低，乙催化剂活性高。"},
        {"type": "reasoning", "reasoning": "α>β"},
        {"type": "reasoning", "reasoning": "α<β"},
        {"type": "text", "text": "结论", "annotations": annotations},
        {"type": "non_standard", "value": {"type": "web_search_call", "action": {"query": "催化剂"}}},
    ]
    with observed_invocation(workspace=tmp_path, entrypoint="self_evolution", stage="reflector",
                             model_label="test", context={}) as (config, cb):
        model = FakeMessagesListChatModel(responses=[AIMessage(
            content=blocks, additional_kwargs={"refusal": "无法核实额外断言"},
            response_metadata={"finish_reason": "length"},
        )])
        model.invoke("question", config=config)
    end, = cb.store.read_events_page(names=["LLM_CALL_END"], limit=10)["events"]
    for block in blocks[:5]:
        assert block["reasoning"] in end["payload"]["reasoning_text"]
    scope = EvolutionTraceScope({cb.run_id: (cb.run_dir, TurnTrace(
        run_id=cb.run_id, thread_id="", entrypoint="self_evolution", status="done",
        user_prompt="question", final_answer="结论", summary="", task_outcome="", outcome_ref="",
    ))})
    row, = scope.execute("SELECT id, payload_json FROM trajectory_events WHERE name='MODEL_RESPONSE'")["rows"]
    generation = json.loads(row["payload_json"])["generations"][0]
    assert next(b for b in generation["content_blocks"] if b["type"] == "text")["annotations"] == annotations
    assert blocks[-1] in generation["content_blocks"]
    assert generation["additional_kwargs"]["refusal"] == "无法核实额外断言"
    assert generation["response_metadata"]["finish_reason"] == "length"
    page = scope.read_event(handle=f"run:{cb.run_id}#event:{row['id']}",
                            field="$.generations[0].content_blocks", offset=0, limit=10000)
    assert "原始证据" in page["segment"]


def test_openrouter_final_sdk_payload_preserves_assistant_blocks():
    from openrouter.components.chatassistantmessage import ChatAssistantMessage

    annotations = [{"type": "citation", "url": "https://example.org/paper"}]
    unknown = {"type": "non_standard", "value": {"type": "custom_result", "result": [1, 2]}}
    message = AIMessage(content=[
        {"type": "reasoning", "reasoning": "检查反应条件。"},
        {"type": "text", "text": "回答", "annotations": annotations},
        unknown,
        {"type": "reasoning", "reasoning": "推理内容", "text": "另一条说明"},
    ])
    original = message.model_dump(mode="json")
    model = build_chat_model(LLMConfig(provider="openrouter", model="test-model", api_key="test-key"))
    converted, _ = model._create_message_dicts([message], None)
    sent = ChatAssistantMessage.model_validate(converted[0]).model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert sent["reasoning"] == "检查反应条件。"
    restored = [json.loads(b["text"]) for b in sent["content"]]
    assert restored == message.content[1:]
    assert message.model_dump(mode="json") == original


def test_openrouter_preserves_native_reasoning_and_tool_fields():
    from openrouter.components.chatassistantmessage import ChatAssistantMessage

    details = [{"type": "reasoning.text", "text": "native reasoning", "signature": "signature", "index": 0}]
    model = build_chat_model(LLMConfig(provider="openrouter", model="test-model", api_key="test-key"))
    message = AIMessage(content=[], additional_kwargs={"reasoning_details": details},
                        tool_calls=[{"id": "call1", "name": "inspect", "args": {"path": "result.md"}}])
    converted, _ = model._create_message_dicts([message], None)
    sent = ChatAssistantMessage.model_validate(converted[0]).model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert sent["reasoning_details"] == details
    assert sent["tool_calls"][0]["function"]["name"] == "inspect"
