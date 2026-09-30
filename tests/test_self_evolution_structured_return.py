"""Exercise the real DeepAgent completion interface without provider calls."""
from __future__ import annotations

import asyncio
import json
from typing import ClassVar

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from catmaster.runtime.self_evolution.agents import (
    _build_self_evolution_deep_agent,
    _agent_response,
)
from catmaster.runtime.self_evolution.models import ReflectionBatch, ProposerResult, ReviewerResult, TextResult


class Model(FakeMessagesListChatModel):
    bound_tools: ClassVar[list[dict]] = []
    seen_messages: ClassVar[list[list]] = []
    choices: ClassVar[list] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        type(self).bound_tools = [convert_to_openai_tool(t) for t in tools]
        type(self).choices.append(tool_choice)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def build(tmp_path, schema, responses, *, writable=False):
    Model.bound_tools = []
    Model.seen_messages = []
    Model.choices = []
    return _build_self_evolution_deep_agent(
        model=Model(responses=responses),
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        tools=[], investigator_tools=[], system_prompt="Inspect the evidence and finish.",
        response_schema=schema, name="self_evolution_test",
        filesystem_tools=("read_file", "write_file", "edit_file") if writable else ("read_file",),
        allow_mutations=writable,
    )


CASES = [
    (ReflectionBatch, {"items": [{"kind": "no_change", "rationale": "Evidence supports no change."}]}),
    (ProposerResult, {"action": "ignore", "rationale": "The fix belongs in product code."}),
    (ReviewerResult, {"recommendation": "reject", "rationale": "Candidate exceeds the evidence."}),
]


@pytest.mark.parametrize("schema,args", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_prose_ending_is_accepted_without_followup(tmp_path, schema, args, asynchronous):
    prose = "No durable update is needed.\nThe current guidance already covers the issue."
    agent = build(tmp_path, schema, [
        AIMessage(content=prose),
        AIMessage(content="", tool_calls=[{"name": schema.__name__, "args": args, "id": "final"}]),
    ])
    request = {"messages": [HumanMessage(content="Assess this episode.")]}
    result = asyncio.run(agent.ainvoke(request)) if asynchronous else agent.invoke(request)

    assert _agent_response(result, schema) == TextResult(text=prose)
    reminders = [m for m in result["messages"] if isinstance(m, HumanMessage)][1:]
    assert not reminders
    assert len(Model.seen_messages) == 1
    assert Model.choices == ["auto"]
    tool_schema = next(t["function"] for t in Model.bound_tools if t["function"]["name"] == schema.__name__)
    assert tool_schema["description"]
    assert "patch" not in tool_schema["parameters"]["properties"]


@pytest.mark.parametrize("schema,args", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_valid_result_needs_no_followup(tmp_path, schema, args, asynchronous):
    agent = build(tmp_path, schema, [AIMessage(content="", tool_calls=[
        {"name": schema.__name__, "args": args, "id": "final"},
    ])])
    request = {"messages": [HumanMessage(content="Assess.")]}
    result = asyncio.run(agent.ainvoke(request)) if asynchronous else agent.invoke(request)
    assert _agent_response(result, schema) == schema.model_validate(args)
    assert len(Model.seen_messages) == 1
    assert Model.choices == ["auto"]


def test_json_in_prose_is_preserved_without_inventing_a_decision(tmp_path):
    prose = '```json\n{"items":[{"kind":"no_change"}]}\n```'
    agent = build(tmp_path, ReflectionBatch, [AIMessage(content=prose), AIMessage(content=prose)])
    result = agent.invoke({"messages": [HumanMessage(content="Assess.")]})
    assert len(Model.seen_messages) == 1
    assert _agent_response(result, ReflectionBatch) == TextResult(text=prose)


def test_candidate_file_edit_survives_text_completion_without_replay(tmp_path):
    (tmp_path / "proposed").mkdir()
    agent = build(tmp_path, ProposerResult, [
        AIMessage(content="", tool_calls=[{"name": "write_file", "id": "edit", "args": {
            "file_path": "/proposed/note.md", "content": "The actual candidate edit.\n",
        }}]),
        AIMessage(content="I edited the candidate and am finished."),
        AIMessage(content="", tool_calls=[{"name": "ProposerResult", "id": "final", "args": {
            "action": "skill", "group": "research_execution", "name": "example",
        }}]),
    ], writable=True)
    result = agent.invoke({"messages": [HumanMessage(content="Edit the candidate.")]})
    assert (tmp_path / "proposed/note.md").read_text() == "The actual candidate edit.\n"
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "write_file"]) == 1
    assert _agent_response(result, ProposerResult).text == "I edited the candidate and am finished."
    assert len(Model.seen_messages) == 2
    assert Model.choices == ["auto", "auto"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_submission_can_be_corrected_in_the_native_loop(tmp_path, asynchronous):
    schema, args = CASES[0]
    agent = build(tmp_path, schema, [
        AIMessage(content="", tool_calls=[{"name": schema.__name__, "args": {"items": []}, "id": "bad"}]),
        AIMessage(content="", tool_calls=[{"name": schema.__name__, "args": args, "id": "good"}]),
    ])
    request = {"messages": [HumanMessage(content="Assess.")]}
    result = asyncio.run(agent.ainvoke(request)) if asynchronous else agent.invoke(request)
    assert _agent_response(result, schema) == schema.model_validate(args)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "bad" and m.status == "error"
               for m in result["messages"])
    assert Model.choices == ["auto", "auto"]


def test_decision_waits_for_parallel_inspection_results(tmp_path):
    (tmp_path / "evidence.md").write_text("The evidence.\n")
    schema, args = CASES[0]
    agent = build(tmp_path, schema, [
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"file_path": "/evidence.md"}, "id": "read"},
            {"name": schema.__name__, "args": args, "id": "early"},
        ]),
        AIMessage(content="", tool_calls=[{"name": schema.__name__, "args": args, "id": "final"}]),
    ])
    result = agent.invoke({"messages": [HumanMessage(content="Assess.")]})
    assert _agent_response(result, schema) == schema.model_validate(args)
    assert len(Model.seen_messages) == 2
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "early" and m.status == "error"
               for m in result["messages"])


def test_empty_or_reasoning_only_response_is_not_a_text_completion():
    for message in (AIMessage(content=""), AIMessage(content=[{"type": "reasoning", "reasoning": "Thinking."}])):
        with pytest.raises(ValueError, match="without a final text"):
            _agent_response({"messages": [message]}, ReflectionBatch)


def test_openrouter_serialized_request_uses_auto_and_accepts_text(tmp_path, monkeypatch):
    import httpx
    from langchain_openrouter import ChatOpenRouter

    requests = []

    def respond(client, request, **kwargs):
        requests.append(json.loads(request.read()))
        return httpx.Response(200, request=request, json={
            "id": "offline-test", "object": "chat.completion", "created": 1,
            "system_fingerprint": None,
            "model": "xiaomi/mimo-v2.6-pro",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "The evidence does not support a change.",
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })

    monkeypatch.setattr(httpx.Client, "send", respond)
    model = ChatOpenRouter(model="xiaomi/mimo-v2.6-pro", api_key="offline-not-sent", max_retries=0)
    agent = _build_self_evolution_deep_agent(
        model=model, backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        tools=[], investigator_tools=[], system_prompt="Inspect and finish.",
        response_schema=ReflectionBatch, name="self_evolution_test",
        filesystem_tools=("read_file",), allow_mutations=False,
    )
    result = agent.invoke({"messages": [HumanMessage(content="Assess.")]})
    assert _agent_response(result, ReflectionBatch).text == "The evidence does not support a change."
    assert len(requests) == 1
    request = requests[0]
    assert request["tool_choice"] == "auto"
    assert "response_format" not in request
    assert "parallel_tool_calls" not in request
    assert "ReflectionBatch" in {tool["function"]["name"] for tool in request["tools"]}
