from __future__ import annotations
import asyncio
import json
from collections import namedtuple
from types import SimpleNamespace
import pytest
from catmaster.webui.run_projection import RunProjection
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.projections.messages import project_message
from catmaster.webui.projections.events import project_event
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadMessage, MessagePart
from catmaster.webui.thread_store import ThreadStore, new_id

def _service(tmp_path):
    workspace = tmp_path / 'workspace'
    (workspace / 'files').mkdir(parents=True, exist_ok=True)
    store = ThreadStore(workspace=workspace)
    service = SimpleNamespace(workspace=workspace, broker=ThreadEventBroker(workspace=workspace),
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id='workspace'))
    return service, None, store

def test_final_reply_preserves_child_lifecycle_and_registers_declared_reports(tmp_path):
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title="Report and background work")
    user = _user_message(thread.thread_id, "current")
    store.append_message(user)
    store.append_message(ThreadMessage(id="answer", thread_id=thread.thread_id,
        role="assistant", status="streaming", parts=[MessagePart(id="text", type="text")]))
    projection = RunProjection(store=store, broker=service.broker,
        artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
        run_id="run", assistant_message_id="answer", text_part_id="text", input_message_id=user.id)
    report = store.workspace / "files" / "reports" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Scientific result", encoding="utf-8")
    projection._project_async_tasks({"child": {"status": "running", "agent_name": "experiment_specialist"}})
    result = projection.finalize(native_status="success", state={"values": {"messages": [
        {"type": "human", "id": user.id, "content": "current"},
        {"type": "ai", "id": "final", "content": "[报告](sandbox:/reports/report.md)\n\n## 输出文件\n- `reports/report.md`"},
    ]}})
    assert next(part for part in result.parts if part.type == "subagent").status == "running"
    assert len([part for part in result.parts if part.type == "artifact"]) == 1


def test_retained_checkpoint_tasks_do_not_publish_lifecycle_events(tmp_path):
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title="Imported conversation")
    store.append_message(ThreadMessage(id="answer", thread_id=thread.thread_id,
        role="assistant", parts=[MessagePart(id="text", type="text")]))
    projection = RunProjection(store=store, broker=service.broker,
        artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
        run_id="run", assistant_message_id="answer", text_part_id="text")
    retained = {"async_tasks": {"old-child": {"status": "running", "agent_name": "litreview_agent"}}}
    for event in ["values", "updates"]:
        projection.process({"event": event, "data": {"model": retained}})
    assert not any(part.type == "subagent" for part in store.get_message(thread.thread_id, "answer").parts)
    assert not any(event.event.startswith("subagent.") for event in service.broker.replay(thread.thread_id))
    # Only actual local execution publishes a child lifecycle transition.
    projection._project_async_tasks({"new-child": {"status": "pending", "agent_name": "writing_specialist"}})
    assert any(event.event == "subagent.started" for event in service.broker.replay(thread.thread_id))


def test_positioned_native_text_is_durable_before_event_cursor(tmp_path):
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title="Reload while streaming")
    store.append_message(ThreadMessage(id="answer", thread_id=thread.thread_id,
        role="assistant", parts=[MessagePart(id="text", type="text")]))
    projection = RunProjection(store=store, broker=service.broker,
        artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
        run_id="run", assistant_message_id="answer", text_part_id="text")
    for fragment in ["🧪 E=-", "409", "0.864"]:
        projection.process({"type": "messages", "ns": [], "data": [
            {"type": "AIMessageChunk", "id": "native", "content": fragment}, {}]})
    separate_reader = ThreadStore(workspace=store.workspace)
    message = separate_reader.get_message(thread.thread_id, "answer")
    assert message.parts[0].text == "🧪 E=-4090.864"
    events = [event for event in service.broker.replay(thread.thread_id) if event.event == "message.delta"]
    assert [event.data["text_offset"] for event in events] == [0, 6, 9]


def test_runtime_notification_keeps_turn_role_and_separate_ui_authorship():
    message = ThreadMessage(id="notice", thread_id="thread", role="user",
        parts=[MessagePart(id="text", type="text", text="Read the completed specialist result.")],
        meta={"origin": "runtime", "notification_title": "Experiment · success"})
    public = project_message(message)
    assert public.role == "user"
    assert public.origin == "runtime"
    assert public.notification_title == "Experiment · success"
    assert public.parts[0].text == message.parts[0].text


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_compaction_is_durable_progress_not_reply_or_final_fallback(tmp_path, version):
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title='Compaction')
    store.append_message(ThreadMessage(id='answer', thread_id=thread.thread_id,
        role='assistant', status='streaming', parts=[MessagePart(id='text', type='text')]))

    def projector():
        return RunProjection(store=store, broker=service.broker,
            artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
            run_id='run', assistant_message_id='answer', text_part_id='text')

    def event(message, metadata):
        if version == 'v1':
            return namedtuple('StreamPart', 'event data')('messages', [message, metadata])
        return {'type': 'messages', 'ns': [], 'data': [message, metadata]}

    projection = projector()
    for fragment in ['## SESSION INTENT', '\nInternal summary']:
        projection.process(event(
            {'type': 'AIMessageChunk', 'id': 'summary', 'content': [
                {'type': 'text', 'text': fragment}, {'type': 'reasoning', 'reasoning': 'Internal reasoning'},
            ]}, {'lc_source': 'summarization', 'lc_internal_call': 'server-token', 'langgraph_node': 'model'},
        ))
    projection.process(event({'type': 'ai', 'id': 'internal', 'content': 'Other internal output'},
                             {'lc_internal_call': 'server-token'}))
    public = project_message(ThreadStore(workspace=store.workspace).get_message(thread.thread_id, 'answer'))
    progress = [part for part in public.parts if part.type == 'progress']
    assert len(progress) == 1 and progress[0].status == 'running'
    assert progress[0].text == '正在压缩上下文…'
    assert projection.final_text_from_state({}) == ('', '')
    assert not any(e.event in {'message.delta', 'reasoning.delta'} for e in service.broker.replay(thread.thread_id))
    assert 'Internal' not in str(public.model_dump())

    # A WebUI restart restores the activity even when rejoin begins after the
    # summarizer's chunks. Ordinary answer text may legitimately name a heading.
    projection = projector()
    reply = 'SESSION INTENT is the heading you asked about.'
    projection.process(event({'type': 'ai', 'id': 'reply', 'content': reply}, {'langgraph_node': 'model'}))
    projected_events = [project_event(e) for e in service.broker.replay(thread.thread_id)]
    updates = [e for e in projected_events if e.event == 'activity.updated' and e.data.part]
    assert updates[-1].data.part.status == 'completed'
    assert updates[-1].data.part.text == '上下文已压缩，继续处理请求。'
    assert projection.final_text_from_state({}) == (reply, 'reply')


def test_real_deepagent_summary_stream_uses_progress_projection(tmp_path):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title='Real summary')
    store.append_message(ThreadMessage(id='answer', thread_id=thread.thread_id,
        role='assistant', status='streaming', parts=[MessagePart(id='text', type='text')]))
    projection = RunProjection(store=store, broker=service.broker,
        artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
        run_id='run', assistant_message_id='answer', text_part_id='text')
    backend = FilesystemBackend(root_dir=store.workspace / 'files', virtual_mode=True)
    graph = create_deep_agent(
        model=Model(responses=[AIMessage(content='Reports are ready')]), backend=backend,
        middleware=[SummarizationMiddleware(
            model=Model(responses=[AIMessage(content='Internal summary sentinel')]),
            backend=backend, trigger=('messages', 2), keep=('messages', 1),
        )],
    )

    async def scenario():
        sources = []
        async for chunk in graph.astream({'messages': [
            HumanMessage(content='Prior request'), AIMessage(content='Prior result'),
            HumanMessage(content='Gather reports'),
        ]}, stream_mode='messages', version='v2'):
            sources.append(chunk['data'][1].get('lc_source'))
            projection.process(chunk)
        assert 'summarization' in sources
        assert projection.final_text_from_state({})[0] == 'Reports are ready'
        public = project_message(store.get_message(thread.thread_id, 'answer'))
        assert 'Internal summary sentinel' not in str(public.model_dump())
        assert any(p.type == 'progress' and p.status == 'completed' for p in public.parts)

    asyncio.run(scenario())


def test_compaction_uses_task_identity_across_provider_message_ids_and_reload(tmp_path):
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Compaction identities')
    store.append_message(ThreadMessage(id='answer', thread_id=thread.thread_id,
        role='assistant', parts=[MessagePart(id='text', type='text')]))

    def projector():
        return RunProjection(store=store, broker=service.broker, artifact_registry=service.artifact_registry,
            thread_id=thread.thread_id, run_id='run', assistant_message_id='answer', text_part_id='text')

    def emit(projection, message_id, *, task='model:first', last=False):
        projection.process({'type': 'messages', 'ns': [], 'data': [
            {'type': 'AIMessageChunk', 'id': message_id, 'content': 'Internal summary',
             **({'chunk_position': 'last'} if last else {})},
            {'lc_source': 'summarization', 'lc_internal_call': 'same-process-marker',
             'langgraph_checkpoint_ns': task},
        ]})

    projection = projector()
    emit(projection, 'resp_first')
    projection = projector()  # Browser/server replay midway through the call.
    emit(projection, 'lc_run--first')
    emit(projection, 'resp_first', last=True)
    parts = store.get_message(thread.thread_id, 'answer').parts
    assert [(p.type, p.status) for p in parts if p.type == 'trace'] == [('trace', 'completed')]
    # No ordinary answer is needed to close the completed summary. Interrupting
    # the next model request must not relabel that summary as interrupted.
    emit(projection, 'resp_second', task='model:second')
    projection.finalize(native_status='interrupted')
    parts = store.get_message(thread.thread_id, 'answer').parts
    assert [p.status for p in parts if p.type == 'trace'] == ['completed', 'interrupted']
    assert not any(p.text == 'Internal summary' for p in parts)


def test_codex_summary_stream_projects_one_activity_per_actual_call(tmp_path):
    import httpx
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.messages import AIMessage, HumanMessage
    from test_provider_request_capture import make_model, response_stream

    requests = []
    def handler(request):
        requests.append(request)
        body = response_stream().replace('resp_test', f'resp_{len(requests)}').replace('msg_test', f'msg_{len(requests)}')
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=body)

    model = make_model(handler)
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Actual Codex stream')
    store.append_message(ThreadMessage(id='answer', thread_id=thread.thread_id,
        role='assistant', parts=[MessagePart(id='text', type='text')]))
    projection = RunProjection(store=store, broker=service.broker, artifact_registry=service.artifact_registry,
        thread_id=thread.thread_id, run_id='run', assistant_message_id='answer', text_part_id='text')
    backend = FilesystemBackend(root_dir=store.workspace / 'files', virtual_mode=True)
    graph = create_deep_agent(model=model, backend=backend, middleware=[SummarizationMiddleware(
        model=model, backend=backend, trigger=('messages', 2), keep=('messages', 1))])

    async def scenario():
        async for chunk in graph.astream({'messages': [HumanMessage(content='Old question'),
            AIMessage(content='Old result'), HumanMessage(content='Continue')]}, stream_mode='messages', version='v2'):
            projection.process(chunk)
    asyncio.run(scenario())
    assert len(requests) == 2  # One summary call followed by the research call.
    progress = [p for p in store.get_message(thread.thread_id, 'answer').parts if p.type == 'trace']
    assert len(progress) == 1 and progress[0].status == 'completed'
    assert projection.final_text_from_state({})[0] == 'OK'


def _user_message(thread_id: str, text: str) -> ThreadMessage:
    return ThreadMessage(
        id=new_id("msg"),
        thread_id=thread_id,
        role="user",
        status="completed",
        parts=[MessagePart(id=new_id("part_text"), type="text", text=text)],
    )


def test_projection_ignores_state_replacement_and_stops_at_next_user(tmp_path: Any) -> None:
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title="Projection")
    user = _user_message(thread.thread_id, "current")
    assistant = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id=new_id("part_text"), type="text", status="streaming")],
    )
    store.append_message(user)
    store.append_message(assistant)
    observed: list[dict[str, Any]] = []
    projection = RunProjection(
        store=store,
        broker=service.broker,
        artifact_registry=service.artifact_registry,
        thread_id=thread.thread_id,
        run_id="run-current",
        assistant_message_id=assistant.id,
        text_part_id=assistant.parts[0].id,
        input_message_id=user.id,
        on_async_task=observed.append,
    )
    projection.process(
        {
            "type": "updates",
            "ns": [],
            "data": {
                "PatchToolCallsMiddleware.before_agent": {
                    "messages": [
                        {"type": "ai", "id": "historical", "content": "must not render"}
                    ]
                }
            },
        }
    )
    projection.process(
        {
            "type": "updates",
            "ns": [],
            "data": {
                "agent": {
                    "async_tasks": {
                        "child": {
                            "agent_name": "experiment_specialist",
                            "thread_id": "child",
                            "run_id": "child-run",
                            "status": "running",
                        }
                    }
                }
            },
        }
    )
    projection.process(
        {
            "type": "messages",
            "ns": [],
            "data": [
                {"type": "ai", "id": "current-answer", "content": "current result"},
                {"run_id": "run-current"},
            ],
        }
    )
    streaming = store.get_message(thread.thread_id, assistant.id)
    assert streaming is not None
    assert next(part for part in streaming.parts if part.type == "text").text == (
        "current result"
    )
    text_events = [
        event
        for event in service.broker.replay(thread.thread_id)
        if event.event == "message.delta"
    ]
    assert [event.data["delta"] for event in text_events] == ["current result"]
    state = {
        "values": {
            "messages": [
                {"type": "human", "id": user.id, "content": "current"},
                {"type": "ai", "id": "current-answer", "content": "current result"},
                {"type": "human", "id": "next-user", "content": "next"},
                {"type": "ai", "id": "next-answer", "content": "wrong run"},
            ]
        }
    }
    completed = projection.finalize(native_status="success", state=state)

    assert [part.text for part in completed.parts if part.type == "text"] == [
        "current result"
    ]
    assert "must not render" not in str(completed.model_dump(mode="json"))
    assert not any(part.type == "subagent" for part in completed.parts)
    assert observed == []
    assert len(
        [message for message in store.list_messages(thread.thread_id) if message.role == "assistant"]
    ) == 1


def test_projection_accepts_sdk_v1_rejoin_parts_and_filters_subgraphs(
    tmp_path: Any,
) -> None:
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title="Rejoin")
    user = _user_message(thread.thread_id, "continue")
    assistant = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id=new_id("part_text"), type="text", status="streaming")],
    )
    store.append_message(user)
    store.append_message(assistant)
    projection = RunProjection(
        store=store,
        broker=service.broker,
        artifact_registry=service.artifact_registry,
        thread_id=thread.thread_id,
        run_id="run-rejoined",
        assistant_message_id=assistant.id,
        text_part_id=assistant.parts[0].id,
        input_message_id=user.id,
    )
    StreamPart = namedtuple("StreamPart", "event data id")
    projection.process(
        StreamPart(
            "messages-tuple|child:worker",
            [
                {"type": "ai", "id": "child-output", "content": "must stay hidden"},
                {},
            ],
            "1",
        )
    )
    projection.process(
        StreamPart(
            "messages-tuple",
            [
                {"type": "ai", "id": "root-output", "content": "rejoined answer"},
                {},
            ],
            "2",
        )
    )
    completed = projection.finalize(
        native_status="success",
        state={
            "values": {
                "messages": [
                    {"type": "human", "id": user.id, "content": "continue"},
                    {"type": "ai", "id": "root-output", "content": "rejoined answer"},
                ]
            }
        },
    )

    assert [part.text for part in completed.parts if part.type == "text"] == [
        "rejoined answer"
    ]
    assert "must stay hidden" not in str(completed.model_dump(mode="json"))


def test_projection_maps_native_custom_tool_content_blocks(tmp_path: Any) -> None:
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title="Custom tool")
    user = _user_message(thread.thread_id, "edit the file")
    assistant = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id=new_id("part_text"), type="text", status="streaming")],
    )
    store.append_message(user)
    store.append_message(assistant)
    projection = RunProjection(
        store=store,
        broker=service.broker,
        artifact_registry=service.artifact_registry,
        thread_id=thread.thread_id,
        run_id="run-custom-tool",
        assistant_message_id=assistant.id,
        text_part_id=assistant.parts[0].id,
        input_message_id=user.id,
    )

    projection.process(
        {
            "type": "messages",
            "ns": [],
            "data": [
                {
                    "type": "ai",
                    "id": "ai-custom-tool",
                    "content": [
                        {
                            "type": "custom_tool_call",
                            "name": "apply_patch",
                            "input": "*** Begin Patch\n*** End Patch",
                            "call_id": "patch-ui",
                        }
                    ],
                },
                {"run_id": "run-custom-tool"},
            ],
        }
    )

    saved = store.get_message(thread.thread_id, assistant.id)
    assert saved is not None
    tool_part = next(part for part in saved.parts if part.type == "tool-call")
    assert tool_part.meta["tool"] == "apply_patch"
    assert tool_part.meta["tool_call_id"] == "patch-ui"
    assert tool_part.meta["input"] == {
        "partial": "*** Begin Patch\n*** End Patch"
    }


def test_projection_coalesces_streaming_tool_call_chunks_by_index(
    tmp_path: Any,
) -> None:
    service, _client, store = _service(tmp_path)
    thread = store.create_thread(title="Streaming tool")
    assistant = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id="part_text", type="text", status="streaming")],
    )
    store.append_message(assistant)
    projection = RunProjection(
        store=store,
        broker=service.broker,
        artifact_registry=service.artifact_registry,
        thread_id=thread.thread_id,
        run_id="run-streamed-tool",
        assistant_message_id=assistant.id,
        text_part_id="part_text",
    )

    for tool_call_chunks in (
        [{"name": "demo", "args": "", "id": "call_streamed", "index": 0}],
        [{"name": None, "args": '{"query": "', "id": None, "index": 0}],
        [{"name": None, "args": 'CO2"}', "id": None, "index": 0}],
    ):
        projection.process(
            {
                "type": "messages",
                "ns": [],
                "data": [
                    {
                        "type": "AIMessageChunk",
                        "id": "model-streamed",
                        "content": [{
                            "type": "function_call",
                            "call_id": tool_call_chunks[0].get("id"),
                            "name": tool_call_chunks[0].get("name"),
                            "arguments": tool_call_chunks[0].get("args"),
                            "index": tool_call_chunks[0].get("index"),
                        }],
                        # LangChain derives this field from tool_call_chunks;
                        # both are present on the serialized chunk.
                        "tool_calls": [
                            {
                                "name": str(tool_call_chunks[0].get("name") or ""),
                                "args": {},
                                "id": tool_call_chunks[0].get("id"),
                            }
                        ],
                        "tool_call_chunks": tool_call_chunks,
                    },
                    {},
                ],
            }
        )
    projection.process(
        {
            "type": "messages",
            "ns": [],
            "data": [
                {
                    "type": "AIMessage",
                    "id": "model-streamed",
                    "content": [
                        {
                            "type": "function_call",
                            "id": "fc_provider_item",
                            "call_id": "call_streamed",
                            "name": "demo",
                            "arguments": '{"query": "CO2"}',
                        }
                    ],
                    "tool_calls": [
                        {
                            "name": "demo",
                            "args": {"query": "CO2"},
                            "id": "call_streamed",
                        }
                    ],
                },
                {},
            ],
        }
    )
    projection.process(
        {
            "type": "messages",
            "ns": [],
            "data": [
                {
                    "type": "ToolMessage",
                    "name": "demo",
                    "tool_call_id": "call_streamed",
                    "content": "done",
                },
                {},
            ],
        }
    )

    saved = store.get_message(thread.thread_id, assistant.id)
    assert saved is not None
    tool_parts = [part for part in saved.parts if part.type == "tool-call"]
    assert len(tool_parts) == 1
    assert tool_parts[0].id == "part_tool_call_streamed"
    assert tool_parts[0].status == "completed"
    assert tool_parts[0].meta["tool"] == "demo"
    assert tool_parts[0].meta["input"] == {"query": "CO2"}
    assert tool_parts[0].meta["output"] == "done"
    event_names = [event.event for event in service.broker.replay(thread.thread_id)]
    assert event_names.count("message.part.created") == 1
    assert event_names.count("tool_call.started") == 1
    assert event_names.count("tool_call.completed") == 1


def test_projection_registers_explicit_final_report_files_once(tmp_path: Any) -> None:
    service, _client, store = _service(tmp_path)
    report = service.workspace / "files" / "reports" / "final.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("evidence", encoding="utf-8")
    thread = store.create_thread(title="Artifacts")
    user = _user_message(thread.thread_id, "finish")
    assistant = ThreadMessage(
        id=new_id("msg"),
        thread_id=thread.thread_id,
        role="assistant",
        status="streaming",
        parts=[MessagePart(id=new_id("part_text"), type="text", status="streaming")],
    )
    store.append_message(user)
    store.append_message(assistant)
    projection = RunProjection(
        store=store,
        broker=service.broker,
        artifact_registry=service.artifact_registry,
        thread_id=thread.thread_id,
        run_id="run-artifact",
        assistant_message_id=assistant.id,
        text_part_id=assistant.parts[0].id,
        input_message_id=user.id,
    )
    text = (
        "## Summary\nFinished.\n\n"
        "## Files\n- `reports/final.md`\n- `(none reported)`"
    )
    state = {
        "values": {
            "messages": [
                {"type": "human", "id": user.id, "content": "finish"},
                {"type": "ai", "id": "native-final", "content": text},
            ]
        }
    }
    projection.process(
        {
            "type": "messages",
            "ns": [],
            "data": [
                {"type": "ai", "id": "native-final", "content": text},
                {"run_id": "run-artifact"},
            ],
        }
    )

    first = projection.finalize(native_status="success", state=state)
    second = projection.finalize(native_status="success", state=state)

    artifact_parts = [part for part in second.parts if part.type == "artifact"]
    assert len(artifact_parts) == 1
    assert artifact_parts[0].path == "files/reports/final.md"
    records = service.artifact_registry.list_artifacts(thread_id=thread.thread_id)
    assert [record.artifact_id for record in records] == [
        artifact_parts[0].artifact_id
    ]
    assert [part.text for part in first.parts if part.type == "text"] == [text]


def test_in_turn_research_notice_does_not_hide_recovered_answer_or_cross_real_next_turn(tmp_path):
    service, _, store = _service(tmp_path)
    thread = store.create_thread(title='Research collaboration')
    projection = RunProjection(store=store, broker=service.broker,
        artifact_registry=service.artifact_registry, thread_id=thread.thread_id,
        run_id='run', assistant_message_id='answer', text_part_id='text', input_message_id='input')
    state = {'values': {'messages': [
        {'type': 'human', 'id': 'input', 'content': 'Research'},
        {'type': 'ai', 'id': 'early', 'content': 'Still comparing'},
        {'type': 'human', 'id': 'notice', 'content': 'Read shared evidence', 'additional_kwargs': {'catmaster_in_turn': True}},
        {'type': 'ai', 'id': 'final', 'content': 'Controls reconciled; report ready'},
        {'type': 'human', 'id': 'next-user', 'content': 'An unrelated request'},
        {'type': 'ai', 'id': 'next-answer', 'content': 'The later answer'},
    ]}}
    assert projection.final_text_from_state(state) == ('Controls reconciled; report ready', 'final')
    projection.process({'type': 'messages', 'ns': [], 'data': [state['values']['messages'][2], {}]})
    assert not service.broker.replay(thread.thread_id)
