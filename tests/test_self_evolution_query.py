from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, ObservabilityStore
from catmaster.runtime.self_evolution.agents import (
    ProposerAgent,
    _OptionalResultMiddleware,
    _self_evolution_backend,
    _self_evolution_filesystem_middleware,
    _trace_query_tools,
)
from catmaster.runtime.self_evolution.models import Observation, ReflectionBatch
from catmaster.runtime.self_evolution.query import EvolutionHistoryScope, EvolutionTraceScope
from catmaster.runtime.self_evolution.storage import SelfEvolutionStore, utc_now
from catmaster.runtime.self_evolution.trace import TurnTrace, semantic_event_payload
from catmaster.runtime.self_evolution.usage import usage_invocation


def _trace(run_id: str) -> TurnTrace:
    return TurnTrace(
        run_id=run_id,
        thread_id=f"thread_{run_id}",
        entrypoint="research",
        status="done",
        user_prompt="Audit this completed run.",
        final_answer="The requested work completed.",
        summary="Completed.",
        task_outcome="verified_success",
        outcome_ref=f"run:{run_id}#event:1",
    )


def _scope_with_semantic_events(tmp_path: Path) -> tuple[EvolutionTraceScope, str]:
    run_id = "run_query"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    store = ObservabilityStore(run_dir)
    store.record_raw_callback(
        "RUN_START",
        category="run",
        run_id=run_id,
        payload={"status": "running", "goal": "Inspect the trajectory."},
    )
    store.record_raw_callback(
        "LLM_RAW_RESPONSE",
        category="model",
        run_id=run_id,
        payload={
            "callback_run_id": "model_call_1",
            "parent_callback_run_id": "agent_turn_1",
            "model": "test-model",
            "generations": [
                {
                    "response_text": "VISIBLE MODEL ANSWER",
                    "content_blocks": [
                        {"type": "text", "text": "VISIBLE MODEL ANSWER"},
                        {
                            "type": "reasoning",
                            "reasoning": "readable reasoning",
                            "encrypted_content": "CIPHERTEXT",
                        },
                    ],
                    "parsed_tool_calls": [
                        {
                            "id": "call_1",
                            "name": "inspect_file",
                            "args": {"path": "notes.md"},
                        }
                    ],
                    "raw_tool_calls": [{"provider_envelope": "DO NOT EXPOSE"}],
                }
            ],
            "provider_response": "DO NOT EXPOSE",
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        },
    )
    # The compact callback record is the same model occurrence and must not be
    # projected a second time when the raw semantic record exists.
    store.record_raw_callback(
        "LLM_CALL_END",
        category="model",
        run_id=run_id,
        payload={
            "callback_run_id": "model_call_1",
            "text_preview": "VISIBLE MODEL ANSWER",
            "tool_calls": ["inspect_file"],
        },
    )
    store.record_raw_callback(
        "TOOL_RAW_INPUT",
        category="tool",
        run_id=run_id,
        payload={
            "callback_run_id": "tool_call_1",
            "parent_callback_run_id": "model_call_1",
            "tool": "inspect_file",
            "params_full": {"path": "notes.md"},
        },
    )
    store.record_raw_callback(
        "TOOL_RAW_OUTPUT",
        category="tool",
        run_id=run_id,
        payload={
            "callback_run_id": "tool_call_1",
            "tool": "inspect_file",
            "raw_output": "RAW TOOL TRANSPORT",
            "projection": {
                "content_text": "MODEL-VISIBLE TOOL RESULT",
                "warnings": [],
                "offload_refs": [],
            },
            "status": "ok",
        },
    )
    store.record_raw_callback(
        "TOOL_CALL_END",
        category="tool",
        run_id=run_id,
        payload={
            "callback_run_id": "tool_call_1",
            "tool": "inspect_file",
            "result": "MODEL-VISIBLE TOOL RESULT",
            "status": "ok",
        },
    )
    return EvolutionTraceScope({run_id: (run_dir, _trace(run_id))}), run_id


def test_trace_query_is_complete_read_only_and_semantically_normalized(
    tmp_path: Path,
) -> None:
    scope, run_id = _scope_with_semantic_events(tmp_path)

    result = scope.execute(
        """
        WITH model_events AS (
            SELECT run_id, id, payload_json
            FROM trajectory_events
            WHERE name = 'MODEL_RESPONSE'
        )
        SELECT r.run_id, e.id,
               json_extract(e.payload_json, '$.generations[0].assistant_text') AS answer
        FROM trajectory_runs AS r
        JOIN model_events AS e USING (run_id)
        ORDER BY e.id
        """
    )
    assert result["rows"] == [
        {"run_id": run_id, "id": result["rows"][0]["id"], "answer": "VISIBLE MODEL ANSWER"}
    ]

    inventory = scope.execute(
        "SELECT name, COUNT(*) AS count FROM trajectory_events GROUP BY name ORDER BY name"
    )["rows"]
    assert inventory == [
        {"name": "MODEL_RESPONSE", "count": 1},
        {"name": "RUN_START", "count": 1},
        {"name": "TOOL_INPUT", "count": 1},
        {"name": "TOOL_RESULT", "count": 1},
    ]
    causal_rows = scope.execute(
        """
        SELECT name, callback_run_id, parent_callback_run_id
        FROM trajectory_events
        WHERE name IN ('MODEL_RESPONSE', 'TOOL_INPUT')
        ORDER BY id
        """
    )["rows"]
    assert causal_rows == [
        {
            "name": "MODEL_RESPONSE",
            "callback_run_id": "model_call_1",
            "parent_callback_run_id": "agent_turn_1",
        },
        {
            "name": "TOOL_INPUT",
            "callback_run_id": "tool_call_1",
            "parent_callback_run_id": "model_call_1",
        },
    ]
    payloads = "\n".join(
        row["payload_json"]
        for row in scope.execute(
            "SELECT payload_json FROM trajectory_events ORDER BY id"
        )["rows"]
    )
    assert "CIPHERTEXT" not in payloads
    assert "DO NOT EXPOSE" not in payloads
    assert "RAW TOOL TRANSPORT" not in payloads
    assert "MODEL-VISIBLE TOOL RESULT" in payloads
    assert '"name": "inspect_file"' in payloads

    with pytest.raises(ValueError, match="rejected"):
        scope.execute("SELECT name FROM sqlite_master")
    with pytest.raises(ValueError, match="rejected"):
        scope.execute("DELETE FROM trajectory_runs")
    with pytest.raises(ValueError, match="rejected"):
        scope.execute("SELECT * FROM trace_0.observation_events")

    index = scope.index(context_label="reflection", skill_context="catalog only")
    assert "VISIBLE MODEL ANSWER" not in index
    assert "MODEL-VISIBLE TOOL RESULT" not in index
    assert f"run:{run_id}#event:" in index
    assert "Counts by event" not in index
    assert "event range" not in index.casefold()


def test_empty_event_payload_is_reported_and_raw_value_stays_reachable(
    tmp_path: Path,
) -> None:
    scope, run_id = _scope_with_semantic_events(tmp_path)
    db_path = tmp_path / run_id / OBSERVABILITY_DB_NAME
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE observation_events SET payload_json = '' WHERE name = 'RUN_START'"
        )

    normalized = scope.execute(
        "SELECT payload_json FROM trajectory_events WHERE name = 'RUN_START'"
    )["rows"][0]["payload_json"]
    raw = scope.execute(
        "SELECT payload_json FROM trajectory_raw_events WHERE name = 'RUN_START'"
    )["rows"][0]["payload_json"]

    diagnostic = json.loads(normalized)["_payload_parse_error"]
    assert diagnostic["error_type"] == "JSONDecodeError"
    assert diagnostic["raw_payload"] == ""
    assert raw == ""


def test_malformed_tool_args_are_diagnostic_instead_of_an_empty_object() -> None:
    payload = semantic_event_payload(
        "LLM_RAW_RESPONSE",
        {
            "generations": [
                {
                    "parsed_tool_calls": [
                        {"id": "call-bad", "name": "inspect", "args_json": ""}
                    ]
                }
            ]
        },
    )

    diagnostic = payload["generations"][0]["tool_calls"][0]["args"][
        "_args_parse_error"
    ]
    assert diagnostic["error_type"] == "JSONDecodeError"
    assert diagnostic["raw_args"] == ""


def test_oversized_trace_query_rejects_without_truncation_and_event_reader_continues(
    tmp_path: Path,
) -> None:
    run_id = "run_large"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    large_text = "0123456789abcdef" * 75_000
    store = ObservabilityStore(run_dir)
    store.record_raw_callback(
        "LLM_RAW_RESPONSE",
        category="model",
        run_id=run_id,
        payload={
            "callback_run_id": "large_model_call",
            "generations": [{"response_text": large_text}],
        },
    )
    scope = EvolutionTraceScope({run_id: (run_dir, _trace(run_id))})
    event = scope.execute(
        "SELECT id, payload_chars FROM trajectory_events WHERE name = 'MODEL_RESPONSE'"
    )["rows"][0]
    handle = f"run:{run_id}#event:{event['id']}"

    with pytest.raises(ValueError, match="above the 100000-byte result boundary"):
        scope.execute(
            "SELECT payload_json FROM trajectory_events WHERE name = 'MODEL_RESPONSE'"
        )

    pieces: list[str] = []
    offset = 0
    while True:
        page = scope.read_event(
            handle=handle,
            field="$.generations[0].assistant_text",
            offset=offset,
            limit=100_000,
        )
        pieces.append(page["segment"])
        if page["next_offset"] is None:
            break
        offset = int(page["next_offset"])
    assert "".join(pieces) == large_text
    assert scope.contains_event(handle)
    assert not scope.contains_event(f"run:outside#event:{event['id']}")


def test_trace_query_tool_schemas_are_non_nullable(tmp_path: Path) -> None:
    scope, _run_id = _scope_with_semantic_events(tmp_path)
    tools = {tool.name: tool for tool in scope.tools()}
    assert set(tools) == {"query_evolution_trace_sql", "read_evolution_event"}
    for tool in tools.values():
        schema = convert_to_openai_tool(tool)["function"]["parameters"]
        assert '"type": "null"' not in json.dumps(schema)
    reader = tools["read_evolution_event"].args_schema.model_json_schema()
    assert reader["properties"]["field"]["default"] == "payload_json"
    assert reader["properties"]["offset"]["default"] == 0
    query = tools["query_evolution_trace_sql"]
    query_schema = query.args_schema.model_json_schema()
    query_surface = query.description + json.dumps(query_schema, sort_keys=True)
    assert "trajectory_runs(run_id, thread_id, entrypoint, status" in query_surface
    assert "trajectory_events(id, run_id, handle," in query_surface


@pytest.mark.parametrize("view", ["trajectory_events", "trajectory_raw_events"])
def test_every_returned_handle_reads_the_same_body_and_metadata(tmp_path: Path, view: str) -> None:
    scope, _ = _scope_with_semantic_events(tmp_path)
    rows = scope.execute(
        f"SELECT handle, name, callback_run_id, payload_json, payload_chars FROM {view} ORDER BY id"
    )["rows"]
    assert rows
    reader = {tool.name: tool for tool in scope.tools()}["read_evolution_event"]
    for event in rows:
        message = reader.invoke({
            "type": "tool_call", "id": "read", "name": reader.name,
            "args": {"handle": event["handle"]},
        })
        assert message.status == "success"
        result = json.loads(message.content)
        assert result["segment"] == event["payload_json"]
        assert result["total_chars"] == event["payload_chars"]
        assert scope.contains_event(event["handle"])
        for field in ("name", "callback_run_id"):
            assert scope.read_event(handle=event["handle"], field=field)["segment"] == event[field]


def test_legacy_raw_only_reference_remains_readable(tmp_path: Path) -> None:
    scope, run_id = _scope_with_semantic_events(tmp_path)
    raw = scope.execute(
        "SELECT id, handle, payload_json FROM trajectory_raw_events WHERE name='LLM_CALL_END'"
    )["rows"][0]
    result = scope.read_event(handle=f"run:{run_id}#event:{raw['id']}")
    assert result["segment"] == raw["payload_json"]
    assert result["handle"] == raw["handle"]
    assert scope.read_event(handle=raw["handle"], field="$.callback_run_id")["segment"] == "model_call_1"


def test_raw_json_continuation_and_real_null_remain_distinct_from_missing(tmp_path: Path) -> None:
    scope, run_id = _scope_with_semantic_events(tmp_path)
    text = "分段内容abc" * 10_000
    with sqlite3.connect(tmp_path / run_id / OBSERVABILITY_DB_NAME) as conn:
        conn.execute("UPDATE observation_events SET payload_json=? WHERE name='LLM_CALL_END'",
                     (json.dumps({"text": text, "optional": None, "flag": False}),))
    event = scope.execute("SELECT handle FROM trajectory_raw_events WHERE name='LLM_CALL_END'")["rows"][0]
    pieces = []
    offset = 0
    while True:
        part = scope.read_event(handle=event["handle"], field="$.text", offset=offset, limit=9000)
        pieces.append(part["segment"])
        if part["next_offset"] is None:
            break
        offset = part["next_offset"]
    assert "".join(pieces) == text
    assert scope.read_event(handle=event["handle"], field="$.optional")["segment"] == "null"
    assert scope.read_event(handle=event["handle"], field="$.flag")["segment"] == "false"


def test_recoverable_query_errors_are_recorded_as_errors(tmp_path: Path) -> None:
    scope, _ = _scope_with_semantic_events(tmp_path)
    query = {tool.name: tool for tool in scope.tools()}["query_evolution_trace_sql"]
    with usage_invocation(
        workspace=tmp_path / "workspace", stage="self_evolution_reflector",
        model_label="test", context={},
    ) as (config, cb):
        message = query.invoke({
            "type": "tool_call", "id": "bad-sql", "name": query.name,
            "args": {"sql": "SELECT absent_column FROM trajectory_events"},
        }, config=config)
        assert message.status == "error"
    with sqlite3.connect(cb.run_dir / OBSERVABILITY_DB_NAME) as conn:
        rows = conn.execute(
            "SELECT payload_json FROM observation_events WHERE name='TOOL_RAW_OUTPUT'"
        ).fetchall()
    assert len(rows) == 1
    payload = json.loads(rows[0][0])
    assert payload["tool_status"] == "error"
    assert payload["raw_output"]["status"] == "error"
    assert json.loads(payload["raw_output"]["content"])["ok"] is False


def test_trace_aggregations_and_pagination_preserve_read_scope(tmp_path: Path) -> None:
    scope, _ = _scope_with_semantic_events(tmp_path)
    for view in ("trajectory_events", "trajectory_raw_events", "trajectory_runs"):
        expected = scope.execute(f"SELECT * FROM {view}")["row_count"]
        assert scope.execute(f"SELECT COUNT(*) AS n FROM {view}")["rows"] == [{"n": expected}]
    assert scope.execute("SELECT COUNT(id) AS n FROM trajectory_raw_events")["rows"] == [{"n": 6}]
    first = scope.execute("SELECT handle FROM trajectory_raw_events ORDER BY id LIMIT 3")["rows"]
    rest = scope.execute("SELECT handle FROM trajectory_raw_events ORDER BY id LIMIT 3 OFFSET 3")["rows"]
    assert first + rest == scope.execute("SELECT handle FROM trajectory_raw_events ORDER BY id")["rows"]
    for sql in (
        "SELECT payload_json FROM trace_selected.observation_events",
        "SELECT name FROM sqlite_master",
        "UPDATE trajectory_runs SET status='failed'",
        "SELECT load_extension('anything')",
    ):
        with pytest.raises(ValueError, match="rejected"):
            scope.execute(sql)


def test_event_errors_distinguish_missing_run_event_and_field(tmp_path: Path) -> None:
    scope, run_id = _scope_with_semantic_events(tmp_path)
    reader = {tool.name: tool for tool in scope.tools()}["read_evolution_event"]
    cases = [
        ({"handle": f"run:{run_id}#event:999"}, "not found"),
        ({"handle": "run:other#event:1"}, "run scope"),
        ({"handle": "malformed"}, "handle"),
        ({"handle": f"run:{run_id}#event:1", "field": "absent_column"}, "Unknown event field"),
        ({"handle": f"run:{run_id}#event:1", "field": "$.absent_json_key"}, "does not exist"),
    ]
    for args, error in cases:
        message = reader.invoke({"type": "tool_call", "id": "bad", "name": reader.name, "args": args})
        result = json.loads(message.content)
        assert message.status == "error"
        assert result["ok"] is False
        assert error in result["error"]
        assert result["recovery"]
        assert "available_schema" not in result
    raw = scope.execute("SELECT handle FROM trajectory_raw_events WHERE name='LLM_RAW_RESPONSE'")["rows"][0]
    with pytest.raises(ValueError, match="Cannot read|does not exist"):
        scope.read_event(handle=raw["handle"], field="$[invalid")


@pytest.mark.parametrize("role", ["reflector", "proposer", "reviewer"])
def test_large_query_exports_are_complete_readable_and_independent(tmp_path: Path, role: str) -> None:
    scope, run_id = _scope_with_semantic_events(tmp_path)
    large_text = "完整结果" * 60_000
    with sqlite3.connect(tmp_path / run_id / OBSERVABILITY_DB_NAME) as conn:
        conn.execute("UPDATE observation_events SET payload_json=? WHERE name='RUN_START'",
                     (json.dumps({"text": large_text}),))
    candidate = tmp_path / "candidate" if role != "reflector" else None
    if candidate:
        candidate.mkdir()
    with _self_evolution_backend(workspace=tmp_path / "workspace", role=role, candidate_root=candidate) as backend:
        query = {tool.name: tool for tool in _trace_query_tools(scope, backend)}["query_evolution_trace_sql"]
        args = {"sql": "SELECT handle, payload_json FROM trajectory_events WHERE name='RUN_START'"}
        with usage_invocation(
            workspace=tmp_path / "workspace", stage=f"self_evolution_{role}",
            model_label="test", context={},
        ) as (config, cb):
            def invoke(index):
                return query.invoke({
                    "type": "tool_call", "id": f"query-{index}", "name": query.name, "args": args,
                }, config=config)

            with ThreadPoolExecutor(max_workers=2) as pool:
                messages = list(pool.map(invoke, range(2)))
        results = [json.loads(message.content) for message in messages]
        assert results[0]["result_path"] != results[1]["result_path"]
        for result in results:
            assert result["ok"] is True
            assert result["row_count"] == 1
            # Read exact bytes through the same backend, then exercise its
            # agent-visible read_file tool on the returned path.
            downloaded = backend.download_files([result["result_path"]])[0]
            content = json.loads(downloaded.content)
            event = content["rows"][0]
            assert json.loads(event["payload_json"])["text"] == large_text
            assert scope.contains_event(event["handle"])
            fs = _self_evolution_filesystem_middleware(backend=backend, tools=["read_file"], writable=False)
            class ReadingModel(FakeMessagesListChatModel):
                def bind_tools(self, tools, **kwargs):
                    return self

            model = ReadingModel(responses=[
                AIMessage(content="", tool_calls=[{
                    "type": "tool_call", "id": "file", "name": "read_file",
                    "args": {"file_path": result["result_path"], "offset": 0, "limit": 5},
                }]),
                AIMessage(content="Read complete."),
            ])
            agent = create_agent(model=model, middleware=[fs])
            output = agent.invoke({"messages": [{"role": "user", "content": "Read the query result."}]})
            message = next(m for m in output["messages"] if isinstance(m, ToolMessage))
            assert message.status == "success"
            assert "columns" in message.content
        if candidate:
            assert not list(candidate.iterdir())
    # The temporary context is gone; original tool I/O must still be inspectable.
    with sqlite3.connect(cb.run_dir / OBSERVABILITY_DB_NAME) as conn:
        outputs = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload_json FROM observation_events WHERE name='TOOL_RAW_OUTPUT'"
        )]
    assert len(outputs) == 2
    for output in outputs:
        artifact = output["raw_output"]["artifact"]
        assert artifact["row_count"] == 1
        assert json.loads(artifact["rows"][0]["payload_json"])["text"] == large_text


def test_target_history_scope_excludes_unrelated_observations_and_jobs(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    for index, target in enumerate(("materials_worker/a", "materials_worker/b"), start=1):
        run_dir = tmp_path / "workspace" / "metadata" / "runs" / f"run-{index}"
        run_dir.mkdir(parents=True)
        job = store.enqueue_job(
            trigger_kind="post_run",
            run_id=f"run-{index}",
            run_dir=run_dir,
        )
        store.write_observation(
            Observation(
                observation_id=f"observation-{index}",
                run_id=f"run-{index}",
                thread_id=f"thread-{index}",
                job_id=job.job_id,
                signal_kind="skill_revision",
                target=target,
                claim=f"claim {index}",
                created_at=utc_now(),
            )
        )

    scoped = EvolutionHistoryScope(
        db_path=store.db_path,
        target="materials_worker/a",
    )

    observations = scoped.execute(
        "SELECT target, run_id FROM evolution_observations ORDER BY run_id"
    )
    jobs = scoped.execute("SELECT run_id FROM evolution_jobs ORDER BY run_id")
    assert observations["rows"] == [
        {"target": "materials_worker/a", "run_id": "run-1"}
    ]
    assert jobs["rows"] == [{"run_id": "run-1"}]


def test_trace_query_tool_returns_recoverable_error_then_accepts_corrected_query(
    tmp_path: Path,
) -> None:
    scope, _run_id = _scope_with_semantic_events(tmp_path)
    query_tool = {tool.name: tool for tool in scope.tools()}[
        "query_evolution_trace_sql"
    ]

    rejected = json.loads(
        query_tool.invoke({"sql": "SELECT event, agent FROM trajectory_events"})
    )
    malformed = json.loads(query_tool.invoke({}))
    corrected = json.loads(
        query_tool.invoke(
            {
                "sql": (
                    "SELECT name, agent_name FROM trajectory_events "
                    "ORDER BY id LIMIT 1"
                )
            }
        )
    )

    assert rejected["ok"] is False
    assert "no such column" in rejected["error"]
    assert "available_schema" not in rejected
    assert rejected["recovery"]
    assert "trajectory_events(" in query_tool.description
    assert corrected["ok"] is True
    assert corrected["columns"] == ["name", "agent_name"]
    assert corrected["row_count"] == 1
    assert malformed["ok"] is False
    assert "sql" in malformed["error"]


def test_reflector_react_loop_can_correct_a_rejected_trace_query(
    tmp_path: Path,
) -> None:
    class _BindableFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tools, tool_choice, kwargs)
            return self

    scope, _run_id = _scope_with_semantic_events(tmp_path)
    model = _BindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_evolution_trace_sql",
                        "args": {"sql": "SELECT event FROM trajectory_events"},
                        "id": "wrong-query",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_evolution_trace_sql",
                        "args": {
                            "sql": "SELECT name FROM trajectory_events ORDER BY id LIMIT 1"
                        },
                        "id": "corrected-query",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReflectionBatch",
                        "args": {
                            "items": [
                                {
                                    "kind": "no_change",
                                    "evidence_refs": [],
                                    "rationale": "The corrected query did not support a durable change.",
                                }
                            ]
                        },
                        "id": "structured-result",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = create_agent(
        model=model,
        tools=scope.tools(),
        system_prompt="Inspect the complete authorized trace.",
        middleware=[_OptionalResultMiddleware(ReflectionBatch)],
        name="recoverable_self_evolution_reflector",
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Reflect."}]})
    tool_messages = {
        message.tool_call_id: message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }

    assert json.loads(tool_messages["wrong-query"].content)["ok"] is False
    assert tool_messages["wrong-query"].status == "error"
    assert json.loads(tool_messages["corrected-query"].content)["ok"] is True
    assert result["structured_response"].items[0].kind == "no_change"


def test_deepagent_reflector_offloads_large_query_result_and_keeps_trace_reachable(
    tmp_path: Path,
) -> None:
    class _CapturingBindableFakeModel(FakeMessagesListChatModel):
        seen_query_results: ClassVar[list[str]] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tools, tool_choice, kwargs)
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            for message in messages:
                if isinstance(message, ToolMessage) and message.name == "query_evolution_trace_sql":
                    self.seen_query_results.append(str(message.content))
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    run_id = "run_deepagent_large_result"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    large_text = "LARGE_EVIDENCE_BLOCK_" * 1_200
    store = ObservabilityStore(run_dir)
    store.record_raw_callback(
        "LLM_RAW_RESPONSE",
        category="model",
        run_id=run_id,
        payload={
            "callback_run_id": "large_deepagent_model_call",
            "generations": [{"response_text": large_text}],
        },
    )
    scope = EvolutionTraceScope({run_id: (run_dir, _trace(run_id))})
    model = _CapturingBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_evolution_trace_sql",
                        "args": {
                            "sql": (
                                "SELECT payload_json FROM trajectory_events "
                                "WHERE name = 'MODEL_RESPONSE'"
                            )
                        },
                        "id": "large-query",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReflectionBatch",
                        "args": {
                            "items": [
                                {
                                    "kind": "no_change",
                                    "evidence_refs": [],
                                    "rationale": "The large event did not establish a durable change.",
                                }
                            ]
                        },
                        "id": "structured-result",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    _CapturingBindableFakeModel.seen_query_results = []
    reflector = ProposerAgent(
        model=model,
        model_label="fake",
        workspace=tmp_path / "workspace",
    )

    result, _meta = reflector.reflect(
        trajectory_markdown=scope.index(
            context_label="reflection",
            skill_context="",
        ),
        skill_catalog="",
        prior_targets=[],
        trace_scope=scope,
    )

    assert result.items[0].kind == "no_change"
    visible_query_result = _CapturingBindableFakeModel.seen_query_results[-1]
    assert "/large_tool_results/" in visible_query_result
    assert large_text not in visible_query_result
    event = scope.execute(
        "SELECT id FROM trajectory_events WHERE name = 'MODEL_RESPONSE'"
    )["rows"][0]
    page = scope.read_event(
        handle=f"run:{run_id}#event:{event['id']}",
        field="$.generations[0].assistant_text",
        limit=100_000,
    )
    assert page["segment"] == large_text
