"""Skills Evo must record paid boundaries, including native nested model calls."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import Field

from catmaster.runtime.self_evolution import agents
from catmaster.runtime.self_evolution.models import ProposerResult, ReflectionBatch, ReflectionResult, ReviewerResult
from catmaster.runtime.self_evolution.usage import usage_invocation
from catmaster.runtime.usage_stats import load_usage_summary, summarize_usage_from_observability


def response(text="answer", **kwargs):
    return AIMessage(content=text, usage_metadata={
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_token_details": {"cache_read": 70},
        "output_token_details": {"reasoning": 8},
    }, **kwargs)


class Model(FakeMessagesListChatModel):
    observed_calls: list[int] = Field(default_factory=list)
    usage_dir: Path | None = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        if self.usage_dir:
            self.observed_calls.append(load_usage_summary(self.usage_dir).get("calls", 0))
        return super()._generate(messages, *args, **kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
def test_native_summary_and_child_are_recorded_before_the_next_call(tmp_path, async_mode):
    model = Model(responses=[
        response("Retained evidence"),
        response("", tool_calls=[{"id": "delegate", "name": "task", "args": {
            "subagent_type": "general-purpose", "description": "Inspect the evidence",
        }}]),
        response("Child finding"),
        response("Final interpretation"),
    ])
    graph = agents._build_self_evolution_deep_agent(
        model=model, backend=FilesystemBackend(root_dir=tmp_path / "files", virtual_mode=True),
        tools=[], investigator_tools=[], system_prompt="Inspect the evidence", response_schema=agents.ReflectionBatch,
        name="self_evolution_reflector", filesystem_tools=agents._SELF_EVOLUTION_REVIEWER_FILESYSTEM_TOOLS,
        allow_mutations=False,
    )
    messages = []
    for i in range(8):
        messages.extend([HumanMessage(content=f"Question {i}"), AIMessage(content=f"Answer {i}")])
    messages[-1] = AIMessage(content="Recent evidence", usage_metadata={
        "input_tokens": 199_990, "output_tokens": 10, "total_tokens": 200_000,
    }, response_metadata={"model_provider": model._get_ls_params()["ls_provider"]})
    messages.append(HumanMessage(content="Continue"))
    with usage_invocation(workspace=tmp_path, stage="self_evolution_reflector",
                          model_label="evo-model", context={"job_id": "job", "source_run_id": "science"}) as (config, cb):
        model.usage_dir = cb.run_dir
        if async_mode:
            result = asyncio.run(graph.ainvoke({"messages": messages}, config))
        else:
            result = graph.invoke({"messages": messages}, config)
        assert result["messages"][-1].content == "Final interpretation"
        assert model.observed_calls == [0, 1, 2, 3]
        saved = load_usage_summary(cb.run_dir)
        assert saved["calls"] == 4
        assert saved["input_tokens"] == 400
        assert saved["input_uncached_tokens"] == 120
        assert saved["input_cached_tokens"] == 280
        assert saved["output_tokens"] == 80
        assert saved["reasoning_tokens"] == 32
        assert saved["total_tokens"] == 480  # Reasoning is already part of output.
        assert not saved["partial"]
        roles = saved["call_counts_by_role"]
        assert roles["self_evolution_reflector/general-purpose"] == 1
        assert roles["self_evolution_reflector/summarization"] == 1
        assert roles["self_evolution_reflector"] == 2
        events = cb.store.read_events_page(names=["LLM_CALL_END"], limit=100)["events"]
        assert len(events) == 4
        assert all(e["payload"]["source_run_id"] == "science" for e in events)
        assert all(e["payload"]["model_label"] == "evo-model" for e in events)
        assert summarize_usage_from_observability(cb.run_dir)["total_tokens"] == 480
        requests = cb.store.read_events_page(names=["LLM_RAW_REQUEST"], limit=100)["events"]
        responses = cb.store.read_events_page(names=["LLM_RAW_RESPONSE"], limit=100)["events"]
        assert len(requests) == len(responses) == 4
        assert {e["payload"]["callback_run_id"] for e in requests} == {
            e["payload"]["callback_run_id"] for e in responses
        }
        assert any("Inspect the evidence" in json.dumps(e["payload"]) for e in requests)
        assert any("Final interpretation" in json.dumps(e["payload"]) for e in responses)
        tools = cb.store.read_events_page(names=["TOOL_RAW_INPUT", "TOOL_RAW_OUTPUT"], limit=100)["events"]
        assert len(tools) == 2
        assert all(e["payload"]["tool_name"] == "task" for e in tools)
        child = next(e for e in events if e["payload"]["agent_name"].endswith("/general-purpose"))
        assert child["payload"]["parent_callback_run_id"]
        assert cb.store.read_events_page(names=["subagent.started", "subagent.completed"], limit=100)["events"]
    assert json.loads((cb.run_dir / "meta.json").read_text())["status"] == "done"
    assert not (tmp_path / "metadata/runs/science").exists()


@pytest.mark.parametrize("stage", ["reflector", "proposer", "reviewer"])
def test_production_agent_methods_record_usage_with_job_context(tmp_path, stage):
    structured = {
        "reflector": ReflectionBatch(items=[ReflectionResult(kind="no_change", rationale="No durable change")]),
        "proposer": ProposerResult(action="ignore", rationale="No durable change"),
        "reviewer": ReviewerResult(recommendation="reject", summary="No supported candidate"),
    }[stage]
    model = Model(responses=[response("", tool_calls=[{
        "id": "structured", "name": type(structured).__name__, "args": structured.model_dump(mode="json"),
    }])])
    context = {"job_id": "same-job", "attempt": 2, "source_run_id": "parent"}
    actor_type = agents.ReviewerAgent if stage == "reviewer" else agents.ProposerAgent
    actor = actor_type(model=model, model_label="test-model", workspace=tmp_path, usage_context=context)
    candidate = tmp_path / "metadata/self_evolution/candidates/test/r0001"
    candidate.mkdir(parents=True)
    if stage == "reflector":
        value, meta = actor.reflect(trajectory_markdown="Evidence", skill_catalog="", prior_targets=[],
                                    trace_scope=SimpleNamespace(tools=lambda **kwargs: []))
    elif stage == "proposer":
        value, meta = actor.propose(candidate_root=candidate)
    else:
        value, meta = actor.review(candidate_root=candidate, action="memory", group="", name="",
                                   rationale="Inspect", validation={})
    assert value == structured
    assert meta["usage"]["calls"] == 1
    run_dir = tmp_path / "metadata/runs" / meta["usage_run_id"]
    assert load_usage_summary(run_dir)["input_cached_tokens"] == 70
    state = json.loads((run_dir / "meta.json").read_text())
    assert state["job_id"] == "same-job" and state["attempt"] == 2
    assert state["agent_name"] == f"self_evolution_{stage}"
    if stage != "reflector":
        assert state["candidate_root"] == str(candidate)
    assert list((tmp_path / "metadata/self_evolution/agent_context").iterdir()) == []


def test_parallel_callbacks_duplicates_and_later_failure_preserve_usage(tmp_path):
    with pytest.raises(RuntimeError, match="later failure"):
        with usage_invocation(workspace=tmp_path, stage="self_evolution_proposer", model_label="test",
                              context={"job_id": "job", "attempt": 1}) as (_, cb):
            def complete(_):
                call_id = uuid4()
                cb.on_chat_model_start({}, [[]], run_id=call_id, metadata={"catmaster_model_label": "test"})
                result = LLMResult(generations=[[ChatGeneration(message=response())]])
                cb.on_llm_end(result, run_id=call_id)
                cb.on_llm_end(result, run_id=call_id)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(complete, range(8)))
            missing_id = uuid4()
            cb.on_chat_model_start({}, [[]], run_id=missing_id)
            cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="No usage"))]]),
                          run_id=missing_id)
            failed_id = uuid4()
            cb.on_chat_model_start({}, [[]], run_id=failed_id)
            cb.on_llm_error(RuntimeError("provider failure"), run_id=failed_id)
            assert load_usage_summary(cb.run_dir)["total_tokens"] == 960
            raise RuntimeError("later failure")
    saved = load_usage_summary(cb.run_dir)
    assert saved["calls"] == 9
    assert saved["total_tokens"] == 960
    assert saved["failed_calls"] == 1
    assert saved["missing_usage_calls"] == 2
    assert saved["pending_calls"] == 0
    assert saved["partial"]
    assert json.loads((cb.run_dir / "meta.json").read_text())["status"] == "error"
    assert len(cb.store.read_events_page(names=["LLM_CALL_END"], limit=100)["events"]) == 9
    assert len(cb.store.read_events_page(names=["LLM_RAW_RESPONSE"], limit=100)["events"]) == 9
    assert cb.store.read_events_page(names=["LLM_ERROR"], limit=100)["events"]
    # A new attempt must not overwrite the earlier attempt or copy its usage.
    with usage_invocation(workspace=tmp_path, stage="self_evolution_proposer", model_label="test",
                          context={"job_id": "job", "attempt": 2}) as (_, retry):
        assert retry.run_dir != cb.run_dir
    assert load_usage_summary(cb.run_dir)["total_tokens"] == 960


def test_completion_timestamps_support_daily_audits_across_midnight(tmp_path, monkeypatch):
    import sqlite3
    from catmaster.runtime import observed_invocation as usage

    midnight = 1_789_401_600
    clock = [midnight - 10]
    monkeypatch.setattr(usage.time, "time", lambda: clock[0])
    with usage_invocation(workspace=tmp_path, stage="self_evolution_reviewer",
                          model_label="test", context={}) as (_, cb):
        for offset in (-1, 1):
            call_id = uuid4()
            cb.on_chat_model_start({}, [[]], run_id=call_id)
            clock[0] = midnight + offset
            cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=response())]]), run_id=call_id)
        # Same discovery and completed-call date filter as the original audit.
        databases = list((tmp_path / "metadata/runs").glob("*/observability.sqlite"))
        assert databases == [cb.run_dir / "observability.sqlite"]
        with sqlite3.connect(databases[0]) as conn:
            after_midnight = conn.execute(
                "SELECT COUNT(*), SUM(json_extract(payload_json, '$.usage.total_tokens')) "
                "FROM observation_events WHERE name='LLM_CALL_END' AND ts>=?", (midnight,),
            ).fetchone()
        assert after_midnight == (1, 120)


def test_coordinator_passes_job_and_attempt_to_real_agent_builder(tmp_path, monkeypatch):
    from catmaster.runtime.self_evolution import pipeline
    from catmaster.runtime.self_evolution.models import SelfEvolutionJob

    monkeypatch.setattr(agents, "build_chat_model", lambda _: Model(responses=[response()]))
    monkeypatch.setattr(agents, "search_tools_for_role", lambda *args, **kwargs: [])
    profile = SimpleNamespace(label_for_role=lambda role: role, config_for_role=lambda role: None)
    monkeypatch.setattr(pipeline.LLMProfile, "from_env_or_file", lambda _: profile)
    coordinator = pipeline.SelfEvolutionCoordinator.__new__(pipeline.SelfEvolutionCoordinator)
    coordinator._proposer = coordinator._reviewer = None
    coordinator.workspace = tmp_path
    coordinator.model_config = ""
    job = SelfEvolutionJob(job_id="job", project_id="project", run_id="science", run_dir="source",
                           thread_id="thread", attempt_count=2)
    proposer, reviewer = coordinator._agents(job)
    expected = {"job_id": "job", "attempt": 2, "source_run_id": "science", "source_thread_id": "thread"}
    assert proposer.usage_context == reviewer.usage_context == expected


def test_accounting_record_is_not_a_research_resume_target(tmp_path):
    from catmaster.webui.session import WebSession

    session = WebSession()
    with usage_invocation(workspace=tmp_path, stage="self_evolution_reflector",
                          model_label="test", context={}) as (_, cb):
        assert not (cb.run_dir / "run_state.json").exists()
        assert session._resolve_resume_dir("research", workspace=tmp_path) is None
        science = tmp_path / "metadata/runs/science"
        science.mkdir()
        (science / "run_state.json").write_text('{"entrypoint":"research","status":"running"}')
        assert session._resolve_resume_dir("research", workspace=tmp_path) == str(science)
