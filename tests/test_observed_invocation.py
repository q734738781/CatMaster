"""Standalone calls and nested tool models share the ordinary debug store."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from catmaster.runtime.artifact_callback import ObservabilityCallbackHandler
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.observed_invocation import observed_invocation
from catmaster.runtime.usage_stats import load_usage_summary, summarize_usage_from_observability
from catmaster.tools.registry import _make_langchain_tool
from catmaster.webui import live_summary_service, thread_title_service
from catmaster.webui.session import WebSession


def model(text):
    return FakeMessagesListChatModel(responses=[AIMessage(
        content=text, usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )])


@pytest.mark.parametrize("failure", [False, True])
def test_title_call_records_success_or_timeout(tmp_path, monkeypatch, failure):
    title_model = model("催化剂筛选")
    if failure:
        class SlowModel(FakeMessagesListChatModel):
            async def _agenerate(self, *args, **kwargs):
                await asyncio.sleep(10)
        title_model = SlowModel(responses=title_model.responses)
        monkeypatch.setattr(thread_title_service, "_TITLE_TIMEOUT_SECONDS", 0.2)
    profile = SimpleNamespace(agents={"thread_title": "title-model"},
                              config_for_role=lambda role: None, label_for_role=lambda role: "title-model")
    monkeypatch.setattr(thread_title_service.LLMProfile, "from_env_or_file", lambda _: profile)
    monkeypatch.setattr(thread_title_service, "build_chat_model", lambda _: title_model)
    call = thread_title_service.generate_semantic_thread_title(
        workspace=tmp_path, thread_id="source-thread", question="筛选催化剂",
        attachment_names=[], entrypoint="research",
    )
    if failure:
        with pytest.raises(TimeoutError):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == "催化剂筛选"
    run_dir, = (tmp_path / "metadata/runs").iterdir()
    state = json.loads((run_dir / "meta.json").read_text())
    assert state["status"] == ("error" if failure else "done")
    assert not (run_dir / "run_state.json").exists()
    events = ObservabilityStore(run_dir).read_events_page(thread_id="source-thread", limit=100)["events"]
    names = {event["name"] for event in events}
    assert {"RUN_START", "RUN_END", "LLM_RAW_REQUEST"} <= names
    assert ("LLM_ERROR" if failure else "LLM_RAW_RESPONSE") in names
    assert load_usage_summary(run_dir)["calls"] == (0 if failure else 1)
    assert load_usage_summary(run_dir)["pending_calls"] == 0
    assert load_usage_summary(run_dir)["failed_calls"] == int(failure)


def test_live_summary_has_independent_record_and_parent_link(tmp_path, monkeypatch):
    source = tmp_path / "metadata/runs/science"
    source.mkdir(parents=True)
    monkeypatch.setattr(live_summary_service, "_get_live_summary_llm", lambda: model('{"live_headline":"Working"}'))
    result = live_summary_service.summarize_live_state(
        {}, enabled=True, run_dir=source, max_events=2, max_params_chars=80,
        max_journal_items=2, timeout_s=5,
    )
    assert result["source"] == "llm"
    run_dir, = source.parent.glob("live_summary_*")
    assert load_usage_summary(run_dir)["total_tokens"] == 13
    assert json.loads((run_dir / "meta.json").read_text())["source_run_id"] == "science"
    assert not (source / "observability.sqlite").exists()


@pytest.mark.parametrize("async_mode", [False, True])
def test_real_review_tool_inherits_observations_without_explicit_callbacks(tmp_path, monkeypatch, async_mode):
    module = importlib.import_module("catmaster.tools.analysis.peer_review_request")
    files = tmp_path / "files"
    files.mkdir()
    (files / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    profile = SimpleNamespace(peer_review_models=["reviewer"], models={"reviewer": SimpleNamespace(model="review-model")})
    monkeypatch.setattr(module.LLMProfile, "from_env_or_file", lambda: profile)
    monkeypatch.setattr(module, "build_chat_model", lambda _: model("Review finding"))
    run_dir = tmp_path / "metadata/runs/science"
    callback = ObservabilityCallbackHandler(run_dir, run_id="science", default_agent_name="peer_review")
    tool = _make_langchain_tool("peer_review_request", module.peer_review_request,
                               module.PeerReviewRequestInput, workspace=str(tmp_path), run_dir=str(run_dir))
    payload = {"pdf_path": "paper.pdf", "review_request": "Review the evidence"}
    config = {"callbacks": [callback]}
    if async_mode:
        result = asyncio.run(tool.ainvoke(payload, config=config))
    else:
        result = tool.invoke(payload, config=config)
    assert "Review finding" in str(result)
    assert summarize_usage_from_observability(run_dir)["calls"] == 1
    requests = callback.store.read_events_page(names=["LLM_RAW_REQUEST"], limit=100)["events"]
    assert len(requests) == 1
    tool_events = callback.store.read_events_page(names=["TOOL_RAW_INPUT"], limit=100)["events"]
    assert requests[0]["payload"]["parent_callback_run_id"] == tool_events[0]["payload"]["callback_run_id"]


def test_standalone_trace_reaches_existing_debug_projection(tmp_path):
    with observed_invocation(workspace=tmp_path, entrypoint="self_evolution", stage="self_evolution_reviewer",
                             model_label="review-model", context={"job_id": "job"}) as (config, callback):
        model("Full review finding").invoke("Evidence to inspect", config=config)
    session = WebSession()
    assert callback.run_id in {name for _, name in session.list_runs(workspace=tmp_path)}
    snapshot = session.read_observability(callback.run_dir, workspace=tmp_path)
    assert snapshot["metrics"]["llm_calls"] == 1
    assert "Full review finding" in json.dumps(snapshot)
    # Raw details remain reachable by ordinary event pagination.
    page = callback.store.read_events_page(names=["LLM_RAW_REQUEST", "LLM_RAW_RESPONSE"], limit=1)
    assert len(page["events"]) == 1
    assert page["has_more"]
    previous = callback.store.read_events_page(
        names=["LLM_RAW_REQUEST", "LLM_RAW_RESPONSE"], limit=1, before_id=page["min_id"],
    )
    assert {page["events"][0]["name"], previous["events"][0]["name"]} == {"LLM_RAW_REQUEST", "LLM_RAW_RESPONSE"}
    assert not previous["has_more"]
    assert not (callback.run_dir / "run_state.json").exists()
