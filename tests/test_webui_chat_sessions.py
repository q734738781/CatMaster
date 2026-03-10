from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.webui.chat_sessions import ChatSessionStore
from catmaster.webui.session import WebSession


def test_chat_session_store_builds_summary_and_recent_messages(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    store = ChatSessionStore(workspace=tmp_path)
    session_id = store.get_active_session_id()

    for idx in range(10):
        role = "user" if idx % 2 == 0 else "assistant"
        store.append_message(session_id, role=role, content=f"message {idx}")

    pack = store.build_history_pack(session_id, recent_messages=4)
    assert pack.session_id == session_id
    assert "Earlier chat session summary:" in pack.summary_text
    assert len(pack.recent_messages) == 4
    assert str(pack.recent_messages[-1].get("content")) == "message 9"


def test_chat_session_store_excludes_hitl_from_chat_and_history(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    store = ChatSessionStore(workspace=tmp_path)
    session_id = store.get_active_session_id()

    store.append_message(session_id, role="user", content="Normal request.", kind="chat")
    store.append_message(session_id, role="assistant", content="Proposal Review\n\nwork packages...", kind="hitl")
    store.append_message(session_id, role="assistant", content="Final answer.", kind="run_result")

    pack = store.build_history_pack(session_id, recent_messages=8)
    assert [str(item.get("content")) for item in pack.recent_messages] == [
        "Normal request.",
        "Final answer.",
    ]
    chat_messages = store.chat_messages(session_id, limit=10)
    assert chat_messages == [
        {"role": "user", "content": "Normal request."},
        {"role": "assistant", "content": "Final answer."},
    ]
    assert "Proposal Review" not in pack.summary_text
    session_payload = store.load_session(session_id)
    assert "Proposal Review" not in str(session_payload.get("summary_text") or "")
    assert str(session_payload.get("title") or "") == "Normal request."


def test_chat_session_store_excludes_run_errors_from_chat_and_history(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    store = ChatSessionStore(workspace=tmp_path)
    session_id = store.get_active_session_id()

    store.append_message(session_id, role="user", content="Normal request.", kind="chat")
    store.append_message(session_id, role="assistant", content="Run crashed badly.", kind="run_error")
    store.append_message(session_id, role="assistant", content="Final answer.", kind="run_result")

    pack = store.build_history_pack(session_id, recent_messages=8)
    assert [str(item.get("content")) for item in pack.recent_messages] == [
        "Normal request.",
        "Final answer.",
    ]
    chat_messages = store.chat_messages(session_id, limit=10)
    assert chat_messages == [
        {"role": "user", "content": "Normal request."},
        {"role": "assistant", "content": "Final answer."},
    ]
    assert "Run crashed badly." not in pack.summary_text
    session_payload = store.load_session(session_id)
    assert "Run crashed badly." not in str(session_payload.get("summary_text") or "")


def test_websession_injects_chat_history_for_standard_lane(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    session._append_chat_message(role="user", content="Earlier user question.")
    session._append_chat_message(role="assistant", content="Earlier assistant answer.")

    fake_llm_config_mod = types.ModuleType("catmaster.llm.config")

    class DummyLLMProfile:
        def __init__(self):
            self.main = SimpleNamespace(model="task-model", provider="openai", base_url=None)
            self.mcp = SimpleNamespace(filesystem=SimpleNamespace(enabled=False))
            self.agent_runtime = SimpleNamespace(max_tool_calls=4, recursion_limit=8, print_state_messages=False)

        @classmethod
        def from_env_or_file(cls, *_a, **_k):
            return cls()

        def config_for_role(self, role: str):
            return SimpleNamespace(model=role, provider="openai", base_url=None)

    fake_llm_config_mod.LLMProfile = DummyLLMProfile
    monkeypatch.setitem(sys.modules, "catmaster.llm.config", fake_llm_config_mod)

    fake_llm_factory_mod = types.ModuleType("catmaster.llm.factory")
    fake_llm_factory_mod.build_chat_model = lambda cfg: {"model": getattr(cfg, "model", "")}
    monkeypatch.setitem(sys.modules, "catmaster.llm.factory", fake_llm_factory_mod)

    captured: dict[str, str] = {}
    fake_runner_factory_mod = types.ModuleType("catmaster.agents.runner_factory")

    class _DummyRunner:
        def run(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["lane"] = str(kwargs.get("lane") or "")
            captured["session_context_text"] = str(kwargs.get("session_context_text") or "")
            return {"status": "done"}

    def fake_build_graph_runner(**kwargs):
        _ = kwargs
        run_dir = system_root(workspace=ws) / "runs" / "run_test"
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            runner=_DummyRunner(),
            run_context=SimpleNamespace(run_id="run_test", run_dir=run_dir, model_name="task-model"),
        )

    fake_runner_factory_mod.build_graph_runner = fake_build_graph_runner
    monkeypatch.setitem(sys.modules, "catmaster.agents.runner_factory", fake_runner_factory_mod)

    msg = session.start_run(
        prompt="Current request.",
        lane="standard",
        run_mode="new_run",
        resume_run_name="",
        proposal_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None
    session.run_thread.join(timeout=5)
    assert captured["lane"] == "standard"
    assert "Current request." in captured["prompt"]
    assert "Relevant conversation history:" in captured["session_context_text"]
    assert "Session ID:" not in captured["session_context_text"]
    assert "Earlier user question." in captured["session_context_text"]
    assert "Earlier assistant answer." in captured["session_context_text"]


def test_websession_injects_chat_history_for_research_lane(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    session._append_chat_message(role="user", content="Earlier user question.")
    pack = session.build_session_context(current_prompt="Research prompt.", lane="research")
    assert pack["session_id"]
    assert "Relevant conversation history:" in pack["context_text"]
    assert "Session ID:" not in pack["context_text"]
    assert "Earlier user question." in pack["context_text"]
    assert int(pack["estimated_tokens"]) > 0


def test_sidebar_snapshot_reads_cached_runs(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    runs_root = system_root(workspace=ws) / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text('{"status":"done"}', encoding="utf-8")
    (run_dir / "meta.json").write_text('{"model_name":"m1","start_time":"2026-03-09T00:00:00Z"}', encoding="utf-8")

    session._mark_sidebar_cache_dirty()
    snapshot = session.get_sidebar_snapshot()
    runs = snapshot.get("runs") or []
    cards = snapshot.get("cards") or []
    assert runs
    assert cards
    assert runs[0][1] == "run_001"
    assert cards[0]["run_name"] == "run_001"


def test_websession_chat_result_prefers_task_state_final_answer(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text(
        '{"summary":"UI card summary.","final_answer":"Only the final answer should go back to chat.","status":"done"}',
        encoding="utf-8",
    )
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "FINAL_REPORT.md").write_text(
        "# Final Report\n\n## User Query\nQ\n\n## Final Answer\nA\n",
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok
    assert session._read_chat_result_text(run_dir) == "Only the final answer should go back to chat."


def test_websession_chat_result_extracts_final_answer_from_report(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "FINAL_REPORT.md").write_text(
        "# Final Report\n\n## User Query\nQ\n\n## Final Answer\nConcise answer.\n\n## Notes\nIgnore.\n",
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok
    assert session._read_chat_result_text(run_dir) == "Concise answer."


def test_snapshot_live_state_falls_back_to_task_state_goal_when_task_events_missing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text(
        '{"status":"running","tasks":[{"task_id":"task_01","goal":"Prepare a 30 A comparison.","task_packet":{"goal":"Prepare a 30 A comparison."},"status":"running"}]}',
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok

    live = session.snapshot_live_state(run_dir)
    assert live.get("current_task_id") == "task_01"
    assert live.get("current_task_goal") == "Prepare a 30 A comparison."


def test_snapshot_live_state_prefers_unified_writing_progress_fields(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text(
        '{"status":"drafting","lane":"writing","current_phase":"drafting","current_work_label":"Write section: Results and Discussion","request":"Write paper."}',
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok

    live = session.snapshot_live_state(run_dir)
    assert live.get("current_task_goal") == "Write section: Results and Discussion"
    assert live.get("current_phase") == "drafting"


def test_snapshot_live_state_prefers_unified_research_progress_fields(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text(
        '{"status":"running","lane":"research","current_phase":"executing","current_work_label":"Experiment: O2 gas-phase comparison","question":"Study O2 on Fe."}',
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok

    live = session.snapshot_live_state(run_dir)
    assert live.get("current_task_goal") == "Experiment: O2 gas-phase comparison"
    assert live.get("current_phase") == "executing"
