from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import catmaster.specialists as specialists_mod
from catmaster.specialists import RUN_STATE_FILE
from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.webui.chat_sessions import ChatSessionStore
from catmaster.webui.session import WebSession


def _install_dummy_llm_profile(monkeypatch) -> None:
    fake_llm_config_mod = types.ModuleType("catmaster.llm.config")

    class DummyLLMProfile:
        def __init__(self):
            self.main = SimpleNamespace(model="task-model", provider="openai", base_url=None)
            self.agent_runtime = SimpleNamespace(max_tool_calls=4, recursion_limit=8, print_state_messages=False)

        @classmethod
        def from_env_or_file(cls, *_a, **_k):
            return cls()

        def config_for_role(self, role: str):
            return SimpleNamespace(model=role, provider="openai", base_url=None)

    fake_llm_config_mod.LLMProfile = DummyLLMProfile
    monkeypatch.setitem(sys.modules, "catmaster.llm.config", fake_llm_config_mod)


def _write_completed_run(
    run_dir: Path,
    *,
    entrypoint: str = "research",
    final_answer: str = "Agent answer.",
    summary: str = "Run finished.",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text(
        json.dumps(
            {
                "entrypoint": entrypoint,
                "status": "done",
                "summary": summary,
                "final_answer": final_answer,
                "text_preview": final_answer,
                "facts": ["fact one"],
                "artifacts": [{"path": "outputs/result.txt", "kind": "file", "description": "reported file"}],
            }
        ),
        encoding="utf-8",
    )


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
    assert [item["role"] for item in chat_messages] == ["user", "assistant"]
    assert [item["content"] for item in chat_messages] == ["Normal request.", "Final answer."]
    assert [item["kind"] for item in chat_messages] == ["chat", "run_result"]
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
    assert [item["role"] for item in chat_messages] == ["user", "assistant"]
    assert [item["content"] for item in chat_messages] == ["Normal request.", "Final answer."]
    assert [item["kind"] for item in chat_messages] == ["chat", "run_result"]
    assert "Run crashed badly." not in pack.summary_text
    session_payload = store.load_session(session_id)
    assert "Run crashed badly." not in str(session_payload.get("summary_text") or "")


def test_websession_binds_chat_session_as_deepagent_thread_for_experiment_lane(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    session._append_chat_message(role="user", content="Earlier user question.")
    session._append_chat_message(role="assistant", content="Earlier assistant answer.", kind="run_result")
    _install_dummy_llm_profile(monkeypatch)

    captured: dict[str, str] = {}
    run_dir = system_root(workspace=ws) / "runs" / "run_test"

    class DummyRunner:
        def run(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["entrypoint"] = str(kwargs.get("entrypoint") or "")
            captured["chat_session_id"] = str(kwargs.get("chat_session_id") or "")
            captured["thread_id"] = str(kwargs.get("thread_id") or "")
            _write_completed_run(run_dir, entrypoint="experiment", final_answer="Agent answer.")
            return {"status": "done"}

    def fake_build_specialist_runner(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            runner=DummyRunner(),
            run_context=SimpleNamespace(run_id="run_test", run_dir=run_dir, model_name="task-model"),
        )

    monkeypatch.setattr(specialists_mod, "build_specialist_runner", fake_build_specialist_runner)

    msg = session.start_run(
        prompt="Current request.",
        lane="experiment",
        run_mode="new_run",
        resume_run_name="",
        proposal_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None
    session.run_thread.join(timeout=5)
    assert captured["entrypoint"] == "experiment"
    assert "Current request." in captured["prompt"]
    assert captured["chat_session_id"] == session.current_chat_session_id()
    assert captured["thread_id"] == session.current_chat_session_id()


def test_websession_builds_thread_binding_for_research_lane(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    session._append_chat_message(role="user", content="Earlier user question.")
    session._append_chat_message(role="assistant", content="Earlier assistant answer.", kind="run_result")
    pack = session.build_thread_binding(lane="research")
    assert pack["session_id"]
    assert pack["thread_id"] == pack["session_id"]


def test_websession_entry_context_status_reports_deepagent_thread(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    for idx in range(5):
        session._append_chat_message(role="user", content=f"Question {idx}")
        session._append_chat_message(role="assistant", content=f"Answer {idx}", kind="run_result")

    status = session.entry_context_status_text(lane="experiment")
    current_session = session.current_chat_session_id()
    assert f"Session `{current_session}`" in status
    assert f"deepagent thread `{current_session}`" in status


def test_websession_lists_and_switches_chat_sessions(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    initial_sessions = session.list_chat_sessions()
    assert len(initial_sessions) == 1
    first_label, first_session_id = initial_sessions[0]
    assert first_session_id == session.current_chat_session_id()
    assert "(active)" in first_label

    session._append_chat_message(role="user", content="Session A")
    second_session_id = session.create_chat_session()
    session._append_chat_message(role="user", content="Session B")

    sessions = session.list_chat_sessions()
    values = {value for _, value in sessions}
    assert first_session_id in values
    assert second_session_id in values
    assert session.current_chat_session_id() == second_session_id

    session.select_chat_session(first_session_id)
    assert session.current_chat_session_id() == first_session_id
    messages = session.get_chat_messages(limit=10)
    assert [item["role"] for item in messages] == ["user"]
    assert [item["content"] for item in messages] == ["Session A"]
    assert [item["kind"] for item in messages] == ["chat"]


def test_sidebar_snapshot_reads_cached_runs(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    runs_root = system_root(workspace=ws) / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text('{"status":"done","entrypoint":"research"}', encoding="utf-8")
    (run_dir / "meta.json").write_text('{"model_name":"m1","start_time":"2026-03-09T00:00:00Z"}', encoding="utf-8")

    session._mark_sidebar_cache_dirty()
    snapshot = session.get_sidebar_snapshot()
    runs = snapshot.get("runs") or []
    cards = snapshot.get("cards") or []
    assert runs
    assert cards
    assert runs[0][1] == "run_001"
    assert cards[0]["run_name"] == "run_001"


def test_websession_reads_persistent_memory_index(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    db_path = system_root(workspace=ws) / "deepagent_memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE store (prefix TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL)")
    prefix = ".".join(("catmaster", session._project_id_for_workspace(ws), "filesystem"))
    payload = {
        "content": [
            "# Persistent Project Memory",
            "",
            "- Prefer MACE screening before VASP unless accuracy is required.",
        ],
    }
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (prefix, "/AGENTS.md", json.dumps(payload).encode("utf-8")),
    )
    conn.commit()
    conn.close()

    text = session.read_memory_index()
    assert "Persistent Memory" in text
    assert "Prefer MACE screening before VASP" in text
    assert "catmaster." in text


def test_list_workspaces_includes_root_when_root_is_project_space(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    session = WebSession()

    ok, _, choices = session.set_workspace_root(str(tmp_path))

    assert ok
    assert choices == [(tmp_path.name, tmp_path.name)]


def test_open_workspace_by_name_accepts_root_project_space_name(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    session = WebSession()
    session.set_workspace_root(str(tmp_path))

    ok, msg = session.open_workspace_by_name(tmp_path.name)

    assert ok, msg
    assert session.current_workspace_path() == str(tmp_path.resolve())


def test_websession_chat_result_prefers_run_state_final_answer(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text(
        '{"summary":"UI card summary.","final_answer":"Only the final answer should go back to chat.","status":"done"}',
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok
    assert session._read_chat_result_text(run_dir) == "Only the final answer should go back to chat."


def test_websession_result_text_formats_summary_facts_and_files(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text(
        json.dumps(
            {
                "status": "done",
                "summary": "Concise answer.",
                "facts": ["fact a", "fact b"],
                "artifacts": [{"path": "outputs/a.txt", "kind": "file", "description": "reported file"}],
            }
        ),
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok
    result_text = session.read_result_text(run_dir)
    assert "## Summary" in result_text
    assert "Concise answer." in result_text
    assert "## Facts" in result_text
    assert "fact a" in result_text
    assert "## Files" in result_text
    assert "outputs/a.txt" in result_text


def test_snapshot_live_state_uses_run_state_text_preview_and_phase(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text(
        '{"status":"running","entrypoint":"research","phase":"executing","text_preview":"Prepare a 30 A comparison."}',
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok

    live = session.snapshot_live_state(run_dir)
    assert live.get("current_task_goal") == "Prepare a 30 A comparison."
    assert live.get("current_phase") == "executing"


def test_snapshot_live_state_falls_back_to_first_todo_item(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ensure_project_space_layout(ws, create=True)
    run_dir = system_root(workspace=ws) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text(
        '{"status":"running","entrypoint":"writing","phase":"drafting","todo_items":["Write section: Results and Discussion","Polish abstract"]}',
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok

    live = session.snapshot_live_state(run_dir)
    assert live.get("current_task_goal") == "Write section: Results and Discussion"
    assert live.get("current_phase") == "drafting"
