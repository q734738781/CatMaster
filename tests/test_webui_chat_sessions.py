from __future__ import annotations

import asyncio
import json
import sqlite3
import shutil
import sys
import time
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
    chat_session_id: str = "",
    user_prompt: str = "",
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
                "chat_session_id": chat_session_id,
                "user_prompt": user_prompt,
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
            _write_completed_run(
                run_dir,
                entrypoint="experiment",
                final_answer="Agent answer.",
                chat_session_id=captured["chat_session_id"],
            )
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
    chat_messages = session.get_chat_messages()
    assert [item["role"] for item in chat_messages[-2:]] == ["user", "assistant"]
    assert [item["kind"] for item in chat_messages[-2:]] == ["chat", "run_result"]
    assert str(chat_messages[-1]["source_run_id"]) == "run_test"


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


def test_websession_backfills_missing_run_results_from_run_state(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    active_session_id = session.current_chat_session_id()
    session._append_chat_message(role="user", content="Prompt one.", session_id=active_session_id)
    session._append_chat_message(role="user", content="Prompt two.", session_id=active_session_id)

    runs_root = system_root(workspace=ws) / "runs"
    _write_completed_run(
        runs_root / "run_20260321_120000_abcd12",
        entrypoint="research",
        final_answer="Recovered result one.",
        chat_session_id=active_session_id,
        user_prompt="Prompt one.",
    )
    _write_completed_run(
        runs_root / "run_20260321_120100_efgh34",
        entrypoint="research",
        final_answer="Recovered result two.",
        chat_session_id=active_session_id,
        user_prompt="Prompt two.",
    )

    chat_messages = session.get_chat_messages()
    assert [item["content"] for item in chat_messages] == [
        "Prompt one.",
        "Recovered result one.",
        "Prompt two.",
        "Recovered result two.",
    ]
    assert [item["kind"] for item in chat_messages] == ["chat", "run_result", "chat", "run_result"]


def test_websession_interrupt_cancels_active_async_run(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok
    _install_dummy_llm_profile(monkeypatch)

    run_dir = system_root(workspace=ws) / "runs" / "run_interrupt"
    started: dict[str, bool] = {"value": False}

    class DummyRunner:
        async def arun(self, prompt: str, **kwargs):
            _ = (prompt, kwargs)
            started["value"] = True
            await asyncio.sleep(60)
            return {"status": "done"}

    def fake_build_specialist_runner(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            runner=DummyRunner(),
            run_context=SimpleNamespace(run_id="run_interrupt", run_dir=run_dir, model_name="task-model"),
        )

    monkeypatch.setattr(specialists_mod, "build_specialist_runner", fake_build_specialist_runner)

    msg = session.start_run(
        prompt="long running task",
        lane="research",
        run_mode="new_run",
        resume_run_name="",
        proposal_review=False,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if started["value"]:
            break
        time.sleep(0.05)
    assert started["value"] is True

    interrupt_msg = session.request_interrupt_current_run(note="stop now")
    assert interrupt_msg == "Interrupt requested."

    session.run_thread.join(timeout=5)
    assert session.run_thread is not None
    assert not session.run_thread.is_alive()
    assert session.run_status == "interrupted_paused"
    interrupt = session.interrupt_status()["interrupt"]
    assert interrupt.get("requested") is True
    assert interrupt.get("acked") is True


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
    assert "Source: `all`" in text
    assert "Prefer MACE screening before VASP" in text
    assert "catmaster." in text


def test_websession_reads_langmem_long_term_memory_index(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    db_path = system_root(workspace=ws) / "deepagent_memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE store (prefix TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL)")
    project_id = session._project_id_for_workspace(ws)
    prefix = ".".join(("catmaster", project_id, "long_term_memory"))
    payload = {"content": "Pt(111) 1/4 ML benchmark was validated under a consistent slab setup."}
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (prefix, "memory-001", json.dumps(payload).encode("utf-8")),
    )
    conn.commit()
    conn.close()

    text = session.read_memory_index(source="langmem")
    assert "long_term_memory" in text
    assert "Source: `langmem`" in text
    assert "Pt(111) 1/4 ML benchmark was validated" in text


def test_websession_reads_instruction_memory_only_when_requested(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    db_path = system_root(workspace=ws) / "deepagent_memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE store (prefix TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL)")
    project_id = session._project_id_for_workspace(ws)
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (
            ".".join(("catmaster", project_id, "filesystem")),
            "/AGENTS.md",
            json.dumps({"content": "Use MACE before VASP when screening."}).encode("utf-8"),
        ),
    )
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (
            ".".join(("catmaster", project_id, "long_term_memory")),
            "memory-001",
            json.dumps({"content": "Durable benchmark fact."}).encode("utf-8"),
        ),
    )
    conn.commit()
    conn.close()

    text = session.read_memory_index(source="instruction")
    assert "Source: `instruction`" in text
    assert "Use MACE before VASP when screening." in text
    assert "Durable benchmark fact." not in text


def test_websession_reads_persistent_memory_index_when_workspace_path_changes(tmp_path: Path) -> None:
    old_root = tmp_path / "old_root"
    new_root = tmp_path / "new_root"
    ws_old = old_root / "Pt_111"
    ws_new = new_root / "Pt_111"
    ensure_project_space_layout(ws_old, create=True)

    run_dir = system_root(workspace=ws_old) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    old_project_id = WebSession._project_id_for_workspace(ws_old)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "project_id": old_project_id,
                "run_id": "run_001",
                "workspace": str(ws_old),
                "model_name": "test-model",
                "start_time": "2026-03-17T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    db_path = system_root(workspace=ws_old) / "deepagent_memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE store (prefix TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL)")
    payload = {
        "content": [
            "# Persistent Project Memory",
            "",
            "- Pt(111) 1/4 ML H adsorption benchmark is durable project context.",
        ],
    }
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (".".join(("catmaster", old_project_id, "filesystem")), "/AGENTS.md", json.dumps(payload).encode("utf-8")),
    )
    conn.commit()
    conn.close()

    shutil.copytree(ws_old, ws_new)

    session = WebSession()
    session.set_workspace_root(str(new_root))
    ok, _ = session.open_workspace(str(ws_new), create=False)
    assert ok
    session.select_run("run_001")

    text = session.read_memory_index()
    assert "Persistent Memory" in text
    assert "Pt(111) 1/4 ML H adsorption benchmark" in text


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
