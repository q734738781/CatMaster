from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.webui.components import unpack_prompt
from catmaster.webui.session import WebSession
from catmaster.webui.web_reporter import PromptBroker


def _wait_for(path: Path, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def test_prompt_broker_supports_persisted_submit(tmp_path: Path) -> None:
    broker = PromptBroker()
    broker.set_store_dir(tmp_path / "hitl")

    captured: dict[str, str] = {}

    def _run_prompt() -> None:
        captured["text"] = broker.request_prompt("hitl", {"report_text": "need feedback"})

    t = threading.Thread(target=_run_prompt, daemon=True)
    t.start()

    deadline = time.time() + 3.0
    pending = None
    while time.time() < deadline:
        pending = broker.get_pending()
        if isinstance(pending, dict):
            break
        time.sleep(0.05)
    assert isinstance(pending, dict)
    prompt_id = str(pending.get("prompt_id") or "")
    assert prompt_id

    assert broker.submit_persisted(prompt_id, "approved")
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert captured.get("text") == "approved"


def test_session_loads_prompt_from_task_state_snapshot(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    run_dir = system_root(workspace=tmp_path) / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text(
        json.dumps(
            {
                "status": "awaiting_human_feedback",
                "last_interrupt": {
                    "type": "proposal_review",
                    "proposal_md": "# proposal",
                    "work_packages": ["wp1", "wp2"],
                    "message": "need review",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path.parent))
    ok, _ = session.open_workspace(str(tmp_path), create=False)
    assert ok
    session.select_run("run_001")
    session.run_status = "running"

    pending = session.get_prompt()
    assert isinstance(pending, dict)
    assert pending.get("kind") == "proposal_review"

    status_text = session.run_status_text()
    assert "awaiting_human_feedback" in status_text


def test_submit_prompt_via_file_is_disabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_002"
    hitl_dir = run_dir / "hitl"
    hitl_dir.mkdir(parents=True, exist_ok=True)

    ok = WebSession._submit_prompt_via_file(run_dir, prompt_id="p_123", text="approved")
    assert ok is False


def test_session_snapshot_prompt_comes_from_task_state_only(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    run_dir = system_root(workspace=tmp_path) / "runs" / "run_003"
    hitl_dir = run_dir / "hitl"
    hitl_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "awaiting_human_feedback",
                "last_interrupt": {
                    "type": "proposal_review",
                    "proposal_md": "# proposal",
                    "work_packages": ["wp1"],
                    "message": "need review",
                },
            },
            ensure_ascii=False,
            indent=2,
            ),
            encoding="utf-8",
        )

    session = WebSession()
    session.set_workspace_root(str(tmp_path.parent))
    ok, _ = session.open_workspace(str(tmp_path), create=False)
    assert ok
    session.select_run("run_003")

    pending = session.get_prompt()
    assert isinstance(pending, dict)
    assert pending.get("kind") == "proposal_review"


def test_session_marks_revised_proposal_review_after_task_intervention(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    run_dir = system_root(workspace=tmp_path) / "runs" / "run_004"
    hitl_dir = run_dir / "hitl"
    hitl_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_state.json").write_text(
        json.dumps(
            {
                "status": "awaiting_human_feedback",
                "hitl_history": [
                    {
                        "interrupt_type": "proposal_review",
                        "feedback": "yes",
                        "approved": True,
                    },
                    {
                        "task_id": "task_01",
                        "feedback": "Retry may help",
                    },
                ],
                "last_interrupt": {
                    "type": "proposal_review",
                    "proposal_md": "# revised proposal",
                    "work_packages": ["wp1"],
                    "message": "need review",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    session = WebSession()
    session.set_workspace_root(str(tmp_path.parent))
    ok, _ = session.open_workspace(str(tmp_path), create=False)
    assert ok
    session.select_run("run_004")

    pending = session.get_prompt()
    assert isinstance(pending, dict)
    payload = pending.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("run_id") == "run_004"
    assert payload.get("is_revised") is True
    assert payload.get("reason") == "replanning after HITL"


def test_unpack_prompt_shows_revised_proposal_review_meta() -> None:
    display = unpack_prompt(
        {
            "prompt_id": "prompt_123",
            "kind": "proposal_review",
            "payload": {
                "run_id": "run_004",
                "prompt_id": "prompt_123",
                "is_revised": True,
                "reason": "replanning after HITL",
                "proposal_description": "# revised proposal",
                "todo": ["wp1", "wp2"],
            },
        }
    )

    assert display.visible is True
    assert display.title == "Revised Proposal Review"
    assert "same run: `run_004`" in display.meta
    assert "reason: replanning after HITL" in display.meta
    assert "prompt id: `prompt_123`" in display.meta
