from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import types

from catmaster.tools.base import system_root
from catmaster.webui.session import WebSession


def _mk_run(path: Path, *, lane: str = "standard") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "task_state.json").write_text(json.dumps({"lane": lane}), encoding="utf-8")
    (path / "meta.json").write_text("{}", encoding="utf-8")


def test_resolve_resume_target_uses_selected_run_lane(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    run_dir = system_root(workspace=ws) / "runs" / "run_fast"
    _mk_run(run_dir, lane="fast")
    session.selected_run_dir = run_dir

    picked, lane, err = session._resolve_resume_target(resume_run_name="", workspace=ws)
    assert err is None
    assert picked == run_dir.resolve()
    assert lane == "fast"


def test_start_run_resume_requires_selected_run(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    msg = session.start_run(
        prompt="",
        lane="standard",
        run_mode="resume_selected_run",
        resume_run_name="",
        plan_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Select a run to resume."
    assert session.run_thread is None


def test_start_run_resume_uses_selected_run_lane(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    sys_root = system_root(workspace=ws)
    resume_run = sys_root / "runs" / "run_fast"
    _mk_run(resume_run, lane="fast")

    captured: dict = {}

    class DummyOrchestrator:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            run_dir = Path(str(kwargs.get("resume_dir") or "")).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            self.run_context = SimpleNamespace(run_id="run_test", run_dir=run_dir, model_name="dummy")

        def run(self, user_request: str, **kwargs):
            captured["run_prompt"] = user_request
            captured["run_kwargs"] = kwargs
            return {"status": "done"}

    fake_orchestrator_mod = types.ModuleType("catmaster.agents.orchestrator")
    fake_orchestrator_mod.Orchestrator = DummyOrchestrator
    monkeypatch.setitem(sys.modules, "catmaster.agents.orchestrator", fake_orchestrator_mod)

    fake_llm_config_mod = types.ModuleType("catmaster.llm.config")

    class DummyLLMProfile:
        @classmethod
        def from_env_or_file(cls, *_a, **_k):
            return object()

    fake_llm_config_mod.LLMProfile = DummyLLMProfile
    monkeypatch.setitem(sys.modules, "catmaster.llm.config", fake_llm_config_mod)

    msg = session.start_run(
        prompt="resume with hint",
        lane="standard",
        run_mode="resume_selected_run",
        resume_run_name="run_fast",
        plan_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None
    session.run_thread.join(timeout=5)
    assert session.run_thread is not None
    assert not session.run_thread.is_alive()

    assert captured["init"]["resume"] is True
    assert Path(str(captured["init"]["resume_dir"])).resolve() == resume_run.resolve()
    assert captured["run_kwargs"]["lane"] == "fast"
    assert captured["run_kwargs"]["resume_feedback"] == "resume with hint"
