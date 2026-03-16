from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import catmaster.specialists as specialists_mod
from catmaster.specialists import RUN_STATE_FILE
from catmaster.tools.base import system_root
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


def _write_completed_run(run_dir: Path, *, entrypoint: str, final_answer: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_STATE_FILE).write_text(
        json.dumps(
            {
                "entrypoint": entrypoint,
                "status": "done",
                "summary": "Completed.",
                "final_answer": final_answer,
                "facts": [],
            }
        ),
        encoding="utf-8",
    )


def _mk_run(path: Path, *, lane: str = "research") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / RUN_STATE_FILE).write_text(json.dumps({"entrypoint": lane, "status": "running"}), encoding="utf-8")
    (path / "meta.json").write_text("{}", encoding="utf-8")


def test_resolve_resume_target_uses_selected_run_lane(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    run_dir = system_root(workspace=ws) / "runs" / "run_experiment"
    _mk_run(run_dir, lane="experiment")
    session.selected_run_dir = run_dir

    picked, lane, err = session._resolve_resume_target(resume_run_name="", workspace=ws)
    assert err is None
    assert picked == run_dir.resolve()
    assert lane == "experiment"


def test_start_run_resume_requires_selected_run(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    msg = session.start_run(
        prompt="",
        lane="research",
        run_mode="resume_selected_run",
        resume_run_name="",
        proposal_review=True,
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
    _install_dummy_llm_profile(monkeypatch)

    resume_run = system_root(workspace=ws) / "runs" / "run_experiment"
    _mk_run(resume_run, lane="experiment")
    captured: dict = {}

    class DummyRunner:
        def resume(self, *, human_feedback: str = ""):
            captured["resume_feedback"] = human_feedback
            _write_completed_run(resume_run, entrypoint="experiment", final_answer="Resume finished.")
            return {"status": "done"}

    def fake_build_specialist_runner(**kwargs):
        captured["init"] = kwargs
        return SimpleNamespace(
            runner=DummyRunner(),
            run_context=SimpleNamespace(run_id="run_test", run_dir=resume_run, model_name="dummy"),
        )

    monkeypatch.setattr(specialists_mod, "build_specialist_runner", fake_build_specialist_runner)

    msg = session.start_run(
        prompt="resume with hint",
        lane="research",
        run_mode="resume_selected_run",
        resume_run_name="run_experiment",
        proposal_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None
    session.run_thread.join(timeout=5)
    assert session.run_thread is not None
    assert not session.run_thread.is_alive()
    assert Path(str(captured["init"]["run_dir"])).resolve() == resume_run.resolve()
    assert captured["init"]["preferred_entrypoint"] == "experiment"
    assert captured["resume_feedback"] == "resume with hint"


def test_start_run_new_lane_uses_specialist_runner(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok
    _install_dummy_llm_profile(monkeypatch)

    run_dir = system_root(workspace=ws) / "runs" / "run_test"
    captured: dict = {}

    class DummyRunner:
        def run(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["run_kwargs"] = kwargs
            _write_completed_run(run_dir, entrypoint="writing", final_answer="Done.")
            return {"status": "done"}

    def fake_build_specialist_runner(**kwargs):
        captured["init"] = kwargs
        return SimpleNamespace(
            runner=DummyRunner(),
            run_context=SimpleNamespace(run_id="run_test", run_dir=run_dir, model_name="task-model"),
        )

    monkeypatch.setattr(specialists_mod, "build_specialist_runner", fake_build_specialist_runner)

    msg = session.start_run(
        prompt="do something",
        lane="writing",
        run_mode="new_run",
        resume_run_name="",
        proposal_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None
    session.run_thread.join(timeout=5)
    assert session.run_thread is not None
    assert not session.run_thread.is_alive()
    assert session.run_status == "done"
    assert captured["init"]["preferred_entrypoint"] == "writing"
    assert captured["run_kwargs"]["entrypoint"] == "writing"
    assert captured["prompt"] == "do something"


def test_start_run_resume_uses_run_state_entrypoint_even_when_ui_lane_differs(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok
    _install_dummy_llm_profile(monkeypatch)

    research_run = system_root(workspace=ws) / "runs" / "run_research"
    _mk_run(research_run, lane="research")
    captured: dict = {}

    class DummyRunner:
        def resume(self, *, human_feedback: str = ""):
            captured["resume_feedback"] = human_feedback
            _write_completed_run(research_run, entrypoint="research", final_answer="Research finished.")
            return {"status": "done"}

    def fake_build_specialist_runner(**kwargs):
        captured["init"] = kwargs
        return SimpleNamespace(
            runner=DummyRunner(),
            run_context=SimpleNamespace(run_id="run_research", run_dir=research_run, model_name="research-model"),
        )

    monkeypatch.setattr(specialists_mod, "build_specialist_runner", fake_build_specialist_runner)

    msg = session.start_run(
        prompt="resume note",
        lane="experiment",
        run_mode="resume_selected_run",
        resume_run_name="run_research",
        proposal_review=True,
        log_llm=False,
        full_auto_major=False,
    )
    assert msg == "Run started."
    assert session.run_thread is not None
    session.run_thread.join(timeout=5)
    assert session.run_thread is not None
    assert not session.run_thread.is_alive()
    assert captured["resume_feedback"] == "resume note"
    assert captured["init"]["preferred_entrypoint"] == "research"
