from __future__ import annotations

import importlib.util
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.modules.setdefault("langchain_openai", types.SimpleNamespace(ChatOpenAI=object))
try:
    _lc_prompts_spec = importlib.util.find_spec("langchain_core.prompts")
except (ModuleNotFoundError, ValueError):
    _lc_prompts_spec = None
if _lc_prompts_spec is None and "langchain_core.prompts" not in sys.modules:
    _lc_prompts = types.ModuleType("langchain_core.prompts")

    class _FakeChatPromptTemplate:
        @staticmethod
        def from_messages(_messages):
            return object()

    _lc_prompts.ChatPromptTemplate = _FakeChatPromptTemplate
    _lc_prompts.__spec__ = importlib.machinery.ModuleSpec("langchain_core.prompts", loader=None)
    _lc_core = types.ModuleType("langchain_core")
    _lc_core.prompts = _lc_prompts
    _lc_core.__spec__ = importlib.machinery.ModuleSpec("langchain_core", loader=None)
    sys.modules["langchain_core"] = _lc_core
    sys.modules["langchain_core.prompts"] = _lc_prompts

from catmaster.agents.orchestrator import Orchestrator


def _make_run_standard_orchestrator(tmp_path: Path, *, decisions: list[dict]) -> tuple[Orchestrator, list[dict]]:
    orch = Orchestrator.__new__(Orchestrator)
    sequence = list(decisions)
    run_dir = tmp_path / "metadata" / "runs" / "run_01"
    run_dir.mkdir(parents=True, exist_ok=True)

    orch.resuming = False
    orch.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    orch.run_context = SimpleNamespace(run_dir=run_dir)
    orch.run_control = SimpleNamespace(clear_interrupt=lambda: None, snapshot=lambda: {})
    orch._emit = lambda *args, **kwargs: None
    orch._trace_event = lambda *args, **kwargs: None
    orch._initialize_memory_goal = lambda _req: None
    orch._create_proposal = lambda _req, log_llm=False: {
        "proposal_md": "# Plan\n\nInitial route.",
        "work_packages": ["task one", "task two"],
    }
    orch._write_proposal = lambda _proposal: "proposal.md"
    orch._write_task_state = lambda _state: None
    orch._proposal_path = lambda: Path("proposal.md")
    orch._director_decide = lambda **kwargs: sequence.pop(0)
    orch._summarize_tasks = lambda _req, _obs, _status: "summary"
    orch._publish_report = lambda _req, _summary: {"final_report": "reports/final_report.md"}
    return orch, sequence


def test_run_standard_commits_initial_director_memory(tmp_path: Path) -> None:
    orch, sequence = _make_run_standard_orchestrator(
        tmp_path,
        decisions=[{"state": "StopAndSynthesize"}],
    )
    commit_calls: list[dict] = []
    orch._commit_director_memory = lambda **kwargs: commit_calls.append(dict(kwargs))

    result = Orchestrator._run_standard(
        orch,
        "user request",
        log_llm=False,
        resume_feedback="",
        proposal_review=False,
        proposal_feedback_provider=None,
        full_auto_major=False,
        defer_ui=True,
        start_ui=lambda *_args: None,
    )

    assert result["status"] == "done"
    assert not sequence
    assert len(commit_calls) == 1
    assert commit_calls[0]["decision_state"] == "InitialPlanCommitted"
    assert commit_calls[0]["proposal_path"] == "proposal.md"


def test_run_standard_commits_major_revise_director_memory(tmp_path: Path) -> None:
    orch, sequence = _make_run_standard_orchestrator(
        tmp_path,
        decisions=[
            {
                "state": "MajorReviseProposal",
                "updated_proposal_md": "# Plan v2\n\nRoute changed.",
                "updated_work_packages": ["task three"],
                "rationale": "Need a different method.",
                "change_log": "Replaced route with new method.",
            },
            {"state": "StopAndSynthesize"},
        ],
    )
    commit_calls: list[dict] = []
    orch._commit_director_memory = lambda **kwargs: commit_calls.append(dict(kwargs))

    result = Orchestrator._run_standard(
        orch,
        "user request",
        log_llm=False,
        resume_feedback="",
        proposal_review=False,
        proposal_feedback_provider=None,
        full_auto_major=True,
        defer_ui=True,
        start_ui=lambda *_args: None,
    )

    assert result["status"] == "done"
    assert not sequence
    assert len(commit_calls) == 2
    assert commit_calls[0]["decision_state"] == "InitialPlanCommitted"
    assert commit_calls[1]["decision_state"] == "MajorReviseProposal"
    assert commit_calls[1]["rationale"] == "Need a different method."


def test_run_standard_failure_auto_replan_returns_to_director(tmp_path: Path) -> None:
    orch, sequence = _make_run_standard_orchestrator(
        tmp_path,
        decisions=[
            {
                "state": "PerformNextTask",
                "task_packet": {
                    "goal": "Investigate failed subset",
                    "task_detail": "Inspect failed subset only and produce actionable retry plan.",
                    "expected_outputs": ["reports/retry_plan.md"],
                    "reference_hint": ["MEMORY/topics/FACTS.md", "rg keywords: failed subset", "done-check: do not rerun successes"],
                    "suggested_tools": ["bash"],
                },
            },
            {"state": "StopAndSynthesize"},
        ],
    )
    commit_calls: list[dict] = []
    orch._commit_director_memory = lambda **kwargs: commit_calls.append(dict(kwargs))

    executed_tasks: list[str] = []

    def _fake_execute_task(*, task_id: str, **_kwargs):
        executed_tasks.append(task_id)
        return {
            "task_id": task_id,
            "outcome": "failure",
            "summary": "Tool-call limit reached (60); auto replan requested.",
            "observation_path": "observations/obs_task_01.md",
            "key_artifacts": [{
                "path": "audit/toolcall_context/run_01_task_01_max_steps.json",
                "description": "max_steps context",
                "kind": "log",
            }],
            "auto_replan": True,
            "failure_kind": "max_steps",
            "event_path": "memory/events.jsonl",
            "memory_merge_failed": False,
            "memory_merge_error": "",
        }

    orch._execute_task = _fake_execute_task

    result = Orchestrator._run_standard(
        orch,
        "user request",
        log_llm=False,
        resume_feedback="",
        proposal_review=False,
        proposal_feedback_provider=None,
        full_auto_major=False,
        defer_ui=True,
        start_ui=lambda *_args: None,
    )

    assert result["status"] == "done"
    assert executed_tasks == ["task_01"]
    assert not sequence
    assert len(result["observations"]) == 1
    assert result["observations"][0]["failure_kind"] == "max_steps"
    assert result["observations"][0]["auto_replan"] is True
    assert commit_calls and commit_calls[0]["decision_state"] == "InitialPlanCommitted"
