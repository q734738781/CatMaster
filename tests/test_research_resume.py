from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.agents.research_nodes import build_dossier_node
from catmaster.agents.research_runner import ResearchRunner
from catmaster.agents.research_schemas import ResearchRequest
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import (
    ConclusionRecord,
    HypothesisRecord,
    ResearchBoard,
    ResearchDossier,
    ResearchStore,
)
from catmaster.runtime.run_context import RunContext


def _profile():
    return SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role, provider="openai", base_url=None),
    )


def _runner(tmp_path: Path) -> ResearchRunner:
    run_context = RunContext.create(workspace=tmp_path, model_name="research-model")
    memory_store = MemoryStore.create_default(workspace=tmp_path)
    memory_store.ensure_exists()
    return ResearchRunner(
        llm_profile=_profile(),
        run_context=run_context,
        memory_store=memory_store,
        history_reader=None,
        skills_runtime=None,
        run_ledger_store=None,
        reporter=None,
    )


def test_research_runner_build_resume_state_routes_to_plan_for_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("catmaster.agents.research_runner.build_chat_model", lambda cfg: object())
    runner = _runner(tmp_path)
    store = runner.store
    request = ResearchRequest(question="Q")
    board = ResearchBoard(
        campaign_id=runner.run_context.run_id,
        question="Q",
        exploration_policy="anchored",
        status="running",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=1,
        hypotheses=[HypothesisRecord(hypothesis_id="H1", text="seed", source="user_seed")],
    )
    store.write_request(request)
    store.save_board(board)

    state = runner._build_resume_state()
    assert state["resume_goto"] == "plan_research"
    assert state["status"] == "running"


def test_research_runner_build_resume_state_summarizes_when_dossier_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("catmaster.agents.research_runner.build_chat_model", lambda cfg: object())
    runner = _runner(tmp_path)
    store = runner.store
    request = ResearchRequest(question="Q", writing_mode="paper_outline")
    board = ResearchBoard(
        campaign_id=runner.run_context.run_id,
        question="Q",
        exploration_policy="anchored",
        status="done",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=1,
    )
    store.write_request(request)
    store.save_board(board)
    store.persist_conclusion(
        ConclusionRecord(
            final_answer_md="Done",
            supported_claims=[],
            open_questions=[],
            recommended_next_steps=[],
            confidence="medium",
        )
    )
    store.persist_dossier(
        ResearchDossier(
            campaign_id=runner.run_context.run_id,
            question="Q",
            exploration_policy="anchored",
            final_answer_md="Done",
            confidence="medium",
        )
    )

    state = runner._build_resume_state()
    assert state["resume_goto"] == "summarize_research"


def test_research_runner_resume_feedback_reenters_planner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("catmaster.agents.research_runner.build_chat_model", lambda cfg: object())
    runner = _runner(tmp_path)
    store = runner.store
    request = ResearchRequest(question="Q")
    board = ResearchBoard(
        campaign_id=runner.run_context.run_id,
        question="Q",
        exploration_policy="anchored",
        status="needs_human",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=1,
        latest_human_questions=["Should we prioritize coverage effects?"],
        action_refs=[
            {"action_id": "ask_001", "kind": "ask_human", "status": "needs_human", "summary": "Need user priority", "ref_path": "request.json", "run_id": None}
        ],
    )
    store.write_request(request)
    store.save_board(board)

    state = runner._build_resume_state(resume_feedback="Prioritize coverage effects before broadening scope.")

    updated_board = ResearchBoard.model_validate(state["board"])
    assert state["resume_goto"] == "plan_research"
    assert state["status"] == "running"
    assert updated_board.human_feedback_summary == "Prioritize coverage effects before broadening scope."
    assert updated_board.action_refs[-1].kind == "human_feedback"
    assert store.load_action_log()[-1]["kind"] == "human_feedback"


@pytest.mark.anyio
async def test_build_dossier_node_falls_back_to_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = ResearchStore(workspace=tmp_path, campaign_id="camp")
    board = ResearchBoard(
        campaign_id="camp",
        question="Q",
        exploration_policy="anchored",
        status="done",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=1,
    )
    store.save_board(board)
    store.persist_conclusion(
        ConclusionRecord(
            final_answer_md="Done",
            supported_claims=["claim"],
            open_questions=["oq"],
            recommended_next_steps=["next"],
            confidence="medium",
        )
    )
    command = await build_dossier_node(
        {"board": board.model_dump(), "request": ResearchRequest(question="Q").model_dump()},
        store=store,
    )
    assert command.goto == "summarize_research"
