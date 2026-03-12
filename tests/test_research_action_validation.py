from __future__ import annotations

import pytest

from catmaster.agents.research_nodes import validate_research_action, validate_research_state_sync
from catmaster.agents.research_schemas import (
    ExperimentBrief,
    NewHypothesisProposal,
    ResearchLeadOutput,
    ResearchRequest,
    ResearchStateSyncOutput,
    RunLiteraturePayload,
    RunWriterPayload,
)
from catmaster.runtime.research import HypothesisRecord, ResearchBoard


def _board() -> ResearchBoard:
    return ResearchBoard(
        campaign_id="camp",
        question="Q",
        exploration_policy="anchored",
        max_cycles=3,
        max_literature_queries=2,
        max_fast_runs=1,
        max_standard_runs=1,
        hypotheses=[HypothesisRecord(hypothesis_id="H1", text="seed", source="user_seed")],
    )


def test_anchored_policy_rejects_sync_new_hypotheses() -> None:
    request = ResearchRequest(question="Q", exploration_policy="anchored")
    sync = ResearchStateSyncOutput(
        current_best_answer_md="Current answer",
        new_hypotheses=[NewHypothesisProposal(text="H2", rationale="expand")],
        open_questions=["Need evidence"],
    )
    with pytest.raises(ValueError, match="anchored policy"):
        validate_research_state_sync(sync=sync, request=request)


def test_deep_report_rejected_when_request_disallows_it() -> None:
    board = _board()
    request = ResearchRequest(question="Q", allow_deep_report=False)
    action = ResearchLeadOutput(
        state="RunLiterature",
        rationale="Need deeper grounding.",
        run_literature=RunLiteraturePayload(query="q", depth="deep_report", why_now="need survey"),
    )
    with pytest.raises(ValueError, match="deep_report"):
        validate_research_action(action=action, request=request, board=board)


def test_research_lead_output_rejects_branch_inconsistency() -> None:
    with pytest.raises(ValueError):
        ResearchLeadOutput(
            state="RunExperiment",
            rationale="bounded experiment",
            run_literature=RunLiteraturePayload(query="q", depth="quick", why_now="x"),
            run_experiment=ExperimentBrief(
                title="exp",
                hypothesis_ids=["H1"],
                lane="fast",
                goal="g",
                task_detail="d",
                expected_outputs=["o"],
                why_now="now",
                stop_condition="stop",
            ),
        )


def test_run_writer_requires_writing_mode() -> None:
    board = _board()
    request = ResearchRequest(question="Q", writing_mode="none")
    action = ResearchLeadOutput(
        state="RunWriter",
        rationale="Enough evidence exists; write now.",
        run_writer=RunWriterPayload(
            request="Write a compact markdown report from the current evidence.",
            writing_mode="internal_report",
            output_format="md",
        ),
    )
    validate_research_action(action=action, request=request, board=board)


def test_run_writer_requires_non_empty_request() -> None:
    board = _board()
    request = ResearchRequest(question="Q", writing_mode="none")
    action = ResearchLeadOutput(
        state="RunWriter",
        rationale="Enough evidence exists; write now.",
        run_writer=RunWriterPayload(
            request="   ",
            writing_mode="internal_report",
            output_format="md",
        ),
    )
    with pytest.raises(ValueError, match="non-empty request"):
        validate_research_action(action=action, request=request, board=board)


def test_local_expand_sync_requires_parent_hypothesis_ids() -> None:
    request = ResearchRequest(question="Q", exploration_policy="local_expand")
    sync = ResearchStateSyncOutput(
        current_best_answer_md="Current answer",
        new_hypotheses=[NewHypothesisProposal(text="H2", rationale="expand")],
    )
    with pytest.raises(ValueError, match="parent_hypothesis_ids"):
        validate_research_state_sync(sync=sync, request=request)
