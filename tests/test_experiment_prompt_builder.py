from __future__ import annotations

from catmaster.agents.research_schemas import ResearchRequest
from catmaster.runtime.research import ExperimentBriefModel, HypothesisRecord, ResearchBoard
from catmaster.runtime.research.experiment_runner import build_experiment_child_request


def test_experiment_child_request_stays_bounded() -> None:
    board = ResearchBoard(
        campaign_id="camp",
        question="What controls CO adsorption on Fe(110)?",
        exploration_policy="anchored",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=1,
        hypotheses=[
            HypothesisRecord(
                hypothesis_id="H1",
                text="Bridge site is most stable.",
                source="user_seed",
            )
        ],
    )
    brief = ExperimentBriefModel(
        title="Bounded CO site check",
        hypothesis_ids=["H1"],
        lane="fast",
        goal="Compare bridge and ontop placements.",
        task_detail="Prepare bounded initial structures and relax only the selected surface.",
        expected_outputs=["ranked structures", "energy summary"],
        why_now="Need a first-pass experiment.",
        stop_condition="Stop when one stable candidate is identified.",
        reference_hint=["Use the existing Fe(110) slab."],
    )
    request = ResearchRequest(question=board.question, seed_hypotheses=["Bridge site is most stable."])

    text = build_experiment_child_request(brief=brief, research_request=request, board=board)

    assert "You are executing one bounded experiment" in text
    assert "Selected hypotheses:" in text
    assert "- H1: Bridge site is most stable." in text
    assert "- Do not run literature research." in text
    assert "- You may update MEMORY/** when the run produces reusable facts, files, or open questions." in text
    assert "- Do not modify the project goal for this bounded child run." in text
