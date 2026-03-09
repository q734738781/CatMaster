from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.research import ExperimentLaneRunner, HypothesisRecord, ResearchBoard
from catmaster.runtime.research.models import ExperimentBriefModel


@pytest.mark.anyio
async def test_experiment_lane_runner_numbers_experiments_globally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeRunner:
        async def arun(self, user_request: str, *, lane: str, proposal_review: bool):
            captured["user_request"] = user_request
            captured["lane"] = lane
            captured["proposal_review"] = proposal_review
            return {
                "status": "done",
                "summary": "ok",
                "observations": [],
                "run_id": "child_run",
                "run_dir": str(tmp_path / "metadata" / "runs" / "child_run"),
                "pending_memory_updates": [],
            }

    def fake_build_graph_runner(**kwargs):
        captured["factory_kwargs"] = kwargs
        return SimpleNamespace(
            runner=_FakeRunner(),
            run_context=SimpleNamespace(run_id="child_run", run_dir=tmp_path / "metadata" / "runs" / "child_run"),
        )

    monkeypatch.setattr("catmaster.runtime.research.experiment_runner.build_graph_runner", fake_build_graph_runner)

    runner = ExperimentLaneRunner(workspace=tmp_path, llm_profile=object(), project_id="proj")
    board = ResearchBoard(
        campaign_id="camp",
        question="Q",
        exploration_policy="anchored",
        max_cycles=4,
        max_literature_queries=2,
        max_fast_runs=2,
        max_standard_runs=2,
        used_fast_runs=1,
        used_standard_runs=1,
        hypotheses=[HypothesisRecord(hypothesis_id="H1", text="seed", source="user_seed")],
        action_refs=[
            {"action_id": "exp_001", "kind": "experiment", "status": "done", "summary": "first", "ref_path": "x"},
            {"action_id": "lit_001", "kind": "literature", "status": "done", "summary": "lit", "ref_path": "y"},
            {"action_id": "exp_002", "kind": "experiment", "status": "done", "summary": "second", "ref_path": "z"},
        ],
    )
    brief = ExperimentBriefModel(
        title="validate",
        hypothesis_ids=["H1"],
        lane="standard",
        goal="g",
        task_detail="d",
        expected_outputs=["o"],
        why_now="now",
        stop_condition="stop",
    )

    pack = await runner.arun(brief=brief, research_request=SimpleNamespace(seed_hypotheses=[]), board=board)

    assert pack.experiment_id == "exp_003"
    assert captured["factory_kwargs"]["reporter"] is runner.reporter
    run_policy = captured["factory_kwargs"]["run_policy"]
    assert run_policy.allow_memory_patch is True
