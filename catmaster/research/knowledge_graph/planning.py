from __future__ import annotations

from typing import Any

from .context import runnable_frontier_ids
from .models import ResearchExperimentEvaluationDraft, ResearchGraphPlanningProposal


def build_planning_preview(
    snapshot: dict[str, Any],
    proposal: ResearchGraphPlanningProposal,
    *,
    focus_node_id: str = "",
    evaluation: ResearchExperimentEvaluationDraft | None = None,
) -> dict[str, Any]:
    """Project one scientific proposal into temporary technology-tree nodes.

    Staging projects scientific branches only. A later evaluator may attach
    current-revision Experiment comparisons to the disposable preview.
    """

    durable_nodes = list(snapshot.get("nodes") or [])
    durable_by_id = {
        str(node["node_id"]): node for node in durable_nodes
    }
    ready_ids = set(
        runnable_frontier_ids(
            durable_nodes,
            list(snapshot.get("edges") or []),
        )
    )
    proposed_hypothesis_ids = {item.proposal_id for item in proposal.hypotheses}
    proposed_experiment_ids = {item.proposal_id for item in proposal.experiments}
    known_ids = {
        *durable_by_id,
        *proposed_hypothesis_ids,
        *proposed_experiment_ids,
    }

    selected_id = str(proposal.recommended_target_id or "")
    if selected_id and selected_id not in {
        *proposed_hypothesis_ids,
        *proposed_experiment_ids,
        *ready_ids,
    }:
        raise ValueError(
            "The recommended route must be a proposed Experiment or an "
            "existing ready Experiment."
        )
    if selected_id in proposed_hypothesis_ids:
        raise ValueError("A planning recommendation must select an Experiment.")

    preview_nodes: list[dict[str, Any]] = []
    for item in proposal.hypotheses:
        preview_nodes.append(
            {
                "node_id": item.proposal_id,
                "kind": "hypothesis",
                "title": item.title or item.claim[:120],
                "state": "",
                "body": item.model_dump(
                    mode="json",
                    exclude={"proposal_id", "title", "refs"},
                ),
                "refs": [
                    ref.model_dump(mode="json")
                    for ref in item.refs
                ],
                "provisional": True,
            }
        )
    for item in proposal.experiments:
        preview_nodes.append(
            {
                "node_id": item.proposal_id,
                "kind": "experiment",
                "title": item.title or item.objective[:120],
                "state": "proposed",
                "body": item.model_dump(
                    mode="json",
                    exclude={
                        "proposal_id",
                        "title",
                        "tests_hypothesis_ids",
                        "depends_on_experiment_ids",
                        "refs",
                    },
                ),
                "refs": [
                    ref.model_dump(mode="json")
                    for ref in item.refs
                ],
                "provisional": True,
            }
        )

    preview_edges: list[dict[str, str]] = []
    for item in proposal.experiments:
        preview_edges.extend(
            {
                "source_node_id": hypothesis_id,
                "target_node_id": item.proposal_id,
                "relation": "tests",
            }
            for hypothesis_id in item.tests_hypothesis_ids
            if hypothesis_id in known_ids
        )
        preview_edges.extend(
            {
                "source_node_id": item.proposal_id,
                "target_node_id": dependency_id,
                "relation": "depends_on",
            }
            for dependency_id in item.depends_on_experiment_ids
            if dependency_id in known_ids
        )
    summary = proposal.recommendation_reason.strip()
    candidate_experiment_ids = [
        *sorted(ready_ids),
        *(item.proposal_id for item in proposal.experiments),
    ]
    score_rows: list[dict[str, Any]] = []
    innovation_recommendation = ""
    conservative_recommendation = ""
    evaluation_memo = ""
    if evaluation is not None:
        score_rows = [
            {
                "experiment_id": experiment_id,
                "innovation_score": float(innovation_score),
                "conservative_score": float(conservative_score),
            }
            for experiment_id, innovation_score, conservative_score in zip(
                evaluation.experiment_ids,
                evaluation.innovation_scores,
                evaluation.conservative_scores,
                strict=True,
            )
        ]
        innovation_recommendation = evaluation.innovation_recommendation
        conservative_recommendation = evaluation.conservative_recommendation
        evaluation_memo = evaluation.evaluation_memo
    return {
        "focus_node_id": str(focus_node_id or ""),
        "nodes": preview_nodes,
        "edges": preview_edges,
        "proposer_recommended_target_id": selected_id,
        "candidate_experiment_ids": candidate_experiment_ids,
        "experiment_evaluations": score_rows,
        "innovation_recommendation": innovation_recommendation,
        "conservative_recommendation": conservative_recommendation,
        "evaluation_memo": evaluation_memo,
        "summary": summary,
    }


__all__ = ["build_planning_preview"]
