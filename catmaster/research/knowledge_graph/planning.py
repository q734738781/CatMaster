from __future__ import annotations

from typing import Any

from .context import ranked_frontier_ids
from .models import ResearchGraphPlanningProposal


def _proposal_route_ids(
    selected_id: str,
    *,
    proposed_hypotheses: dict[str, Any],
    proposed_experiments: dict[str, Any],
) -> list[str]:
    """Return the selected proposal and only its scientific prerequisites."""

    if selected_id in proposed_hypotheses:
        return [selected_id]
    if selected_id not in proposed_experiments:
        return [selected_id] if selected_id else []

    route: list[str] = []
    included: set[str] = set()

    def append_once(node_id: str) -> None:
        if node_id not in included:
            route.append(node_id)
            included.add(node_id)

    def include_experiment(experiment_id: str) -> None:
        if experiment_id in included:
            return
        item = proposed_experiments[experiment_id]
        for dependency_id in item.depends_on_experiment_ids:
            if dependency_id in proposed_experiments:
                include_experiment(dependency_id)
        for hypothesis_id in item.tests_hypothesis_ids:
            if hypothesis_id in proposed_hypotheses:
                append_once(hypothesis_id)
        append_once(experiment_id)

    include_experiment(selected_id)
    return route


def build_planning_preview(
    snapshot: dict[str, Any],
    proposal: ResearchGraphPlanningProposal,
    *,
    focus_node_id: str = "",
) -> dict[str, Any]:
    """Project one scientific proposal into temporary technology-tree nodes.

    The proposer chooses a route and explains it in scientific language. This
    projection never invents a scalar utility or re-scores branches.
    """

    durable_nodes = list(snapshot.get("nodes") or [])
    durable_by_id = {
        str(node["node_id"]): node for node in durable_nodes
    }
    ready_ids = set(
        ranked_frontier_ids(
            durable_nodes,
            list(snapshot.get("edges") or []),
        )
    )
    proposed_hypotheses = {
        item.proposal_id: item for item in proposal.hypotheses
    }
    proposed_experiments = {
        item.proposal_id: item for item in proposal.experiments
    }
    known_ids = {
        *durable_by_id,
        *proposed_hypotheses,
        *proposed_experiments,
    }

    selected_id = str(proposal.recommended_target_id or "")
    if selected_id and selected_id not in {
        *proposed_hypotheses,
        *proposed_experiments,
        *ready_ids,
    }:
        raise ValueError(
            "The recommended route must be a proposal in this plan or an "
            "existing ready experiment."
        )
    route_ids = _proposal_route_ids(
        selected_id,
        proposed_hypotheses=proposed_hypotheses,
        proposed_experiments=proposed_experiments,
    )
    route_id_set = set(route_ids)

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
                "recommended": item.proposal_id in route_id_set,
                "planning_reason": (
                    proposal.recommendation_reason
                    if item.proposal_id == selected_id
                    else ""
                ),
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
                "recommended": item.proposal_id in route_id_set,
                "planning_reason": (
                    proposal.recommendation_reason
                    if item.proposal_id == selected_id
                    else ""
                ),
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
    if (
        focus_node_id
        and durable_by_id.get(focus_node_id, {}).get("kind") == "result"
    ):
        preview_edges.extend(
            {
                "source_node_id": focus_node_id,
                "target_node_id": item.proposal_id,
                "relation": "suggests",
            }
            for item in proposal.hypotheses
        )

    selected_title = ""
    if selected_id in proposed_hypotheses:
        item = proposed_hypotheses[selected_id]
        selected_title = item.title or item.claim[:120]
    elif selected_id in proposed_experiments:
        item = proposed_experiments[selected_id]
        selected_title = item.title or item.objective[:120]
    elif selected_id in durable_by_id:
        selected_title = str(durable_by_id[selected_id].get("title") or selected_id)

    summary = (
        f"Recommended next: {selected_title}. {proposal.recommendation_reason}".strip()
        if selected_id
        else "No route was recommended; the temporary branches remain available for review."
    )
    return {
        "focus_node_id": str(focus_node_id or ""),
        "nodes": preview_nodes,
        "edges": preview_edges,
        "recommended_target_id": selected_id,
        "recommended_proposal_id": (
            selected_id
            if selected_id in proposed_hypotheses
            or selected_id in proposed_experiments
            else ""
        ),
        "recommended_existing_node_id": (
            selected_id if selected_id in durable_by_id else ""
        ),
        "route_ids": route_ids,
        "summary": summary,
    }


__all__ = ["build_planning_preview"]
