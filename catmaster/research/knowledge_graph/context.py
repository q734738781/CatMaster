from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import NodeKind
from .store import ResearchGraphStore


def runnable_frontier_ids(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[str]:
    """Return every eligible Experiment in a stable, non-preferential order."""

    by_id = {str(node["node_id"]): node for node in nodes}
    dependencies: dict[str, list[str]] = {}
    for edge in edges:
        if str(edge["relation"]) != "depends_on":
            continue
        dependencies.setdefault(str(edge["source_node_id"]), []).append(
            str(edge["target_node_id"])
        )

    eligible = [
        str(node["node_id"])
        for node in nodes
        if node["kind"] == NodeKind.EXPERIMENT.value
        and node["state"] == "ready"
        and all(
            by_id.get(dependency_id, {}).get("state") == "has_results"
            for dependency_id in dependencies.get(str(node["node_id"]), [])
        )
    ]
    return sorted(eligible)


class ResearchGraphContextBuilder:
    """Build a semantic, explicitly partial focus snippet for one graph."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        store: ResearchGraphStore | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.store = store or ResearchGraphStore(self.workspace)

    @staticmethod
    def _public_node(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "node_id": str(node["node_id"]),
            "kind": str(node["kind"]),
            "title": str(node["title"]),
            "state": str(node.get("state") or ""),
            "body": dict(node.get("body") or {}),
            "revision": int(node["revision"]),
        }

    @staticmethod
    def _public_edge(edge: dict[str, Any]) -> dict[str, str]:
        return {
            "source_node_id": str(edge["source_node_id"]),
            "target_node_id": str(edge["target_node_id"]),
            "relation": str(edge["relation"]),
        }

    @staticmethod
    def _node_markdown(node: dict[str, Any]) -> list[str]:
        body = dict(node.get("body") or {})
        lines = [
            f"### {node['title']} ({node['node_id']})",
            f"- Kind: {node['kind']}",
        ]
        if str(node.get("state") or ""):
            lines.append(f"- State: {node['state']}")
        for key, value in body.items():
            if value in ("", [], {}):
                continue
            if isinstance(value, list):
                lines.append(f"- {key}:")
                lines.extend(f"  - {item}" for item in value)
            else:
                lines.append(f"- {key}: {value}")
        return lines

    @staticmethod
    def _focus_neighborhood_ids(
        focus_node_id: str,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> set[str]:
        if not focus_node_id:
            return set()
        by_id = {str(node["node_id"]): node for node in nodes}
        selected = {focus_node_id}

        def include_incident(node_id: str, relations: set[str] | None = None) -> None:
            for edge in edges:
                relation = str(edge["relation"])
                if relations is not None and relation not in relations:
                    continue
                source = str(edge["source_node_id"])
                target = str(edge["target_node_id"])
                if source == node_id:
                    selected.add(target)
                if target == node_id:
                    selected.add(source)

        include_incident(focus_node_id)
        focus = by_id[focus_node_id]
        related_hypotheses: set[str] = set()
        if focus["kind"] == NodeKind.RESULT.value:
            producing_experiments = {
                str(edge["source_node_id"])
                for edge in edges
                if edge["relation"] == "produces"
                and str(edge["target_node_id"]) == focus_node_id
            }
            for experiment_id in producing_experiments:
                selected.add(experiment_id)
                for edge in edges:
                    if (
                        edge["relation"] == "tests"
                        and str(edge["target_node_id"]) == experiment_id
                    ):
                        hypothesis_id = str(edge["source_node_id"])
                        selected.add(hypothesis_id)
                        related_hypotheses.add(hypothesis_id)
            related_hypotheses.update(
                str(edge["target_node_id"])
                for edge in edges
                if edge["relation"] in {"supports", "opposes", "inconclusive"}
                and str(edge["source_node_id"]) == focus_node_id
            )
        elif focus["kind"] == NodeKind.EXPERIMENT.value:
            related_hypotheses.update(
                str(edge["source_node_id"])
                for edge in edges
                if edge["relation"] == "tests"
                and str(edge["target_node_id"]) == focus_node_id
            )
        elif focus["kind"] == NodeKind.HYPOTHESIS.value:
            related_hypotheses.add(focus_node_id)

        # Include every directly related Result for the hypotheses relevant to
        # the focus. This exposes conflicts without turning the snippet into a
        # whole-graph projection.
        for edge in edges:
            if (
                edge["relation"] in {"supports", "opposes", "inconclusive"}
                and str(edge["target_node_id"]) in related_hypotheses
            ):
                selected.add(str(edge["source_node_id"]))
                selected.add(str(edge["target_node_id"]))
        return selected

    def build(
        self,
        graph_id: str,
        *,
        focus_node_id: str = "",
    ) -> dict[str, Any]:
        snapshot = self.store.get_snapshot(graph_id)
        graph = snapshot["graph"]
        all_nodes = list(snapshot["nodes"])
        all_edges = list(snapshot["edges"])
        all_refs = list(snapshot["refs"])
        by_id = {str(node["node_id"]): node for node in all_nodes}
        if focus_node_id and focus_node_id not in by_id:
            raise KeyError(f"Research focus node not found: {focus_node_id}")

        frontier = runnable_frontier_ids(all_nodes, all_edges)
        selected_ids = self._focus_neighborhood_ids(
            focus_node_id,
            nodes=all_nodes,
            edges=all_edges,
        )
        if not focus_node_id:
            selected_ids.update(
                str(node["node_id"])
                for node in all_nodes
                if node["kind"] == NodeKind.HYPOTHESIS.value
            )

        selected_nodes = [
            node for node in all_nodes if str(node["node_id"]) in selected_ids
        ]
        selected_edges = [
            edge
            for edge in all_edges
            if str(edge["source_node_id"]) in selected_ids
            and str(edge["target_node_id"]) in selected_ids
        ]
        selected_refs = [
            ref for ref in all_refs if str(ref["node_id"]) in selected_ids
        ]

        lines = [
            "# Active Research Graph: partial focus snippet",
            f"Graph: {graph['title']} ({graph['graph_id']}, revision {graph['revision']})",
            f"Question: {graph['question']}",
            f"Completion criterion: {graph['completion_criterion']}",
            f"Completion state: {'satisfied' if graph['completed'] else 'open'}",
            (
                "Counts: "
                f"{sum(node['kind'] == 'hypothesis' for node in all_nodes)} hypotheses, "
                f"{sum(node['kind'] == 'experiment' for node in all_nodes)} experiments, "
                f"{sum(node['kind'] == 'result' for node in all_nodes)} results."
            ),
            "",
            "This is an explicitly partial semantic focus snippet, not the full graph. "
            "Use query_research_graph_sql and the research-graph-query skill for any "
            "node, edge, ref, launch, planning, or referenced owner row not shown here.",
            "",
            "## Focus and directly related scientific nodes",
        ]
        if selected_nodes:
            for node in selected_nodes:
                lines.extend(["", *self._node_markdown(node)])
        else:
            lines.append("- No focus node is bound.")

        lines.extend(["", "## Direct typed relations"])
        if selected_edges:
            lines.extend(
                f"- {edge['source_node_id']} --{edge['relation']}--> {edge['target_node_id']}"
                for edge in selected_edges
            )
        else:
            lines.append("- No direct relation is present in this focus snippet.")

        lines.extend(["", "## Focus sources"])
        if selected_refs:
            lines.extend(
                f"- {ref['node_id']} -> {ref['ref_kind']}:{ref['ref_id']}"
                for ref in selected_refs
            )
        else:
            lines.append("- No source ref is attached to the shown nodes.")

        lines.extend(["", "## Complete runnable frontier"])
        if frontier:
            for node_id in frontier:
                node = by_id[node_id]
                objective = str(dict(node.get("body") or {}).get("objective") or "")
                lines.append(f"- {node['title']} ({node_id}): {objective}")
        else:
            lines.append("- No ready Experiment currently satisfies its dependencies.")

        public_graph = {
            key: graph[key]
            for key in (
                "graph_id",
                "title",
                "question",
                "completion_criterion",
                "completed",
                "orchestration_mode",
                "archived",
                "revision",
            )
        }
        return {
            "markdown": "\n".join(lines).strip(),
            "presentation": {
                "graph": public_graph,
                "partial": True,
                "focus_node_id": focus_node_id,
                "nodes": [self._public_node(node) for node in selected_nodes],
                "edges": [self._public_edge(edge) for edge in selected_edges],
                "refs": [
                    {
                        "node_id": str(ref["node_id"]),
                        "ref_kind": str(ref["ref_kind"]),
                        "ref_id": str(ref["ref_id"]),
                    }
                    for ref in selected_refs
                ],
                "frontier_node_ids": frontier,
                "shown_count": len(selected_nodes),
                "total_count": len(all_nodes),
            },
        }


__all__ = ["ResearchGraphContextBuilder", "runnable_frontier_ids"]
