from __future__ import annotations

import re
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any

from catmaster.storage import connect_workspace_db

from .models import NodeKind, RefKind
from .store import ResearchGraphStore

_TOKEN_RE = re.compile(r"[\w\u3400-\u9fff]{2,}", re.UNICODE)
_PRIORITY_RANK = {"": 0, "low": 1, "medium": 2, "high": 3}
_COMPUTE_COST_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "": 4}


def _node_search_text(node: dict[str, Any]) -> str:
    body = node.get("body") if isinstance(node.get("body"), dict) else {}
    values = [str(node.get("title") or "")]
    for key in ("claim", "rationale", "objective", "plan_summary", "decision_rule", "summary"):
        value = body.get(key)
        if value:
            values.append(str(value))
    predictions = body.get("predictions")
    if isinstance(predictions, list):
        values.extend(str(item) for item in predictions)
    return "\n".join(values)


def ranked_frontier_ids(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[str]:
    """Return runnable experiments in explicit value/cost order.

    The graph may retain many competing branches. This ordering only chooses
    which ready experiment an auto graph starts next; it does not prune nodes
    or claim a probabilistic utility score.
    """

    by_id = {str(node["node_id"]): node for node in nodes}
    dependencies: dict[str, list[str]] = {}
    tested_hypotheses: dict[str, list[str]] = {}
    for edge in edges:
        relation = str(edge["relation"])
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])
        if relation == "depends_on":
            dependencies.setdefault(source, []).append(target)
        elif relation == "tests":
            tested_hypotheses.setdefault(target, []).append(source)

    creation_order = {
        str(node["node_id"]): index for index, node in enumerate(nodes)
    }
    runnable: list[str] = []
    for node in nodes:
        node_id = str(node["node_id"])
        if node["kind"] != NodeKind.EXPERIMENT.value or node["state"] != "ready":
            continue
        if all(
            by_id.get(dependency, {}).get("state") == "has_results"
            for dependency in dependencies.get(node_id, [])
        ):
            runnable.append(node_id)

    def order_key(node_id: str) -> tuple[int, int, int, int, str]:
        node = by_id[node_id]
        body = node.get("body") if isinstance(node.get("body"), dict) else {}
        hypothesis_importance = max(
            (
                _PRIORITY_RANK.get(
                    str(
                        (
                            by_id.get(hypothesis_id, {}).get("body")
                            if isinstance(
                                by_id.get(hypothesis_id, {}).get("body"),
                                dict,
                            )
                            else {}
                        ).get("importance", "")
                    ),
                    0,
                )
                for hypothesis_id in tested_hypotheses.get(node_id, [])
            ),
            default=0,
        )
        expected_value = _PRIORITY_RANK.get(
            str(body.get("expected_value") or ""),
            0,
        )
        compute_cost = _COMPUTE_COST_RANK.get(
            str(body.get("estimated_compute_cost") or ""),
            4,
        )
        return (
            -hypothesis_importance,
            -expected_value,
            compute_cost,
            creation_order[node_id],
            node_id,
        )

    return sorted(runnable, key=order_key)


class ResearchGraphContextBuilder:
    """Build deterministic bounded graph context for one explicitly selected graph."""

    def __init__(self, *, workspace: Path | str, store: ResearchGraphStore | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.store = store or ResearchGraphStore(self.workspace)

    def _fts_matches(
        self,
        *,
        nodes: list[dict[str, Any]],
        query: str,
        limit: int,
        extra_text_by_node: dict[str, str] | None = None,
    ) -> list[str]:
        terms = list(dict.fromkeys(token.lower() for token in _TOKEN_RE.findall(query)))
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:16])
        try:
            with connect_workspace_db(self.workspace) as connection:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE temp.research_context_fts
                    USING fts5(node_id UNINDEXED, search_text)
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO temp.research_context_fts(node_id, search_text)
                    VALUES (?, ?)
                    """,
                    [
                        (
                            str(node["node_id"]),
                            "\n".join(
                                [
                                    _node_search_text(node),
                                    str(
                                        (extra_text_by_node or {}).get(
                                            str(node["node_id"]), ""
                                        )
                                    ),
                                ]
                            ),
                        )
                        for node in nodes
                    ],
                )
                rows = connection.execute(
                    """
                    SELECT node_id
                    FROM temp.research_context_fts
                    WHERE research_context_fts MATCH ?
                    ORDER BY bm25(research_context_fts), node_id
                    LIMIT ?
                    """,
                    (expression, max(1, int(limit))),
                ).fetchall()
            return [str(row["node_id"]) for row in rows]
        except sqlite3.Error:
            # Some minimal SQLite builds omit FTS5. Keep the fallback
            # deterministic and token-based rather than silently selecting the
            # whole graph.
            scored: list[tuple[int, str]] = []
            for node in nodes:
                text = "\n".join(
                    [
                        _node_search_text(node),
                        str(
                            (extra_text_by_node or {}).get(
                                str(node["node_id"]), ""
                            )
                        ),
                    ]
                ).lower()
                score = sum(text.count(term) for term in terms)
                if score:
                    scored.append((-score, str(node["node_id"])))
            scored.sort()
            return [node_id for _score, node_id in scored[:limit]]

    @staticmethod
    def _frontier(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[str]:
        return ranked_frontier_ids(nodes, edges)

    @staticmethod
    def _neighbors(
        edges: list[dict[str, Any]],
    ) -> dict[str, list[tuple[str, str]]]:
        adjacency: dict[str, list[tuple[str, str]]] = {}
        for edge in edges:
            source = str(edge["source_node_id"])
            target = str(edge["target_node_id"])
            relation = str(edge["relation"])
            adjacency.setdefault(source, []).append((target, relation))
            adjacency.setdefault(target, []).append((source, relation))
        for rows in adjacency.values():
            rows.sort(key=lambda item: (item[1], item[0]))
        return adjacency

    @staticmethod
    def _focus_path(
        focus_node_id: str,
        edges: list[dict[str, Any]],
        *,
        max_depth: int = 8,
    ) -> list[str]:
        if not focus_node_id:
            return []
        predecessors: dict[str, list[str]] = {}
        for edge in edges:
            predecessors.setdefault(str(edge["target_node_id"]), []).append(
                str(edge["source_node_id"])
            )
        for values in predecessors.values():
            values.sort()
        queue: deque[list[str]] = deque([[focus_node_id]])
        visited = {focus_node_id}
        best = [focus_node_id]
        while queue:
            path = queue.popleft()
            best = path
            current = path[-1]
            parents = [
                parent
                for parent in predecessors.get(current, [])
                if parent not in visited
            ]
            if not parents or len(path) >= max_depth:
                return list(reversed(path))
            for parent in parents:
                visited.add(parent)
                queue.append([*path, parent])
        return list(reversed(best))

    def _note_excerpt(self, ref_id: str, *, limit: int = 600) -> str:
        raw = str(ref_id or "").strip().replace("\\", "/").lstrip("/")
        if not raw:
            return ""
        candidate = self.workspace.joinpath(*Path(raw).parts).resolve()
        files_root = (self.workspace / "files").resolve()
        if not candidate.exists() and not raw.startswith("files/"):
            candidate = files_root.joinpath(*Path(raw).parts).resolve()
        try:
            candidate.relative_to(files_root)
        except ValueError:
            return ""
        if not candidate.is_file():
            return ""
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 1)].rstrip() + "…"

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        cap = max(1, int(limit))
        if len(text) <= cap:
            return text
        return text[: max(1, cap - 1)].rstrip() + "…"

    @classmethod
    def _node_line(
        cls,
        node: dict[str, Any],
        evidence_state: str = "",
        *,
        detail_limit: int = 480,
    ) -> str:
        body = node["body"]
        kind = node["kind"]
        if kind == "hypothesis":
            details = [str(body["claim"])]
            if str(body.get("importance") or "").strip():
                details.append(f"importance: {body['importance']}")
            detail = " | ".join(details)
        elif kind == "experiment":
            details = [str(body["objective"])]
            if str(body.get("decision_rule") or "").strip():
                details.append(f"decision: {body['decision_rule']}")
            details.append(f"state: {node['state']}")
            if (
                node["state"] == "blocked"
                and str(body.get("blocking_reason") or "").strip()
            ):
                details.append(f"blocked because: {body['blocking_reason']}")
            if str(body.get("expected_value") or "").strip():
                details.append(f"value: {body['expected_value']}")
            if str(body.get("estimated_compute_cost") or "").strip():
                details.append(f"compute: {body['estimated_compute_cost']}")
            detail = " | ".join(details)
        else:
            detail = body["summary"]
        suffix = f" | evidence: {evidence_state}" if evidence_state else ""
        title = cls._compact_text(node["title"], 120)
        detail = cls._compact_text(detail, detail_limit)
        return (
            f"- [{kind}] {title} ({node['node_id']}): "
            f"{detail}{suffix}"
        )

    @staticmethod
    def _evidence_states(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, str]:
        result_edges: dict[str, set[str]] = {}
        for edge in edges:
            relation = str(edge["relation"])
            if relation in {"supports", "opposes", "inconclusive"}:
                result_edges.setdefault(str(edge["target_node_id"]), set()).add(relation)
        states: dict[str, str] = {}
        for node in nodes:
            if node["kind"] != "hypothesis":
                continue
            relations = result_edges.get(str(node["node_id"]), set())
            if "supports" in relations and "opposes" in relations:
                states[str(node["node_id"])] = "conflicting evidence"
            elif "supports" in relations:
                states[str(node["node_id"])] = "supporting evidence available"
            elif "opposes" in relations:
                states[str(node["node_id"])] = "opposing evidence available"
            elif "inconclusive" in relations:
                states[str(node["node_id"])] = "not yet distinguished"
            else:
                states[str(node["node_id"])] = "no results yet"
        return states

    def build(
        self,
        graph_id: str,
        *,
        focus_node_id: str = "",
        query: str = "",
        max_nodes: int = 24,
        max_chars: int = 12_000,
        hops: int = 2,
        planning: bool = False,
    ) -> dict[str, Any]:
        snapshot = self.store.get_snapshot(graph_id)
        graph = snapshot["graph"]
        all_nodes = snapshot["nodes"]
        edges = snapshot["edges"]
        refs = snapshot["refs"]
        node_by_id = {str(node["node_id"]): node for node in all_nodes}
        if focus_node_id and focus_node_id not in node_by_id:
            raise KeyError(f"Research focus node not found: {focus_node_id}")
        max_nodes = min(100, max(4, int(max_nodes)))
        max_chars = min(40_000, max(2_000, int(max_chars)))
        hops = min(2, max(1, int(hops)))

        frontier = self._frontier(all_nodes, edges)
        note_text_by_node: dict[str, str] = {}
        for ref in refs:
            if ref["ref_kind"] != RefKind.NOTE.value:
                continue
            excerpt = self._note_excerpt(str(ref["ref_id"]))
            if excerpt:
                node_id = str(ref["node_id"])
                note_text_by_node[node_id] = "\n".join(
                    filter(
                        None,
                        [note_text_by_node.get(node_id, ""), excerpt],
                    )
                )
        matches = self._fts_matches(
            nodes=all_nodes,
            query=query,
            limit=max_nodes,
            extra_text_by_node=note_text_by_node,
        )
        adjacency = self._neighbors(edges)
        seeds: list[str] = []
        for node_id in [focus_node_id, *matches, *frontier]:
            if node_id and node_id not in seeds:
                seeds.append(node_id)

        # With no focus, text match, or runnable work, show the earliest
        # hypotheses rather than guessing a semantic neighborhood.
        if not seeds:
            seeds.extend(
                str(node["node_id"])
                for node in all_nodes
                if node["kind"] == "hypothesis"
            )

        distance: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for seed in seeds:
            if seed in node_by_id and seed not in distance:
                distance[seed] = 0
                queue.append((seed, 0))
        while queue:
            node_id, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor, _relation in adjacency.get(node_id, []):
                if neighbor not in distance:
                    distance[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        tests_by_experiment: dict[str, list[str]] = {}
        producer_by_result: dict[str, str] = {}
        evidence_by_hypothesis: dict[str, dict[str, list[str]]] = {}
        for edge in edges:
            source = str(edge["source_node_id"])
            target = str(edge["target_node_id"])
            relation = str(edge["relation"])
            if relation == "tests":
                tests_by_experiment.setdefault(target, []).append(source)
            elif relation == "produces":
                producer_by_result[target] = source
            elif relation in {"supports", "opposes", "inconclusive"}:
                evidence_by_hypothesis.setdefault(target, {}).setdefault(
                    relation, []
                ).append(source)
        for rows in tests_by_experiment.values():
            rows.sort()
        for relation_rows in evidence_by_hypothesis.values():
            for rows in relation_rows.values():
                rows.sort()

        def related_hypotheses(node_id: str) -> list[str]:
            node = node_by_id.get(node_id)
            if node is None:
                return []
            if node["kind"] == "hypothesis":
                return [node_id]
            if node["kind"] == "experiment":
                return list(tests_by_experiment.get(node_id, []))
            hypotheses: list[str] = []
            for edge in edges:
                if (
                    str(edge["source_node_id"]) == node_id
                    and str(edge["relation"])
                    in {"supports", "opposes", "inconclusive"}
                ):
                    target = str(edge["target_node_id"])
                    if target not in hypotheses:
                        hypotheses.append(target)
            producer = producer_by_result.get(node_id, "")
            for hypothesis_id in tests_by_experiment.get(producer, []):
                if hypothesis_id not in hypotheses:
                    hypotheses.append(hypothesis_id)
            return hypotheses

        priority: list[str] = []

        def add_priority(node_id: str) -> None:
            if node_id and node_id in node_by_id and node_id not in priority:
                priority.append(node_id)

        def add_evidence_cluster(node_ids: list[str]) -> None:
            hypotheses: list[str] = []
            for node_id in node_ids:
                for hypothesis_id in related_hypotheses(node_id):
                    if hypothesis_id not in hypotheses:
                        hypotheses.append(hypothesis_id)
            for hypothesis_id in hypotheses:
                add_priority(hypothesis_id)
                relation_rows = evidence_by_hypothesis.get(
                    hypothesis_id, {}
                )
                # One supporting and one opposing result are reserved before
                # extra evidence so a large frontier cannot hide a conflict.
                for relation in ("supports", "opposes", "inconclusive"):
                    rows = relation_rows.get(relation, [])
                    if rows:
                        add_priority(rows[0])
                for relation in ("supports", "opposes", "inconclusive"):
                    for result_id in relation_rows.get(relation, [])[1:]:
                        add_priority(result_id)

        critical_seeds: list[str] = []
        if focus_node_id:
            critical_seeds.append(focus_node_id)
        if matches and matches[0] not in critical_seeds:
            critical_seeds.append(matches[0])
        for node_id in critical_seeds:
            add_priority(node_id)
        add_evidence_cluster(critical_seeds)

        # Remaining query hits and their evidence still precede all runnable
        # frontier nodes. FTS rank therefore cannot be displaced merely
        # because the graph contains many ready experiments.
        for node_id in matches:
            add_priority(node_id)
            add_evidence_cluster([node_id])
        for node_id in frontier:
            add_priority(node_id)
        if planning:
            # Scientific planning needs the whole concise hypothesis landscape,
            # not only the focused neighborhood. Runnable experiments stay
            # first so a large set of dormant hypotheses cannot hide work that
            # can actually run. Detailed notes and logs remain behind refs.
            for node in all_nodes:
                if node["kind"] == NodeKind.HYPOTHESIS.value:
                    add_priority(str(node["node_id"]))
        for node_id, _depth in sorted(distance.items(), key=lambda item: (item[1], item[0])):
            add_priority(node_id)
        if not priority:
            for node in all_nodes:
                if node["kind"] == "hypothesis":
                    add_priority(str(node["node_id"]))

        evidence_states = self._evidence_states(all_nodes, edges)
        focus_path = self._focus_path(focus_node_id, edges)
        prefix_lines = [
            "# Active Research Graph",
            (
                f"Graph: {self._compact_text(graph['title'], 160)} "
                f"({graph['graph_id']}, revision {graph['revision']})"
            ),
            f"Question: {self._compact_text(graph['question'], 360)}",
            (
                "Completion criterion: "
                f"{self._compact_text(graph['completion_criterion'], 360)}"
            ),
            f"Completion state: {'satisfied' if graph['completed'] else 'open'}",
        ]
        if focus_node_id:
            focus = node_by_id[focus_node_id]
            prefix_lines.append(
                f"Focus: {self._compact_text(focus['title'], 120)} "
                f"({focus_node_id})"
            )
            focus_refs = [
                ref for ref in refs if str(ref["node_id"]) == focus_node_id
            ]
            if focus_refs:
                source_budget = max(320, min(1_600, max_chars // 5))
                focus_source_lines = ["", "## Focus sources"]
                shown_sources = 0
                for ref in focus_refs:
                    label_limit = min(
                        360,
                        max(
                            80,
                            source_budget
                            - len("\n".join(focus_source_lines))
                            - 4,
                        ),
                    )
                    label = self._compact_text(
                        f"{ref['ref_kind']}:{ref['ref_id']}",
                        label_limit,
                    )
                    candidate = f"- {label}"
                    if len("\n".join([*focus_source_lines, candidate])) > source_budget:
                        break
                    focus_source_lines.append(candidate)
                    shown_sources += 1
                    if ref["ref_kind"] == RefKind.NOTE.value:
                        excerpt = self._compact_text(
                            note_text_by_node.get(focus_node_id, ""),
                            240,
                        )
                        excerpt_line = f"  Note excerpt: {excerpt}"
                        if excerpt and len(
                            "\n".join([*focus_source_lines, excerpt_line])
                        ) <= source_budget:
                            focus_source_lines.append(excerpt_line)
                if shown_sources:
                    omitted_sources = len(focus_refs) - shown_sources
                    omitted_line = f"- {omitted_sources} more focus source(s) omitted."
                    if omitted_sources and len(
                        "\n".join([*focus_source_lines, omitted_line])
                    ) <= source_budget:
                        focus_source_lines.append(omitted_line)
                    prefix_lines.extend(focus_source_lines)
        prefix_lines.extend(["", "## Focus path"])
        if focus_path:
            path_text = "- Research question -> " + " -> ".join(
                f"{self._compact_text(node_by_id[node_id]['title'], 80)} "
                f"({node_id})"
                for node_id in focus_path
                if node_id in node_by_id
            )
            prefix_lines.append(self._compact_text(path_text, 420))
        elif frontier:
            path_text = "- Research question -> runnable frontier: " + ", ".join(
                f"{self._compact_text(node_by_id[node_id]['title'], 80)} "
                f"({node_id})"
                for node_id in frontier[:3]
                if node_id in node_by_id
            )
            prefix_lines.append(self._compact_text(path_text, 420))
        else:
            prefix_lines.append("- Research question -> no runnable experiment.")
        prefix_lines.extend(["", "## Relevant scientific nodes"])

        frontier_lines = ["", "## Runnable frontier"]
        if frontier:
            frontier_lines.extend(
                f"- {self._compact_text(node_by_id[node_id]['title'], 100)} "
                f"({node_id})"
                for node_id in frontier[:3]
                if node_id in node_by_id
            )
            if len(frontier) > 3:
                frontier_lines.append(
                    f"- {len(frontier) - 3} more runnable experiment(s) are "
                    "outside this bounded frontier preview."
                )
        else:
            frontier_lines.append("- No ready experiment is currently runnable.")

        target_count = max(1, min(max_nodes, len(priority) or 1))
        detail_limit = max(
            80,
            min(480, int(max_chars / target_count) - 180),
        )
        selected_ids: list[str] = []
        node_lines: list[str] = []

        def footer_for(shown_count: int) -> str:
            omitted = max(0, len(all_nodes) - shown_count)
            return (
                f"Bounded view: {shown_count} of {len(all_nodes)} nodes; "
                f"{omitted} omitted. Use inspect_research_graph with this graph "
                "ID and a focus node or query to inspect another neighborhood."
            )

        def mandatory_markdown(
            candidate_lines: list[str],
            *,
            shown_count: int,
        ) -> str:
            return "\n".join(
                [
                    *prefix_lines,
                    *candidate_lines,
                    *frontier_lines,
                    "",
                    footer_for(shown_count),
                ]
            ).strip()

        for node_id in priority[:max_nodes]:
            line = self._node_line(
                node_by_id[node_id],
                evidence_states.get(node_id, ""),
                detail_limit=detail_limit,
            )
            trial_lines = [*node_lines, line]
            if len(
                mandatory_markdown(
                    trial_lines,
                    shown_count=len(selected_ids) + 1,
                )
            ) > max_chars:
                break
            selected_ids.append(node_id)
            node_lines.append(line)

        # The minimum character budget always has room for one compact node in
        # ordinary graphs. Keep a defensive compact fallback for unusually
        # long IDs/titles so count metadata can still describe visible content.
        if priority and not selected_ids:
            node_id = priority[0]
            line = self._node_line(
                node_by_id[node_id],
                evidence_states.get(node_id, ""),
                detail_limit=80,
            )
            base_without_node = mandatory_markdown([], shown_count=0)
            available = max(
                80,
                max_chars - len(base_without_node) - 4,
            )
            compact_line = self._compact_text(line, available)
            if len(
                mandatory_markdown([compact_line], shown_count=1)
            ) <= max_chars:
                selected_ids.append(node_id)
                node_lines.append(compact_line)

        selected_set = set(selected_ids)
        selected_nodes = [node_by_id[node_id] for node_id in selected_ids]
        selected_edges = [
            edge
            for edge in edges
            if edge["source_node_id"] in selected_set
            and edge["target_node_id"] in selected_set
        ]
        selected_refs = [ref for ref in refs if ref["node_id"] in selected_set]
        omitted_count = max(0, len(all_nodes) - len(selected_nodes))

        optional_lines: list[str] = []

        def with_optional(candidate: list[str]) -> str:
            return "\n".join(
                [
                    *prefix_lines,
                    *node_lines,
                    *candidate,
                    *frontier_lines,
                    "",
                    footer_for(len(selected_nodes)),
                ]
            ).strip()

        optional_refs = [
            ref for ref in selected_refs
            if str(ref["node_id"]) != focus_node_id
        ]
        if optional_refs:
            source_lines: list[str] = []
            for ref in optional_refs:
                label = self._compact_text(
                    f"{ref['ref_kind']}:{ref['ref_id']}",
                    260,
                )
                line = f"- {ref['node_id']} -> {label}"
                candidate = ["", "## Sources", *source_lines, line]
                if len(with_optional([*optional_lines, *candidate])) > max_chars:
                    break
                source_lines.append(line)
                if ref["ref_kind"] == RefKind.NOTE.value:
                    excerpt = self._compact_text(
                        note_text_by_node.get(str(ref["node_id"]), ""),
                        240,
                    )
                    excerpt_line = f"  Note excerpt: {excerpt}"
                    if excerpt and len(
                        with_optional(
                            [
                                *optional_lines,
                                "",
                                "## Sources",
                                *source_lines,
                                excerpt_line,
                            ]
                        )
                    ) <= max_chars:
                        source_lines.append(excerpt_line)
            if source_lines:
                optional_lines.extend(["", "## Sources", *source_lines])

        relation_candidates = (
            [
                f"- {edge['source_node_id']} --{edge['relation']}--> "
                f"{edge['target_node_id']}"
                for edge in selected_edges
            ]
            if selected_edges
            else ["- No relation is present in this bounded view."]
        )
        relation_section = ["", "## Typed relations"]
        if len(with_optional([*optional_lines, *relation_section])) <= max_chars:
            relation_lines: list[str] = []
            for line in relation_candidates:
                if len(
                    with_optional(
                        [
                            *optional_lines,
                            *relation_section,
                            *relation_lines,
                            line,
                        ]
                    )
                ) > max_chars:
                    break
                relation_lines.append(line)
            if relation_lines:
                optional_lines.extend([*relation_section, *relation_lines])

        markdown = with_optional(optional_lines)
        if len(markdown) > max_chars:
            # All node counts above are based on complete visible node lines.
            # This defensive branch can only trim optional tail material.
            markdown = mandatory_markdown(
                node_lines,
                shown_count=len(selected_nodes),
            )

        return {
            "markdown": markdown,
            "presentation": {
                "graph": {
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
                },
                "focus_node_id": focus_node_id,
                "nodes": [
                    {
                        "node_id": str(node["node_id"]),
                        "kind": str(node["kind"]),
                        "title": str(node["title"]),
                        "state": str(node.get("state") or ""),
                        "body": dict(node.get("body") or {}),
                        "revision": int(node["revision"]),
                    }
                    for node in selected_nodes
                ],
                "edges": [
                    {
                        "source_node_id": str(edge["source_node_id"]),
                        "target_node_id": str(edge["target_node_id"]),
                        "relation": str(edge["relation"]),
                    }
                    for edge in selected_edges
                ],
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
                "omitted_count": omitted_count,
                "inspect_hint": (
                    "Choose a node or search phrase to inspect another "
                    "one-to-two-hop neighborhood."
                ),
            },
        }


__all__ = ["ResearchGraphContextBuilder", "ranked_frontier_ids"]
