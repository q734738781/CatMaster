from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from catmaster.storage import connect_workspace_db, ensure_workspace_ui_events

from .models import (
    EdgeRelation,
    ExperimentState,
    NodeKind,
    OrchestrationMode,
    RefKind,
    validate_node_body,
)

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{3,160}$")
_ACTIVE_LAUNCH_STATUSES = ("claimed", "submitting", "running", "unknown")
_TERMINAL_PLANNING_STATUSES = ("finished", "no_change", "stale")
_PLANNING_LEASE_SECONDS = 120
_SCHEMA_COMPONENT = "research_knowledge_graph"
_SCHEMA_VERSION = 2


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _safe_id(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"Invalid {label}: {value!r}")
    return text


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ResearchGraphConflict(ValueError):
    def __init__(self, *, expected_revision: int, current_revision: int) -> None:
        self.expected_revision = int(expected_revision)
        self.current_revision = int(current_revision)
        super().__init__(
            "This research graph changed in another thread. "
            f"Refresh and retry your edit (expected revision {expected_revision}, "
            f"current revision {current_revision})."
        )


class ResearchGraphStore:
    """Transactional persistence for one workspace's research graphs."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self._init_schema()

    def _init_schema(self) -> None:
        with connect_workspace_db(self.workspace) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_graphs (
                    graph_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    orchestration_mode TEXT NOT NULL
                        CHECK (orchestration_mode IN ('manual', 'auto')),
                    archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
                    revision INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_nodes (
                    graph_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    kind TEXT NOT NULL
                        CHECK (kind IN ('hypothesis', 'experiment', 'result')),
                    title TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT '',
                    body_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (graph_id, node_id),
                    FOREIGN KEY (graph_id) REFERENCES research_graphs(graph_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_edges (
                    graph_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relation TEXT NOT NULL CHECK (
                        relation IN (
                            'tests', 'produces', 'supports', 'opposes',
                            'inconclusive', 'suggests', 'depends_on'
                        )
                    ),
                    PRIMARY KEY (
                        graph_id, source_node_id, relation, target_node_id
                    ),
                    FOREIGN KEY (graph_id, source_node_id)
                        REFERENCES research_nodes(graph_id, node_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (graph_id, target_node_id)
                        REFERENCES research_nodes(graph_id, node_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_refs (
                    graph_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    ref_kind TEXT NOT NULL CHECK (
                        ref_kind IN (
                            'thread', 'message', 'artifact', 'run',
                            'note', 'doi', 'url'
                        )
                    ),
                    ref_id TEXT NOT NULL,
                    PRIMARY KEY (graph_id, node_id, ref_kind, ref_id),
                    FOREIGN KEY (graph_id, node_id)
                        REFERENCES research_nodes(graph_id, node_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_launches (
                    launch_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    experiment_node_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'claimed', 'submitting', 'running', 'completed',
                            'blocked', 'unknown'
                        )
                    ),
                    thread_id TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_until REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (graph_id, experiment_node_id)
                        REFERENCES research_nodes(graph_id, node_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_planning (
                    planning_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    start_revision INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'claimed', 'attached', 'finished',
                            'no_change', 'stale'
                        )
                    ),
                    thread_id TEXT NOT NULL DEFAULT '',
                    lease_until REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (graph_id) REFERENCES research_graphs(graph_id)
                        ON DELETE CASCADE
                );

                CREATE UNIQUE INDEX IF NOT EXISTS research_launch_one_active
                    ON research_launches(graph_id, experiment_node_id)
                    WHERE status IN ('claimed', 'submitting', 'running', 'unknown');
                CREATE UNIQUE INDEX IF NOT EXISTS research_planning_one_active
                    ON research_planning(graph_id)
                    WHERE status IN ('claimed', 'attached');

                CREATE INDEX IF NOT EXISTS research_nodes_graph_kind
                    ON research_nodes(graph_id, kind);
                CREATE INDEX IF NOT EXISTS research_edges_graph_target
                    ON research_edges(graph_id, target_node_id, relation);
                CREATE INDEX IF NOT EXISTS research_launches_graph_status
                    ON research_launches(graph_id, status);
                CREATE INDEX IF NOT EXISTS research_planning_graph_status
                    ON research_planning(graph_id, status, updated_at);

                """
            )
            ensure_workspace_ui_events(connection)
            connection.execute(
                """
                INSERT INTO schema_migrations(component, version)
                VALUES (?, ?)
                ON CONFLICT(component) DO UPDATE SET version=excluded.version
                """,
                (_SCHEMA_COMPONENT, _SCHEMA_VERSION),
            )

    @staticmethod
    def _graph_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "graph_id": str(row["graph_id"]),
            "title": str(row["title"]),
            "question": str(row["question"]),
            "orchestration_mode": str(row["orchestration_mode"]),
            "archived": bool(row["archived"]),
            "revision": int(row["revision"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _node_row(row: sqlite3.Row) -> dict[str, Any]:
        kind = NodeKind(str(row["kind"]))
        body = validate_node_body(kind, json.loads(str(row["body_json"])))
        return {
            "graph_id": str(row["graph_id"]),
            "node_id": str(row["node_id"]),
            "kind": kind.value,
            "title": str(row["title"]),
            "state": str(row["state"]),
            "body": body,
            "revision": int(row["revision"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _edge_row(row: sqlite3.Row) -> dict[str, str]:
        return {
            "graph_id": str(row["graph_id"]),
            "source_node_id": str(row["source_node_id"]),
            "target_node_id": str(row["target_node_id"]),
            "relation": str(row["relation"]),
        }

    @staticmethod
    def _ref_row(row: sqlite3.Row) -> dict[str, str]:
        return {
            "graph_id": str(row["graph_id"]),
            "node_id": str(row["node_id"]),
            "ref_kind": str(row["ref_kind"]),
            "ref_id": str(row["ref_id"]),
        }

    @staticmethod
    def _launch_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "launch_id": str(row["launch_id"]),
            "graph_id": str(row["graph_id"]),
            "experiment_node_id": str(row["experiment_node_id"]),
            "idempotency_key": str(row["idempotency_key"]),
            "status": str(row["status"]),
            "thread_id": str(row["thread_id"]),
            "run_id": str(row["run_id"]),
            "lease_owner": str(row["lease_owner"]),
            "lease_until": float(row["lease_until"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _planning_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "planning_id": str(row["planning_id"]),
            "graph_id": str(row["graph_id"]),
            "revision": int(row["start_revision"]),
            "thread_id": str(row["thread_id"]),
            "status": str(row["status"]),
            "lease_until": float(row["lease_until"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def _get_graph_row(
        self,
        connection: sqlite3.Connection,
        graph_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM research_graphs WHERE graph_id = ?",
            (_safe_id(graph_id, label="graph_id"),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Research graph not found: {graph_id}")
        return row

    def _get_node_row(
        self,
        connection: sqlite3.Connection,
        graph_id: str,
        node_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT * FROM research_nodes
            WHERE graph_id = ? AND node_id = ?
            """,
            (
                _safe_id(graph_id, label="graph_id"),
                _safe_id(node_id, label="node_id"),
            ),
        ).fetchone()
        if row is None:
            raise KeyError(f"Research node not found: {node_id}")
        return row

    def _bump_graph(
        self,
        connection: sqlite3.Connection,
        *,
        graph_id: str,
        expected_revision: int,
    ) -> int:
        now = _now()
        cursor = connection.execute(
            """
            UPDATE research_graphs
            SET revision = revision + 1, updated_at = ?
            WHERE graph_id = ? AND revision = ?
            """,
            (now, graph_id, int(expected_revision)),
        )
        if int(cursor.rowcount or 0) == 1:
            return int(expected_revision) + 1
        current = self._get_graph_row(connection, graph_id)
        raise ResearchGraphConflict(
            expected_revision=int(expected_revision),
            current_revision=int(current["revision"]),
        )

    @staticmethod
    def _prune_events(
        connection: sqlite3.Connection,
        *,
        graph_id: str,
        newest_event_id: int,
    ) -> None:
        # ui_events is shared with thread streams, so global event-id gaps say
        # nothing about how much replay history this graph retains. Keep the
        # newest 5,000 rows for this graph, regardless of unrelated traffic.
        _ = newest_event_id
        connection.execute(
            """
            DELETE FROM ui_events
            WHERE graph_id = ?
              AND event_id IN (
                  SELECT event_id
                  FROM ui_events
                  WHERE graph_id = ?
                  ORDER BY event_id DESC
                  LIMIT -1 OFFSET 5000
              )
            """,
            (graph_id, graph_id),
        )

    @classmethod
    def _write_event(
        cls,
        connection: sqlite3.Connection,
        *,
        graph_id: str,
        revision: int,
        change: str,
        thread_id: str = "",
        node_ids: Iterable[str] = (),
        launch_id: str = "",
    ) -> int:
        payload = {
            "graph_id": graph_id,
            "revision": int(revision),
            "change": str(change),
            "node_ids": [str(item) for item in node_ids if str(item)],
        }
        if launch_id:
            payload["launch_id"] = str(launch_id)
        cursor = connection.execute(
            """
            INSERT INTO ui_events (
                event_type, thread_id, graph_id, payload_json, created_at
            ) VALUES ('research_graph.updated', ?, ?, ?, ?)
            """,
            (str(thread_id or ""), graph_id, _json(payload), _now()),
        )
        event_id = int(cursor.lastrowid)
        # The outbox is a bounded delivery aid, not an event-sourced copy of
        # the graph. Keep enough history for reconnects while preventing an
        # indefinitely growing audit log.
        cls._prune_events(
            connection,
            graph_id=graph_id,
            newest_event_id=event_id,
        )
        return event_id

    @classmethod
    def _write_planning_event(
        cls,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        planning: dict[str, Any],
        recovered: bool = False,
    ) -> int:
        """Publish a disposable UI notice for durable planning state."""

        payload = {
            "graph_id": str(planning["graph_id"]),
            "planning_id": str(planning["planning_id"]),
            "revision": int(planning["revision"]),
        }
        if recovered:
            payload["recovered"] = True
        cursor = connection.execute(
            """
            INSERT INTO ui_events (
                event_type, thread_id, graph_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(event_type),
                str(planning.get("thread_id") or ""),
                str(planning["graph_id"]),
                _json(payload),
                _now(),
            ),
        )
        event_id = int(cursor.lastrowid)
        cls._prune_events(
            connection,
            graph_id=str(planning["graph_id"]),
            newest_event_id=event_id,
        )
        return event_id

    def create_graph(
        self,
        *,
        title: str,
        question: str,
        orchestration_mode: OrchestrationMode | str = OrchestrationMode.MANUAL,
        initial_hypotheses: list[dict[str, Any]] | None = None,
        graph_id: str = "",
    ) -> dict[str, Any]:
        graph_id = _safe_id(graph_id, label="graph_id") if graph_id else _new_id("graph")
        question = str(question or "").strip()
        title = str(title or "").strip() or question[:120]
        if not question:
            raise ValueError("Research question is required.")
        mode = OrchestrationMode(orchestration_mode).value
        hypotheses = list(initial_hypotheses or [])
        now = _now()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO research_graphs (
                    graph_id, title, question, orchestration_mode,
                    archived, revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, 1, ?, ?)
                """,
                (graph_id, title, question, mode, now, now),
            )
            created_ids: list[str] = []
            for seed in hypotheses:
                body = validate_node_body(
                    NodeKind.HYPOTHESIS,
                    {
                        "claim": seed.get("claim"),
                        "rationale": seed.get("rationale", ""),
                        "predictions": seed.get("predictions", []),
                        "importance": seed.get("importance", "medium"),
                    },
                )
                node_id = _new_id("hyp")
                node_title = str(seed.get("title") or body["claim"][:120]).strip()
                connection.execute(
                    """
                    INSERT INTO research_nodes (
                        graph_id, node_id, kind, title, state, body_json,
                        revision, created_at, updated_at
                    ) VALUES (?, ?, 'hypothesis', ?, '', ?, 1, ?, ?)
                    """,
                    (graph_id, node_id, node_title, _json(body), now, now),
                )
                created_ids.append(node_id)
            event_id = self._write_event(
                connection,
                graph_id=graph_id,
                revision=1,
                change="graph.created",
                node_ids=created_ids,
            )
        graph = self.get_graph(graph_id)
        graph["event_id"] = event_id
        return graph

    def list_graphs(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived = 0"
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM research_graphs
                {where}
                ORDER BY archived ASC, updated_at DESC, graph_id ASC
                """
            ).fetchall()
        return [self._graph_row(row) for row in rows]

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        with connect_workspace_db(self.workspace) as connection:
            return self._graph_row(self._get_graph_row(connection, graph_id))

    def get_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        with connect_workspace_db(self.workspace) as connection:
            return self._node_row(self._get_node_row(connection, graph_id, node_id))

    def get_snapshot(self, graph_id: str) -> dict[str, Any]:
        graph_id = _safe_id(graph_id, label="graph_id")
        with connect_workspace_db(self.workspace) as connection:
            graph = self._graph_row(self._get_graph_row(connection, graph_id))
            nodes = [
                self._node_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM research_nodes
                    WHERE graph_id = ?
                    ORDER BY created_at ASC, node_id ASC
                    """,
                    (graph_id,),
                ).fetchall()
            ]
            edges = [
                self._edge_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM research_edges
                    WHERE graph_id = ?
                    ORDER BY source_node_id, relation, target_node_id
                    """,
                    (graph_id,),
                ).fetchall()
            ]
            refs = [
                self._ref_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM research_refs
                    WHERE graph_id = ?
                    ORDER BY node_id, ref_kind, ref_id
                    """,
                    (graph_id,),
                ).fetchall()
            ]
            launches = [
                self._launch_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM research_launches
                    WHERE graph_id = ?
                    ORDER BY created_at DESC, launch_id DESC
                    """,
                    (graph_id,),
                ).fetchall()
            ]
        return {
            "graph": graph,
            "nodes": nodes,
            "edges": edges,
            "refs": refs,
            "launches": launches,
        }

    def update_graph(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        graph_id = _safe_id(graph_id, label="graph_id")
        allowed = {"title", "question", "orchestration_mode", "archived"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        if not updates:
            return self.get_graph(graph_id)
        if "title" in updates:
            updates["title"] = str(updates["title"] or "").strip()
            if not updates["title"]:
                raise ValueError("Graph title is required.")
        if "question" in updates:
            updates["question"] = str(updates["question"] or "").strip()
            if not updates["question"]:
                raise ValueError("Research question is required.")
        if "orchestration_mode" in updates:
            updates["orchestration_mode"] = OrchestrationMode(
                str(updates["orchestration_mode"])
            ).value
        if "archived" in updates:
            updates["archived"] = int(bool(updates["archived"]))
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            assignments = ", ".join(f"{key} = ?" for key in updates)
            connection.execute(
                f"UPDATE research_graphs SET {assignments} WHERE graph_id = ?",
                (*updates.values(), graph_id),
            )
            self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="graph.updated",
            )
        return self.get_graph(graph_id)

    @staticmethod
    def _validate_node_state(kind: NodeKind, state: str) -> str:
        value = str(state or "").strip()
        if kind is NodeKind.EXPERIMENT:
            return ExperimentState(value or ExperimentState.DRAFT.value).value
        if value:
            raise ValueError(f"{kind.value} nodes do not have an execution state.")
        return ""

    @staticmethod
    def _expected_relation_shape(
        relation: EdgeRelation,
    ) -> tuple[NodeKind, NodeKind]:
        return {
            EdgeRelation.TESTS: (NodeKind.HYPOTHESIS, NodeKind.EXPERIMENT),
            EdgeRelation.PRODUCES: (NodeKind.EXPERIMENT, NodeKind.RESULT),
            EdgeRelation.SUPPORTS: (NodeKind.RESULT, NodeKind.HYPOTHESIS),
            EdgeRelation.OPPOSES: (NodeKind.RESULT, NodeKind.HYPOTHESIS),
            EdgeRelation.INCONCLUSIVE: (NodeKind.RESULT, NodeKind.HYPOTHESIS),
            EdgeRelation.SUGGESTS: (NodeKind.RESULT, NodeKind.HYPOTHESIS),
            EdgeRelation.DEPENDS_ON: (NodeKind.EXPERIMENT, NodeKind.EXPERIMENT),
        }[relation]

    def _validate_edge(
        self,
        connection: sqlite3.Connection,
        *,
        graph_id: str,
        source_node_id: str,
        target_node_id: str,
        relation: EdgeRelation,
    ) -> None:
        source = self._get_node_row(connection, graph_id, source_node_id)
        target = self._get_node_row(connection, graph_id, target_node_id)
        expected_source, expected_target = self._expected_relation_shape(relation)
        actual = (NodeKind(str(source["kind"])), NodeKind(str(target["kind"])))
        if actual != (expected_source, expected_target):
            raise ValueError(
                f"{relation.value} requires {expected_source.value} -> "
                f"{expected_target.value}, got {actual[0].value} -> {actual[1].value}."
            )
        if relation is EdgeRelation.DEPENDS_ON:
            if source_node_id == target_node_id:
                raise ValueError("An experiment cannot depend on itself.")
            cycle = connection.execute(
                """
                WITH RECURSIVE reachable(node_id) AS (
                    SELECT target_node_id
                    FROM research_edges
                    WHERE graph_id = ?
                      AND source_node_id = ?
                      AND relation = 'depends_on'
                    UNION
                    SELECT edge.target_node_id
                    FROM research_edges AS edge
                    JOIN reachable ON edge.source_node_id = reachable.node_id
                    WHERE edge.graph_id = ?
                      AND edge.relation = 'depends_on'
                )
                SELECT 1 FROM reachable WHERE node_id = ? LIMIT 1
                """,
                (graph_id, target_node_id, graph_id, source_node_id),
            ).fetchone()
            if cycle is not None:
                raise ValueError(
                    "This dependency would create a cycle. Experiment dependencies "
                    "must remain acyclic."
                )

    def add_node_bundle(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        kind: NodeKind | str,
        title: str,
        body: dict[str, Any],
        state: str = "",
        edges: list[dict[str, str]] | None = None,
        refs: list[dict[str, str]] | None = None,
        node_id: str = "",
        change: str = "node.added",
    ) -> tuple[dict[str, Any], int]:
        graph_id = _safe_id(graph_id, label="graph_id")
        node_id = _safe_id(node_id, label="node_id") if node_id else _new_id(
            {
                NodeKind.HYPOTHESIS: "hyp",
                NodeKind.EXPERIMENT: "exp",
                NodeKind.RESULT: "res",
            }[NodeKind(kind)]
        )
        node_kind = NodeKind(kind)
        node_title = str(title or "").strip()
        if not node_title:
            raise ValueError("Node title is required.")
        node_body = validate_node_body(node_kind, body)
        node_state = self._validate_node_state(node_kind, state)
        edge_rows = list(edges or [])
        ref_rows = list(refs or [])
        now = _now()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._get_graph_row(connection, graph_id)
            connection.execute(
                """
                INSERT INTO research_nodes (
                    graph_id, node_id, kind, title, state, body_json,
                    revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    graph_id,
                    node_id,
                    node_kind.value,
                    node_title,
                    node_state,
                    _json(node_body),
                    now,
                    now,
                ),
            )
            for edge in edge_rows:
                relation = EdgeRelation(edge["relation"])
                source_id = _safe_id(
                    str(edge["source_node_id"]), label="source_node_id"
                )
                target_id = _safe_id(
                    str(edge["target_node_id"]), label="target_node_id"
                )
                self._validate_edge(
                    connection,
                    graph_id=graph_id,
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation=relation,
                )
                connection.execute(
                    """
                    INSERT INTO research_edges (
                        graph_id, source_node_id, target_node_id, relation
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (graph_id, source_id, target_id, relation.value),
                )
            for ref in ref_rows:
                connection.execute(
                    """
                    INSERT INTO research_refs (
                        graph_id, node_id, ref_kind, ref_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        graph_id,
                        node_id,
                        RefKind(ref["ref_kind"]).value,
                        str(ref["ref_id"]).strip(),
                    ),
                )
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            event_id = self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change=change,
                node_ids=[node_id],
            )
        return self.get_node(graph_id, node_id), event_id

    def update_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        expected_revision: int,
        expected_node_revision: int,
        title: str,
        state: str,
        body: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        graph_id = _safe_id(graph_id, label="graph_id")
        node_id = _safe_id(node_id, label="node_id")
        title = str(title or "").strip()
        if not title:
            raise ValueError("Node title is required.")
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_node_row(connection, graph_id, node_id)
            kind = NodeKind(str(current["kind"]))
            validated_body = validate_node_body(kind, body)
            validated_state = self._validate_node_state(kind, state)
            if kind is NodeKind.EXPERIMENT:
                current_state = str(current["state"])
                editable_states = {
                    "draft": {"draft", "ready"},
                    "ready": {"draft", "ready"},
                    "blocked": {"blocked", "draft", "ready"},
                    "running": {"running"},
                    "has_results": {"has_results"},
                }[current_state]
                if validated_state not in editable_states:
                    raise ValueError(
                        f"Experiment state cannot change from {current_state} "
                        f"to {validated_state} through a content edit. Use Run, "
                        "Record result, Mark blocked, or Run replicate."
                    )
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            cursor = connection.execute(
                """
                UPDATE research_nodes
                SET title = ?, state = ?, body_json = ?,
                    revision = revision + 1, updated_at = ?
                WHERE graph_id = ? AND node_id = ? AND revision = ?
                """,
                (
                    title,
                    validated_state,
                    _json(validated_body),
                    _now(),
                    graph_id,
                    node_id,
                    int(expected_node_revision),
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                latest = self._get_node_row(connection, graph_id, node_id)
                raise ValueError(
                    "This node changed in another thread. "
                    f"Refresh and retry (current node revision {latest['revision']})."
                )
            event_id = self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="node.updated",
                node_ids=[node_id],
            )
        return self.get_node(graph_id, node_id), event_id

    def add_edge(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        source_node_id: str,
        target_node_id: str,
        relation: EdgeRelation | str,
    ) -> int:
        graph_id = _safe_id(graph_id, label="graph_id")
        source_node_id = _safe_id(source_node_id, label="source_node_id")
        target_node_id = _safe_id(target_node_id, label="target_node_id")
        edge_relation = EdgeRelation(relation)
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_edge(
                connection,
                graph_id=graph_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relation=edge_relation,
            )
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            connection.execute(
                """
                INSERT INTO research_edges (
                    graph_id, source_node_id, target_node_id, relation
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    graph_id,
                    source_node_id,
                    target_node_id,
                    edge_relation.value,
                ),
            )
            return self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="edge.added",
                node_ids=[source_node_id, target_node_id],
            )

    def add_ref(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        node_id: str,
        ref_kind: RefKind | str,
        ref_id: str,
    ) -> int:
        graph_id = _safe_id(graph_id, label="graph_id")
        node_id = _safe_id(node_id, label="node_id")
        kind = RefKind(ref_kind)
        reference = str(ref_id or "").strip()
        if not reference:
            raise ValueError("Reference target is required.")
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._get_node_row(connection, graph_id, node_id)
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            connection.execute(
                """
                INSERT INTO research_refs (
                    graph_id, node_id, ref_kind, ref_id
                ) VALUES (?, ?, ?, ?)
                """,
                (graph_id, node_id, kind.value, reference),
            )
            return self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="ref.added",
                node_ids=[node_id],
            )

    def mark_experiment_blocked(
        self,
        graph_id: str,
        experiment_node_id: str,
        *,
        expected_revision: int,
        reason: str,
        refs: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], int]:
        graph_id = _safe_id(graph_id, label="graph_id")
        experiment_node_id = _safe_id(
            experiment_node_id, label="experiment_node_id"
        )
        summary = str(reason or "").strip()
        if not summary:
            raise ValueError("A blocking reason is required.")
        result_id = _new_id("res")
        now = _now()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            experiment = self._get_node_row(
                connection, graph_id, experiment_node_id
            )
            if NodeKind(str(experiment["kind"])) is not NodeKind.EXPERIMENT:
                raise ValueError("Only experiment nodes can be marked blocked.")
            current_state = ExperimentState(str(experiment["state"]))
            if current_state in {
                ExperimentState.BLOCKED,
                ExperimentState.HAS_RESULTS,
            }:
                raise ValueError(
                    f"Experiment in {current_state.value} state cannot be "
                    "marked blocked."
                )
            connection.execute(
                """
                UPDATE research_nodes
                SET state = 'blocked', revision = revision + 1, updated_at = ?
                WHERE graph_id = ? AND node_id = ?
                """,
                (now, graph_id, experiment_node_id),
            )
            body = validate_node_body(NodeKind.RESULT, {"summary": summary})
            connection.execute(
                """
                INSERT INTO research_nodes (
                    graph_id, node_id, kind, title, state, body_json,
                    revision, created_at, updated_at
                ) VALUES (?, ?, 'result', 'Execution blocked', '', ?, 1, ?, ?)
                """,
                (graph_id, result_id, _json(body), now, now),
            )
            connection.execute(
                """
                INSERT INTO research_edges (
                    graph_id, source_node_id, target_node_id, relation
                ) VALUES (?, ?, ?, 'produces')
                """,
                (graph_id, experiment_node_id, result_id),
            )
            connection.execute(
                """
                UPDATE research_launches
                SET status = 'blocked', lease_owner = '', lease_until = 0,
                    updated_at = ?
                WHERE graph_id = ? AND experiment_node_id = ?
                  AND status IN ('claimed', 'submitting', 'running', 'unknown')
                """,
                (now, graph_id, experiment_node_id),
            )
            for ref in list(refs or []):
                connection.execute(
                    """
                    INSERT INTO research_refs (
                        graph_id, node_id, ref_kind, ref_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        graph_id,
                        result_id,
                        RefKind(ref["ref_kind"]).value,
                        str(ref["ref_id"]).strip(),
                    ),
                )
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            event_id = self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="experiment.blocked",
                node_ids=[experiment_node_id, result_id],
            )
        return self.get_node(graph_id, result_id), event_id

    def claim_launch(
        self,
        graph_id: str,
        experiment_node_id: str,
        *,
        expected_revision: int,
        replicate: bool,
        lease_owner: str,
        lease_seconds: int = 120,
    ) -> tuple[dict[str, Any], bool]:
        graph_id = _safe_id(graph_id, label="graph_id")
        experiment_node_id = _safe_id(
            experiment_node_id, label="experiment_node_id"
        )
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                f"""
                SELECT * FROM research_launches
                WHERE graph_id = ? AND experiment_node_id = ?
                  AND status IN ({','.join('?' for _ in _ACTIVE_LAUNCH_STATUSES)})
                ORDER BY created_at DESC LIMIT 1
                """,
                (graph_id, experiment_node_id, *_ACTIVE_LAUNCH_STATUSES),
            ).fetchone()
            if existing is not None:
                return self._launch_row(existing), False
            experiment = self._get_node_row(
                connection, graph_id, experiment_node_id
            )
            if NodeKind(str(experiment["kind"])) is not NodeKind.EXPERIMENT:
                raise ValueError("Only experiment nodes can be launched.")
            state = ExperimentState(str(experiment["state"]))
            allowed = {ExperimentState.READY}
            if replicate:
                allowed.add(ExperimentState.HAS_RESULTS)
            if state not in allowed:
                action = "replicated" if replicate else "run"
                raise ValueError(
                    f"Experiment must be ready before it can be {action}; "
                    f"current state is {state.value}."
                )
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            launch_id = _new_id("launch")
            idempotency_key = _new_id(
                "replicate" if replicate else "experiment"
            )
            now = _now()
            connection.execute(
                """
                INSERT INTO research_launches (
                    launch_id, graph_id, experiment_node_id, idempotency_key,
                    status, thread_id, run_id, lease_owner, lease_until,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'claimed', '', '', ?, ?, ?, ?)
                """,
                (
                    launch_id,
                    graph_id,
                    experiment_node_id,
                    idempotency_key,
                    str(lease_owner or ""),
                    now + max(30, int(lease_seconds)),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE research_nodes
                SET state = 'running', revision = revision + 1, updated_at = ?
                WHERE graph_id = ? AND node_id = ?
                """,
                (now, graph_id, experiment_node_id),
            )
            self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="experiment.launch_claimed",
                node_ids=[experiment_node_id],
                launch_id=launch_id,
            )
            row = connection.execute(
                "SELECT * FROM research_launches WHERE launch_id = ?",
                (launch_id,),
            ).fetchone()
            assert row is not None
            return self._launch_row(row), True

    def update_launch(
        self,
        launch_id: str,
        *,
        status: str,
        thread_id: str = "",
        run_id: str = "",
        lease_owner: str = "",
        lease_until: float = 0,
    ) -> dict[str, Any]:
        launch_id = _safe_id(launch_id, label="launch_id")
        normalized_status = str(status or "").strip()
        if normalized_status not in {
            "claimed",
            "submitting",
            "running",
            "completed",
            "blocked",
            "unknown",
        }:
            raise ValueError(f"Invalid launch status: {status}")
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM research_launches WHERE launch_id = ?",
                (launch_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Research launch not found: {launch_id}")
            current_status = str(current["status"])
            if (
                current_status in {"completed", "blocked"}
                and normalized_status != current_status
            ):
                # Result/blocked writeback may finish a very fast child before
                # the submitter advances its local launch object to running.
                # Terminal launch states are absorbing and must never be
                # downgraded by that race or by recovery.
                return self._launch_row(current)
            connection.execute(
                """
                UPDATE research_launches
                SET status = ?,
                    thread_id = CASE WHEN ? = '' THEN thread_id ELSE ? END,
                    run_id = CASE WHEN ? = '' THEN run_id ELSE ? END,
                    lease_owner = ?,
                    lease_until = ?,
                    updated_at = ?
                WHERE launch_id = ?
                """,
                (
                    normalized_status,
                    str(thread_id or ""),
                    str(thread_id or ""),
                    str(run_id or ""),
                    str(run_id or ""),
                    str(lease_owner or ""),
                    float(lease_until or 0),
                    _now(),
                    launch_id,
                ),
            )
            graph = self._get_graph_row(connection, str(current["graph_id"]))
            event_id = self._write_event(
                connection,
                graph_id=str(current["graph_id"]),
                revision=int(graph["revision"]),
                change=f"launch.{normalized_status}",
                thread_id=str(thread_id or current["thread_id"] or ""),
                node_ids=[str(current["experiment_node_id"])],
                launch_id=launch_id,
            )
            row = connection.execute(
                "SELECT * FROM research_launches WHERE launch_id = ?",
                (launch_id,),
            ).fetchone()
            assert row is not None
            result = self._launch_row(row)
            result["event_id"] = event_id
            return result

    def complete_experiment_after_result(
        self,
        connection: sqlite3.Connection,
        *,
        graph_id: str,
        experiment_node_id: str,
    ) -> None:
        connection.execute(
            """
            UPDATE research_nodes
            SET state = 'has_results', revision = revision + 1, updated_at = ?
            WHERE graph_id = ? AND node_id = ? AND kind = 'experiment'
            """,
            (_now(), graph_id, experiment_node_id),
        )
        connection.execute(
            """
            UPDATE research_launches
            SET status = 'completed', lease_owner = '', lease_until = 0,
                updated_at = ?
            WHERE graph_id = ? AND experiment_node_id = ?
              AND status IN ('claimed', 'submitting', 'running', 'unknown')
            """,
            (_now(), graph_id, experiment_node_id),
        )

    def add_result_bundle(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        title: str,
        body: dict[str, Any],
        experiment_node_id: str,
        judgments: list[dict[str, str]],
        refs: list[dict[str, str]] | None = None,
        node_id: str = "",
    ) -> tuple[dict[str, Any], int]:
        graph_id = _safe_id(graph_id, label="graph_id")
        experiment_node_id = _safe_id(
            experiment_node_id, label="experiment_node_id"
        )
        result_id = _safe_id(node_id, label="node_id") if node_id else _new_id(
            "res"
        )
        result_body = validate_node_body(NodeKind.RESULT, body)
        title = str(title or "").strip() or result_body["summary"][:120]
        now = _now()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            experiment = self._get_node_row(
                connection, graph_id, experiment_node_id
            )
            if NodeKind(str(experiment["kind"])) is not NodeKind.EXPERIMENT:
                raise ValueError("A result must be produced by an experiment node.")
            experiment_state = ExperimentState(str(experiment["state"]))
            if experiment_state not in {
                ExperimentState.READY,
                ExperimentState.RUNNING,
                ExperimentState.HAS_RESULTS,
            }:
                raise ValueError(
                    "A result can be recorded only for a ready, running, or "
                    f"previously completed experiment; current state is "
                    f"{experiment_state.value}."
                )
            connection.execute(
                """
                INSERT INTO research_nodes (
                    graph_id, node_id, kind, title, state, body_json,
                    revision, created_at, updated_at
                ) VALUES (?, ?, 'result', ?, '', ?, 1, ?, ?)
                """,
                (graph_id, result_id, title, _json(result_body), now, now),
            )
            connection.execute(
                """
                INSERT INTO research_edges (
                    graph_id, source_node_id, target_node_id, relation
                ) VALUES (?, ?, ?, 'produces')
                """,
                (graph_id, experiment_node_id, result_id),
            )
            for judgment in judgments:
                relation = EdgeRelation(judgment["relation"])
                if relation not in {
                    EdgeRelation.SUPPORTS,
                    EdgeRelation.OPPOSES,
                    EdgeRelation.INCONCLUSIVE,
                }:
                    raise ValueError("Result judgments must be supports, opposes, or inconclusive.")
                hypothesis_id = _safe_id(
                    str(judgment["hypothesis_node_id"]),
                    label="hypothesis_node_id",
                )
                self._validate_edge(
                    connection,
                    graph_id=graph_id,
                    source_node_id=result_id,
                    target_node_id=hypothesis_id,
                    relation=relation,
                )
                connection.execute(
                    """
                    INSERT INTO research_edges (
                        graph_id, source_node_id, target_node_id, relation
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (graph_id, result_id, hypothesis_id, relation.value),
                )
            for ref in list(refs or []):
                connection.execute(
                    """
                    INSERT INTO research_refs (
                        graph_id, node_id, ref_kind, ref_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        graph_id,
                        result_id,
                        RefKind(ref["ref_kind"]).value,
                        str(ref["ref_id"]).strip(),
                    ),
                )
            self.complete_experiment_after_result(
                connection,
                graph_id=graph_id,
                experiment_node_id=experiment_node_id,
            )
            revision = self._bump_graph(
                connection,
                graph_id=graph_id,
                expected_revision=expected_revision,
            )
            event_id = self._write_event(
                connection,
                graph_id=graph_id,
                revision=revision,
                change="result.recorded",
                node_ids=[experiment_node_id, result_id],
            )
        return self.get_node(graph_id, result_id), event_id

    def list_events(
        self,
        *,
        graph_id: str,
        after_event_id: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        graph_id = _safe_id(graph_id, label="graph_id")
        capped = min(500, max(1, int(limit)))
        with connect_workspace_db(self.workspace) as connection:
            self._get_graph_row(connection, graph_id)
            rows = connection.execute(
                """
                SELECT * FROM ui_events
                WHERE graph_id = ? AND event_id > ?
                ORDER BY event_id ASC
                LIMIT ?
                """,
                (graph_id, max(0, int(after_event_id)), capped),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except Exception:
                payload = {}
            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "event_type": str(row["event_type"]),
                    "thread_id": str(row["thread_id"]),
                    "graph_id": str(row["graph_id"]),
                    "payload": payload if isinstance(payload, dict) else {},
                    "created_at": float(row["created_at"]),
                }
            )
        return events

    def latest_event_id(self, graph_id: str) -> int:
        graph_id = _safe_id(graph_id, label="graph_id")
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                "SELECT MAX(event_id) AS event_id FROM ui_events WHERE graph_id = ?",
                (graph_id,),
            ).fetchone()
        return int(row["event_id"] or 0) if row is not None else 0

    def active_launches(self) -> list[dict[str, Any]]:
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM research_launches
                WHERE status IN ({','.join('?' for _ in _ACTIVE_LAUNCH_STATUSES)})
                ORDER BY created_at ASC
                """,
                _ACTIVE_LAUNCH_STATUSES,
            ).fetchall()
        return [self._launch_row(row) for row in rows]

    def get_launch(self, launch_id: str) -> dict[str, Any]:
        launch_id = _safe_id(launch_id, label="launch_id")
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                "SELECT * FROM research_launches WHERE launch_id = ?",
                (launch_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Research launch not found: {launch_id}")
        return self._launch_row(row)

    def find_launch_by_thread(self, thread_id: str) -> dict[str, Any] | None:
        thread_id = _safe_id(thread_id, label="thread_id")
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                """
                SELECT * FROM research_launches
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return self._launch_row(row) if row is not None else None

    def claim_planning(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        lease_owner: str,
        stale_after_seconds: int = 900,
        recovery_lease_seconds: int = 120,
        allow_same_revision_after_no_change: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically claim or recover one graph-planning child."""

        graph_id = _safe_id(graph_id, label="graph_id")
        now = _now()
        lease_seconds = max(30, int(recovery_lease_seconds))
        stale_seconds = max(60, int(stale_after_seconds))
        # Ownership is intentionally not persisted: BEGIN IMMEDIATE, the
        # partial unique index, and lease expiry are sufficient for recovery.
        _ = lease_owner
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            graph = self._get_graph_row(connection, graph_id)
            current_revision = int(graph["revision"])
            if current_revision != int(expected_revision):
                raise ResearchGraphConflict(
                    expected_revision=int(expected_revision),
                    current_revision=current_revision,
                )
            active_row = connection.execute(
                """
                SELECT * FROM research_planning
                WHERE graph_id = ? AND status IN ('claimed', 'attached')
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (graph_id,),
            ).fetchone()
            if active_row is not None:
                active = self._planning_row(active_row)
                if float(active["lease_until"]) > now:
                    return active, False
                age = now - float(active["updated_at"])
                if age <= stale_seconds:
                    connection.execute(
                        """
                        UPDATE research_planning
                        SET lease_until = ?, updated_at = ?
                        WHERE planning_id = ?
                          AND status IN ('claimed', 'attached')
                        """,
                        (
                            now + lease_seconds,
                            now,
                            active["planning_id"],
                        ),
                    )
                    recovered_row = connection.execute(
                        """
                        SELECT * FROM research_planning
                        WHERE planning_id = ?
                        """,
                        (active["planning_id"],),
                    ).fetchone()
                    recovered = self._planning_row(recovered_row)
                    self._write_planning_event(
                        connection,
                        event_type="research_graph.planning_started",
                        planning=recovered,
                        recovered=True,
                    )
                    return {**recovered, "recovered": True}, True
                connection.execute(
                    """
                    UPDATE research_planning
                    SET status = 'stale', lease_until = 0, updated_at = ?
                    WHERE planning_id = ?
                      AND status IN ('claimed', 'attached')
                    """,
                    (now, active["planning_id"]),
                )
                stale = {
                    **active,
                    "status": "stale",
                    "lease_until": 0.0,
                    "updated_at": now,
                }
                self._write_planning_event(
                    connection,
                    event_type="research_graph.planning_stale",
                    planning=stale,
                )

            if not allow_same_revision_after_no_change:
                no_change = connection.execute(
                    """
                    SELECT 1
                    FROM research_planning
                    WHERE graph_id = ?
                      AND start_revision = ?
                      AND status = 'no_change'
                    LIMIT 1
                    """,
                    (graph_id, current_revision),
                ).fetchone()
                if no_change is not None:
                    return {
                        "planning_id": "",
                        "graph_id": graph_id,
                        "revision": current_revision,
                        "thread_id": "",
                        "status": "no_change",
                        "lease_until": 0.0,
                        "created_at": now,
                        "updated_at": now,
                    }, False

            planning_id = _new_id("planning")
            connection.execute(
                """
                INSERT INTO research_planning (
                    planning_id, graph_id, start_revision, status,
                    thread_id, lease_until, created_at, updated_at
                ) VALUES (?, ?, ?, 'claimed', '', ?, ?, ?)
                """,
                (
                    planning_id,
                    graph_id,
                    current_revision,
                    now + lease_seconds,
                    now,
                    now,
                ),
            )
            planning_row = connection.execute(
                """
                SELECT * FROM research_planning
                WHERE planning_id = ?
                """,
                (planning_id,),
            ).fetchone()
            planning = self._planning_row(planning_row)
            self._write_planning_event(
                connection,
                event_type="research_graph.planning_started",
                planning=planning,
            )
            return planning, True

    def update_planning(
        self,
        graph_id: str,
        planning_id: str,
        *,
        start_revision: int,
        status: str,
        thread_id: str = "",
    ) -> int:
        graph_id = _safe_id(graph_id, label="graph_id")
        planning_id = _safe_id(planning_id, label="planning_id")
        event_type = {
            "attached": "research_graph.planning_attached",
            "finished": "research_graph.planning_finished",
            "no_change": "research_graph.planning_no_change",
            "stale": "research_graph.planning_stale",
        }.get(str(status or "").strip())
        if not event_type:
            raise ValueError(f"Invalid planning status: {status}")
        normalized_status = str(status or "").strip()
        now = _now()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._get_graph_row(connection, graph_id)
            row = connection.execute(
                """
                SELECT * FROM research_planning
                WHERE graph_id = ? AND planning_id = ?
                """,
                (graph_id, planning_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Research planning state not found: {planning_id}")
            planning = self._planning_row(row)
            if int(planning["revision"]) != int(start_revision):
                raise ValueError(
                    "Planning start revision does not match durable state."
                )
            if planning["status"] in _TERMINAL_PLANNING_STATUSES:
                return 0
            next_thread_id = str(thread_id or planning["thread_id"])
            next_lease = (
                now + _PLANNING_LEASE_SECONDS
                if normalized_status == "attached"
                else 0
            )
            cursor = connection.execute(
                """
                UPDATE research_planning
                SET status = ?, thread_id = ?, lease_until = ?, updated_at = ?
                WHERE planning_id = ?
                  AND status IN ('claimed', 'attached')
                """,
                (
                    normalized_status,
                    next_thread_id,
                    next_lease,
                    now,
                    planning_id,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                return 0
            updated_row = connection.execute(
                """
                SELECT * FROM research_planning
                WHERE planning_id = ?
                """,
                (planning_id,),
            ).fetchone()
            updated = self._planning_row(updated_row)
            return self._write_planning_event(
                connection,
                event_type=event_type,
                planning=updated,
            )

    def scheduler_snapshot(self, graph_id: str) -> dict[str, Any]:
        graph = self.get_graph(graph_id)
        with connect_workspace_db(self.workspace) as connection:
            active_planning_row = connection.execute(
                """
                SELECT * FROM research_planning
                WHERE graph_id = ? AND status IN ('claimed', 'attached')
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (graph_id,),
            ).fetchone()
            last_row = connection.execute(
                """
                SELECT MAX(start_revision) AS revision
                FROM research_planning
                WHERE graph_id = ?
                  AND status IN ('finished', 'no_change', 'stale')
                """,
                (graph_id,),
            ).fetchone()
        active_planning_id = (
            str(active_planning_row["planning_id"])
            if active_planning_row is not None
            else ""
        )
        last_planning_revision = (
            int(last_row["revision"])
            if last_row is not None and last_row["revision"] is not None
            else -1
        )
        active_launch = next(
            (
                launch
                for launch in self.get_snapshot(graph_id)["launches"]
                if launch["status"] in _ACTIVE_LAUNCH_STATUSES
            ),
            None,
        )
        return {
            "enabled": graph["orchestration_mode"] == "auto" and not graph["archived"],
            "current_launch_id": (
                str(active_launch["launch_id"])
                if active_launch is not None
                else active_planning_id
            ),
            "last_planning_revision": last_planning_revision,
        }

    def find_planning_by_thread(self, thread_id: str) -> dict[str, Any] | None:
        thread_id = _safe_id(thread_id, label="thread_id")
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                """
                SELECT * FROM research_planning
                WHERE thread_id = ?
                  AND status IN ('claimed', 'attached')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return self._planning_row(row) if row is not None else None


__all__ = [
    "ResearchGraphConflict",
    "ResearchGraphStore",
]
