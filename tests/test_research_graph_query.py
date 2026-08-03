from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.research.knowledge_graph.models import GraphCreateRequest
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.tools.base import ensure_project_space_layout
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_models import MessagePart, ThreadMessage
from catmaster.webui.thread_store import ThreadStore


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    return workspace


def test_bound_sql_has_no_hidden_row_or_body_truncation_and_supports_sqlite(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    long_claim = "mechanistic detail " * 2_000
    service = ResearchGraphService(workspace=workspace)
    created = service.create_graph(
        GraphCreateRequest(
            question="Which of the many mechanisms remains viable?",
            initial_hypotheses=[
                {"claim": long_claim},
                *[
                    {"claim": f"Mechanism {index:03d} remains viable."}
                    for index in range(140)
                ],
            ],
        )
    )
    graph_id = created["graph"]["graph_id"]
    service.create_graph(
        GraphCreateRequest(
            question="This graph must remain outside the bound SQL projection.",
            initial_hypotheses=[{"claim": "An unrelated mechanism is hidden."}],
        )
    )
    query = ResearchGraphSQLQuery(workspace)

    all_rows = query.execute(
        graph_id=graph_id,
        sql="SELECT node_id FROM research_nodes ORDER BY node_id",
    )
    assert all_rows["row_count"] == 141
    assert len(all_rows["rows"]) == 141
    assert all_rows["revision"] == created["graph"]["revision"]

    second_page = query.execute(
        graph_id=graph_id,
        sql="SELECT node_id FROM research_nodes ORDER BY node_id LIMIT 50 OFFSET 50",
    )
    assert second_page["rows"] == all_rows["rows"][50:100]

    body = query.execute(
        graph_id=graph_id,
        sql=(
            "SELECT length(json_extract(body_json, '$.claim')) AS claim_length "
            "FROM research_nodes WHERE node_id = '"
            + next(
                node["node_id"]
                for node in created["nodes"]
                if node["body"]["claim"] == long_claim.strip()
            )
            + "'"
        ),
    )
    assert body["rows"] == [{"claim_length": len(long_claim.strip())}]

    advanced = query.execute(
        graph_id=graph_id,
        sql=(
            "WITH RECURSIVE seq(x) AS ("
            "SELECT 1 UNION ALL SELECT x + 1 FROM seq WHERE x < 4"
            ") SELECT x, row_number() OVER (ORDER BY x DESC) AS rank, "
            "json_extract('{\"ok\": true}', '$.ok') AS json_ok FROM seq"
        ),
    )
    assert [row["x"] for row in advanced["rows"]] == [4, 3, 2, 1]
    assert [row["rank"] for row in advanced["rows"]] == [1, 2, 3, 4]
    assert {row["json_ok"] for row in advanced["rows"]} == {1}


def test_bound_sql_rejects_direct_storage_introspection_and_mutation(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    created = ResearchGraphService(workspace=workspace).create_graph(
        GraphCreateRequest(
            question="Can the query surface remain read-only?",
            initial_hypotheses=[{"claim": "The graph remains unchanged."}],
        )
    )
    graph_id = created["graph"]["graph_id"]
    query = ResearchGraphSQLQuery(workspace)

    rejected = [
        "SELECT * FROM main.research_nodes",
        (
            "WITH research_nodes AS (SELECT * FROM main.research_nodes) "
            "SELECT graph_id FROM research_nodes"
        ),
        "SELECT * FROM temp.research_nodes",
        "SELECT * FROM main /* disguised qualifier */ . research_nodes",
        "SELECT name FROM sqlite_master",
        "PRAGMA table_info(research_nodes)",
        "SELECT * FROM pragma_table_info('research_nodes')",
        "SELECT load_extension('anything')",
        "SELECT readfile('/etc/passwd')",
        "VALUES (1)",
        "EXPLAIN SELECT node_id FROM research_nodes",
        "DELETE FROM research_nodes",
        "ATTACH DATABASE ':memory:' AS other",
        "SELECT node_id FROM research_nodes; SELECT graph_id FROM research_graphs",
    ]
    for statement in rejected:
        with pytest.raises(ValueError, match="rejected|SELECT or WITH"):
            query.execute(graph_id=graph_id, sql=statement)

    unchanged = query.execute(
        graph_id=graph_id,
        sql="SELECT count(*) AS count FROM research_nodes",
    )
    assert unchanged["rows"] == [{"count": 1}]


def test_message_owner_projection_requires_an_unambiguous_referenced_owner(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    threads = ThreadStore(workspace=workspace, workspace_id="default")
    first = threads.create_thread(title="First owner", entrypoint="research")
    second = threads.create_thread(title="Second owner", entrypoint="research")
    for thread, text in ((first, "first payload"), (second, "second payload")):
        threads.append_message(
            ThreadMessage(
                id="shared_message",
                thread_id=thread.thread_id,
                role="assistant",
                status="completed",
                parts=[
                    MessagePart(
                        id=f"part_{thread.thread_id}",
                        type="text",
                        text=text,
                        status="completed",
                    )
                ],
            )
        )

    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    created = service.create_graph(
        GraphCreateRequest(
            question="Which owner supplied the cited message?",
            initial_hypotheses=[{"claim": "The first owner supplied it."}],
        )
    )
    graph_id = created["graph"]["graph_id"]
    node_id = created["nodes"][0]["node_id"]
    with pytest.raises(ValueError, match="thread_id:message_id"):
        service.validate_ref({"ref_kind": "message", "ref_id": "shared_message"})

    service.add_ref(
        graph_id,
        expected_revision=created["graph"]["revision"],
        node_id=node_id,
        ref={
            "ref_kind": "message",
            "ref_id": f"{first.thread_id}:shared_message",
        },
    )
    rows = ResearchGraphSQLQuery(workspace).execute(
        graph_id=graph_id,
        sql=(
            "SELECT thread_id, message_id FROM thread_messages "
            "ORDER BY thread_id, message_id"
        ),
    )["rows"]
    assert rows == [
        {"thread_id": first.thread_id, "message_id": "shared_message"}
    ]


def test_owner_views_expose_only_rows_referenced_by_the_bound_graph(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    threads = ThreadStore(workspace=workspace, workspace_id="default")
    first = threads.create_thread(title="Referenced thread", entrypoint="research")
    second = threads.create_thread(title="One referenced message", entrypoint="research")
    hidden = threads.create_thread(title="Unreferenced thread", entrypoint="research")

    def append(thread_id: str, message_id: str, text: str) -> None:
        threads.append_message(
            ThreadMessage(
                id=message_id,
                thread_id=thread_id,
                role="assistant",
                status="completed",
                parts=[
                    MessagePart(
                        id=f"part_{message_id}",
                        type="text",
                        text=text,
                        status="completed",
                    )
                ],
            )
        )

    append(first.thread_id, "first_one", "visible through the thread ref")
    append(first.thread_id, "first_two", "also visible through the thread ref")
    append(second.thread_id, "second_selected", "visible by canonical message ref")
    append(second.thread_id, "second_hidden", "not referenced")
    append(hidden.thread_id, "third_hidden", "not referenced")

    files = workspace / "files"
    (files / "visible.txt").write_text("visible artifact", encoding="utf-8")
    (files / "hidden.txt").write_text("hidden artifact", encoding="utf-8")
    artifacts = ArtifactRegistry(workspace=workspace, workspace_id="default")
    visible_artifact = artifacts.register_path(
        "visible.txt",
        thread_id=first.thread_id,
    )
    artifacts.register_path("hidden.txt", thread_id=hidden.thread_id)

    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    created = service.create_graph(
        GraphCreateRequest(
            question="Which referenced owners can this graph read?",
            initial_hypotheses=[{"claim": "Only explicit owners are visible."}],
        )
    )
    graph_id = created["graph"]["graph_id"]
    node_id = created["nodes"][0]["node_id"]
    current = service.add_ref(
        graph_id,
        expected_revision=created["graph"]["revision"],
        node_id=node_id,
        ref={"ref_kind": "thread", "ref_id": first.thread_id},
    )
    current = service.add_ref(
        graph_id,
        expected_revision=current["graph"]["revision"],
        node_id=node_id,
        ref={
            "ref_kind": "message",
            "ref_id": f"{second.thread_id}:second_selected",
        },
    )
    service.add_ref(
        graph_id,
        expected_revision=current["graph"]["revision"],
        node_id=node_id,
        ref={"ref_kind": "artifact", "ref_id": visible_artifact.artifact_id},
    )

    query = ResearchGraphSQLQuery(workspace)
    artifact_rows = query.execute(
        graph_id=graph_id,
        sql="SELECT artifact_id, thread_id FROM workspace_artifacts",
    )["rows"]
    message_rows = query.execute(
        graph_id=graph_id,
        sql=(
            "SELECT thread_id, message_id FROM thread_messages "
            "ORDER BY thread_id, message_id"
        ),
    )["rows"]

    assert artifact_rows == [
        {
            "artifact_id": visible_artifact.artifact_id,
            "thread_id": first.thread_id,
        }
    ]
    assert message_rows == sorted(
        [
            {"thread_id": first.thread_id, "message_id": "first_one"},
            {"thread_id": first.thread_id, "message_id": "first_two"},
            {
                "thread_id": second.thread_id,
                "message_id": "second_selected",
            },
        ],
        key=lambda row: (row["thread_id"], row["message_id"]),
    )
