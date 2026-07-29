from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from catmaster.research.knowledge_graph.context import ResearchGraphContextBuilder
from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
    HypothesisCreateRequest,
    ResearchGraphPlanningProposal,
    ResultCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.store import (
    ResearchGraphConflict,
    ResearchGraphStore,
)
from catmaster.storage import connect_workspace_db, workspace_database_path
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import (
    ensure_project_space_layout,
    system_root,
    workspace_scope,
)
from catmaster.tools.misc.research_graph import record_bound_research_result
from catmaster.webui.thread_models import MessagePart, ThreadMessage, ThreadStatus
from catmaster.webui.thread_store import ThreadStore


def _workspace(tmp_path: Path, name: str = "default") -> Path:
    workspace = tmp_path / name
    ensure_project_space_layout(workspace, create=True)
    return workspace


def _seed_graph(service: ResearchGraphService) -> dict:
    return service.create_graph(
        GraphCreateRequest(
            question="Does treatment A increase response B?",
            title="A to B",
            initial_hypotheses=[
                {
                    "claim": "Treatment A increases response B.",
                    "rationale": "A activates the proposed pathway.",
                    "predictions": ["B is higher than in the control."],
                }
            ],
        )
    )


def _ready_experiment(
    service: ResearchGraphService,
    graph: dict,
    hypothesis_id: str,
    *,
    title: str = "Measure B",
) -> dict:
    return service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=graph["graph"]["revision"],
            title=title,
            objective="Measure response B after treatment A.",
            plan_summary="Compare treated and matched control samples.",
            decision_rule="A reproducible increase supports the hypothesis.",
            execution_lane="experiment",
            state="ready",
            tests_hypothesis_ids=[hypothesis_id],
        ),
    )


def test_workspace_graph_preserves_full_scientific_cycle_and_competing_evidence(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    first = _seed_graph(service)
    graph_id = first["graph"]["graph_id"]
    h1 = first["nodes"][0]
    with_experiment = _ready_experiment(service, first, h1["node_id"])
    e1 = with_experiment["node"]

    supporting = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=with_experiment["graph"]["revision"],
            title="First measurement",
            summary="B increased in the treated samples.",
            experiment_node_id=e1["node_id"],
            judgments=[
                {
                    "hypothesis_node_id": h1["node_id"],
                    "relation": "supports",
                }
            ],
            refs=[{"ref_kind": "url", "ref_id": "https://example.org/run-1"}],
        ),
    )
    r1 = supporting["node"]
    opposing = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=supporting["graph"]["revision"],
            title="Independent replicate",
            summary="The independent replicate found a decrease in B.",
            experiment_node_id=e1["node_id"],
            judgments=[
                {
                    "hypothesis_node_id": h1["node_id"],
                    "relation": "opposes",
                }
            ],
        ),
    )
    h1_view = next(
        node for node in opposing["nodes"] if node["node_id"] == h1["node_id"]
    )
    assert h1_view["evidence_state"] == "conflicting_evidence"

    second_hypothesis = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=opposing["graph"]["revision"],
            title="Context dependence",
            claim="The effect of A on B depends on sample context.",
            rationale="The two experiments used different contexts.",
            predictions=["Context stratification resolves the sign change."],
            suggested_by_result_ids=[r1["node_id"]],
        ),
    )
    h2 = second_hypothesis["node"]
    second_experiment = _ready_experiment(
        service,
        second_hypothesis,
        h2["node_id"],
        title="Stratified replicate",
    )

    relations = {
        (
            edge["source_node_id"],
            edge["relation"],
            edge["target_node_id"],
        )
        for edge in second_experiment["edges"]
    }
    assert (h1["node_id"], "tests", e1["node_id"]) in relations
    assert (e1["node_id"], "produces", r1["node_id"]) in relations
    assert (r1["node_id"], "suggests", h2["node_id"]) in relations
    assert (
        h2["node_id"],
        "tests",
        second_experiment["node"]["node_id"],
    ) in relations
    assert sum(node["kind"] == "result" for node in opposing["nodes"]) == 2


def test_frontier_keeps_all_branches_but_auto_runs_one_ranked_experiment(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    submissions: list[str] = []

    class _Loop:
        async def submit(self, *, thread_id, payload):
            submissions.append(thread_id)
            return {}

    service = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which branch should be tested next?",
            orchestration_mode="auto",
            initial_hypotheses=[
                {"claim": "Lower-priority mechanism.", "importance": "low"},
                {"claim": "Higher-priority mechanism.", "importance": "high"},
            ],
        )
    )
    graph_id = graph["graph"]["graph_id"]
    low_hypothesis = next(
        node for node in graph["nodes"] if node["body"]["importance"] == "low"
    )
    high_hypothesis = next(
        node for node in graph["nodes"] if node["body"]["importance"] == "high"
    )

    def add_ready(
        current: dict,
        *,
        hypothesis_id: str,
        title: str,
        expected_value: str,
        estimated_compute_cost: str,
    ) -> dict:
        return service.add_experiment(
            graph_id,
            ExperimentCreateRequest(
                expected_revision=current["graph"]["revision"],
                title=title,
                objective=f"Test {title}.",
                plan_summary="Run the bounded discriminating check.",
                decision_rule="The observed outcome distinguishes the branch.",
                state="ready",
                tests_hypothesis_ids=[hypothesis_id],
                expected_value=expected_value,
                estimated_compute_cost=estimated_compute_cost,
            ),
        )

    low_branch = add_ready(
        graph,
        hypothesis_id=low_hypothesis["node_id"],
        title="Low hypothesis, high value",
        expected_value="high",
        estimated_compute_cost="none",
    )
    high_expensive = add_ready(
        low_branch,
        hypothesis_id=high_hypothesis["node_id"],
        title="High hypothesis, high value, high compute",
        expected_value="high",
        estimated_compute_cost="high",
    )
    high_cheap = add_ready(
        high_expensive,
        hypothesis_id=high_hypothesis["node_id"],
        title="High hypothesis, high value, low compute",
        expected_value="high",
        estimated_compute_cost="low",
    )
    high_lower_value = add_ready(
        high_cheap,
        hypothesis_id=high_hypothesis["node_id"],
        title="High hypothesis, lower value",
        expected_value="low",
        estimated_compute_cost="none",
    )

    frontier = [item["node_id"] for item in high_lower_value["graph"]["frontier"]]
    assert frontier == [
        high_cheap["node"]["node_id"],
        high_expensive["node"]["node_id"],
        high_lower_value["node"]["node_id"],
        low_branch["node"]["node_id"],
    ]

    asyncio.run(service.tick())
    snapshot = service.store.get_snapshot(graph_id)
    active = [
        launch
        for launch in snapshot["launches"]
        if launch["status"] in {"claimed", "submitting", "running", "unknown"}
    ]
    assert len(active) == 1
    assert active[0]["experiment_node_id"] == high_cheap["node"]["node_id"]
    assert submissions == [active[0]["thread_id"]]


def test_planner_schema_retains_multi_branch_output_bounds() -> None:
    schema = ResearchGraphPlanningProposal.model_json_schema()
    assert schema["properties"]["hypotheses"]["maxItems"] == 12
    assert schema["properties"]["experiments"]["maxItems"] == 24


def test_thread_binding_is_formal_cross_thread_and_not_ownership(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    node_id = graph["nodes"][0]["node_id"]
    threads = ThreadStore(workspace=workspace, workspace_id="default")
    thread_a = threads.create_thread(title="A", entrypoint="research")
    thread_b = threads.create_thread(title="B", entrypoint="research")
    threads.update_thread(thread_a.thread_id, status=ThreadStatus.RUNNING)

    service.bind_thread(
        thread_a.thread_id,
        graph_id=graph_id,
        focus_node_id=node_id,
    )
    service.bind_thread(thread_b.thread_id, graph_id=graph_id)
    catalog = service.catalog(current_thread_id=thread_b.thread_id)

    assert catalog[0]["bound_thread_count"] == 2
    assert catalog[0]["bound_to_current_thread"] is True
    assert service.store.get_graph(graph_id)["graph_id"] == graph_id
    assert threads.get_thread(thread_a.thread_id).status is ThreadStatus.RUNNING
    # A running source thread does not lock workspace graph mutations.
    changed = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=graph["graph"]["revision"],
            claim="A second cross-thread hypothesis.",
            rationale="It is independently falsifiable.",
            predictions=["A distinct measurement changes."],
        ),
    )
    assert changed["graph"]["revision"] == graph["graph"]["revision"] + 1


def test_graph_cas_and_dependency_dag_are_enforced_transactionally(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    h1 = graph["nodes"][0]
    first = _ready_experiment(service, graph, h1["node_id"], title="E1")
    second = service.add_experiment(
        graph_id,
        ExperimentCreateRequest(
            expected_revision=first["graph"]["revision"],
            objective="Second check",
            plan_summary="Run after E1.",
            decision_rule="The output discriminates the hypothesis.",
            state="draft",
            depends_on_experiment_ids=[first["node"]["node_id"]],
        ),
    )
    before_revision = second["graph"]["revision"]
    before_events = service.store.latest_event_id(graph_id)
    with pytest.raises(ValueError, match="cycle"):
        service.store.add_edge(
            graph_id,
            expected_revision=before_revision,
            source_node_id=first["node"]["node_id"],
            target_node_id=second["node"]["node_id"],
            relation="depends_on",
        )
    assert service.store.get_graph(graph_id)["revision"] == before_revision
    assert service.store.latest_event_id(graph_id) == before_events

    service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=before_revision,
            claim="Concurrent edit wins.",
            rationale="CAS test.",
            predictions=[],
        ),
    )
    with pytest.raises(ResearchGraphConflict) as conflict:
        service.add_hypothesis(
            graph_id,
            HypothesisCreateRequest(
                expected_revision=before_revision,
                claim="Stale edit.",
                rationale="Should not overwrite.",
                predictions=[],
            ),
        )
    assert conflict.value.current_revision == before_revision + 1


def test_two_workers_claim_only_one_active_launch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    graph_id = graph["graph"]["graph_id"]
    revision = experiment["graph"]["revision"]
    experiment_id = experiment["node"]["node_id"]

    def claim(worker: str):
        return ResearchGraphStore(workspace).claim_launch(
            graph_id,
            experiment_id,
            expected_revision=revision,
            replicate=False,
            lease_owner=worker,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(claim, ["one", "two"]))
    assert sum(claimed for _launch, claimed in rows) == 1
    assert len({launch["launch_id"] for launch, _claimed in rows}) == 1
    snapshot = service.store.get_snapshot(graph_id)
    assert sum(
        launch["status"] in {"claimed", "submitting", "running", "unknown"}
        for launch in snapshot["launches"]
    ) == 1


def test_refs_are_workspace_scoped_and_missing_sources_remain_visible(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    other = _workspace(tmp_path, "other")
    (other / "files" / "foreign.md").write_text("foreign", encoding="utf-8")
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    node_id = graph["nodes"][0]["node_id"]
    with pytest.raises(ValueError, match="workspace"):
        service.add_ref(
            graph["graph"]["graph_id"],
            expected_revision=graph["graph"]["revision"],
            node_id=node_id,
            ref={"ref_kind": "note", "ref_id": "../other/files/foreign.md"},
        )

    # Persisted refs are never silently deleted if their owner later vanishes.
    (workspace / "files" / "source.md").write_text("source text", encoding="utf-8")
    added = service.add_ref(
        graph["graph"]["graph_id"],
        expected_revision=graph["graph"]["revision"],
        node_id=node_id,
        ref={"ref_kind": "note", "ref_id": "files/source.md"},
    )
    (workspace / "files" / "source.md").unlink()
    view = service.presentation(graph["graph"]["graph_id"])
    ref = next(node for node in view["nodes"] if node["node_id"] == node_id)[
        "refs"
    ][0]
    assert ref["available"] is False
    assert ref["label"] == "Source unavailable"
    assert added["graph"]["revision"] == graph["graph"]["revision"] + 1


def test_agent_and_web_presentations_exclude_runtime_and_audit_fields(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    created = _seed_graph(service)
    graph_id = created["graph"]["graph_id"]

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    web_view = service.presentation(graph_id)
    catalog = service.catalog()
    built_context = service.context_builder.build(graph_id)
    context = built_context["presentation"]
    forbidden = {
        "created_at",
        "updated_at",
        "idempotency_key",
        "lease_owner",
        "lease_until",
        "worker_id",
        "planning",
        "events",
        "actions",
        "launches",
    }
    assert not forbidden & keys(web_view)
    assert not forbidden & keys(context)
    assert not (forbidden - {"updated_at"}) & keys(catalog)
    assert all("created_at" not in graph for graph in catalog)
    assert all("updated_at" in graph for graph in catalog)
    assert not any(field in built_context["markdown"] for field in forbidden)
    assert web_view["graph"]["revision"] >= 1
    assert all("revision" in node for node in web_view["nodes"])


def test_bounded_graphrag_uses_fts_notes_keeps_evidence_and_reports_omissions(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    revision = graph["graph"]["revision"]
    for index in range(110):
        result = service.add_hypothesis(
            graph_id,
            HypothesisCreateRequest(
                expected_revision=revision,
                title=f"Hypothesis {index:03d}",
                claim=f"Mechanism {index:03d} controls response.",
                rationale="Deterministic context budget fixture.",
                predictions=[f"Observable {index:03d} changes."],
            ),
        )
        revision = result["graph"]["revision"]
    target = result["node"]
    note = workspace / "files" / "target-note.md"
    note.write_text("Rare zirconium fingerprint appears only here.", encoding="utf-8")
    service.add_ref(
        graph_id,
        expected_revision=revision,
        node_id=target["node_id"],
        ref={"ref_kind": "note", "ref_id": "files/target-note.md"},
    )
    context = ResearchGraphContextBuilder(workspace=workspace).build(
        graph_id,
        query="zirconium fingerprint",
        max_nodes=12,
        max_chars=4_000,
    )
    assert target["node_id"] in {
        node["node_id"] for node in context["presentation"]["nodes"]
    }
    assert context["presentation"]["shown_count"] <= 12
    assert context["presentation"]["omitted_count"] >= 99
    assert len(context["markdown"]) <= 4_000
    assert "Question:" in context["markdown"]
    assert "Focus path" in context["markdown"]
    assert "omitted" in context["markdown"]
    assert not context["markdown"].lstrip().startswith("{")


def test_graphrag_query_and_conflicting_evidence_precede_large_frontier(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    base_hypothesis_id = graph["nodes"][0]["node_id"]
    revision = graph["graph"]["revision"]
    for index in range(16):
        proposal = service.add_experiment(
            graph_id,
            ExperimentCreateRequest(
                expected_revision=revision,
                title=f"Frontier experiment {index:02d}",
                objective=f"Run frontier check {index:02d}.",
                plan_summary="A ready experiment that must not displace search evidence.",
                decision_rule="Record the bounded observation.",
                execution_lane="experiment",
                state="ready",
                tests_hypothesis_ids=[base_hypothesis_id],
            ),
        )
        revision = proposal["graph"]["revision"]

    target = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=revision,
            title="Zirconium target",
            claim="The rare zirconium fingerprint controls the response.",
            rationale="This is the query-ranked hypothesis.",
            predictions=["The fingerprint tracks the response."],
        ),
    )
    target_hypothesis_id = target["node"]["node_id"]
    target_experiment = service.add_experiment(
        graph_id,
        ExperimentCreateRequest(
            expected_revision=target["graph"]["revision"],
            title="Target measurement",
            objective="Measure the rare zirconium fingerprint.",
            plan_summary="Run two independent determinations.",
            decision_rule="A positive association supports; a negative association opposes.",
            execution_lane="experiment",
            state="ready",
            tests_hypothesis_ids=[target_hypothesis_id],
        ),
    )
    supporting = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=target_experiment["graph"]["revision"],
            title="Supporting determination",
            summary="The fingerprint correlated positively with the response.",
            experiment_node_id=target_experiment["node"]["node_id"],
            judgments=[
                {
                    "hypothesis_node_id": target_hypothesis_id,
                    "relation": "supports",
                }
            ],
        ),
    )
    opposing = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=supporting["graph"]["revision"],
            title="Opposing determination",
            summary="The replicate showed the opposite association.",
            experiment_node_id=target_experiment["node"]["node_id"],
            judgments=[
                {
                    "hypothesis_node_id": target_hypothesis_id,
                    "relation": "opposes",
                }
            ],
        ),
    )

    context = service.context_builder.build(
        graph_id,
        query="rare zirconium fingerprint",
        max_nodes=4,
        max_chars=4_000,
    )
    selected_ids = [
        node["node_id"] for node in context["presentation"]["nodes"]
    ]
    assert target_hypothesis_id in selected_ids
    assert supporting["node"]["node_id"] in selected_ids
    assert opposing["node"]["node_id"] in selected_ids
    assert len(context["presentation"]["frontier_node_ids"]) == 16
    assert context["presentation"]["shown_count"] == len(selected_ids)
    assert context["presentation"]["omitted_count"] == (
        context["presentation"]["total_count"] - len(selected_ids)
    )
    assert all(node_id in context["markdown"] for node_id in selected_ids)
    assert sum(
        line.startswith("- [")
        for line in context["markdown"].splitlines()
    ) == len(selected_ids)
    assert len(context["markdown"]) <= 4_000


def test_research_domain_schema_has_only_minimal_tables(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    ResearchGraphStore(workspace)
    connection = sqlite3.connect(workspace_database_path(workspace))
    try:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table'
                  AND (name LIKE 'research_%' OR name = 'ui_events')
                """
            )
        }
        assert tables == {
            "research_graphs",
            "research_nodes",
            "research_edges",
            "research_refs",
            "research_launches",
            "research_planning",
            "ui_events",
        }
        node_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(research_nodes)")
        }
        graph_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(research_graphs)")
        }
        planning_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(research_planning)")
        }
    finally:
        connection.close()
    forbidden = {
        "metadata",
        "confidence",
        "novelty",
        "importance",
        "model",
        "prompt",
        "tokens",
        "checksum",
        "layout",
        "x",
        "y",
    }
    assert node_columns.isdisjoint(forbidden)
    assert graph_columns.isdisjoint(forbidden)
    assert "body_json" in node_columns
    assert planning_columns == {
        "planning_id",
        "graph_id",
        "start_revision",
        "status",
        "thread_id",
        "lease_until",
        "created_at",
        "updated_at",
    }


def test_planner_claim_is_single_recovers_identity_and_no_change_does_not_loop(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    revision = graph["graph"]["revision"]

    def claim(worker: str):
        return ResearchGraphStore(workspace).claim_planning(
            graph_id,
            expected_revision=revision,
            lease_owner=worker,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, ["one", "two"]))
    assert sum(claimed for _planning, claimed in claims) == 1
    planning_ids = {row["planning_id"] for row, _claimed in claims}
    assert len(planning_ids) == 1
    planning_id = planning_ids.pop()

    service.store.update_planning(
        graph_id,
        planning_id,
        start_revision=revision,
        status="attached",
        thread_id="thread_planner_recovery",
    )
    with connect_workspace_db(workspace) as connection:
        connection.execute(
            """
            UPDATE research_planning
            SET lease_until = 0, updated_at = ?
            WHERE planning_id = ?
            """,
            (time.time() - 180, planning_id),
        )
    recovered, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
        lease_owner="recovery-worker",
        recovery_lease_seconds=30,
    )
    assert claimed is True
    assert recovered["planning_id"] == planning_id
    assert recovered["thread_id"] == "thread_planner_recovery"

    service.store.update_planning(
        graph_id,
        planning_id,
        start_revision=revision,
        status="no_change",
        thread_id="thread_planner_recovery",
    )
    suppressed, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
        lease_owner="next-worker",
    )
    assert claimed is False
    assert suppressed["planning_id"] == ""

    changed = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=revision,
            claim="A graph change opens a new planning opportunity.",
            rationale="The scheduler suppression is revision-scoped.",
            predictions=[],
        ),
    )
    next_planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=changed["graph"]["revision"],
        lease_owner="next-worker",
    )
    assert claimed is True
    assert next_planning["planning_id"] != planning_id


def test_planning_recovery_survives_more_events_than_the_outbox_retains(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    revision = graph["graph"]["revision"]
    planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
        lease_owner="first-worker",
    )
    assert claimed is True
    planning_id = planning["planning_id"]
    service.store.update_planning(
        graph_id,
        planning_id,
        start_revision=revision,
        status="attached",
        thread_id="thread_planning_high_volume",
    )

    with connect_workspace_db(workspace) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            """
            INSERT INTO ui_events (
                event_type, thread_id, graph_id, payload_json, created_at
            ) VALUES ('research_graph.updated', '', ?, '{}', ?)
            """,
            [(graph_id, time.time()) for _ in range(5_200)],
        )
        newest = connection.execute(
            "SELECT MAX(event_id) AS event_id FROM ui_events"
        ).fetchone()
        ResearchGraphStore._prune_events(
            connection,
            graph_id=graph_id,
            newest_event_id=int(newest["event_id"]),
        )
        connection.execute(
            """
            UPDATE research_planning
            SET lease_until = 0, updated_at = ?
            WHERE planning_id = ?
            """,
            (time.time() - 180, planning_id),
        )
        remaining_planning_events = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM ui_events
            WHERE graph_id = ?
              AND event_type LIKE 'research_graph.planning_%'
            """,
            (graph_id,),
        ).fetchone()
        graph_event_count = connection.execute(
            "SELECT COUNT(*) AS count FROM ui_events WHERE graph_id = ?",
            (graph_id,),
        ).fetchone()
    assert int(remaining_planning_events["count"]) == 0
    assert int(graph_event_count["count"]) == 5_000

    recovered, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
        lease_owner="recovery-worker",
        recovery_lease_seconds=30,
    )
    assert claimed is True
    assert recovered["planning_id"] == planning_id
    assert recovered["thread_id"] == "thread_planning_high_volume"
    assert recovered["recovered"] is True


def test_graph_event_pruning_ignores_unrelated_thread_event_id_gaps(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    first_graph_event = service.store.latest_event_id(graph_id)
    with connect_workspace_db(workspace) as connection:
        connection.executemany(
            """
            INSERT INTO ui_events (
                event_type, thread_id, graph_id, payload_json, created_at
            ) VALUES ('message.delta', 'thread_gap_fixture', '', '{}', ?)
            """,
            [(time.time(),) for _ in range(6_000)],
        )

    service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=graph["graph"]["revision"],
            claim="Unrelated event traffic must not shorten graph replay.",
            rationale="The outbox cursor is global but retention is graph-local.",
            predictions=[],
        ),
    )
    retained_ids = {
        event["event_id"]
        for event in service.store.list_events(
            graph_id=graph_id,
            after_event_id=0,
            limit=100,
        )
    }
    assert first_graph_event in retained_ids


def test_launch_recovery_reuses_child_and_never_blindly_resubmits(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    submissions: list[str] = []
    thread_store = ThreadStore(workspace=workspace, workspace_id="default")

    class _Loop:
        async def submit(self, *, thread_id, payload):
            submissions.append(thread_id)
            thread_store.append_message(
                ThreadMessage(
                    id=f"msg_{len(submissions):04d}",
                    thread_id=thread_id,
                    role="user",
                    status="completed",
                    parts=[
                        MessagePart(
                            id=f"part_{len(submissions):04d}",
                            type="text",
                            text=payload.text,
                            status="completed",
                        )
                    ],
                )
            )
            return {}

    service = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    launch, claimed = service.store.claim_launch(
        graph["graph"]["graph_id"],
        experiment["node"]["node_id"],
        expected_revision=experiment["graph"]["revision"],
        replicate=False,
        lease_owner="crashed-worker",
    )
    assert claimed is True
    with connect_workspace_db(workspace) as connection:
        connection.execute(
            "UPDATE research_launches SET lease_until = 0 WHERE launch_id = ?",
            (launch["launch_id"],),
        )

    asyncio.run(service.tick())
    recovered = service.store.get_launch(launch["launch_id"])
    assert recovered["status"] == "running"
    assert recovered["thread_id"].startswith("thread_rg_")
    assert submissions == [recovered["thread_id"]]
    service.reconcile_finished_child(
        child_thread_id=recovered["thread_id"],
        terminal_status="interrupted",
        run_id="run_waiting_for_approval",
    )
    assert service.store.get_launch(launch["launch_id"])["status"] == "running"

    # A worker restart sees the existing thread/message identity. A terminal
    # child without the required Result is a blocked experiment, never success.
    asyncio.run(service.tick())
    asyncio.run(service.tick())
    assert submissions == [recovered["thread_id"]]
    assert service.store.get_launch(launch["launch_id"])["status"] == "blocked"
    snapshot = service.store.get_snapshot(graph["graph"]["graph_id"])
    assert any(
        node["kind"] == "result"
        and "completed without recording" in node["body"]["summary"]
        for node in snapshot["nodes"]
    )


def test_service_launch_child_writeback_completes_result_with_sources(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    submissions: list[tuple[str, str, str]] = []
    writeback_payload: dict[str, object] = {}
    writeback_content: list[str] = []
    thread_store = ThreadStore(workspace=workspace, workspace_id="default")
    (system_root(workspace) / "runs" / "run_bound_child").mkdir(parents=True)

    class _Loop:
        async def submit(self, *, thread_id, payload):
            submissions.append((thread_id, payload.entrypoint, payload.text))
            thread_store.append_message(
                ThreadMessage(
                    id="msg_launch",
                    thread_id=thread_id,
                    role="user",
                    status="completed",
                    parts=[
                        MessagePart(
                            id="part_launch",
                            type="text",
                            text=payload.text,
                            status="completed",
                        )
                    ],
                )
            )
            with workspace_scope(workspace), toolcall_context(
                "tool_bound_result",
                context={
                    "thread_id": thread_id,
                    "entrypoint": payload.entrypoint,
                    "run_id": "run_bound_child",
                },
            ):
                content, _artifact = record_bound_research_result(
                    writeback_payload
                )
            writeback_content.append(content)
            return {}

    service = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    hypothesis_id = graph["nodes"][0]["node_id"]
    experiment = _ready_experiment(
        service,
        graph,
        hypothesis_id,
    )
    writeback_payload.update(
        {
            "title": "Completed measurement",
            "summary": "Response B increased reproducibly.",
            "judgments": [
                {
                    "hypothesis_node_id": hypothesis_id,
                    "relation": "supports",
                }
            ],
            "refs": [
                {
                    "ref_kind": "url",
                    "ref_id": "https://example.org/completed-run",
                }
            ],
        }
    )
    launched = asyncio.run(
        service.launch_experiment(
            graph["graph"]["graph_id"],
            experiment["node"]["node_id"],
            expected_revision=experiment["graph"]["revision"],
        )
    )
    child = launched["thread"]
    assert child.active_research_graph_id == graph["graph"]["graph_id"]
    assert child.research_focus_node_id == experiment["node"]["node_id"]
    assert len(submissions) == 1
    assert submissions[0][0] == child.thread_id
    assert submissions[0][1] == "experiment"
    assert "record a concise Result" in submissions[0][2]
    assert launched["launch"]["status"] == "completed"
    assert len(writeback_content) == 1
    assert "Recorded the bound experiment result" in writeback_content[0]

    service.reconcile_finished_child(
        child_thread_id=child.thread_id,
        terminal_status="done",
        run_id="run_completed",
    )
    snapshot = service.store.get_snapshot(graph["graph"]["graph_id"])
    results = [node for node in snapshot["nodes"] if node["kind"] == "result"]
    assert len(results) == 1
    result_id = results[0]["node_id"]
    assert {
        (ref["ref_kind"], ref["ref_id"])
        for ref in snapshot["refs"]
        if ref["node_id"] == result_id
    } == {
        ("run", "run_bound_child"),
        ("thread", child.thread_id),
        ("url", "https://example.org/completed-run"),
    }
    assert service.store.get_launch(launched["launch"]["launch_id"])[
        "status"
    ] == "completed"


def test_user_can_launch_a_planner_from_an_explicit_result_focus(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    submissions: list[tuple[str, str]] = []
    thread_store = ThreadStore(workspace=workspace, workspace_id="default")

    class _Loop:
        async def submit(self, *, thread_id, payload):
            submissions.append((thread_id, payload.text))
            thread_store.append_message(
                ThreadMessage(
                    id="msg_plan",
                    thread_id=thread_id,
                    role="user",
                    status="completed",
                    parts=[
                        MessagePart(
                            id="part_plan",
                            type="text",
                            text=payload.text,
                            status="completed",
                        )
                    ],
                )
            )
            return {}

    service = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    result = service.record_result(
        graph["graph"]["graph_id"],
        ResultCreateRequest(
            expected_revision=experiment["graph"]["revision"],
            summary="The first observation requires a follow-up.",
            experiment_node_id=experiment["node"]["node_id"],
            judgments=[],
        ),
    )

    planned = asyncio.run(
        service.plan_next_step(
            graph["graph"]["graph_id"],
            expected_revision=result["graph"]["revision"],
            focus_node_id=result["node"]["node_id"],
        )
    )
    child = planned["thread"]
    assert child.active_research_graph_id == graph["graph"]["graph_id"]
    assert child.research_focus_node_id == result["node"]["node_id"]
    assert child.entrypoint == "research"
    assert len(submissions) == 1
    assert submissions[0][0] == child.thread_id
    assert "bounded portfolio" in submissions[0][1]
    assert "scheduler" not in planned


def test_graph_identity_survives_workspace_rename_and_thread_removal(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, "before")
    service = ResearchGraphService(workspace=workspace, workspace_id="before")
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    thread = service.thread_store.create_thread(title="Temporary owner")
    service.bind_thread(thread.thread_id, graph_id=graph_id)
    service.thread_store.thread_dir(thread.thread_id).rename(
        workspace / "metadata" / "removed-thread"
    )

    renamed = tmp_path / "after"
    workspace.rename(renamed)
    reopened = ResearchGraphStore(renamed)
    assert reopened.get_graph(graph_id)["graph_id"] == graph_id


def test_context_is_explicitly_graph_scoped_and_node_bodies_are_strict(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    first = service.create_graph(
        GraphCreateRequest(
            question="Question alpha?",
            initial_hypotheses=[{"claim": "Alpha-only mechanism."}],
        )
    )
    second = service.create_graph(
        GraphCreateRequest(
            question="Question beta?",
            initial_hypotheses=[{"claim": "Beta-only mechanism."}],
        )
    )
    context = service.context_builder.build(
        first["graph"]["graph_id"],
        query="beta",
    )
    assert "Question alpha?" in context["markdown"]
    assert "Beta-only" not in context["markdown"]

    node = first["nodes"][0]
    with pytest.raises(ValueError, match="Extra inputs"):
        service.store.update_node(
            first["graph"]["graph_id"],
            node["node_id"],
            expected_revision=first["graph"]["revision"],
            expected_node_revision=node["revision"],
            title=node["title"],
            state="",
            body={
                **node["body"],
                "confidence": 0.9,
            },
        )
    assert service.store.get_graph(second["graph"]["graph_id"])[
        "question"
    ] == "Question beta?"
