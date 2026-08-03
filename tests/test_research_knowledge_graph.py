from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from catmaster.research.knowledge_graph.context import (
    ResearchGraphContextBuilder,
    runnable_frontier_ids,
)
from catmaster.research.knowledge_graph.planning import build_planning_preview
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
    GraphPatchRequest,
    HypothesisCreateRequest,
    ResearchExperimentEvaluationDraft,
    ResearchGraphPlanningProposal,
    ResultCreateRequest,
    ResultJudgmentSetRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.store import (
    ResearchGraphConflict,
    ResearchGraphStore,
)
from catmaster.storage import connect_workspace_db, workspace_database_path
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
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


def test_one_sentence_inputs_create_useful_drafts_and_ready_requires_details(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    created = service.create_graph(
        GraphCreateRequest(question="What controls the observed selectivity?")
    )
    graph_id = created["graph"]["graph_id"]
    assert created["graph"]["title"] == "What controls the observed selectivity?"
    assert created["graph"]["completion_criterion"].startswith(
        "Reach a defensible answer"
    )
    assert created["nodes"] == []

    hypothesis = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=created["graph"]["revision"],
            claim="The interfacial site controls selectivity.",
        ),
    )
    assert hypothesis["node"]["body"]["rationale"] == ""
    assert hypothesis["node"]["body"]["predictions"] == []

    draft = service.add_experiment(
        graph_id,
        ExperimentCreateRequest(
            expected_revision=hypothesis["graph"]["revision"],
            objective="Compare selectivity with and without the interface.",
            tests_hypothesis_ids=[hypothesis["node"]["node_id"]],
        ),
    )
    assert draft["node"]["state"] == "draft"
    assert draft["node"]["body"]["plan_summary"] == ""
    assert draft["node"]["body"]["decision_rule"] == ""

    with pytest.raises(ValueError, match="ready experiment requires"):
        service.add_experiment(
            graph_id,
            ExperimentCreateRequest(
                expected_revision=draft["graph"]["revision"],
                objective="Run an underspecified experiment.",
                state="ready",
            ),
        )
    assert service.store.get_graph(graph_id)["revision"] == draft["graph"]["revision"]

    result = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=draft["graph"]["revision"],
            summary="The collaborator observed a reversible selectivity shift.",
        ),
    )
    assert result["node"]["title"].startswith("The collaborator observed")


def test_store_migrates_retired_expected_value_and_stales_old_preview(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    graph_id = graph["graph"]["graph_id"]
    experiment_id = experiment["node"]["node_id"]
    planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=experiment["graph"]["revision"],
    )
    assert claimed is True
    old_body = dict(experiment["node"]["body"])
    old_body["expected_value"] = "high"
    with connect_workspace_db(workspace) as connection:
        connection.execute(
            """
            UPDATE research_nodes SET body_json = ?
            WHERE graph_id = ? AND node_id = ?
            """,
            (json.dumps(old_body), graph_id, experiment_id),
        )
        connection.execute(
            """
            UPDATE research_planning
            SET preview_json = ?
            WHERE planning_id = ?
            """,
            (
                json.dumps(
                    {
                        "proposal": {
                            "hypotheses": [],
                            "experiments": [{"expected_value": "high"}],
                        }
                    }
                ),
                planning["planning_id"],
            ),
        )
        connection.execute(
            """
            UPDATE schema_migrations SET version = 4
            WHERE component = 'research_knowledge_graph'
            """
        )

    migrated = ResearchGraphStore(workspace)
    snapshot = migrated.get_snapshot(graph_id)
    migrated_experiment = next(
        node for node in snapshot["nodes"] if node["node_id"] == experiment_id
    )
    assert "expected_value" not in migrated_experiment["body"]
    assert snapshot["graph"]["revision"] == experiment["graph"]["revision"] + 1
    migrated_planning = migrated.get_planning(graph_id, planning["planning_id"])
    assert migrated_planning["status"] == "stale"
    assert migrated_planning["preview"] == {}
    with connect_workspace_db(workspace) as connection:
        version = connection.execute(
            """
            SELECT version FROM schema_migrations
            WHERE component = 'research_knowledge_graph'
            """
        ).fetchone()["version"]
    assert version == 5


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


def test_sourced_external_observation_does_not_require_a_graph_experiment(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    hypothesis_id = graph["nodes"][0]["node_id"]

    recorded = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=graph["graph"]["revision"],
            title="Collaborator operando observation",
            summary=(
                "The collaborator observed a reversible interface band under "
                "reactive gas."
            ),
            judgments=[
                {
                    "hypothesis_node_id": hypothesis_id,
                    "relation": "inconclusive",
                }
            ],
            refs=[
                {
                    "ref_kind": "url",
                    "ref_id": "https://example.org/collaborator-observation",
                }
            ],
        ),
    )

    result_id = recorded["node"]["node_id"]
    snapshot = service.store.get_snapshot(graph_id)
    assert not any(
        edge["relation"] == "produces"
        and edge["target_node_id"] == result_id
        for edge in snapshot["edges"]
    )
    assert {
        (edge["source_node_id"], edge["relation"], edge["target_node_id"])
        for edge in snapshot["edges"]
    } >= {(result_id, "inconclusive", hypothesis_id)}
    assert {
        (ref["ref_kind"], ref["ref_id"])
        for ref in snapshot["refs"]
        if ref["node_id"] == result_id
    } == {("url", "https://example.org/collaborator-observation")}
    context = service.context_builder.build(
        graph_id,
        focus_node_id=result_id,
    )
    assert "reversible interface band" in context["markdown"]


def test_frontier_keeps_all_branches_but_auto_does_not_invent_a_selection(
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
                estimated_compute_cost=estimated_compute_cost,
            ),
        )

    low_branch = add_ready(
        graph,
        hypothesis_id=low_hypothesis["node_id"],
        title="Low hypothesis, high value",
        estimated_compute_cost="none",
    )
    high_expensive = add_ready(
        low_branch,
        hypothesis_id=high_hypothesis["node_id"],
        title="High hypothesis, high value, high compute",
        estimated_compute_cost="high",
    )
    high_cheap = add_ready(
        high_expensive,
        hypothesis_id=high_hypothesis["node_id"],
        title="High hypothesis, high value, low compute",
        estimated_compute_cost="low",
    )
    high_lower_value = add_ready(
        high_cheap,
        hypothesis_id=high_hypothesis["node_id"],
        title="High hypothesis, lower value",
        estimated_compute_cost="none",
    )

    frontier = [item["node_id"] for item in high_lower_value["graph"]["frontier"]]
    assert frontier == sorted(
        [
            high_cheap["node"]["node_id"],
            high_expensive["node"]["node_id"],
            high_lower_value["node"]["node_id"],
            low_branch["node"]["node_id"],
        ]
    )

    asyncio.run(service.tick())
    snapshot = service.store.get_snapshot(graph_id)
    active = [
        launch
        for launch in snapshot["launches"]
        if launch["status"] in {"claimed", "submitting", "running", "unknown"}
    ]
    assert active == []
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.active_research_graph_id == graph_id
        and thread.title.startswith("Plan next step:")
    )
    service.reconcile_finished_child(
        child_thread_id=planning_thread.thread_id,
        terminal_status="idle",
    )
    asyncio.run(service.tick())
    snapshot = service.store.get_snapshot(graph_id)
    active = [
        launch
        for launch in snapshot["launches"]
        if launch["status"] in {"claimed", "submitting", "running", "unknown"}
    ]
    assert active == []
    assert submissions == [planning_thread.thread_id]


def test_planning_write_boundary_keeps_parallel_branches_without_route_scores() -> None:
    schema = ResearchGraphPlanningProposal.model_json_schema()
    assert "maxItems" not in schema["properties"]["hypotheses"]
    assert "maxItems" not in schema["properties"]["experiments"]
    experiment_schema = schema["$defs"]["ResearchExperimentProposal"]
    assert "plan_summary" not in experiment_schema["required"]
    assert "decision_rule" not in experiment_schema["required"]
    encoded = json.dumps(schema)
    assert "self_consistency" not in encoded
    assert "llm_value" not in encoded
    assert "evaluations" not in schema["properties"]
    wide_proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "hypotheses": [
                {
                    "proposal_id": f"hypothesis {index}",
                    "claim": f"Mechanism {index} is active.",
                }
                for index in range(13)
            ],
            "experiments": [
                {
                    "proposal_id": f"experiment {index}",
                    "objective": f"Test mechanism {index}.",
                }
                for index in range(25)
            ],
        }
    )
    assert len(wide_proposal.hypotheses) == 13
    assert len(wide_proposal.experiments) == 25
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "hypotheses": [
                {"proposal_id": "hyp_a", "claim": "Mechanism A is active."},
                {"proposal_id": "hyp_b", "claim": "Mechanism B is active."},
            ],
        }
    )
    assert [item.proposal_id for item in proposal.hypotheses] == [
        "hyp_a",
        "hyp_b",
    ]
    with pytest.raises(ValueError, match="at most once"):
        ResultCreateRequest(
            expected_revision=1,
            summary="One observation.",
            judgments=[
                {"hypothesis_node_id": "hyp_same", "relation": "supports"},
                {"hypothesis_node_id": "hyp_same", "relation": "opposes"},
            ],
        )


def test_planning_staging_has_no_hidden_64_node_cap(
    tmp_path: Path,
) -> None:
    class _Loop:
        async def submit(self, **_kwargs):
            return {}

    service = ResearchGraphService(
        workspace=_workspace(tmp_path),
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    planned = asyncio.run(
        service.plan_next_step(
            graph["graph"]["graph_id"],
            expected_revision=graph["graph"]["revision"],
        )
    )
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "hypotheses": [
                {
                    "proposal_id": f"branch {index}",
                    "claim": f"Distinct mechanism {index} is active.",
                }
                for index in range(65)
            ]
        }
    )
    staged = service.stage_planning_proposal(
        graph["graph"]["graph_id"],
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planned["thread"].thread_id,
        proposal=proposal,
    )
    assert staged["staged"] == {"hypotheses": 65, "experiments": 0}
    assert len(service.store.get_snapshot(graph["graph"]["graph_id"])["nodes"]) == 1


def test_materialized_planning_experiment_can_remain_a_draft(
    tmp_path: Path,
) -> None:
    class _Loop:
        async def submit(self, **_kwargs):
            return {}

    service = ResearchGraphService(
        workspace=_workspace(tmp_path),
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    planned = asyncio.run(
        service.plan_next_step(
            graph_id,
            expected_revision=graph["graph"]["revision"],
        )
    )
    proposal_id = "实验草案 一"
    staged = service.stage_planning_proposal(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planned["thread"].thread_id,
        proposal=ResearchGraphPlanningProposal.model_validate(
            {
                "experiments": [
                    {
                        "proposal_id": proposal_id,
                        "objective": "先记录一个需要继续完善的表征思路。",
                    }
                ],
                "recommended_target_id": proposal_id,
                "recommendation_reason": "该思路值得保留，但执行细节尚未确定。",
            }
        ),
    )
    materialized = service.materialize_planning_proposal(
        graph_id,
        staged["planning_id"],
        expected_revision=graph["graph"]["revision"],
        proposal_id=proposal_id,
    )
    node_id = materialized["node_ids"][proposal_id]
    node = service.store.get_node(graph_id, node_id)
    assert node["state"] == "draft"
    assert node["body"]["plan_summary"] == ""
    assert node["body"]["decision_rule"] == ""
    assert materialized["next_experiment_node_id"] == ""


def test_auto_planning_leaves_an_incomplete_recommendation_provisional(
    tmp_path: Path,
) -> None:
    class _Loop:
        async def submit(self, **_kwargs):
            return {}

    service = ResearchGraphService(
        workspace=_workspace(tmp_path),
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which incomplete idea should be developed?",
            orchestration_mode="auto",
        )
    )
    graph_id = graph["graph"]["graph_id"]
    asyncio.run(service.tick())
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.title.startswith("Plan next step:")
    )
    staged = service.stage_planning_proposal(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        proposal=ResearchGraphPlanningProposal.model_validate(
            {
                "experiments": [
                    {
                        "proposal_id": "draft idea",
                        "objective": "Develop an underspecified measurement.",
                    }
                ],
                "recommended_target_id": "draft idea",
                "recommendation_reason": "It is promising but not yet runnable.",
            }
        ),
    )
    assert "materialized" not in staged
    assert service.store.get_snapshot(graph_id)["nodes"] == []
    preview_nodes = service.presentation(graph_id)["planning_preview"]["nodes"]
    assert len(preview_nodes) == 1
    assert preview_nodes[0]["provisional"] is True


def test_planning_preview_preserves_the_scientific_recommendation_without_rescoring(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = service.create_graph(
        GraphCreateRequest(
            question="What should run next?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "The durable mechanism is testable."}],
        )
    )
    durable_hypothesis_id = graph["nodes"][0]["node_id"]
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "hypotheses": [
                {
                    "proposal_id": "hyp_high_value",
                    "claim": "A high-value but not yet testable alternative.",
                    "importance": "high",
                }
            ],
            "experiments": [
                {
                    "proposal_id": "exp_concrete",
                    "objective": "Run the available discriminating measurement.",
                    "plan_summary": "Compare matched samples.",
                    "decision_rule": "The sign of the response distinguishes the branch.",
                    "tests_hypothesis_ids": [durable_hypothesis_id],
                    "estimated_compute_cost": "high",
                }
            ],
            "recommended_target_id": "exp_concrete",
            "recommendation_reason": "It is the available discriminating check.",
        }
    )
    snapshot = service.store.get_snapshot(graph["graph"]["graph_id"])
    automatic = build_planning_preview(snapshot, proposal)
    assert automatic["proposer_recommended_target_id"] == "exp_concrete"

    assert "available discriminating check" in automatic["summary"]


def test_planning_preview_projects_recommendation_without_persisting_scores(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    hypothesis_id = graph["nodes"][0]["node_id"]
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "hypotheses": [
                {
                    "proposal_id": "hyp_context",
                    "title": "Context-dependent mechanism",
                    "claim": "The mechanism changes with the sample environment.",
                    "rationale": "The literature reports environment-sensitive kinetics.",
                    "predictions": ["The response changes after stratification."],
                    "refs": [
                        {
                            "ref_kind": "url",
                            "ref_id": "https://example.org/context-mechanism",
                        }
                    ],
                }
            ],
            "experiments": [
                {
                    "proposal_id": "exp_context",
                    "title": "Stratified comparison",
                    "objective": "Compare matched environments.",
                    "plan_summary": "Run the same bounded measurement in two environments.",
                    "decision_rule": "A reproducible interaction distinguishes the branch.",
                    "tests_hypothesis_ids": ["hyp_context"],
                    "estimated_compute_cost": "low",
                },
                {
                    "proposal_id": "exp_repeat",
                    "title": "Undifferentiated repeat",
                    "objective": "Repeat the original measurement.",
                    "plan_summary": "Repeat without stratification.",
                    "decision_rule": "Record whether the original response repeats.",
                    "tests_hypothesis_ids": [hypothesis_id],
                    "estimated_compute_cost": "low",
                },
            ],
            "recommended_target_id": "exp_context",
            "recommendation_reason": (
                "Search evidence makes stratification discriminating."
            ),
        }
    )
    preview = build_planning_preview(
        service.store.get_snapshot(graph["graph"]["graph_id"]),
        proposal,
    )

    assert preview["proposer_recommended_target_id"] == "exp_context"
    assert {
        "proposer_recommended_proposal_id",
        "proposer_recommended_existing_node_id",
        "route_ids",
    }.isdisjoint(preview)
    assert all(
        "proposer_recommended" not in node and "planning_reason" not in node
        for node in preview["nodes"]
    )
    encoded = json.dumps(preview)
    assert "llm_value" not in encoded
    assert "visits" not in encoded
    assert not any(node.get("kind") == "result" for node in preview["nodes"])


def test_planning_does_not_invent_a_recommendation_from_branch_shape(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which equal-value verification should run first?",
            initial_hypotheses=[
                {"claim": "Mechanism A controls the response."},
                {"claim": "Mechanism B controls the response."},
            ],
        )
    )
    first_hypothesis_id, second_hypothesis_id = [
        node["node_id"] for node in graph["nodes"]
    ]
    experiments = [
        {
            "proposal_id": "exp_a_first",
            "objective": "Run the first check for mechanism A.",
            "plan_summary": "Run one bounded matched comparison.",
            "decision_rule": "The marker distinguishes the mechanism.",
            "tests_hypothesis_ids": [first_hypothesis_id],
        },
        {
            "proposal_id": "exp_a_second",
            "objective": "Run the second check for mechanism A.",
            "plan_summary": "Run one bounded matched comparison.",
            "decision_rule": "The marker distinguishes the mechanism.",
            "tests_hypothesis_ids": [first_hypothesis_id],
        },
        {
            "proposal_id": "exp_b_only",
            "objective": "Run the only check for mechanism B.",
            "plan_summary": "Run one bounded matched comparison.",
            "decision_rule": "The marker distinguishes the mechanism.",
            "tests_hypothesis_ids": [second_hypothesis_id],
        },
    ]
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "experiments": experiments,
        }
    )

    preview = build_planning_preview(
        service.store.get_snapshot(graph["graph"]["graph_id"]),
        proposal,
    )

    assert preview["proposer_recommended_target_id"] == ""
    assert "visits" not in json.dumps(preview)


def test_new_result_can_change_the_explained_route_without_saved_scores(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    hypothesis_id = graph["nodes"][0]["node_id"]
    first = _ready_experiment(
        service,
        graph,
        hypothesis_id,
        title="Adsorption-energy comparison",
    )
    second = _ready_experiment(
        service,
        first,
        hypothesis_id,
        title="Operando reconstruction probe",
    )
    first_id = first["node"]["node_id"]
    second_id = second["node"]["node_id"]

    before = build_planning_preview(
        service.store.get_snapshot(graph_id),
        ResearchGraphPlanningProposal.model_validate(
            {
                "recommended_target_id": first_id,
                "recommendation_reason": (
                    "The adsorption contrast is the best next discriminator."
                ),
            }
        ),
    )
    assert before["proposer_recommended_target_id"] == first_id

    result = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=second["graph"]["revision"],
            summary=(
                "A collaborator observed reversible restructuring under the "
                "reaction atmosphere."
            ),
        ),
    )
    after = build_planning_preview(
        service.store.get_snapshot(graph_id),
        ResearchGraphPlanningProposal.model_validate(
            {
                "recommended_target_id": second_id,
                "recommendation_reason": (
                    "The observed restructuring makes this probe decisive."
                ),
            }
        ),
        focus_node_id=result["node"]["node_id"],
    )

    assert after["proposer_recommended_target_id"] == second_id
    assert "restructuring makes this probe decisive" in after["summary"]
    durable = json.dumps(service.store.get_snapshot(graph_id))
    assert "llm_value" not in durable
    assert "visits" not in durable


def test_manual_planning_preview_stays_temporary_until_selected_route_is_materialized(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    class _Loop:
        async def submit(self, *, thread_id, payload):
            return {"thread_id": thread_id, "text": payload.text}

    service = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    planned = asyncio.run(
        service.plan_next_step(
            graph_id,
            expected_revision=graph["graph"]["revision"],
            focus_node_id=graph["nodes"][0]["node_id"],
        )
    )
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "hypotheses": [
                {
                    "proposal_id": "hyp_literature_branch",
                    "title": "Literature-grounded alternative",
                    "claim": "An alternative pathway controls response B.",
                    "rationale": "A public source reports the alternative pathway.",
                    "predictions": ["A pathway-specific marker tracks response B."],
                    "refs": [
                        {
                            "ref_kind": "url",
                            "ref_id": "https://example.org/alternative",
                        }
                    ],
                }
            ],
            "experiments": [
                {
                    "proposal_id": "exp_marker",
                    "title": "Measure the pathway marker",
                    "objective": "Measure the pathway-specific marker.",
                    "plan_summary": "Compare matched treated and control samples.",
                    "decision_rule": "Marker-response coupling supports the alternative.",
                    "tests_hypothesis_ids": ["hyp_literature_branch"],
                    "estimated_compute_cost": "low",
                    "refs": [
                        {
                            "ref_kind": "url",
                            "ref_id": "https://example.org/marker-method",
                        }
                    ],
                }
            ],
            "recommended_target_id": "exp_marker",
            "recommendation_reason": (
                "The cited marker directly separates the branches."
            ),
        }
    )
    staged = service.stage_planning_proposal(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planned["thread"].thread_id,
        proposal=proposal,
    )

    assert len(service.store.get_snapshot(graph_id)["nodes"]) == 1
    preview = service.presentation(graph_id)["planning_preview"]
    assert preview["proposer_recommended_target_id"] == "exp_marker"
    assert all(
        node["provisional"] is True
        for node in preview["nodes"]
    )
    assert "llm_value" not in json.dumps(preview)

    materialized = service.materialize_planning_proposal(
        graph_id,
        staged["planning_id"],
        expected_revision=graph["graph"]["revision"],
        proposal_id="exp_marker",
    )
    snapshot = service.store.get_snapshot(graph_id)
    assert len(snapshot["nodes"]) == 3
    assert {node["kind"] for node in snapshot["nodes"]} == {
        "hypothesis",
        "experiment",
    }
    assert materialized["next_experiment_node_id"]
    assert "planning_preview" not in service.presentation(graph_id)
    assert {
        (edge["relation"], edge["source_node_id"], edge["target_node_id"])
        for edge in snapshot["edges"]
    } >= {
        (
            "tests",
            materialized["node_ids"]["hyp_literature_branch"],
            materialized["node_ids"]["exp_marker"],
        )
    }


def test_result_driven_suggests_edge_appears_only_after_route_materialization(
    tmp_path: Path,
) -> None:
    class _Loop:
        async def submit(self, **_kwargs):
            return {}

    service = ResearchGraphService(
        workspace=_workspace(tmp_path),
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    result = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=graph["graph"]["revision"],
            summary="The observed response requires a distinct explanation.",
        ),
    )
    planned = asyncio.run(
        service.plan_next_step(
            graph_id,
            expected_revision=result["graph"]["revision"],
            focus_node_id=result["node"]["node_id"],
        )
    )
    staged = service.stage_planning_proposal(
        graph_id,
        expected_revision=result["graph"]["revision"],
        planning_thread_id=planned["thread"].thread_id,
        proposal=ResearchGraphPlanningProposal.model_validate(
            {
                "hypotheses": [
                    {
                        "proposal_id": "hyp_followup",
                        "claim": "A distinct reversible state controls the response.",
                    }
                ],
                "experiments": [
                    {
                        "proposal_id": "exp_followup",
                        "objective": "Distinguish the reversible state.",
                        "plan_summary": "Measure the state under matched cycling.",
                        "decision_rule": "Reversibility distinguishes the explanation.",
                        "tests_hypothesis_ids": ["hyp_followup"],
                    }
                ],
                "recommended_target_id": "exp_followup",
                "recommendation_reason": "It directly tests the new explanation.",
            }
        ),
    )
    preview = service.presentation(graph_id)["planning_preview"]
    assert all(edge["relation"] != "suggests" for edge in preview["edges"])
    assert all(edge["relation"] != "suggests" for edge in service.store.get_snapshot(graph_id)["edges"])

    materialized = service.materialize_planning_proposal(
        graph_id,
        staged["planning_id"],
        expected_revision=result["graph"]["revision"],
        proposal_id="exp_followup",
    )
    assert {
        (edge["source_node_id"], edge["relation"], edge["target_node_id"])
        for edge in service.store.get_snapshot(graph_id)["edges"]
    } >= {
        (
            result["node"]["node_id"],
            "suggests",
            materialized["node_ids"]["hyp_followup"],
        )
    }


def test_only_the_current_planning_preview_is_retained(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    revision = graph["graph"]["revision"]

    first, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
    )
    assert claimed is True
    service.store.set_planning_preview(
        graph_id,
        first["planning_id"],
        start_revision=revision,
        preview={"start_revision": revision, "summary": "first"},
    )
    service.store.update_planning(
        graph_id,
        first["planning_id"],
        start_revision=revision,
        status="no_change",
    )

    second, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
        allow_same_revision_after_no_change=True,
    )
    assert claimed is True
    service.store.set_planning_preview(
        graph_id,
        second["planning_id"],
        start_revision=revision,
        preview={"start_revision": revision, "summary": "second"},
    )

    with pytest.raises(KeyError):
        service.store.get_planning(graph_id, first["planning_id"])
    latest = service.store.latest_planning_preview(graph_id)
    assert latest is not None
    assert latest["planning_id"] == second["planning_id"]
    assert latest["preview"]["summary"] == "second"


def test_auto_planning_materializes_only_selected_branch_then_launches_one_experiment(
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
            question="Which mechanism should be tested?",
            completion_criterion="One discriminating experiment resolves the mechanisms.",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "Mechanism A controls the response."}],
        )
    )
    graph_id = graph["graph"]["graph_id"]
    hypothesis_id = graph["nodes"][0]["node_id"]
    asyncio.run(service.tick())
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.title.startswith("Plan next step:")
    )
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "experiments": [
                {
                    "proposal_id": "exp_discriminating",
                    "title": "Discriminating measurement",
                    "objective": "Measure the mechanism-specific observable.",
                    "plan_summary": "Run the bounded mechanism-specific measurement.",
                    "decision_rule": "The observable distinguishes A from its alternative.",
                    "tests_hypothesis_ids": [hypothesis_id],
                    "estimated_compute_cost": "low",
                },
                {
                    "proposal_id": "exp_low_value",
                    "title": "Low-value repeat",
                    "objective": "Repeat a nondiscriminating measurement.",
                    "plan_summary": "Repeat the original measurement.",
                    "decision_rule": "Record the repeated value.",
                    "tests_hypothesis_ids": [hypothesis_id],
                    "estimated_compute_cost": "high",
                },
            ],
            "recommended_target_id": "exp_discriminating",
            "recommendation_reason": (
                "This directly tests the mechanism-specific prediction."
            ),
        }
    )
    staged = service.stage_planning_proposal(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        proposal=proposal,
    )
    assert "materialized" not in staged
    assert not [
        node
        for node in service.store.get_snapshot(graph_id)["nodes"]
        if node["kind"] == "experiment"
    ]
    service.stage_planning_evaluation(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        evaluation=ResearchExperimentEvaluationDraft(
            experiment_ids=["exp_discriminating", "exp_low_value"],
            innovation_scores=[0.91, 0.18],
            conservative_scores=[0.83, 0.35],
            innovation_recommendation="exp_discriminating",
            conservative_recommendation="exp_discriminating",
            evaluation_memo="The discriminating measurement wins under both policies.",
        ),
    )

    service.reconcile_finished_child(
        child_thread_id=planning_thread.thread_id,
        terminal_status="idle",
    )
    asyncio.run(service.tick())
    materialized = service.store.get_planning(
        graph_id,
        staged["planning_id"],
    )["preview"]
    next_experiment_id = materialized["materialized_next_experiment_id"]
    assert next_experiment_id
    assert materialized["materialized_revision"] == (
        service.store.get_graph(graph_id)["revision"]
    )
    assert "evaluation" not in materialized
    assert "proposal" not in materialized
    asyncio.run(service.tick())
    active = [
        launch
        for launch in service.store.get_snapshot(graph_id)["launches"]
        if launch["status"] in {"claimed", "submitting", "running", "unknown"}
    ]
    assert len(active) == 1
    assert active[0]["experiment_node_id"] == next_experiment_id
    assert submissions == [planning_thread.thread_id, active[0]["thread_id"]]


def test_auto_scheduler_never_substitutes_a_materialized_dependency(
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
            question="Which step in the route should run?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "The route has a measurable intermediate."}],
        )
    )
    graph_id = graph["graph"]["graph_id"]
    hypothesis_id = graph["nodes"][0]["node_id"]
    asyncio.run(service.tick())
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.title.startswith("Plan next step:")
    )
    proposal = ResearchGraphPlanningProposal.model_validate(
        {
            "experiments": [
                {
                    "proposal_id": "exp_prerequisite",
                    "objective": "Prepare and verify the intermediate.",
                    "plan_summary": "Measure the intermediate under matched conditions.",
                    "decision_rule": "A resolved intermediate permits the downstream test.",
                    "tests_hypothesis_ids": [hypothesis_id],
                },
                {
                    "proposal_id": "exp_selected_downstream",
                    "objective": "Run the downstream discriminating test.",
                    "plan_summary": "Measure the downstream response after preparation.",
                    "decision_rule": "The response distinguishes the candidate mechanism.",
                    "tests_hypothesis_ids": [hypothesis_id],
                    "depends_on_experiment_ids": ["exp_prerequisite"],
                },
            ],
            "recommended_target_id": "exp_selected_downstream",
            "recommendation_reason": "The downstream test is the evaluator's target.",
        }
    )
    staged = service.stage_planning_proposal(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        proposal=proposal,
    )
    service.stage_planning_evaluation(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        evaluation=ResearchExperimentEvaluationDraft(
            experiment_ids=["exp_prerequisite", "exp_selected_downstream"],
            innovation_scores=[0.4, 0.9],
            conservative_scores=[0.5, 0.8],
            innovation_recommendation="exp_selected_downstream",
            conservative_recommendation="exp_selected_downstream",
        ),
    )
    service.reconcile_finished_child(
        child_thread_id=planning_thread.thread_id,
        terminal_status="idle",
    )

    asyncio.run(service.tick())
    receipt = service.store.get_planning(
        graph_id,
        staged["planning_id"],
    )["preview"]
    mapping = receipt["materialized_node_ids"]
    assert receipt["materialized_next_experiment_id"] == ""
    assert service._frontier_ids(service.store.get_snapshot(graph_id)) == [
        mapping["exp_prerequisite"]
    ]

    asyncio.run(service.tick())
    assert service.store.get_snapshot(graph_id)["launches"] == []
    assert len(submissions) == 2
    assert submissions[0] == planning_thread.thread_id
    assert submissions[1] != planning_thread.thread_id


def test_materialized_scheduler_receipt_is_stale_after_another_mutation(
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
            question="Can a materialized route survive an unrelated mutation?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "The selected route is revision-bound."}],
        )
    )
    graph_id = graph["graph"]["graph_id"]
    asyncio.run(service.tick())
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.title.startswith("Plan next step:")
    )
    service.stage_planning_proposal(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        proposal=ResearchGraphPlanningProposal.model_validate(
            {
                "experiments": [
                    {
                        "proposal_id": "exp_revision_bound",
                        "objective": "Run one revision-bound check.",
                        "plan_summary": "Measure the selected observable.",
                        "decision_rule": "The observable resolves the branch.",
                        "tests_hypothesis_ids": [graph["nodes"][0]["node_id"]],
                    }
                ],
                "recommended_target_id": "exp_revision_bound",
                "recommendation_reason": "It is the explicit current-revision route.",
            }
        ),
    )
    service.stage_planning_evaluation(
        graph_id,
        expected_revision=graph["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        evaluation=ResearchExperimentEvaluationDraft(
            experiment_ids=["exp_revision_bound"],
            innovation_scores=[0.8],
            conservative_scores=[0.8],
            innovation_recommendation="exp_revision_bound",
            conservative_recommendation="exp_revision_bound",
        ),
    )
    service.reconcile_finished_child(
        child_thread_id=planning_thread.thread_id,
        terminal_status="idle",
    )
    asyncio.run(service.tick())
    materialized_revision = service.store.get_graph(graph_id)["revision"]
    service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=materialized_revision,
            claim="A concurrent observation changes the route comparison.",
        ),
    )

    asyncio.run(service.tick())
    assert service.store.get_snapshot(graph_id)["launches"] == []
    assert len(submissions) == 2
    assert submissions[0] == planning_thread.thread_id
    assert submissions[1] != planning_thread.thread_id


def test_auto_scheduler_uses_the_conservative_evaluator_choice(
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
            question="Which ready experiment should run?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "The alternatives can be distinguished."}],
        )
    )
    hypothesis_id = graph["nodes"][0]["node_id"]
    innovative = _ready_experiment(
        service,
        graph,
        hypothesis_id,
        title="Risky high-information experiment",
    )
    conservative = _ready_experiment(
        service,
        innovative,
        hypothesis_id,
        title="Reliable discriminating experiment",
    )
    graph_id = graph["graph"]["graph_id"]
    asyncio.run(service.tick())
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.title.startswith("Plan next step:")
    )
    service.stage_planning_proposal(
        graph_id,
        expected_revision=conservative["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        proposal=ResearchGraphPlanningProposal(
            recommended_target_id=innovative["node"]["node_id"],
            recommendation_reason="The proposer prefers its potential information gain.",
        ),
    )
    candidate_ids = [
        innovative["node"]["node_id"],
        conservative["node"]["node_id"],
    ]
    service.stage_planning_evaluation(
        graph_id,
        expected_revision=conservative["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        evaluation=ResearchExperimentEvaluationDraft(
            experiment_ids=candidate_ids,
            innovation_scores=[0.94, 0.62],
            conservative_scores=[0.31, 0.88],
            innovation_recommendation=innovative["node"]["node_id"],
            conservative_recommendation=conservative["node"]["node_id"],
            evaluation_memo="The policies select different ready experiments.",
        ),
    )
    preview = service.presentation(graph_id)["planning_preview"]
    assert preview["innovation_recommendation"] == innovative["node"]["node_id"]
    assert preview["conservative_recommendation"] == conservative["node"]["node_id"]
    assert len(preview["experiment_evaluations"]) == 2
    assert all(
        "innovation_score" not in node and "conservative_score" not in node
        for node in preview["nodes"]
    )

    service.reconcile_finished_child(
        child_thread_id=planning_thread.thread_id,
        terminal_status="idle",
    )
    asyncio.run(service.tick())
    active = [
        launch
        for launch in service.store.get_snapshot(graph_id)["launches"]
        if launch["status"] in {"claimed", "submitting", "running", "unknown"}
    ]
    assert len(active) == 1
    assert active[0]["experiment_node_id"] == conservative["node"]["node_id"]


def test_planning_scores_become_invisible_and_cannot_launch_after_revision_change(
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
            question="Can a stale evaluation ever authorize execution?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "The evaluation must match the graph revision."}],
        )
    )
    ready = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    graph_id = graph["graph"]["graph_id"]
    asyncio.run(service.tick())
    planning_thread = next(
        thread
        for thread in service.thread_store.list_threads()
        if thread.title.startswith("Plan next step:")
    )
    service.stage_planning_proposal(
        graph_id,
        expected_revision=ready["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        proposal=ResearchGraphPlanningProposal(
            recommended_target_id=ready["node"]["node_id"],
            recommendation_reason="It is currently runnable.",
        ),
    )
    service.stage_planning_evaluation(
        graph_id,
        expected_revision=ready["graph"]["revision"],
        planning_thread_id=planning_thread.thread_id,
        evaluation=ResearchExperimentEvaluationDraft(
            experiment_ids=[ready["node"]["node_id"]],
            innovation_scores=[0.7],
            conservative_scores=[0.8],
            innovation_recommendation=ready["node"]["node_id"],
            conservative_recommendation=ready["node"]["node_id"],
        ),
    )
    changed = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=ready["graph"]["revision"],
            claim="A new observation changes the candidate comparison.",
        ),
    )
    assert changed["graph"]["revision"] > ready["graph"]["revision"]
    assert "planning_preview" not in service.presentation(graph_id)

    service.reconcile_finished_child(
        child_thread_id=planning_thread.thread_id,
        terminal_status="idle",
    )
    asyncio.run(service.tick())
    assert service.store.get_snapshot(graph_id)["launches"] == []
    assert len(submissions) == 2
    assert submissions[0] == planning_thread.thread_id
    assert submissions[1] != planning_thread.thread_id


def test_new_result_schedules_a_fresh_auto_route_evaluation(
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
            question="Which mechanism should guide the next verification?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "Mechanism A controls the response."}],
        )
    )
    graph_id = graph["graph"]["graph_id"]

    asyncio.run(service.tick())
    first_thread_id = submissions[-1]
    service.reconcile_finished_child(
        child_thread_id=first_thread_id,
        terminal_status="idle",
    )
    current = service.store.get_snapshot(graph_id)
    result = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=current["graph"]["revision"],
            summary=(
                "A collaborator observed a reversible phase change under "
                "reaction conditions."
            ),
        ),
    )

    asyncio.run(service.tick())

    assert len(submissions) == 2
    assert submissions[1] != first_thread_id
    planning_thread = service.thread_store.get_thread(submissions[1])
    assert planning_thread.research_focus_node_id == result["node"]["node_id"]


def test_completion_criterion_is_explicit_and_stops_auto_orchestration(
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
            question="Does A control B?",
            completion_criterion=(
                "A recorded discriminating Result with a traceable source "
                "answers whether A controls B."
            ),
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "A controls B."}],
        )
    )
    assert graph["graph"]["completion_criterion"].startswith(
        "A recorded discriminating Result"
    )
    with pytest.raises(ValueError, match="before it records a Result"):
        service.patch_graph(
            graph["graph"]["graph_id"],
            GraphPatchRequest(
                expected_revision=graph["graph"]["revision"],
                completed=True,
            ),
        )

    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    result = service.record_result(
        graph["graph"]["graph_id"],
        ResultCreateRequest(
            expected_revision=experiment["graph"]["revision"],
            summary="The discriminating observation supports A controlling B.",
            experiment_node_id=experiment["node"]["node_id"],
            judgments=[
                {
                    "hypothesis_node_id": graph["nodes"][0]["node_id"],
                    "relation": "supports",
                }
            ],
            refs=[{"ref_kind": "url", "ref_id": "https://example.org/result"}],
        ),
    )
    completed = service.patch_graph(
        graph["graph"]["graph_id"],
        GraphPatchRequest(
            expected_revision=result["graph"]["revision"],
            completed=True,
        ),
    )
    assert completed["graph"]["completed"] is True
    asyncio.run(service.tick())
    assert submissions == []
    context = service.context_builder.build(graph["graph"]["graph_id"])
    assert "Completion state: satisfied" in context["markdown"]


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


def test_result_judgment_is_replaceable_clearable_and_reopens_completion(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    hypothesis_id = graph["nodes"][0]["node_id"]
    recorded = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=graph["graph"]["revision"],
            summary="The observation is real, but its interpretation is pending.",
        ),
    )
    result_id = recorded["node"]["node_id"]
    completed = service.patch_graph(
        graph_id,
        GraphPatchRequest(
            expected_revision=recorded["graph"]["revision"],
            completed=True,
        ),
    )
    assert completed["graph"]["completed"] is True

    supported = service.set_result_judgment(
        graph_id,
        result_id,
        hypothesis_id,
        ResultJudgmentSetRequest(
            expected_revision=completed["graph"]["revision"],
            relation="supports",
        ),
    )
    assert supported["graph"]["completed"] is False
    assert {
        edge["relation"]
        for edge in supported["edges"]
        if edge["source_node_id"] == result_id
        and edge["target_node_id"] == hypothesis_id
    } == {"supports"}

    opposed = service.set_result_judgment(
        graph_id,
        result_id,
        hypothesis_id,
        ResultJudgmentSetRequest(
            expected_revision=supported["graph"]["revision"],
            relation="opposes",
        ),
    )
    assert {
        edge["relation"]
        for edge in opposed["edges"]
        if edge["source_node_id"] == result_id
        and edge["target_node_id"] == hypothesis_id
    } == {"opposes"}

    cleared = service.set_result_judgment(
        graph_id,
        result_id,
        hypothesis_id,
        ResultJudgmentSetRequest(
            expected_revision=opposed["graph"]["revision"],
            relation="unjudged",
        ),
    )
    assert not any(
        edge["source_node_id"] == result_id
        and edge["target_node_id"] == hypothesis_id
        and edge["relation"] in {"supports", "opposes", "inconclusive"}
        for edge in cleared["edges"]
    )


def test_completed_graph_reopens_only_for_scientific_change_and_archive_is_read_only(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    result = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=graph["graph"]["revision"],
            summary="A result sufficient to exercise lifecycle semantics.",
        ),
    )
    completed = service.patch_graph(
        graph_id,
        GraphPatchRequest(
            expected_revision=result["graph"]["revision"],
            completed=True,
        ),
    )
    sourced = service.add_ref(
        graph_id,
        expected_revision=completed["graph"]["revision"],
        node_id=result["node"]["node_id"],
        ref={"ref_kind": "url", "ref_id": "https://example.org/result"},
    )
    assert sourced["graph"]["completed"] is True

    expanded = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=sourced["graph"]["revision"],
            claim="A new observation motivates another mechanism.",
        ),
    )
    assert expanded["graph"]["completed"] is False

    archived = service.patch_graph(
        graph_id,
        GraphPatchRequest(
            expected_revision=expanded["graph"]["revision"],
            archived=True,
        ),
    )
    archived_revision = archived["graph"]["revision"]
    with pytest.raises(ValueError, match="archived"):
        service.add_hypothesis(
            graph_id,
            HypothesisCreateRequest(
                expected_revision=archived_revision,
                claim="This must not be written while archived.",
            ),
        )
    with pytest.raises(ValueError, match="archived"):
        service.add_ref(
            graph_id,
            expected_revision=archived_revision,
            node_id=result["node"]["node_id"],
            ref={"ref_kind": "url", "ref_id": "https://example.org/blocked"},
        )
    assert service.store.get_graph(graph_id)["revision"] == archived_revision

    restored = service.patch_graph(
        graph_id,
        GraphPatchRequest(
            expected_revision=archived_revision,
            archived=False,
        ),
    )
    assert restored["graph"]["archived"] is False


def test_blocked_experiment_records_reason_without_fabricating_a_result(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )

    blocked = service.mark_experiment_blocked(
        graph["graph"]["graph_id"],
        experiment["node"]["node_id"],
        expected_revision=experiment["graph"]["revision"],
        reason="The required operando cell is unavailable.",
    )

    blocked_node = next(
        node
        for node in blocked["nodes"]
        if node["node_id"] == experiment["node"]["node_id"]
    )
    assert blocked_node["state"] == "blocked"
    assert blocked_node["body"]["blocking_reason"] == (
        "The required operando cell is unavailable."
    )
    assert not any(node["kind"] == "result" for node in blocked["nodes"])
    assert not any(edge["relation"] == "produces" for edge in blocked["edges"])


def test_schema_migration_removes_only_exact_legacy_blocker_result(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace)
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    graph_id = graph["graph"]["graph_id"]
    experiment_id = experiment["node"]["node_id"]
    legacy_result_id = "res_legacy_blocker"
    user_result_id = "res_user_named_blocker"
    same_timestamp = time.time()

    with connect_workspace_db(workspace) as connection:
        connection.execute(
            """
            UPDATE research_nodes
            SET state = 'blocked', updated_at = ?
            WHERE graph_id = ? AND node_id = ?
            """,
            (same_timestamp, graph_id, experiment_id),
        )
        connection.execute(
            """
            INSERT INTO research_nodes (
                graph_id, node_id, kind, title, state, body_json,
                revision, created_at, updated_at
            ) VALUES (?, ?, 'result', 'Execution blocked', '', ?, 1, ?, ?)
            """,
            (
                graph_id,
                legacy_result_id,
                json.dumps({"summary": "The operando cell was unavailable."}),
                same_timestamp,
                same_timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO research_edges (
                graph_id, source_node_id, target_node_id, relation
            ) VALUES (?, ?, ?, 'produces')
            """,
            (graph_id, experiment_id, legacy_result_id),
        )
        connection.execute(
            """
            INSERT INTO research_refs (
                graph_id, node_id, ref_kind, ref_id
            ) VALUES (?, ?, 'thread', 'thread_legacy')
            """,
            (graph_id, legacy_result_id),
        )
        connection.execute(
            """
            INSERT INTO research_nodes (
                graph_id, node_id, kind, title, state, body_json,
                revision, created_at, updated_at
            ) VALUES (?, ?, 'result', 'Execution blocked', '', ?, 1, ?, ?)
            """,
            (
                graph_id,
                user_result_id,
                json.dumps({"summary": "A user-authored result with this title."}),
                same_timestamp + 1,
                same_timestamp + 1,
            ),
        )
        connection.execute(
            """
            INSERT INTO research_edges (
                graph_id, source_node_id, target_node_id, relation
            ) VALUES (?, ?, ?, 'produces')
            """,
            (graph_id, experiment_id, user_result_id),
        )
        connection.execute(
            """
            UPDATE schema_migrations
            SET version = 3
            WHERE component = 'research_knowledge_graph'
            """
        )

    migrated = ResearchGraphStore(workspace).get_snapshot(graph_id)
    migrated_experiment = next(
        node for node in migrated["nodes"] if node["node_id"] == experiment_id
    )
    assert migrated_experiment["body"]["blocking_reason"] == (
        "The operando cell was unavailable."
    )
    assert legacy_result_id not in {
        node["node_id"] for node in migrated["nodes"]
    }
    assert user_result_id in {node["node_id"] for node in migrated["nodes"]}
    assert {
        (ref["node_id"], ref["ref_kind"], ref["ref_id"])
        for ref in migrated["refs"]
    } >= {(experiment_id, "thread", "thread_legacy")}


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
    assert not any(
        field in built_context["markdown"]
        for field in forbidden - {"planning", "launches"}
    )
    assert web_view["graph"]["revision"] >= 1
    assert all("revision" in node for node in web_view["nodes"])


def test_partial_focus_snippet_defers_unrelated_nodes_to_bound_sql(
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
        focus_node_id=graph["nodes"][0]["node_id"],
    )
    assert target["node_id"] not in {
        node["node_id"] for node in context["presentation"]["nodes"]
    }
    assert context["presentation"]["partial"] is True
    assert "omitted_count" not in context["presentation"]
    assert "Question:" in context["markdown"]
    assert "explicitly partial" in context["markdown"]
    assert "Rare zirconium fingerprint" not in context["markdown"]
    assert not context["markdown"].lstrip().startswith("{")
    queried = ResearchGraphSQLQuery(workspace).execute(
        graph_id=graph_id,
        sql=(
            "SELECT node_id, title FROM research_nodes "
            "WHERE node_id = '" + target["node_id"] + "'"
        ),
    )
    assert queried["rows"] == [
        {"node_id": target["node_id"], "title": target["title"]}
    ]


def test_focus_snippet_keeps_conflicting_evidence_and_complete_frontier(
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
        focus_node_id=target_hypothesis_id,
    )
    selected_ids = [
        node["node_id"] for node in context["presentation"]["nodes"]
    ]
    assert target_hypothesis_id in selected_ids
    assert supporting["node"]["node_id"] in selected_ids
    assert opposing["node"]["node_id"] in selected_ids
    assert len(context["presentation"]["frontier_node_ids"]) == 16
    assert context["presentation"]["shown_count"] == len(selected_ids)
    assert all(node_id in context["markdown"] for node_id in selected_ids)
    assert "Complete runnable frontier" in context["markdown"]


def test_focus_source_handle_is_preserved_without_loading_source_body(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    source = "https://example.org/" + ("focus-evidence-" * 12)
    focused = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=graph["graph"]["revision"],
            summary="A focused result with a long but important source.",
            refs=[{"ref_kind": "url", "ref_id": source}],
        ),
    )
    revision = focused["graph"]["revision"]
    for index in range(18):
        added = service.add_hypothesis(
            graph_id,
            HypothesisCreateRequest(
                expected_revision=revision,
                claim=f"Distractor mechanism {index} with verbose context.",
            ),
        )
        revision = added["graph"]["revision"]

    context = service.context_builder.build(
        graph_id,
        focus_node_id=focused["node"]["node_id"],
    )
    markdown = context["markdown"]
    assert "## Focus sources" in markdown
    assert source in markdown
    assert markdown.count(source) == 1
    assert "explicitly partial" in markdown


def test_default_planning_focus_follows_the_latest_scientific_change(
    tmp_path: Path,
) -> None:
    service = ResearchGraphService(workspace=_workspace(tmp_path))
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    result = service.record_result(
        graph_id,
        ResultCreateRequest(
            expected_revision=graph["graph"]["revision"],
            summary="An older result should not permanently dominate focus.",
        ),
    )
    latest = service.add_hypothesis(
        graph_id,
        HypothesisCreateRequest(
            expected_revision=result["graph"]["revision"],
            claim="The newest scientific input is a follow-up hypothesis.",
            suggested_by_result_ids=[result["node"]["node_id"]],
        ),
    )
    snapshot = service.store.get_snapshot(graph_id)
    assert service._planning_focus(snapshot) == latest["node"]["node_id"]


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
        "preview_json",
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

    def claim(_marker: str):
        return ResearchGraphStore(workspace).claim_planning(
            graph_id,
            expected_revision=revision,
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
            # A real literature-planning turn can remain inside provider
            # backoff for more than 15 minutes without becoming abandoned.
            (time.time() - 1_200, planning_id),
        )
    recovered, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
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
        launch_id=launch["launch_id"],
    )
    assert service.store.get_launch(launch["launch_id"])["status"] == "running"
    service.reconcile_finished_child(
        child_thread_id=recovered["thread_id"],
        terminal_status="steered",
        run_id="run_waiting_for_approval",
        launch_id=launch["launch_id"],
    )
    assert service.store.get_launch(launch["launch_id"])["status"] == "running"

    # An idle child remains associated with the same unfinished launch so a
    # later turn can continue it without a duplicate submission.
    asyncio.run(service.tick())
    asyncio.run(service.tick())
    assert submissions == [recovered["thread_id"]]
    assert service.store.get_launch(launch["launch_id"])["status"] == "running"
    snapshot = service.store.get_snapshot(graph["graph"]["graph_id"])
    assert not any(node["kind"] == "result" for node in snapshot["nodes"])
    experiment_node = next(
        node
        for node in snapshot["nodes"]
        if node["node_id"] == experiment["node"]["node_id"]
    )
    assert experiment_node["state"] == "running"

    # The same child can still report a late real observation without a manual
    # graph-state edit.
    with workspace_scope(workspace), toolcall_context(
        "tool_late_result",
        context={
            "thread_id": recovered["thread_id"],
            "entrypoint": "experiment",
            "research_graph_id": graph["graph"]["graph_id"],
            "research_focus_node_id": experiment["node"]["node_id"],
            "research_launch_id": launch["launch_id"],
        },
    ):
        record_bound_research_result(
            {"summary": "The delayed measurement completed successfully."}
        )
    assert service.store.get_launch(launch["launch_id"])["status"] == "completed"
    snapshot = service.store.get_snapshot(graph["graph"]["graph_id"])
    assert any(
        node["kind"] == "result"
        and "delayed measurement" in node["body"]["summary"]
        for node in snapshot["nodes"]
    )


def test_service_launch_child_writeback_completes_result_with_sources(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    submissions: list[tuple[str, str, str]] = []
    writeback_payload: dict[str, object] = {}
    writeback_context: dict[str, str] = {}
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
            active_launch = service.store.find_active_launch_by_thread(thread_id)
            assert active_launch is not None
            with workspace_scope(workspace), toolcall_context(
                "tool_bound_result",
                context={
                    "thread_id": thread_id,
                    "entrypoint": payload.entrypoint,
                    "run_id": "run_bound_child",
                    "research_graph_id": writeback_context["graph_id"],
                    "research_focus_node_id": writeback_context[
                        "experiment_node_id"
                    ],
                    "research_launch_id": active_launch["launch_id"],
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
    writeback_context.update(
        {
            "graph_id": graph["graph"]["graph_id"],
            "experiment_node_id": experiment["node"]["node_id"],
        }
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
    assert "collect a concise Result" in submissions[0][2]
    assert "shared evidence judge" in submissions[0][2]
    assert launched["launch"]["status"] == "completed"
    assert len(writeback_content) == 1
    assert "Recorded the bound research result" in writeback_content[0]

    service.reconcile_finished_child(
        child_thread_id=child.thread_id,
        terminal_status="done",
        run_id="run_completed",
        launch_id=launched["launch"]["launch_id"],
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
    finished_launch = service.store.get_launch(launched["launch"]["launch_id"])
    assert finished_launch["status"] == "completed"
    assert finished_launch["run_id"] == "run_bound_child"


def test_bound_result_writeback_does_not_retry_a_concurrent_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    graph_id = graph["graph"]["graph_id"]
    thread = service.thread_store.create_thread(
        title="Concurrent bound writeback",
        entrypoint="experiment",
    )
    service.thread_store.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id=experiment["node"]["node_id"],
    )

    original_record_result = ResearchGraphService.record_result
    mutation_count = 0

    def race_record_result(self, target_graph_id, request, **kwargs):
        nonlocal mutation_count
        mutation_count += 1
        current = self.store.get_graph(target_graph_id)
        self.add_hypothesis(
            target_graph_id,
            HypothesisCreateRequest(
                expected_revision=current["revision"],
                claim="A concurrent observation changes the live graph.",
            ),
        )
        return original_record_result(self, target_graph_id, request, **kwargs)

    monkeypatch.setattr(ResearchGraphService, "record_result", race_record_result)
    with workspace_scope(workspace), toolcall_context(
        "tool_concurrent_result",
        context={
            "thread_id": thread.thread_id,
            "entrypoint": "experiment",
            "research_graph_id": graph_id,
            "research_focus_node_id": experiment["node"]["node_id"],
            "research_launch_id": "",
        },
    ), pytest.raises(CatMasterToolExecutionError, match="changed in another thread"):
        record_bound_research_result(
            {"summary": "This judgment was made against the previous revision."}
        )

    assert mutation_count == 1
    snapshot = service.store.get_snapshot(graph_id)
    assert not any(node["kind"] == "result" for node in snapshot["nodes"])


def test_bound_result_can_be_standalone_without_an_experiment_focus(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    graph = _seed_graph(service)
    graph_id = graph["graph"]["graph_id"]
    hypothesis_id = graph["nodes"][0]["node_id"]
    thread = service.thread_store.create_thread(
        title="Bound literature synthesis",
        entrypoint="literature_review",
    )

    with workspace_scope(workspace), toolcall_context(
        "tool_standalone_result",
        context={
            "thread_id": thread.thread_id,
            "entrypoint": "literature_review",
            "research_graph_id": graph_id,
            "research_focus_node_id": hypothesis_id,
            "research_launch_id": "",
        },
    ):
        record_bound_research_result(
            {
                "summary": "Matched literature evidence favors the proposed pathway.",
                "judgments": [
                    {
                        "hypothesis_node_id": hypothesis_id,
                        "relation": "supports",
                    }
                ],
                "refs": [
                    {
                        "ref_kind": "doi",
                        "ref_id": "10.1000/example",
                    }
                ],
            }
        )

    snapshot = service.store.get_snapshot(graph_id)
    result = next(node for node in snapshot["nodes"] if node["kind"] == "result")
    assert not any(
        edge["relation"] == "produces"
        and edge["target_node_id"] == result["node_id"]
        for edge in snapshot["edges"]
    )
    assert {
        (ref["ref_kind"], ref["ref_id"])
        for ref in snapshot["refs"]
        if ref["node_id"] == result["node_id"]
    } == {
        ("doi", "10.1000/example"),
        ("thread", thread.thread_id),
    }


def test_attached_result_without_a_launch_does_not_complete_another_launch(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    graph = _seed_graph(service)
    experiment = _ready_experiment(
        service,
        graph,
        graph["nodes"][0]["node_id"],
    )
    graph_id = graph["graph"]["graph_id"]
    experiment_id = experiment["node"]["node_id"]
    launch, claimed = service.store.claim_launch(
        graph_id,
        experiment_id,
        expected_revision=experiment["graph"]["revision"],
        replicate=False,
        lease_owner="worker-a",
    )
    assert claimed is True
    managed_thread = service.thread_store.create_thread(
        title="Managed execution",
        entrypoint="experiment",
    )
    service.store.update_launch(
        launch["launch_id"],
        status="running",
        thread_id=managed_thread.thread_id,
    )
    attached_thread = service.thread_store.create_thread(
        title="Attached analysis",
        entrypoint="experiment",
    )

    with workspace_scope(workspace), toolcall_context(
        "tool_attached_result",
        context={
            "thread_id": attached_thread.thread_id,
            "entrypoint": "experiment",
            "research_graph_id": graph_id,
            "research_focus_node_id": experiment_id,
            "research_launch_id": "",
        },
    ):
        record_bound_research_result(
            {"summary": "An independent attached analysis found the same trend."}
        )

    assert service.store.get_launch(launch["launch_id"])["status"] == "running"
    assert service.store.get_node(graph_id, experiment_id)["state"] == "has_results"


def test_runnable_frontier_ignores_priority_and_cost_fields() -> None:
    nodes = [
        {
            "node_id": "hyp_unknown",
            "kind": "hypothesis",
            "state": "",
            "body": {"importance": ""},
        },
        {
            "node_id": "hyp_low",
            "kind": "hypothesis",
            "state": "",
            "body": {"importance": "low"},
        },
        {
            "node_id": "exp_unknown",
            "kind": "experiment",
            "state": "ready",
            "body": {"estimated_compute_cost": ""},
        },
        {
            "node_id": "exp_low",
            "kind": "experiment",
            "state": "ready",
            "body": {
                "estimated_compute_cost": "high",
            },
        },
    ]
    edges = [
        {
            "source_node_id": "hyp_unknown",
            "target_node_id": "exp_unknown",
            "relation": "tests",
        },
        {
            "source_node_id": "hyp_low",
            "target_node_id": "exp_low",
            "relation": "tests",
        },
    ]

    assert runnable_frontier_ids(nodes, edges) == ["exp_low", "exp_unknown"]


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
    assert "hypothesis_proposer" in submissions[0][1]
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
