from __future__ import annotations

import json
from pathlib import Path

import pytest

from catmaster.tools.base import ensure_project_space_layout, system_root, workspace_scope
from catmaster.research.knowledge_graph.models import (
    BoundExperimentCreateRequest,
    GraphCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.webui.thread_store import ThreadStore
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.misc.research_graph import (
    add_research_experiment,
    create_bound_research_experiment,
    create_research_graph,
    list_research_graphs,
    mark_bound_research_experiment_failed,
    mark_research_planning_no_change,
    query_research_graph_sql,
    record_bound_research_result,
    record_research_experiment_comparison,
    record_research_result,
    resume_bound_research_experiment,
    retract_bound_research_result,
    set_research_graph_focus,
    set_research_result_judgment,
    update_bound_research_result,
    update_research_graph_scope,
)
from catmaster.tools.registry import ToolRegistry


TOOL_NAMES = [
    "list_research_graphs",
    "create_research_graph",
    "query_research_graph_sql",
    "add_research_hypothesis",
    "add_research_experiment",
    "record_research_result",
    "set_research_result_judgment",
    "revise_research_claim",
    "record_research_disposition",
    "record_research_review",
    "mark_research_experiment_failed",
    "stage_research_plan",
    "record_research_experiment_comparison",
    "mark_research_planning_no_change",
    "set_research_graph_completion",
    "record_bound_research_result",
    "mark_bound_research_experiment_failed",
    "set_research_graph_focus",
    "create_bound_research_experiment",
    "update_bound_research_result",
    "resume_bound_research_experiment",
    "retract_bound_research_result",
    "update_research_graph_scope",
]


def _contains_null_schema(value) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "null":
            return True
        return any(_contains_null_schema(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_null_schema(item) for item in value)
    return False


def test_research_graph_tools_run_real_flow_without_raw_graph_artifacts(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    threads = ThreadStore(workspace=tmp_path, workspace_id="default")
    thread = threads.create_thread(title="Research", entrypoint="research")
    with workspace_scope(tmp_path), toolcall_context(
        "tool_research_flow",
        context={"thread_id": thread.thread_id, "entrypoint": "research"},
    ):
        content, created = create_research_graph(
            {
                "question": "Does catalyst A improve conversion?",
                "title": "Catalyst A",
                "completion_criterion": (
                    "A sourced matched comparison determines whether catalyst "
                    "A improves conversion."
                ),
                "orchestration_mode": "manual",
                "initial_hypotheses": [
                    {
                        "claim": "Catalyst A improves conversion.",
                        "rationale": "A changes the active site.",
                        "predictions": ["Conversion rises under matched conditions."],
                    }
                ],
            }
        )
        graph_id = created["data"]["graph_id"]
        assert "Created Research Graph" in content
        assert set(created["data"]) == {"graph_id", "revision", "changed"}

        _content, inspected = query_research_graph_sql(
            {"sql": "SELECT graph_id, revision FROM research_graphs"}
        )
        revision = inspected["data"]["revision"]
        assert inspected["data"]["rows"] == [
            {"graph_id": graph_id, "revision": revision}
        ]
        hypothesis_id = ResearchGraphStore(tmp_path).get_snapshot(graph_id)[
            "nodes"
        ][0]["node_id"]
        _content, experiment = add_research_experiment(
            {
                "graph_id": graph_id,
                "expected_revision": revision,
                "objective": "Measure conversion.",
                "plan_summary": "Compare A with a matched control.",
                "decision_rule": "Higher conversion supports the hypothesis.",
                "execution_lane": "experiment",
                "state": "ready",
                "tests_hypothesis_ids": [hypothesis_id],
                "depends_on_experiment_ids": [],
                "refs": [],
                "title": "Conversion measurement",
            }
        )
        revision = experiment["data"]["revision"]
        experiment_id = experiment["data"]["changed"]["node"]["node_id"]
        content, result = record_research_result(
            {
                "graph_id": graph_id,
                "expected_revision": revision,
                "summary": "Conversion increased reproducibly.",
                "experiment_node_id": experiment_id,
                "judgments": [
                    {
                        "hypothesis_node_id": hypothesis_id,
                        "relation": "supports",
                    }
                ],
                "refs": [{"ref_kind": "url", "ref_id": "https://example.org/result"}],
                "title": "Conversion result",
            }
        )
        assert "Recorded result" in content
        assert set(result["data"]) == {"graph_id", "revision", "changed"}
        revision = result["data"]["revision"]
        result_id = result["data"]["changed"]["node"]["node_id"]
        content, _judged = set_research_result_judgment(
            {
                "graph_id": graph_id,
                "expected_revision": revision,
                "result_node_id": result_id,
                "hypothesis_node_id": hypothesis_id,
                "relation": "opposes",
            }
        )
        assert "Result judgment set to opposes" in content
        snapshot = ResearchGraphStore(tmp_path).get_snapshot(graph_id)
        assert {
            edge["relation"]
            for edge in snapshot["edges"]
            if edge["source_node_id"] == result_id
            and edge["target_node_id"] == hypothesis_id
        } == {"opposes"}
        listed, artifact = list_research_graphs({"include_archived": False})
        assert graph_id in listed
        assert artifact["data"]["graph_count"] == 1


def test_catalog_and_query_report_the_same_accepted_binding_after_thread_switch(tmp_path):
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    old = service.create_graph(GraphCreateRequest(question="Earlier completed study"))["graph"]
    new = service.create_graph(GraphCreateRequest(question="Brand new question"))["graph"]
    service.store.update_graph(old["graph_id"], expected_revision=old["revision"], changes={"completed": True})
    thread = service.thread_store.create_thread(title="Research", entrypoint="research")
    service.bind_thread(thread.thread_id, graph_id=new["graph_id"])
    accepted = {"thread_id": thread.thread_id, "research_graph_id": old["graph_id"]}
    with workspace_scope(tmp_path), toolcall_context("accepted_turn", context=accepted):
        # The bound graph remains explicit even off-page or filtered out.
        for params in ({"limit": 1}, {"query": "Brand new"}, {"offset": 999}):
            content, artifact = list_research_graphs(params)
            assert artifact["data"]["bound_graph_id"] == old["graph_id"]
            assert old["graph_id"] in content.splitlines()[0]
        query_text, query = query_research_graph_sql({"sql": "SELECT graph_id, completed FROM research_graphs"})
        assert json.loads(query_text)["graph_id"] == old["graph_id"]
        assert query["data"]["rows"] == [{"graph_id": old["graph_id"], "completed": 1}]
    assert service.thread_store.get_thread(thread.thread_id).active_research_graph_id == new["graph_id"]

    # An explicitly unbound turn must not fall back to a later thread selection.
    with workspace_scope(tmp_path), toolcall_context("unbound_turn", context={**accepted, "research_graph_id": ""}):
        _content, artifact = list_research_graphs({})
        assert artifact["data"]["bound_graph_id"] == ""
        with pytest.raises(Exception, match="not bound"):
            query_research_graph_sql({"sql": "SELECT graph_id FROM research_graphs"})

    # Legacy/direct callers without a turn snapshot retain the thread fallback.
    with workspace_scope(tmp_path), toolcall_context("direct_thread", context={"thread_id": thread.thread_id}):
        _content, artifact = list_research_graphs({})
        assert artifact["data"]["bound_graph_id"] == new["graph_id"]
    with workspace_scope(tmp_path), toolcall_context("cli", context={"thread_id": "checkpoint_only"}):
        _content, artifact = list_research_graphs({})
        assert artifact["data"]["bound_graph_id"] == ""
        assert artifact["data"]["graph_count"] == 2

    tools = {t.name: t for t in ToolRegistry().as_langchain_tools(
        allowlist=["list_research_graphs", "query_research_graph_sql"],
        workspace=str(tmp_path), runtime_context=accepted)}
    from langchain_core.utils.function_calling import convert_to_openai_tool
    query_schema = convert_to_openai_tool(tools["query_research_graph_sql"])["function"]["parameters"]
    assert set(query_schema["properties"]) == {"sql"}
    assert query_schema["required"] == ["sql"]
    assert "bound_graph_id" in tools["list_research_graphs"].description


def test_pair_thread_query_stays_clean_after_outcome_writeback(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(
        GraphCreateRequest(
            question="Should the current ready Experiment run now?",
            initial_hypotheses=[{"claim": "A measurable effect is present."}],
        )
    )
    graph_id = created["graph"]["graph_id"]
    revision = created["graph"]["revision"]
    planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
    )
    assert claimed
    comparison_thread_id = "thread_rgcmp_clean_tool_test"
    service.thread_store.create_thread(
        thread_id=comparison_thread_id,
        title="Clean pair",
        entrypoint="research",
        meta={"internal_kind": "research_graph_experiment_comparison"},
    )
    service.thread_store.update_thread(
        comparison_thread_id,
        active_research_graph_id=graph_id,
    )
    selection = service._initial_selection_state(
        revision=revision,
        candidate_ids=["exp_clean_candidate"],
    )
    selection.update(
        {
            "status": "comparing",
            "active_comparison": {
                "purpose": "wait_forward",
                "candidate_a_id": "exp_clean_candidate",
                "candidate_b_id": "__wait__",
                "thread_id": comparison_thread_id,
                "sequence": 1,
            },
        }
    )
    service.store.set_planning_preview(
        graph_id,
        planning["planning_id"],
        start_revision=revision,
        preview={"selection": selection},
    )

    with workspace_scope(tmp_path), toolcall_context(
        "tool_pair_query_clean",
        context={
            "thread_id": comparison_thread_id,
            "entrypoint": "research",
            "research_graph_id": graph_id,
        },
    ):
        with pytest.raises(Exception, match="not research_planning"):
            query_research_graph_sql(
                {"sql": "SELECT planning_id FROM research_planning"}
            )
        _content, visible = query_research_graph_sql(
            {"sql": "SELECT kind, title FROM research_nodes"}
        )
        assert visible["data"]["rows"] == [
            {"kind": "hypothesis", "title": "A measurable effect is present."}
        ]

        record_research_experiment_comparison(
            {
                "outcome": "a",
                "reason": "The proposed measurement can resolve the current question.",
                "decisive_source_refs": [],
                "unresolved_tradeoff": "",
            }
        )
        with pytest.raises(Exception, match="not research_planning"):
            query_research_graph_sql(
                {"sql": "SELECT planning_id FROM research_planning"}
            )


def test_research_tool_final_schemas_are_non_nullable_and_minimal(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    registry = ToolRegistry()
    openai_tools = registry.as_openai_tools(allowlist=TOOL_NAMES)
    assert {tool["name"] for tool in openai_tools} == set(TOOL_NAMES)
    for tool in openai_tools:
        assert not _contains_null_schema(tool["parameters"])
        assert "metadata" not in tool["parameters"].get("properties", {})
        assert "source_thread_id" not in tool["parameters"].get("properties", {})
        properties = tool["parameters"].get("properties", {})
        if tool["name"] == "create_research_graph":
            assert "completion_criterion" not in tool["parameters"].get(
                "required",
                [],
            )
            assert "decision_preferences" not in tool["parameters"].get(
                "required",
                [],
            )
        if tool["name"] == "add_research_hypothesis":
            importance = properties["importance"]
            assert "not a computational recipe" in properties[
                "claim"
            ]["description"]
            assert importance["default"] == ""
            assert importance["enum"] == ["", "low", "medium", "high"]
            assert importance["type"] == "string"
            assert "confidence" in importance["description"]
        if tool["name"] == "add_research_experiment":
            assert "plan_summary" not in tool["parameters"].get("required", [])
            assert "decision_rule" not in tool["parameters"].get("required", [])
            assert properties["plan_summary"]["type"] == "string"
            assert properties["plan_summary"]["default"] == ""
            assert "expected_value" not in properties
            assert properties["estimated_compute_cost"]["enum"] == [
                "",
                "none",
                "low",
                "medium",
                "high",
            ]
        if tool["name"] == "stage_research_plan":
            assert "graph_id" not in properties
            assert "expected_revision" not in properties
            assert "evaluations" not in properties
            assert "maxItems" not in properties["hypotheses"]
            assert "maxItems" not in properties["experiments"]
            serialized = json.dumps(tool["parameters"])
            assert "proposal_id" not in serialized
            assert "recommended_target_id" not in serialized
            assert "tests_hypothesis_ids" not in serialized
            assert "depends_on_experiment_ids" not in serialized
            assert "recommended_route" in properties
            experiment_schema = tool["parameters"]["$defs"][
                "ResearchExperimentDraft"
            ]
            hypothesis_schema = tool["parameters"]["$defs"][
                "ResearchHypothesisDraft"
            ]
            assert "plan_summary" not in experiment_schema["required"]
            assert "decision_rule" not in experiment_schema["required"]
            assert "not a computational recipe" in hypothesis_schema[
                "properties"
            ]["claim"]["description"]
            assert experiment_schema["properties"]["plan_summary"]["type"] == "string"
            assert experiment_schema["properties"]["plan_summary"]["description"]
            assert "not computational recipes" in properties["experiments"][
                "description"
            ]
            assert "tests_hypotheses" in experiment_schema["properties"]
            assert "depends_on_experiments" in experiment_schema["properties"]
            assert "expected_value" not in experiment_schema["properties"]
            assert "blocking_reason" not in experiment_schema["properties"]
            assert '"external"' in serialized
        if tool["name"] == "query_research_graph_sql":
            assert set(properties) == {"sql"}
            assert tool["parameters"]["required"] == ["sql"]
            assert properties["sql"]["type"] == "string"
            sql_description = properties["sql"]["description"]
            assert "research_nodes(graph_id, node_id, kind, title, state, body_json" in sql_description
            assert "workspace_artifacts(artifact_id, thread_id, payload_json" in sql_description
            assert "json_extract(n.body_json, '$.claim')" in sql_description
            assert "json_extract(a.payload_json, '$.path')" in sql_description
            assert "sqlite_master" in sql_description
        if tool["name"] == "record_research_experiment_comparison":
            assert set(properties) == {
                "outcome",
                "reason",
                "decisive_source_refs",
                "unresolved_tradeoff",
            }
            assert set(tool["parameters"]["required"]) == {"outcome", "reason"}
            assert properties["outcome"]["enum"] == [
                "a",
                "b",
                "indistinguishable",
                "neither",
            ]
            assert properties["decisive_source_refs"]["type"] == "array"
            assert "decisive_source_refs" not in tool["parameters"]["required"]
            assert {
                "score",
                "confidence",
                "rank",
                "experiment_ids",
            }.isdisjoint(properties)
        if tool["name"] == "record_research_result":
            assert properties["experiment_node_id"]["type"] == "string"
            assert properties["experiment_node_id"]["default"] == ""
            assert "experiment_node_id" not in tool["parameters"].get(
                "required",
                [],
            )
            assert "global evidence grade" in properties["summary"]["description"]
            assert "evidence_level" not in properties
            assert "evidence_strength" not in properties
        if tool["name"] == "record_bound_research_result":
            assert "global evidence grade" in properties["summary"]["description"]
            assert "evidence_level" not in properties
            assert "evidence_strength" not in properties
        if tool["name"] in {"record_research_result", "record_bound_research_result", "update_bound_research_result"}:
            for field in ("methods", "conclusion"):
                assert properties[field]["type"] == "string"
                assert "anyOf" not in properties[field]
        if tool["name"] == "set_research_result_judgment":
            assert properties["relation"]["enum"] == [
                "supports",
                "opposes",
                "inconclusive",
                "unjudged",
            ]
        if tool["name"] == "stage_research_plan":
            assert "self_consistency" not in json.dumps(tool["parameters"])
        if tool["name"] in {
            "record_bound_research_result",
            "mark_bound_research_experiment_failed",
            "set_research_graph_focus",
            "create_bound_research_experiment",
            "update_bound_research_result",
            "resume_bound_research_experiment",
            "retract_bound_research_result",
            "update_research_graph_scope",
        }:
            assert {
                "graph_id",
                "thread_id",
                "experiment_node_id",
                "expected_revision",
            }.isdisjoint(tool["parameters"].get("properties", {}))
    langchain_tools = registry.as_langchain_tools(
        allowlist=TOOL_NAMES,
        workspace=str(tmp_path),
    )
    for tool in langchain_tools:
        assert not _contains_null_schema(tool.args_schema)


def test_bound_graph_focus_result_correction_resume_and_retraction_flow(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(
        GraphCreateRequest(
            question="Which surface state controls selectivity?",
            initial_hypotheses=[
                {
                    "claim": "The reconstructed surface controls selectivity.",
                    "predictions": ["A reconstruction marker tracks selectivity."],
                }
            ],
        )
    )
    graph_id = created["graph"]["graph_id"]
    hypothesis_id = created["nodes"][0]["node_id"]
    threads = ThreadStore(workspace=tmp_path, workspace_id=tmp_path.name)
    thread = threads.create_thread(
        thread_id="thread_bound_execution",
        title="Bound experiment",
        entrypoint="experiment",
    )
    threads.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id="",
    )
    run_id = "run_bound_execution"
    (system_root(tmp_path) / "runs" / run_id).mkdir(parents=True)
    runtime_context = {
        "thread_id": thread.thread_id,
        "entrypoint": "experiment",
        "research_graph_id": graph_id,
        "research_focus_node_id": "",
        "run_id": run_id,
    }

    with workspace_scope(tmp_path), toolcall_context(
        "tool_bound_graph_flow",
        context=runtime_context,
    ):
        with pytest.raises(Exception, match="explicit Experiment focus"):
            record_bound_research_result({"summary": "A premature observation."})

        _content, focused_hypothesis = set_research_graph_focus(
            {"node_id": hypothesis_id}
        )
        assert focused_hypothesis["data"]["focus"]["node_id"] == hypothesis_id
        assert focused_hypothesis["data"]["neighbors"] == []
        with pytest.raises(Exception, match="explicit Experiment focus"):
            record_bound_research_result({"summary": "Still not bound to an Experiment."})

        _content, adopted = create_bound_research_experiment(
            {
                "title": "Operando reconstruction test",
                "objective": "Test whether reconstruction tracks selectivity.",
                "plan_summary": "Measure structure and selectivity together.",
                "decision_rule": "A reversible correlation supports the hypothesis.",
                "tests_hypothesis_ids": [hypothesis_id],
                "depends_on_experiment_ids": [],
                "refs": [],
            }
        )
        experiment_id = adopted["data"]["changed"]["node"]["node_id"]
        assert adopted["data"]["changed"]["node"]["state"] == "ready"
        assert threads.get_thread(thread.thread_id).research_focus_node_id == experiment_id

        _content, recorded = record_bound_research_result(
            {
                "title": "Operando observation",
                "summary": "The marker and selectivity rose together.",
                "judgments": [],
                "refs": [{"ref_kind": "url", "ref_id": "https://example.org/raw"}],
            }
        )
        result_id = recorded["data"]["changed"]["node"]["node_id"]
        before_update = service.store.get_snapshot(graph_id)
        produces_before = [
            edge
            for edge in before_update["edges"]
            if edge["source_node_id"] == experiment_id
            and edge["target_node_id"] == result_id
            and edge["relation"] == "produces"
        ]
        assert len(produces_before) == 1
        result_refs = {
            (ref["ref_kind"], ref["ref_id"])
            for ref in before_update["refs"]
            if ref["node_id"] == result_id
        }
        assert ("thread", thread.thread_id) in result_refs
        assert ("run", run_id) in result_refs

        _content, corrected = update_bound_research_result(
            {
                "result_node_id": result_id,
                "summary": "The marker rose before selectivity under matched conditions.",
                "title": "",
                "refs": [{"ref_kind": "url", "ref_id": "https://example.org/analysis"}],
            }
        )
        corrected_node = corrected["data"]["changed"]["node"]
        assert corrected_node["revision"] == 2
        assert corrected_node["title"] == "Operando observation"
        after_update = service.store.get_snapshot(graph_id)
        assert produces_before == [
            edge
            for edge in after_update["edges"]
            if edge["source_node_id"] == experiment_id
            and edge["target_node_id"] == result_id
            and edge["relation"] == "produces"
        ]

        current_graph = service.store.get_graph(graph_id)
        current_result = service.store.get_node(graph_id, result_id)
        with pytest.raises(ValueError, match="not created by the current run"):
            service.retract_result(
                graph_id=graph_id,
                result_node_id=result_id,
                expected_revision=int(current_graph["revision"]),
                expected_node_revision=int(current_result["revision"]),
                reason="Wrong owner must not be allowed to retract this Result.",
                focused_experiment_node_id=experiment_id,
                owner_run_id="run_other_execution",
                owner_thread_id=thread.thread_id,
            )

        _content, retracted = retract_bound_research_result(
            {
                "result_node_id": result_id,
                "reason": "This was an analysis checkpoint, not a scientific Result.",
            }
        )
        assert retracted["data"]["deleted_result"]["node_id"] == result_id
        assert service.store.get_node(graph_id, experiment_id)["state"] == "ready"
        with pytest.raises(KeyError):
            service.store.get_node(graph_id, result_id)

        mark_bound_research_experiment_failed(
            {
                "reason": "The authorized instrument session expired.",
                "refs": [],
            }
        )
        assert service.store.get_node(graph_id, experiment_id)["state"] == "blocked"
        _content, resumed = resume_bound_research_experiment(
            {
                "reason": "A new authorized instrument session is active.",
                "refs": [],
            }
        )
        assert resumed["data"]["changed"]["node"]["state"] == "ready"
        assert resumed["data"]["changed"]["node"]["body"]["blocking_reason"] == ""

        _content, neighborhood = set_research_graph_focus(
            {"node_id": experiment_id}
        )
        assert {node["node_id"] for node in neighborhood["data"]["neighbors"]} == {
            hypothesis_id
        }
        _content, cleared = set_research_graph_focus({"node_id": ""})
        assert cleared["data"]["focus"] is None
        assert threads.get_thread(thread.thread_id).research_focus_node_id == ""

    history = service.store.list_mutation_history(graph_id=graph_id)
    changes = [item["payload"].get("change") for item in history]
    assert "result.updated" in changes
    assert "result.retracted" in changes
    assert "experiment.resumed" in changes
    retraction = next(
        item for item in history if item["payload"].get("change") == "result.retracted"
    )
    assert retraction["payload"]["details"]["reason"].startswith(
        "This was an analysis checkpoint"
    )


def test_bound_literature_review_records_standalone_result_without_experiment_focus(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(
        GraphCreateRequest(question="Which oxidation mechanism is reported for ZrC?")
    )
    graph_id = created["graph"]["graph_id"]
    threads = ThreadStore(workspace=tmp_path, workspace_id=tmp_path.name)
    thread = threads.create_thread(
        thread_id="thread_bound_literature",
        title="Bound literature review",
        entrypoint="literature_review",
    )
    threads.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id="",
    )
    run_id = "run_bound_literature"
    (system_root(tmp_path) / "runs" / run_id).mkdir(parents=True)

    with workspace_scope(tmp_path), toolcall_context(
        "tool_bound_literature_result",
        context={
            "thread_id": thread.thread_id,
            "entrypoint": "literature_review",
            "research_graph_id": graph_id,
            "research_focus_node_id": "",
            "research_launch_id": "",
            "run_id": run_id,
        },
    ):
        _content, recorded = record_bound_research_result(
            {
                "title": "Reported ZrC oxidation regime",
                "summary": (
                    "The selected source reports a temperature-dependent "
                    "oxidation regime."
                ),
                "judgments": [],
                "refs": [
                    {
                        "ref_kind": "url",
                        "ref_id": "https://example.org/zrc-oxidation",
                    }
                ],
            }
        )

    result_id = recorded["data"]["changed"]["node"]["node_id"]
    assert recorded["data"]["changed"]["experiment_node_id"] == ""
    snapshot = service.store.get_snapshot(graph_id)
    assert not any(
        edge["target_node_id"] == result_id and edge["relation"] == "produces"
        for edge in snapshot["edges"]
    )
    result_refs = {
        (ref["ref_kind"], ref["ref_id"])
        for ref in snapshot["refs"]
        if ref["node_id"] == result_id
    }
    assert ("thread", thread.thread_id) in result_refs
    assert ("run", run_id) in result_refs
    assert ("url", "https://example.org/zrc-oxidation") in result_refs

    with workspace_scope(tmp_path), toolcall_context(
        "tool_bound_literature_result_revision",
        context={
            "thread_id": thread.thread_id,
            "entrypoint": "literature_review",
            "research_graph_id": graph_id,
            "research_focus_node_id": "",
            "research_launch_id": "",
            "run_id": run_id,
        },
    ):
        _content, revised = update_bound_research_result(
            {
                "result_node_id": result_id,
                "summary": "The source reports two temperature-dependent oxidation regimes.",
                "title": "",
                "refs": [],
            }
        )
        assert revised["data"]["changed"]["node"]["revision"] == 2
        _content, retracted = retract_bound_research_result(
            {
                "result_node_id": result_id,
                "reason": "The selected passage was outside the requested material scope.",
            }
        )

    assert retracted["data"]["deleted_result"]["node_id"] == result_id
    with pytest.raises(KeyError):
        service.store.get_node(graph_id, result_id)


def test_bound_experiment_create_rolls_back_when_focus_binding_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(
        GraphCreateRequest(
            question="Which state is active?",
            initial_hypotheses=[{"claim": "State A is active."}],
        )
    )
    graph_id = created["graph"]["graph_id"]
    hypothesis_id = created["nodes"][0]["node_id"]
    thread = service.thread_store.create_thread(
        thread_id="thread_atomic_binding",
        title="Atomic binding",
        entrypoint="experiment",
    )
    service.thread_store.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id=hypothesis_id,
    )
    before = service.store.get_snapshot(graph_id)

    def _fail_binding(*args, **kwargs):
        raise RuntimeError("thread binding failed")

    monkeypatch.setattr(service.thread_store, "update_thread", _fail_binding)
    with pytest.raises(RuntimeError, match="thread binding failed"):
        service.add_bound_experiment(
            thread_id=thread.thread_id,
            graph_id=graph_id,
            expected_revision=before["graph"]["revision"],
            request=BoundExperimentCreateRequest(
                objective="Measure State A.",
                plan_summary="Acquire matched spectra.",
                decision_rule="The State A marker must track activity.",
                tests_hypothesis_ids=[hypothesis_id],
            ),
        )

    after = service.store.get_snapshot(graph_id)
    assert after["graph"]["revision"] == before["graph"]["revision"]
    assert after["nodes"] == before["nodes"]
    assert after["edges"] == before["edges"]


def test_only_research_entries_can_update_bound_graph_scope(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(GraphCreateRequest(question="Original question?"))
    graph_id = created["graph"]["graph_id"]
    thread = service.thread_store.create_thread(
        thread_id="thread_scope_owner",
        title="Scope owner",
        entrypoint="research",
    )
    service.thread_store.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id="",
    )
    base_context = {
        "thread_id": thread.thread_id,
        "research_graph_id": graph_id,
    }

    with workspace_scope(tmp_path), toolcall_context(
        "experiment_scope_attempt",
        context={**base_context, "entrypoint": "experiment"},
    ):
        with pytest.raises(Exception, match="Only Research"):
            update_research_graph_scope({"question": "Unauthorized change?"})

    with workspace_scope(tmp_path), toolcall_context(
        "research_scope_update",
        context={**base_context, "entrypoint": "research"},
    ):
        update_research_graph_scope(
            {
                "question": "Explicitly corrected question?",
                "completion_criterion": "One sourced discriminating Result.",
                "decision_preferences": "Prefer non-destructive measurements when equally decisive.",
            }
        )
    graph = service.store.get_graph(graph_id)
    assert graph["question"] == "Explicitly corrected question?"
    assert graph["completion_criterion"] == "One sourced discriminating Result."
    assert graph["decision_preferences"] == (
        "Prefer non-destructive measurements when equally decisive."
    )

    with workspace_scope(tmp_path), toolcall_context(
        "persistent_research_scope_update",
        context={**base_context, "entrypoint": "persistent_research"},
    ):
        update_research_graph_scope(
            {"question": "Persistent Research may steer this question?"}
        )
    graph = service.store.get_graph(graph_id)
    assert graph["question"] == "Persistent Research may steer this question?"


def test_create_research_graph_binds_host_injected_current_thread(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    threads = ThreadStore(workspace=tmp_path, workspace_id="default")
    thread = threads.create_thread(title="Research", entrypoint="research")
    registry = ToolRegistry()
    tool = next(
        item
        for item in registry.as_langchain_tools(
            allowlist=["create_research_graph"],
            workspace=str(tmp_path),
            runtime_context={
                "thread_id": thread.thread_id,
                "entrypoint": "research",
            },
        )
        if item.name == "create_research_graph"
    )

    tool.invoke(
        {
            "question": "Which pathway controls selectivity?",
            "completion_criterion": (
                "A sourced discriminating Result identifies which pathway "
                "controls selectivity."
            ),
            "initial_hypotheses": [
                {
                    "claim": "Pathway A controls selectivity.",
                    "rationale": "It has the lower barrier.",
                    "predictions": ["The A marker tracks selectivity."],
                }
            ],
        }
    )

    bound = threads.get_thread(thread.thread_id)
    assert bound.active_research_graph_id.startswith("graph_")
    assert bound.research_focus_node_id.startswith("hyp_")
    schema = tool.args_schema
    assert "thread_id" not in schema.get("properties", {})
    assert "active_research_graph_id" not in schema.get("properties", {})


def test_planning_child_stages_from_bound_thread_without_protocol_ids(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(
        GraphCreateRequest(
            question="Which surface state controls selectivity?",
            initial_hypotheses=[
                {
                    "claim": "The reconstructed surface controls selectivity.",
                    "predictions": ["An operando reconstruction marker tracks selectivity."],
                }
            ],
        )
    )
    graph_id = created["graph"]["graph_id"]
    revision = created["graph"]["revision"]
    planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
    )
    assert claimed is True
    threads = ThreadStore(workspace=tmp_path, workspace_id="default")
    thread = threads.create_thread(
        thread_id="thread_bound_planning",
        title="Plan next step",
        entrypoint="research",
    )
    threads.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
        research_focus_node_id=created["nodes"][0]["node_id"],
    )
    service.store.update_planning(
        graph_id,
        planning["planning_id"],
        start_revision=revision,
        status="attached",
        thread_id=thread.thread_id,
    )
    registry = ToolRegistry()
    tool = next(
        item
        for item in registry.as_langchain_tools(
            allowlist=["stage_research_plan"],
            workspace=str(tmp_path),
            runtime_context={
                "thread_id": thread.thread_id,
                "entrypoint": "research",
            },
        )
        if item.name == "stage_research_plan"
    )

    tool.invoke(
        {
            "hypotheses": [
                {
                    "claim": "A static ensemble controls selectivity.",
                    "predictions": [
                        "Selectivity remains stable when reconstruction is suppressed."
                    ],
                }
            ],
            "experiments": [
                {
                    "objective": "Test whether reconstruction tracks selectivity.",
                    "plan_summary": "Measure structure and selectivity together.",
                    "decision_rule": (
                        "A reversible marker-selectivity correlation supports the branch."
                    ),
                    "execution_lane": "literature_review",
                    "tests_hypotheses": [
                        "The reconstructed surface controls selectivity.",
                        "A static ensemble controls selectivity.",
                    ],
                }
            ],
            "recommended_route": "Test whether reconstruction tracks selectivity.",
            "recommendation_reason": (
                "It directly separates reconstruction from a static-site explanation."
            ),
        }
    )

    staged = service.store.find_planning_by_thread(thread.thread_id)
    assert staged is not None
    proposal = staged["preview"]["proposal"]
    hypothesis = proposal["hypotheses"][0]
    experiment = proposal["experiments"][0]
    assert proposal["recommended_target_id"] == experiment["proposal_id"]
    assert experiment["tests_hypothesis_ids"] == [
        created["nodes"][0]["node_id"],
        hypothesis["proposal_id"],
    ]
    assert hypothesis["proposal_id"].startswith(
        f"{planning['planning_id']}_hypothesis_"
    )
    assert experiment["proposal_id"].startswith(
        f"{planning['planning_id']}_experiment_"
    )
    assert "graph_id" not in tool.args_schema.get("properties", {})
    assert "expected_revision" not in tool.args_schema.get("properties", {})


def test_planning_child_can_explicitly_finish_without_a_new_branch(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    created = service.create_graph(
        GraphCreateRequest(question="Is another discriminating branch justified?")
    )
    graph_id = created["graph"]["graph_id"]
    revision = created["graph"]["revision"]
    planning, claimed = service.store.claim_planning(
        graph_id,
        expected_revision=revision,
    )
    assert claimed is True
    thread = service.thread_store.create_thread(
        title="Plan next step",
        entrypoint="research",
    )
    service.thread_store.update_thread(
        thread.thread_id,
        active_research_graph_id=graph_id,
    )
    service.store.update_planning(
        graph_id,
        planning["planning_id"],
        start_revision=revision,
        status="attached",
        thread_id=thread.thread_id,
    )

    with workspace_scope(tmp_path), toolcall_context(
        "tool_planning_no_change",
        context={"thread_id": thread.thread_id, "entrypoint": "research"},
    ):
        content, artifact = mark_research_planning_no_change(
            {"reason": "Existing Results already exhaust the distinguishable branches."}
        )

    assert "No new or existing ready branch" in content
    assert artifact["data"]["planning_id"] == planning["planning_id"]
    assert artifact["data"]["selection_started"] is False
    assert service.store.get_planning(graph_id, planning["planning_id"])["status"] == "no_change"
    assert service.store.planning_covers_current_graph(graph_id) is True
    preview = service.presentation(graph_id)["planning_preview"]
    assert preview["status"] == "no_change"
    registry_tool = next(
        item
        for item in ToolRegistry().as_langchain_tools(
            allowlist=["mark_research_planning_no_change"],
            workspace=str(tmp_path),
        )
        if item.name == "mark_research_planning_no_change"
    )
    assert set(registry_tool.args_schema.get("properties", {})) == {"reason"}
