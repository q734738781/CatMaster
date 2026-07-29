from __future__ import annotations

from pathlib import Path

from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.webui.thread_store import ThreadStore
from catmaster.tools.misc.research_graph import (
    add_research_experiment,
    create_research_graph,
    inspect_research_graph,
    list_research_graphs,
    record_research_result,
)
from catmaster.tools.registry import ToolRegistry


TOOL_NAMES = [
    "list_research_graphs",
    "create_research_graph",
    "inspect_research_graph",
    "add_research_hypothesis",
    "add_research_experiment",
    "record_research_result",
    "mark_research_experiment_failed",
    "record_bound_research_result",
    "mark_bound_research_experiment_failed",
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
    with workspace_scope(tmp_path):
        content, created = create_research_graph(
            {
                "question": "Does catalyst A improve conversion?",
                "title": "Catalyst A",
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
        graph_id = created["data"]["graph"]["graph_id"]
        assert "Active Research Graph" in content
        assert "nodes" not in created["data"]

        _content, inspected = inspect_research_graph(
            {"graph_id": graph_id, "max_nodes": 24, "max_chars": 12_000}
        )
        revision = inspected["data"]["graph"]["revision"]
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
        revision = experiment["data"]["graph"]["revision"]
        experiment_id = experiment["data"]["focus_node_id"]
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
        assert "supporting evidence available" in content
        assert result["data"]["omitted_count"] == 0
        listed, artifact = list_research_graphs({"include_archived": False})
        assert graph_id in listed
        assert artifact["data"]["graph_count"] == 1


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
        if tool["name"] == "add_research_hypothesis":
            importance = properties["importance"]
            assert importance["default"] == "medium"
            assert importance["enum"] == ["low", "medium", "high"]
            assert importance["type"] == "string"
            assert "confidence" in importance["description"]
        if tool["name"] == "add_research_experiment":
            assert properties["expected_value"]["enum"] == [
                "low",
                "medium",
                "high",
            ]
            assert properties["estimated_compute_cost"]["enum"] == [
                "none",
                "low",
                "medium",
                "high",
            ]
        if tool["name"] in {
            "record_bound_research_result",
            "mark_bound_research_experiment_failed",
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
