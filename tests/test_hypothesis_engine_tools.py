from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from catmaster.research.hypothesis_engine.storage import engine_path, load_engine
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.misc.hypothesis_engine import (
    advance_hypothesis_campaign,
    extend_hypothesis_campaign,
    initialize_hypothesis_campaign,
    inspect_hypothesis_campaign,
    record_hypothesis_result,
)
from catmaster.tools.registry import ToolRegistry


TOOL_NAMES = [
    "initialize_hypothesis_campaign",
    "extend_hypothesis_campaign",
    "inspect_hypothesis_campaign",
    "advance_hypothesis_campaign",
    "record_hypothesis_result",
]


def _hypothesis(hypothesis_id: str) -> dict:
    return {
        "id": hypothesis_id,
        "claim": f"Mechanism {hypothesis_id} controls the result.",
        "rationale": f"Scientific rationale for {hypothesis_id}.",
        "predictions": [f"Observable predicted only by {hypothesis_id}."],
        "derived_from": [],
    }


def _action(
    action_id: str,
    targets: list[str],
    *,
    cost: str = "low",
    prerequisites: list[str] | None = None,
    task_suffix: str = "",
) -> dict:
    return {
        "id": action_id,
        "executor": "literature",
        "question": f"Which target survives {action_id}?",
        "task": f"Find the decisive primary result for {action_id}.{task_suffix}",
        "target_hypotheses": targets,
        "decision_rule": (
            "Outcome one supports the first target and opposes the second; "
            "missing discrimination is inconclusive."
        ),
        "prerequisite_action_ids": prerequisites or [],
        "information_value": "high",
        "cost": cost,
    }


def _effects(verdicts: dict[str, str]) -> list[dict]:
    return [
        {
            "hypothesis_id": hypothesis_id,
            "verdict": verdict,
            "reason": f"The returned result is {verdict} for {hypothesis_id}.",
        }
        for hypothesis_id, verdict in verdicts.items()
    ]


def _initialize_payload(thread_id: str) -> dict:
    return {
        "thread_id": thread_id,
        "question": "Which mechanism is supported?",
        "hypotheses": [
            _hypothesis("h1"),
            _hypothesis("h2"),
            _hypothesis("h-scope"),
        ],
        "actions": [_action("a-literature", ["h1", "h2"])],
    }


def test_tools_run_separated_propose_execute_judge_revise_flow(
    tmp_path: Path,
) -> None:
    thread_id = "thread-scientific-flow"
    with workspace_scope(tmp_path):
        content, initialized = initialize_hypothesis_campaign(
            _initialize_payload(thread_id)
        )

        assert "from hypothesis_proposer" in content
        assert "EXECUTION PACKET" not in content
        assert initialized["data"]["state"]["active_action_id"] == ""

        content, initialized = advance_hypothesis_campaign(
            {
                "thread_id": thread_id,
                "expected_revision": initialized["data"]["state"]["revision"],
                "action_id": "a-literature",
            }
        )
        assert "EXECUTION PACKET" in content
        assert "- delegate_to: litreview_agent" in content
        assert "- target hypotheses:" in content
        assert "- decision rule:" in content
        assert "evidence_judge" in content
        assert "Do not create a new hypothesis in the result call" in content
        assert initialized["data"]["state"]["active_action_id"] == "a-literature"
        assert engine_path(tmp_path / "files", thread_id).exists()

        first_revision = initialized["data"]["state"]["revision"]
        content, first_result = record_hypothesis_result(
            {
                "thread_id": thread_id,
                "expected_revision": first_revision,
                "action_id": "a-literature",
                "outcome": "completed",
                "evidence_summary": "The primary source supports h1 and opposes h2.",
                "source": "doi:10.1021/example",
                "effects": _effects({"h1": "supports", "h2": "opposes"}),
            }
        )

        assert "Recorded the independent evidence judgment" in content
        assert first_result["data"]["controller"]["status"] == (
            "needs_hypothesis_revision"
        )
        assert first_result["data"]["state"]["active_action_id"] == ""
        assert first_result["data"]["state"]["hypotheses"][0]["status"] == "supported"
        assert first_result["data"]["state"]["hypotheses"][1]["status"] == "rejected"
        assert first_result["data"]["state"]["hypotheses"][2]["status"] == "open"

        content, revised = extend_hypothesis_campaign(
            {
                "thread_id": thread_id,
                "expected_revision": first_result["data"]["state"]["revision"],
                "hypotheses": [],
                "actions": [
                    _action(
                        "a-scope",
                        ["h-scope"],
                        prerequisites=["a-literature"],
                    )
                ],
            }
        )
        assert "Applied hypothesis_proposer revision" in content
        assert "EXECUTION PACKET" not in content
        assert revised["data"]["state"]["active_action_id"] == ""

        content, revised = advance_hypothesis_campaign(
            {
                "thread_id": thread_id,
                "expected_revision": revised["data"]["state"]["revision"],
                "action_id": "a-scope",
            }
        )
        assert "EXECUTION PACKET" in content

        final_content, final = record_hypothesis_result(
            {
                "thread_id": thread_id,
                "expected_revision": revised["data"]["state"]["revision"],
                "action_id": "a-scope",
                "outcome": "completed",
                "evidence_summary": "The source supports only the stated system scope.",
                "source": "doi:10.1021/example",
                "effects": _effects({"h-scope": "supports"}),
            }
        )

        assert "EXECUTION PACKET" not in final_content
        assert final["data"]["controller"]["status"] == "complete"
        persisted = load_engine(tmp_path / "files", thread_id)
        assert len(persisted.state.evidence) == 2
        assert persisted.state.revision == 5
        assert not hasattr(persisted.state, "runs")
        assert not hasattr(persisted.state, "budget")


def test_result_tool_forbids_inline_hypothesis_generation(tmp_path: Path) -> None:
    thread_id = "thread-no-inline-branch"
    with workspace_scope(tmp_path):
        _, initialized = initialize_hypothesis_campaign(
            _initialize_payload(thread_id)
        )
        with pytest.raises(
            CatMasterToolExecutionError,
            match="new_hypotheses",
        ):
            record_hypothesis_result(
                {
                    "thread_id": thread_id,
                    "action_id": "a-literature",
                    "outcome": "completed",
                    "evidence_summary": "Result.",
                    "source": "doi:test",
                    "effects": _effects(
                        {"h1": "supports", "h2": "opposes"}
                    ),
                    "new_hypotheses": [_hypothesis("h4")],
                }
            )
        persisted = load_engine(tmp_path / "files", thread_id)
        assert persisted.state.revision == initialized["data"]["state"]["revision"]
        assert len(persisted.state.hypotheses) == 3


def test_high_cost_tool_action_uses_the_same_explicit_selection_contract(
    tmp_path: Path,
) -> None:
    thread_id = "thread-interactive"
    payload = _initialize_payload(thread_id)
    payload["actions"] = [_action("a-expensive", ["h1", "h2"], cost="high")]
    with workspace_scope(tmp_path):
        content, initialized = initialize_hypothesis_campaign(payload)
        assert "EXECUTION PACKET" not in content
        assert initialized["data"]["controller"]["phase"] == "ready"

        with pytest.raises(
            CatMasterToolExecutionError,
            match="validation error",
        ):
            advance_hypothesis_campaign(
                {
                    "thread_id": thread_id,
                    "action_id": "",
                }
            )

        content, advanced = advance_hypothesis_campaign(
            {
                "thread_id": thread_id,
                "action_id": "a-expensive",
            }
        )
        assert "EXECUTION PACKET" in content
        assert advanced["data"]["controller"]["active_packet"]["cost"] == "high"


def test_failure_records_only_scientific_failure_and_moves_on(
    tmp_path: Path,
) -> None:
    thread_id = "thread-failure"
    payload = _initialize_payload(thread_id)
    payload["actions"].append(
        _action("z-alternative", ["h1", "h2"], task_suffix=" Independent route.")
    )
    with workspace_scope(tmp_path):
        _, initialized = initialize_hypothesis_campaign(payload)
        _, initialized = advance_hypothesis_campaign(
            {
                "thread_id": thread_id,
                "expected_revision": initialized["data"]["state"]["revision"],
                "action_id": "a-literature",
            }
        )
        _, failed = record_hypothesis_result(
            {
                "thread_id": thread_id,
                "expected_revision": initialized["data"]["state"]["revision"],
                "action_id": "a-literature",
                "outcome": "failed",
                "failure_reason": "publisher endpoint unavailable",
            }
        )

        assert failed["data"]["state"]["actions"][0]["status"] == "failed"
        assert failed["data"]["state"]["actions"][0]["failure_reason"] == (
            "publisher endpoint unavailable"
        )
        assert failed["data"]["state"]["evidence"] == []
        assert failed["data"]["state"]["active_action_id"] == ""
        assert failed["data"]["controller"]["recommended_action_id"] == "z-alternative"


def test_controller_tools_expose_science_not_autopilot_state(tmp_path: Path) -> None:
    thread_id = "thread-controller-boundary"
    with workspace_scope(tmp_path):
        _, initialized = initialize_hypothesis_campaign(
            _initialize_payload(thread_id)
        )

        state = initialized["data"]["state"]
        assert "mode" not in state
        assert "paused" not in state
        assert "autopilot" not in state
        assert set(TOOL_NAMES) == {
            "initialize_hypothesis_campaign",
            "extend_hypothesis_campaign",
            "inspect_hypothesis_campaign",
            "advance_hypothesis_campaign",
            "record_hypothesis_result",
        }


def test_inspection_does_not_mutate_revision(tmp_path: Path) -> None:
    thread_id = "thread-inspect"
    with workspace_scope(tmp_path):
        _, initialized = initialize_hypothesis_campaign(
            _initialize_payload(thread_id)
        )
        revision = initialized["data"]["state"]["revision"]

        _, inspected = inspect_hypothesis_campaign({"thread_id": thread_id})

        assert inspected["data"]["state"]["revision"] == revision
        assert inspected["data"]["controller"]["recommended_action_id"] == (
            "a-literature"
        )


def test_stale_map_revision_is_rejected_before_mutation(tmp_path: Path) -> None:
    thread_id = "thread-stale-map"
    with workspace_scope(tmp_path):
        _, initialized = initialize_hypothesis_campaign(
            _initialize_payload(thread_id)
        )
        revision = initialized["data"]["state"]["revision"]
        extend_hypothesis_campaign(
            {
                "thread_id": thread_id,
                "expected_revision": revision,
                "hypotheses": [],
                "actions": [
                    _action(
                        "a-independent",
                        ["h1", "h2"],
                        task_suffix=" Use an independent route.",
                    )
                ],
            }
        )

        with pytest.raises(
            CatMasterToolExecutionError,
            match=r"stale campaign revision: expected 0, current 1",
        ):
            advance_hypothesis_campaign(
                {
                    "thread_id": thread_id,
                    "expected_revision": revision,
                    "action_id": "a-literature",
                }
            )

        persisted = load_engine(tmp_path / "files", thread_id)
        assert persisted.state.revision == 1
        assert persisted.state.active_action_id == ""


def test_langchain_surface_returns_scientific_packet_to_parent_agent(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    tools = {
        item.name: item
        for item in registry.as_langchain_tools(
            allowlist=[
                "initialize_hypothesis_campaign",
                "advance_hypothesis_campaign",
            ],
            workspace=str(tmp_path),
            audience="research_specialist",
        )
    }
    initialize_tool = tools["initialize_hypothesis_campaign"]
    payload = _initialize_payload("thread-langchain-surface")

    initialized = initialize_tool.invoke(
        {
            "type": "tool_call",
            "name": initialize_tool.name,
            "id": "call-initialize",
            "args": payload,
        }
    )
    assert isinstance(initialized, ToolMessage)

    advance_tool = tools["advance_hypothesis_campaign"]
    result = advance_tool.invoke(
        {
            "type": "tool_call",
            "name": advance_tool.name,
            "id": "call-advance",
            "args": {
                "thread_id": "thread-langchain-surface",
                "expected_revision": initialized.artifact["data"]["state"]["revision"],
                "action_id": "a-literature",
            },
        }
    )

    assert isinstance(result, ToolMessage)
    assert "EXECUTION PACKET" in str(result.content)
    assert "rationale:" in str(result.content)
    assert "predictions:" in str(result.content)
    packet = result.artifact["data"]["controller"]["active_packet"]
    assert packet["action_id"] == "a-literature"
    assert "run_id" not in packet


def _null_markers(value, path: str = "$") -> list[str]:
    markers: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "null":
            markers.append(path)
        if value.get("default", object()) is None:
            markers.append(f"{path}.default")
        for key in ("anyOf", "oneOf"):
            variants = value.get(key)
            if isinstance(variants, list) and any(
                isinstance(item, dict) and item.get("type") == "null"
                for item in variants
            ):
                markers.append(f"{path}.{key}")
        for key, item in value.items():
            markers.extend(_null_markers(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            markers.extend(_null_markers(item, f"{path}[{index}]"))
    return markers


def test_final_agent_tool_schemas_are_lean_tagged_and_nonnullable() -> None:
    registry = ToolRegistry()
    openai_tools = {
        item["name"]: item
        for item in registry.as_openai_tools(allowlist=TOOL_NAMES)
    }
    langchain_tools = {
        item.name: item
        for item in registry.as_langchain_tools(allowlist=TOOL_NAMES)
    }

    assert set(openai_tools) == set(TOOL_NAMES)
    assert set(langchain_tools) == set(TOOL_NAMES)
    for name in TOOL_NAMES:
        openai_tool = openai_tools[name]
        assert openai_tool["description"].startswith("[research/control]")
        assert not _null_markers(openai_tool["parameters"]), name
        args_schema = langchain_tools[name].args_schema
        if hasattr(args_schema, "model_json_schema"):
            args_schema = args_schema.model_json_schema()
        assert not _null_markers(args_schema), name

    initialize_schema = openai_tools["initialize_hypothesis_campaign"]["parameters"]
    action_ref = initialize_schema["properties"]["actions"]["items"]["$ref"]
    action_definition = initialize_schema["$defs"][action_ref.rsplit("/", 1)[-1]]
    assert set(action_definition["properties"]) == {
        "id",
        "executor",
        "question",
        "task",
        "target_hypotheses",
        "decision_rule",
        "prerequisite_action_ids",
        "information_value",
        "cost",
    }
    hypothesis_ref = initialize_schema["properties"]["hypotheses"]["items"]["$ref"]
    hypothesis_definition = initialize_schema["$defs"][
        hypothesis_ref.rsplit("/", 1)[-1]
    ]
    assert set(hypothesis_definition["properties"]) == {
        "id",
        "claim",
        "rationale",
        "predictions",
        "derived_from",
    }
    record_properties = openai_tools["record_hypothesis_result"][
        "parameters"
    ]["properties"]
    assert "new_hypotheses" not in record_properties
    assert "new_actions" not in record_properties
    assert "actual_resource_usage" not in record_properties
    assert "provenance_ref" not in record_properties
