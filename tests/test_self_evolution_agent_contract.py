from __future__ import annotations

import json
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool

from catmaster.runtime.self_evolution.agents import (
    _prepare_skill_tool,
    _OptionalResultMiddleware,
)
from catmaster.runtime.self_evolution.models import (
    ProposerResult,
    ReflectionBatch,
    ReflectionResult,
    ReviewerResult,
    SKILL_GROUPS,
)
from catmaster.runtime.skills.catalog import CatMasterSkillsRuntime, SkillCatalog
from catmaster.runtime.skills.roots import ACTIVE_SKILL_GROUPS
from catmaster.specialists.runtime import _SKILL_GROUPS


def _agent_schema(
    model_type: type[ReflectionResult] | type[ReflectionBatch] | type[ProposerResult] | type[ReviewerResult],
) -> dict:
    tool = _OptionalResultMiddleware(model_type).tools[0]
    return convert_to_openai_tool(tool)["function"]["parameters"]


def test_reflection_schema_is_small_nonnullable_and_has_no_scores_or_routing_metadata() -> None:
    schema = _agent_schema(ReflectionResult)
    schema_text = json.dumps(schema, sort_keys=True)
    properties = schema["properties"]

    assert '"type": "null"' not in schema_text
    assert set(properties) == {
        "kind",
        "group",
        "name",
        "change",
        "evidence_refs",
        "rationale",
    }
    assert properties["kind"]["enum"] == [
        "no_change",
        "execution_lapse",
        "workspace_preference",
        "skill_revision",
        "skill_discovery",
    ]
    assert not {
        "confidence",
        "score",
        "route_hint",
        "supporting_ids",
        "counterexample_ids",
        "embedding",
    } & properties.keys()


def test_reflection_batch_schema_is_nonnullable_and_supports_multiple_items() -> None:
    schema = _agent_schema(ReflectionBatch)
    schema_text = json.dumps(schema, sort_keys=True)

    assert '"type": "null"' not in schema_text
    assert set(schema["properties"]) == {"items"}
    assert schema["properties"]["items"]["type"] == "array"
    assert "maxItems" not in schema["properties"]["items"]

    batch = ReflectionBatch(
        items=[
            ReflectionResult(
                kind="skill_revision",
                group="materials_worker",
                name="same-owner",
                change=f"Independent change {index}",
                evidence_refs=[f"run:one#event:{index}"],
            )
            for index in (1, 2)
        ]
    )
    assert len(batch.items) == 2


def test_proposer_agent_schema_exposes_only_the_bounded_delta() -> None:
    schema = _agent_schema(ProposerResult)
    schema_text = json.dumps(schema, sort_keys=True)
    properties = schema["properties"]

    assert '"type": "null"' not in schema_text
    assert {
        "delta_operation",
        "applicability_boundary",
        "non_applicability",
        "expected_step_change",
    } <= properties.keys()
    assert properties["delta_operation"]["type"] == "string"
    assert "enum" not in properties["delta_operation"]
    assert "evaluation_cases" not in properties
    assert "EvaluationCase" not in schema.get("$defs", {})
    assert properties["action"]["enum"] == ["defer", "ignore", "memory", "skill"]

    parsed = ProposerResult.model_validate(
        {
            "action": "skill",
            "group": None,
            "name": None,
            "rationale": None,
            "delta_operation": None,
            "applicability_boundary": None,
            "non_applicability": None,
            "expected_step_change": None,
        }
    )

    assert parsed.group == ""
    assert parsed.name == ""
    assert parsed.delta_operation == "replace"
    assert parsed.applicability_boundary == []
    assert parsed.non_applicability == []
    assert ProposerResult(action="skill", delta_operation="simplify").delta_operation == "simplify"


def test_reviewer_agent_schema_covers_auto_eligibility_and_scope_evidence() -> None:
    schema = _agent_schema(ReviewerResult)
    schema_text = json.dumps(schema, sort_keys=True)
    properties = schema["properties"]
    recommendation = properties["recommendation"]

    assert '"type": "null"' not in schema_text
    assert {
        "evidence_sufficiency",
        "scope_assessment",
        "proportionality_assessment",
        "counterexamples",
        "concerns",
        "human_checks",
    } <= properties.keys()
    assert recommendation["enum"] == ["approve", "reject", "needs_revision"]
    assert "eligible for workspace auto selection" in recommendation["description"]
    assert "bounded automatic revision" in recommendation["description"]
    assert "decision" not in properties
    assert "evaluation_assessment" not in properties


def _candidate_workspace(tmp_path: Path) -> Path:
    candidate_root = tmp_path / "candidate"
    (candidate_root / "current" / "skills").mkdir(parents=True)
    (candidate_root / "proposed").mkdir()
    return candidate_root


def test_prepare_new_skill_creates_only_an_empty_agent_editable_directory(tmp_path: Path) -> None:
    candidate_root = _candidate_workspace(tmp_path)
    tool = _prepare_skill_tool(candidate_root)

    result = tool.invoke({"group": "materials_worker", "name": "bounded-method"})
    destination = candidate_root / "proposed" / "materials_worker" / "bounded-method"

    assert "Created an empty candidate directory" in result
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert set(tool.args_schema.model_json_schema()["properties"]) == {"group", "name"}


def test_prepare_existing_skill_keeps_full_copy_path_without_new_skill_fields(tmp_path: Path) -> None:
    candidate_root = _candidate_workspace(tmp_path)
    source = candidate_root / "current" / "skills" / "materials_worker" / "existing-method"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: existing-method\ndescription: Existing complete method.\n---\n# Existing\n",
        encoding="utf-8",
    )
    (source / "reference.txt").write_text("preserve this asset\n", encoding="utf-8")
    tool = _prepare_skill_tool(candidate_root)

    result = tool.invoke({"group": "materials_worker", "name": "existing-method"})

    destination = candidate_root / "proposed" / "materials_worker" / "existing-method"
    assert "Copied the complete staged bundle" in result
    assert (destination / "reference.txt").read_text(encoding="utf-8") == "preserve this asset\n"


def test_allowed_tools_metadata_is_inert_in_catalog_visibility(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skill = repo_root / "skills" / "materials_worker" / "inert-metadata"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: inert-metadata\n"
        "description: A visible SOP despite an unknown advisory declaration.\n"
        "allowed-tools: definitely_not_a_registered_tool\n"
        "---\n"
        "Use the SOP when its semantic trigger applies.\n",
        encoding="utf-8",
    )
    runtime = CatMasterSkillsRuntime(
        catalog=SkillCatalog.create_default(repo_root=repo_root)
    )
    runtime.refresh_catalog()

    visible = runtime.visible_skills("task_runner")
    meta = next(item for item in visible if item.name == "inert-metadata")
    assert not hasattr(meta, "allowed_tools")


def test_active_skill_roots_have_one_runtime_source_of_truth() -> None:
    assert SKILL_GROUPS is ACTIVE_SKILL_GROUPS
    assert _SKILL_GROUPS is ACTIVE_SKILL_GROUPS
