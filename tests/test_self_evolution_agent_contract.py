from __future__ import annotations

import json
from pathlib import Path

from langchain.agents.structured_output import ToolStrategy

from catmaster.runtime.self_evolution.agents import (
    _load_prompt,
    _prepare_skill_tool,
)
from catmaster.runtime.self_evolution.models import (
    ProposerResult,
    ReflectionResult,
    ReviewerResult,
)


def _agent_schema(
    model_type: type[ReflectionResult] | type[ProposerResult] | type[ReviewerResult],
) -> dict:
    return ToolStrategy(model_type).schema_specs[0].json_schema


def test_self_evolution_prompts_use_complete_trajectories_and_advisory_review() -> None:
    reflector = _load_prompt("reflector").casefold()
    proposer = _load_prompt("proposer").casefold()
    reviewer = _load_prompt("reviewer").casefold()

    assert "complete recorded episode trajectory" in reflector
    assert "execution_lapse" in reflector
    assert "regular expressions" in reflector
    assert "embedding similarity" in reflector
    assert "fixed recurrence count" in reflector

    assert "/evidence.md" in proposer
    assert "complete recorded semantic trajectory" in proposer
    assert "not a complete thread" not in proposer
    assert "attribution hypothesis" in proposer
    assert "prefer `replace`, `delete`, or `merge`" in proposer
    assert "recurrence\nthreshold" in proposer
    assert "evaluation_cases" not in proposer
    assert "a/b/c" not in proposer
    assert "checksum" not in proposer
    assert "hash comparison" not in proposer
    assert "one frozen interaction trace" not in proposer

    assert "/evidence.md" in reviewer
    assert "complete recorded semantic trajectory" in reviewer
    assert "fixed\n  episode count" in reviewer
    assert "/evaluation.json" not in reviewer
    assert "evidence sufficiency" in reviewer
    assert "counterexamples and non-applicability" in reviewer
    assert "human\ncanary" in reviewer
    assert "a/b/c" not in reviewer
    assert "advisory decision support" in reviewer
    assert "reject terminally" in reviewer
    assert "checksum" not in reviewer
    assert "hash comparison" not in reviewer
    assert "one frozen interaction trace" not in reviewer


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
    assert properties["delta_operation"]["enum"] == ["add", "delete", "replace", "merge"]
    assert "evaluation_cases" not in properties
    assert "EvaluationCase" not in schema.get("$defs", {})

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


def test_reviewer_agent_schema_is_advisory_and_covers_scope_evidence() -> None:
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
    assert "Advisory recommendation" in recommendation["description"]
    assert "never changes a terminal state" in recommendation["description"]
    assert "never authorizes canary or stable promotion" in recommendation["description"]
    assert "decision" not in properties
    assert "evaluation_assessment" not in properties


def _candidate_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_root = tmp_path / "candidate"
    repo_root = tmp_path / "repo"
    self_develop_root = tmp_path / "self_develop"
    (candidate_root / "current").mkdir(parents=True)
    (candidate_root / "proposed").mkdir()
    (repo_root / "skills").mkdir(parents=True)
    self_develop_root.mkdir()
    return candidate_root, repo_root, self_develop_root


def test_prepare_new_skill_refuses_missing_or_placeholder_content_without_writing(tmp_path: Path) -> None:
    candidate_root, repo_root, self_develop_root = _candidate_workspace(tmp_path)
    tool = _prepare_skill_tool(
        candidate_root,
        repo_root=repo_root,
        self_develop_root=self_develop_root,
    )

    missing_result = tool.invoke({"group": "materials_worker", "name": "bounded-method"})
    placeholder_result = tool.invoke(
        {
            "group": "materials_worker",
            "name": "placeholder-method",
            "description": "Replace this with a useful description.",
            "applicability": "TODO fill this in",
            "non_applicability": "A concrete nearby negative case.",
            "expected_step_change": "Use the candidate-specific bounded step.",
        }
    )

    assert "No candidate files were written" in missing_result
    assert "No candidate files were written" in placeholder_result
    assert not (candidate_root / "proposed" / "materials_worker" / "bounded-method").exists()
    assert not (candidate_root / "proposed" / "materials_worker" / "placeholder-method").exists()


def test_prepare_new_skill_writes_complete_minimal_bundle_from_proposal(tmp_path: Path) -> None:
    candidate_root, repo_root, self_develop_root = _candidate_workspace(tmp_path)
    tool = _prepare_skill_tool(
        candidate_root,
        repo_root=repo_root,
        self_develop_root=self_develop_root,
    )
    result = tool.invoke(
        {
            "group": "materials_worker",
            "name": "bounded-method",
            "description": "Use a bounded method only for tasks with an explicit transfer contract.",
            "applicability": "A transfer task explicitly requires an integrity receipt.",
            "non_applicability": "Ordinary local analysis without a transfer or checkpoint contract.",
            "expected_step_change": "Create the required receipt after the transfer completes.",
        }
    )

    skill_md = candidate_root / "proposed" / "materials_worker" / "bounded-method" / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")

    assert "Created a complete bounded new skill" in result
    assert "replace this" not in content.casefold()
    assert "todo" not in content.casefold()
    for heading in (
        "## Overview",
        "## Quick Start",
        "## Workflow",
        "## Method-critical defaults",
        "## Output Contract",
        "## References",
    ):
        section = content.split(heading, 1)[1].split("\n## ", 1)[0].strip()
        assert section, heading

    schema = tool.args_schema.model_json_schema()
    for field_name in ("description", "applicability", "non_applicability", "expected_step_change"):
        assert schema["properties"][field_name]["type"] == "string"
        assert "anyOf" not in schema["properties"][field_name]


def test_prepare_existing_skill_keeps_full_copy_path_without_new_skill_fields(tmp_path: Path) -> None:
    candidate_root, repo_root, self_develop_root = _candidate_workspace(tmp_path)
    source = repo_root / "skills" / "materials_worker" / "existing-method"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: existing-method\ndescription: Existing complete method.\n---\n# Existing\n",
        encoding="utf-8",
    )
    (source / "reference.txt").write_text("preserve this asset\n", encoding="utf-8")
    tool = _prepare_skill_tool(
        candidate_root,
        repo_root=repo_root,
        self_develop_root=self_develop_root,
    )

    result = tool.invoke({"group": "materials_worker", "name": "existing-method"})

    destination = candidate_root / "proposed" / "materials_worker" / "existing-method"
    assert "Copied the complete current built_in bundle" in result
    assert (destination / "reference.txt").read_text(encoding="utf-8") == "preserve this asset\n"
