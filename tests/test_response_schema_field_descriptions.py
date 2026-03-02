from __future__ import annotations

from catmaster.agents.response_schemas import (
    DirectorOutput,
    PerformNextTaskPayload,
    ProposalOutput,
    ReviseProposalPayload,
    StopAndSynthesizePayload,
    TaskDecisionRecord,
    TaskFileRecord,
    TaskOutput,
    TaskPacket,
)


def test_task_packet_field_descriptions_are_semantic() -> None:
    schema = TaskPacket.model_json_schema()
    props = schema.get("properties", {})

    task_detail_desc = str(props.get("task_detail", {}).get("description", ""))
    expected_outputs_desc = str(props.get("expected_outputs", {}).get("description", ""))
    reference_hint_desc = str(props.get("reference_hint", {}).get("description", ""))

    assert "invariants" in task_detail_desc
    assert "done criteria" in task_detail_desc
    assert "avoid copying long context" in task_detail_desc

    assert "Flat list" in expected_outputs_desc
    assert "Do not use nested objects" in expected_outputs_desc
    assert "Use [] only when truly none" in expected_outputs_desc

    assert "list of short strings" in reference_hint_desc
    assert "do not write long narrative paragraphs" in reference_hint_desc
    assert "Use [] when no hint is needed" in reference_hint_desc


def test_task_file_record_semantics_allow_file_or_dir() -> None:
    schema = TaskFileRecord.model_json_schema()
    props = schema.get("properties", {})
    path_desc = str(props.get("path", {}).get("description", ""))
    kind_desc = str(props.get("kind", {}).get("description", ""))
    assert "file or directory" in path_desc
    assert "project-relative paths" in path_desc
    assert "dir" in kind_desc


def test_director_rationale_description_is_local_not_structural() -> None:
    schema = DirectorOutput.model_json_schema()
    desc = str(schema.get("properties", {}).get("rationale", {}).get("description", ""))
    assert "Very brief decision rationale" in desc
    assert "do not repeat long context" in desc


def test_proposal_output_descriptions_capture_status_placeholders() -> None:
    schema = ProposalOutput.model_json_schema()
    props = schema.get("properties", {})
    proposal_md_desc = str(props.get("proposal_md", {}).get("description", ""))
    work_packages_desc = str(props.get("work_packages", {}).get("description", ""))
    error_desc = str(props.get("error", {}).get("description", ""))
    needs_human_desc = str(props.get("needs_human", {}).get("description", ""))
    assert "status=success" in proposal_md_desc
    assert "status=fail" in proposal_md_desc
    assert "use []" in work_packages_desc
    assert "empty string" in error_desc
    assert "set false" in needs_human_desc


def test_task_output_descriptions_capture_field_quality_rules() -> None:
    schema = TaskOutput.model_json_schema()
    props = schema.get("properties", {})
    summary_desc = str(props.get("summary", {}).get("description", ""))
    facts_desc = str(props.get("facts", {}).get("description", ""))
    open_questions_desc = str(props.get("open_questions", {}).get("description", ""))
    decisions_desc = str(props.get("decisions", {}).get("description", ""))
    error_desc = str(props.get("error", {}).get("description", ""))
    hint_desc = str(props.get("hint", {}).get("description", ""))
    assert "Do not paste long tables/logs/scripts" in summary_desc
    assert "Do not restate command traces" in facts_desc
    assert "speculative questions" in open_questions_desc
    assert "avoid duplicating summary/facts content" in decisions_desc
    assert "status=done" in error_desc
    assert "status=done" in hint_desc


def test_director_payload_field_assignment() -> None:
    perform_schema = PerformNextTaskPayload.model_json_schema()
    stop_schema = StopAndSynthesizePayload.model_json_schema()
    perform_props = set(perform_schema.get("properties", {}).keys())
    stop_props = set(stop_schema.get("properties", {}).keys())
    assert "task_packet" in perform_props
    assert "deliverables" not in perform_props
    assert "deliverables" not in stop_props


def test_revise_and_stop_payload_descriptions_capture_compactness() -> None:
    revise_schema = ReviseProposalPayload.model_json_schema()
    stop_schema = StopAndSynthesizePayload.model_json_schema()
    revise_props = revise_schema.get("properties", {})
    stop_props = stop_schema.get("properties", {})
    change_log_desc = str(revise_props.get("change_log", {}).get("description", ""))
    questions_desc = str(revise_props.get("questions_for_human", {}).get("description", ""))
    stop_reason_desc = str(stop_props.get("stop_reason", {}).get("description", ""))
    assert "Short change summary" in change_log_desc
    assert "Short question strings" in questions_desc
    assert "Short stop reason" in stop_reason_desc


def _assert_provider_schema_shape(schema: dict) -> None:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False
    assert set(required) == set(props.keys())


def _assert_no_ref_sibling_keywords(schema: dict) -> None:
    def _walk(node: object) -> None:
        if isinstance(node, dict):
            if "$ref" in node:
                assert set(node.keys()) == {"$ref"}
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(schema)


def test_structured_output_models_are_provider_compatible_shape() -> None:
    for cls in (
        TaskFileRecord,
        TaskDecisionRecord,
        TaskPacket,
        ProposalOutput,
        PerformNextTaskPayload,
        ReviseProposalPayload,
        StopAndSynthesizePayload,
        DirectorOutput,
        TaskOutput,
    ):
        schema = cls.model_json_schema()
        _assert_provider_schema_shape(schema)
        _assert_no_ref_sibling_keywords(schema)
