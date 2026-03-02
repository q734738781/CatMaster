from __future__ import annotations

import pytest
from pydantic import ValidationError

from catmaster.agents.response_schemas import ProposalOutput, TaskOutput


def test_proposal_output_success_status_requires_success_placeholders() -> None:
    payload = ProposalOutput(
        status="success",
        proposal_md="# Proposal\n\ncontent",
        work_packages=["wp1"],
        error="",
        needs_human=False,
    )
    assert payload.status == "success"

    with pytest.raises(ValidationError, match="status=success requires error=''"):
        ProposalOutput(
            status="success",
            proposal_md="# Proposal\n\ncontent",
            work_packages=["wp1"],
            error="not empty",
            needs_human=False,
        )


def test_proposal_output_fail_status_requires_failure_placeholders() -> None:
    payload = ProposalOutput(
        status="fail",
        proposal_md="",
        work_packages=[],
        error="missing required context",
        needs_human=True,
    )
    assert payload.status == "fail"

    with pytest.raises(ValidationError, match="status=fail requires proposal_md=''"):
        ProposalOutput(
            status="fail",
            proposal_md="# Proposal\n\nshould be empty",
            work_packages=[],
            error="missing required context",
            needs_human=True,
        )


def test_task_output_done_status_requires_done_placeholders() -> None:
    payload = TaskOutput(
        status="done",
        summary="completed",
        facts=[],
        files=[],
        constraints=[],
        open_questions=[],
        decisions=[],
        next_steps=[],
        artifacts=[],
        error="",
        needs_human=False,
        hint="",
    )
    assert payload.status == "done"

    with pytest.raises(ValidationError, match="status=done requires hint=''"):
        TaskOutput(
            status="done",
            summary="completed",
            facts=[],
            files=[],
            constraints=[],
            open_questions=[],
            decisions=[],
            next_steps=[],
            artifacts=[],
            error="",
            needs_human=False,
            hint="should be empty",
        )


def test_task_output_blocked_status_requires_error() -> None:
    payload = TaskOutput(
        status="blocked",
        summary="blocked by missing files",
        facts=[],
        files=[],
        constraints=[],
        open_questions=["which pseudopotential to use?"],
        decisions=[],
        next_steps=[],
        artifacts=[],
        error="required file not found",
        needs_human=True,
        hint="confirm file source",
    )
    assert payload.status == "blocked"

    with pytest.raises(ValidationError, match="status=blocked requires non-empty error"):
        TaskOutput(
            status="blocked",
            summary="blocked by missing files",
            facts=[],
            files=[],
            constraints=[],
            open_questions=["which pseudopotential to use?"],
            decisions=[],
            next_steps=[],
            artifacts=[],
            error="",
            needs_human=True,
            hint="confirm file source",
        )
