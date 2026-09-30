from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from .common import humanize_identifier, redact_internal_text


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_text(value: Any, *, workspace: Path | None, limit: int = 4_000) -> str:
    return redact_internal_text(value, workspace=workspace, limit=limit)


def _label(value: Any, *, fallback: str = "") -> str:
    return humanize_identifier(str(value or ""), fallback=fallback)


def _project_review(value: Any, *, workspace: Path | None) -> dict[str, Any]:
    raw = _record(value)
    proportionality = _record(raw.get("proportionality_assessment"))
    change_points: list[dict[str, str]] = []
    for point in _items(raw.get("change_points")):
        if not isinstance(point, dict):
            continue
        change_points.append(
            {
                "title": _safe_text(point.get("title"), workspace=workspace, limit=240),
                "before": _safe_text(point.get("before"), workspace=workspace),
                "after": _safe_text(point.get("after"), workspace=workspace),
                "evidence": _safe_text(point.get("evidence"), workspace=workspace),
                "evidence_source": _safe_text(
                    point.get("evidence_source"),
                    workspace=workspace,
                    limit=400,
                ),
                "impact": _safe_text(point.get("impact"), workspace=workspace),
            }
        )
    is_text = raw.get("format") == "text"
    recommendation = str(raw.get("recommendation") or ("not_submitted" if is_text else "unavailable")).strip()
    return {
        "available": bool(raw),
        "recommendation": recommendation,
        "recommendation_label": "Text review completed" if is_text else _label(recommendation, fallback="Review unavailable"),
        "summary": redact_internal_text(raw.get("text"), workspace=workspace, limit=None) if is_text else _safe_text(raw.get("summary"), workspace=workspace, limit=1_200),
        "change_points": change_points,
        "evidence_sufficiency": _safe_text(
            raw.get("evidence_sufficiency"),
            workspace=workspace,
            limit=1_500,
        ),
        "scope_assessment": _safe_text(
            raw.get("scope_assessment"),
            workspace=workspace,
            limit=1_500,
        ),
        "proportionality": {
            "status": str(proportionality.get("status") or "unavailable"),
            "status_label": _label(
                proportionality.get("status"),
                fallback="Not assessed",
            ),
            "explanation": _safe_text(
                proportionality.get("explanation"),
                workspace=workspace,
                limit=1_500,
            ),
        },
        "counterexamples": [
            _safe_text(item, workspace=workspace, limit=1_200)
            for item in _items(raw.get("counterexamples"))
            if str(item or "").strip()
        ][:20],
        "concerns": [
            _safe_text(item, workspace=workspace, limit=1_200)
            for item in _items(raw.get("concerns"))
            if str(item or "").strip()
        ][:20],
        "human_checks": [
            _safe_text(item, workspace=workspace, limit=1_200)
            for item in _items(raw.get("human_checks"))
            if str(item or "").strip()
        ][:20],
        "rationale": _safe_text(raw.get("rationale"), workspace=workspace, limit=1_500),
    }


def project_self_evolution_observation(
    value: Any,
    *,
    workspace: Path | None = None,
) -> dict[str, Any]:
    raw = value.to_dict() if hasattr(value, "to_dict") else _record(value)
    refs: list[dict[str, Any]] = []
    for ref in _items(raw.get("evidence_refs"))[:8]:
        if not isinstance(ref, dict):
            continue
        refs.append(
            {
                "source_ref": str(ref.get("source_ref") or ref.get("ref") or ""),
                "reason": _label(ref.get("reason"), fallback="Evidence"),
                "excerpt": _safe_text(
                    ref.get("excerpt"),
                    workspace=workspace,
                    limit=900,
                ),
            }
        )
    signal = str(raw.get("signal_kind") or "")
    status = str(raw.get("status") or "open")
    return {
        "observation_id": str(raw.get("observation_id") or ""),
        "target": str(raw.get("target") or ""),
        "resolved_target": str(raw.get("resolved_target") or raw.get("target") or ""),
        "claim": _safe_text(raw.get("claim"), workspace=workspace, limit=2_000),
        "signal": signal,
        "signal_label": {
            "workspace_preference": "Workspace preference",
            "skill_revision": "Existing skill revision",
            "skill_discovery": "New reusable method",
        }.get(signal, "Learning evidence"),
        "status": status,
        "status_label": {
            "open": "Available for proposal",
            "consolidated": "Included in a candidate revision",
        }.get(status, _label(status, fallback="Open")),
        "evidence": refs,
        "outcome_ref": str(raw.get("outcome_ref") or ""),
        "created_at": str(raw.get("created_at") or ""),
    }


def project_self_evolution_candidate(
    value: Any,
    *,
    workspace: Path | None = None,
    ctx: str = "",
    workspace_name: str = "",
) -> dict[str, Any]:
    raw = value.to_dict() if hasattr(value, "to_dict") else dict(value or {})
    candidate_id = str(raw.get("candidate_id") or "")
    action = str(raw.get("action") or "skill")
    group = str(raw.get("group") or "")
    name = str(raw.get("name") or "")
    revision = max(1, int(raw.get("revision") or 1))
    version = str(raw.get("version") or f"r{revision:04d}")
    title = (
        "Workspace preference"
        if action == "memory"
        else " · ".join(
            item
            for item in (
                humanize_identifier(group, fallback=""),
                humanize_identifier(name, fallback=""),
            )
            if item
        )
    )
    proposal = _record(raw.get("proposal"))
    readiness = _record(raw.get("promotion_readiness"))
    detail_ref = ""
    diff_ref = ""
    revision_file_refs: dict[str, str] = {}
    if ctx and candidate_id:
        project_query = f"?project_space={workspace_name}" if workspace_name else ""
        base = (
            f"/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
            f"/revisions/{revision}"
        )
        detail_ref = base + project_query
        diff_ref = base + "/diff" + project_query
        for file_name in _items(raw.get("revision_files")):
            name_value = str(file_name or "").strip()
            if not name_value:
                continue
            revision_file_refs[name_value] = (
                base
                + "/files/content?file="
                + quote(name_value)
                + (f"&project_space={quote(workspace_name)}" if workspace_name else "")
            )
    evidence = [
        project_self_evolution_observation(item, workspace=workspace)
        for item in _items(raw.get("evidence"))
        if isinstance(item, dict)
    ]
    status = str(raw.get("status") or "pending")
    route = str(raw.get("route") or "")
    return {
        "candidate_id": candidate_id,
        "revision": revision,
        "version": version,
        "title": title or "Skill revision",
        "target_label": title or "Skill revision",
        "target": {
            "action": action,
            "group": group,
            "name": name,
            "exact_version": f"{candidate_id}@{version}",
        },
        "status": status,
        "status_label": _label(status, fallback="Pending review"),
        "route": route,
        "route_label": {
            "workspace_preference": "Workspace preference",
            "amend_existing_skill": "Amend existing skill",
            "new_skill": "New skill",
        }.get(route, _label(route, fallback="Learning candidate")),
        "behavior_change": _safe_text(
            proposal.get("expected_step_change") or raw.get("rationale"),
            workspace=workspace,
            limit=1_500,
        ),
        "why_now": _safe_text(raw.get("rationale"), workspace=workspace, limit=1_500),
        "evidence": evidence,
        "evidence_summary": (
            f"{len(evidence)} complete episode observation"
            f"{'s' if len(evidence) != 1 else ''} for this exact target."
        ),
        "applicability_boundary": [
            _safe_text(item, workspace=workspace, limit=900)
            for item in _items(proposal.get("applicability_boundary"))
            if str(item or "").strip()
        ],
        "non_applicability": [
            _safe_text(item, workspace=workspace, limit=900)
            for item in _items(proposal.get("non_applicability"))
            if str(item or "").strip()
        ],
        "delta_operation": str(proposal.get("delta_operation") or ""),
        "content_parent_version": str(proposal.get("content_parent_version") or ""),
        "delta_operation_label": _label(
            proposal.get("delta_operation"),
            fallback="Candidate revision",
        ),
        "review": _project_review(
            raw.get("review"),
            workspace=workspace,
        ),
        "validation": {
            "valid": bool(_record(raw.get("validation")).get("valid")),
            "errors": [
                _safe_text(item, workspace=workspace, limit=1_000)
                for item in _items(_record(raw.get("validation")).get("errors"))
            ][:20],
        },
        "promotion_readiness": {
            "ready": bool(readiness.get("ready")),
            "canary_ready": bool(readiness.get("canary_ready")),
            "reason": _safe_text(readiness.get("reason"), workspace=workspace, limit=1_500),
            "canary_actual_use": _record(readiness.get("canary_actual_use")),
        },
        "allowed_actions": [
            str(item).replace("_", "-")
            for item in _items(raw.get("allowed_actions"))
            if str(item).strip()
        ],
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
        "detail_ref": detail_ref,
        "diff_ref": diff_ref,
        "revision_file_refs": revision_file_refs,
        "technical_details_available": bool(diff_ref),
    }


def project_self_evolution_job(value: Any, *, workspace: Path | None = None) -> dict[str, Any]:
    raw = value.to_dict() if hasattr(value, "to_dict") else dict(value or {})
    trigger = str(raw.get("trigger_kind") or "learning")
    status = str(raw.get("status") or "unknown")
    outcome = _record(raw.get("outcome"))
    text_responses = [
        {**item, "text": redact_internal_text(item["text"], workspace=workspace, limit=None)}
        for item in _items(outcome.get("text_responses"))
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    reflection_items: list[dict[str, Any]] = []
    item_status_labels = {
        "completed": "Candidate created",
        "text_completed": "Completed with a text response",
        "activated": "Activated for the next run",
        "dormant": "Approved and dormant",
        "waiting_for_clarification": "Waiting for an in-chat clarification",
        "needs_revision": "Automatic repair budget exhausted",
        "rejected": "Rejected by independent review",
        "deferred": "Deferred for more real evidence",
        "ignored": "Ignored as non-durable",
        "no_change": "No durable change supported",
        "execution_lapse": "Covered by existing guidance",
        "error": "Needs attention",
        "pending": "Processing",
    }
    item_kind_labels = {
        "no_change": "No durable change",
        "execution_lapse": "Execution lapse",
        "workspace_preference": "Workspace preference",
        "skill_revision": "Existing skill revision",
        "skill_discovery": "New reusable skill",
    }
    for item in _items(outcome.get("reflection_items")):
        if not isinstance(item, dict):
            continue
        item_kind = str(item.get("kind") or "")
        item_status = str(item.get("status") or "unknown")
        target = _safe_text(item.get("target"), workspace=workspace, limit=240)
        resolved_target = _safe_text(
            item.get("resolved_target"),
            workspace=workspace,
            limit=240,
        )
        error = _safe_text(item.get("error"), workspace=workspace, limit=800)
        finding = _record(item.get("finding"))
        finding_refs: list[dict[str, str]] = []
        for ref in _items(finding.get("evidence_refs"))[:20]:
            if not isinstance(ref, dict):
                continue
            source_ref = str(ref.get("source_ref") or ref.get("ref") or "").strip()
            if not source_ref:
                continue
            finding_refs.append(
                {
                    "source_ref": source_ref,
                    "reason": _label(ref.get("reason"), fallback="Evidence"),
                    "excerpt": _safe_text(
                        ref.get("excerpt"),
                        workspace=workspace,
                        limit=900,
                    ),
                }
            )
        reason = _safe_text(
            item.get("reason") or finding.get("rationale") or finding.get("change"),
            workspace=workspace,
            limit=1_200,
        )
        reflection_items.append(
            {
                "item_ref": str(item.get("item_ref") or ""),
                "kind": item_kind,
                "title": target or item_kind_labels.get(item_kind, "Learning finding"),
                "target": target,
                "resolved_target": resolved_target,
                "status": item_status,
                "status_label": item_status_labels.get(
                    item_status,
                    _label(item_status, fallback="Unknown"),
                ),
                "observation_id": str(item.get("observation_id") or ""),
                "candidate_id": str(item.get("candidate_id") or ""),
                "error": error,
                "reason": reason,
                "finding": {
                    "change": _safe_text(
                        finding.get("change"),
                        workspace=workspace,
                        limit=1_500,
                    ),
                    "rationale": _safe_text(
                        finding.get("rationale"),
                        workspace=workspace,
                        limit=1_500,
                    ),
                    "evidence_refs": finding_refs,
                },
                "retry_allowed": bool(
                    str(item.get("item_ref") or "").strip()
                    and (item_status == "error" or bool(error))
                ),
            }
        )
    failed_item_count = sum(
        1 for item in reflection_items if item["retry_allowed"]
    )
    completed_item_count = sum(
        1
        for item in reflection_items
        if item["status"] in {
            "completed",
            "text_completed",
            "activated",
            "dormant",
            "waiting_for_clarification",
            "needs_revision",
            "rejected",
            "deferred",
            "ignored",
            "no_change",
            "execution_lapse",
        }
    )
    if failed_item_count and completed_item_count:
        outcome_state = "partial"
    elif status in {"error", "recovery_review"}:
        outcome_state = "error"
    elif status == "done":
        outcome_state = "complete"
    else:
        outcome_state = "pending"
    candidate_ids = {
        str(item.get("candidate_id") or "").strip()
        for item in reflection_items
        if str(item.get("candidate_id") or "").strip()
    }
    revision_candidate_id = str(outcome.get("revision_candidate_id") or "").strip()
    if revision_candidate_id:
        candidate_ids.add(revision_candidate_id)
    kinds = {
        str(item.get("kind") or "")
        for item in reflection_items
        if str(item.get("kind") or "")
    }
    terminal_item_statuses = {item["status"] for item in reflection_items}
    revision_status = str(outcome.get("revision_status") or "")
    if outcome_state == "pending":
        result_kind = "pending"
    elif outcome_state == "partial":
        result_kind = "partial"
    elif outcome_state == "error":
        result_kind = "error"
    elif text_responses and not (terminal_item_statuses - {"text_completed", "no_change", "execution_lapse"}):
        result_kind = "text_completed"
    elif "needs_revision" in terminal_item_statuses or revision_status == "needs_revision":
        result_kind = "needs_revision"
    elif candidate_ids:
        result_kind = "candidate_created"
    elif kinds == {"no_change"} or (
        not reflection_items and bool(outcome.get("no_change"))
    ):
        result_kind = "no_change"
    elif kinds == {"execution_lapse"}:
        result_kind = "execution_lapse"
    elif terminal_item_statuses == {"deferred"}:
        result_kind = "deferred"
    elif terminal_item_statuses == {"ignored"}:
        result_kind = "ignored"
    elif reflection_items:
        result_kind = "completed_without_candidate"
    else:
        result_kind = "legacy_complete"
    result_labels = {
        "pending": "Processing",
        "partial": "Completed with issues",
        "error": "Needs attention",
        "candidate_created": (
            "Candidate revision completed"
            if revision_candidate_id
            else (
                "Learning candidate created"
                if len(candidate_ids) == 1
                else "Learning candidates created"
            )
        ),
        "needs_revision": "Further revision needed",
        "no_change": "No change recommended",
        "execution_lapse": "Existing guidance was sufficient",
        "deferred": "Waiting for more real evidence",
        "ignored": "No durable change retained",
        "completed_without_candidate": "Completed without a candidate",
        "legacy_complete": "Legacy result",
        "text_completed": "Completed with a text response",
    }
    if result_kind == "text_completed":
        summary = "\n\n".join(item["text"] for item in text_responses)
    elif result_kind == "partial":
        summary = "The run was checked, but one or more independent findings failed. Retry only the failed item."
    elif result_kind == "error":
        summary = _safe_text(raw.get("error"), workspace=workspace, limit=1_200) or (
            "Self-evolution failed before it could return a learning result."
        )
    elif result_kind == "needs_revision":
        summary = (
            "The reviewer requested another revision after the bounded automatic repair "
            "budget ended. The latest exact revision remains visible and dormant."
        )
    elif result_kind == "candidate_created":
        activated_count = sum(item["status"] == "activated" for item in reflection_items)
        dormant_count = sum(item["status"] == "dormant" for item in reflection_items)
        if activated_count:
            summary = (
                f"{activated_count} reviewer-approved revision"
                f"{' was' if activated_count == 1 else 's were'} selected for the next run."
            )
        elif dormant_count:
            summary = (
                f"{dormant_count} reviewer-approved revision"
                f"{' is' if dormant_count == 1 else 's are'} recorded and dormant under the current workspace mode or target policy."
            )
        elif revision_candidate_id:
            summary = "The requested immutable candidate revision completed."
        else:
            summary = (
                f"The run produced {len(candidate_ids)} immutable learning candidate"
                f"{' revision' if len(candidate_ids) == 1 else ' revisions'}."
            )
    elif result_kind == "no_change":
        summary = next(
            (item["reason"] for item in reflection_items if item.get("reason")),
            "The run was checked. Its evidence did not support a durable skill or workspace-memory change.",
        )
    elif result_kind == "execution_lapse":
        summary = next(
            (item["reason"] for item in reflection_items if item.get("reason")),
            "The run was checked. Existing guidance already covered the issue, so no skill change was proposed.",
        )
    elif result_kind == "deferred":
        summary = next(
            (item["reason"] for item in reflection_items if item.get("reason")),
            "The finding remains open until later ordinary use supplies enough evidence.",
        )
    elif result_kind == "ignored":
        summary = next(
            (item["reason"] for item in reflection_items if item.get("reason")),
            "The finding was recorded and consumed as non-durable evidence.",
        )
    elif result_kind == "completed_without_candidate":
        summary = next(
            (item["reason"] for item in reflection_items if item.get("reason")),
            "The run was checked, but no learning candidate was created from the supported findings.",
        )
    elif result_kind == "legacy_complete":
        summary = "This older job completed before detailed learning results were recorded."
    else:
        summary = "Evidence is being processed."
    return {
        "job_id": str(raw.get("job_id") or ""),
        "run_id": str(raw.get("run_id") or ""),
        "episode_id": str(raw.get("episode_id") or ""),
        "selected_item_ref": str(raw.get("selected_item_ref") or ""),
        "predecessor_job_id": str(raw.get("predecessor_job_id") or ""),
        "title": {
            "post_run": "Post-run evidence extraction",
            "explicit_learn": "Requested durable correction",
            "candidate_revision": "Requested candidate revision",
            "selected_retry": "Selected job retry",
        }.get(trigger, "Learning review"),
        "status": status,
        "status_label": result_labels[result_kind],
        "result_kind": result_kind,
        "result_label": result_labels[result_kind],
        "attempt_count": int(raw.get("attempt_count") or 0),
        "retry_allowed": (
            status in {"error", "recovery_review"}
            and failed_item_count == 0
        ),
        "outcome": {
            "no_change": bool(outcome.get("no_change")),
            "state": outcome_state,
            "failed_item_count": failed_item_count,
            "completed_item_count": completed_item_count,
            "candidate_count": len(candidate_ids),
            "result_kind": result_kind,
            "reflection_items": reflection_items,
            "text_responses": text_responses,
        },
        "summary": summary,
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
    }


def project_self_evolution_payload(
    value: Any,
    *,
    workspace: Path | None = None,
    ctx: str = "",
    workspace_name: str = "",
) -> dict[str, Any]:
    raw = dict(value or {})
    candidates = [
        project_self_evolution_candidate(
            item,
            workspace=workspace,
            ctx=ctx,
            workspace_name=workspace_name,
        )
        for item in _items(raw.get("candidates"))
        if isinstance(item, dict)
    ]
    observations = [
        project_self_evolution_observation(item, workspace=workspace)
        for item in _items(raw.get("observations"))
        if isinstance(item, dict)
    ]
    jobs = [
        project_self_evolution_job(item, workspace=workspace)
        for item in _items(raw.get("jobs"))
        if isinstance(item, dict)
    ]
    return {
        "enabled": bool(raw.get("enabled", True)),
        "disabled_reason": _safe_text(
            raw.get("disabled_reason"),
            workspace=workspace,
            limit=1_200,
        ),
        "mode": str(raw.get("mode") or "auto"),
        "mode_source": str(raw.get("mode_source") or "deployment_default"),
        "scope": "workspace",
        "activation": "next_selected_run",
        "candidates": candidates,
        "candidate_count": int(raw.get("candidate_count") or len(candidates)),
        "next_cursor": str(raw.get("next_cursor") or ""),
        "status_counts": {
            str(key): int(count or 0)
            for key, count in _record(raw.get("status_counts")).items()
            if isinstance(count, (int, float)) and not isinstance(count, bool)
        },
        "observations": observations,
        "observation_count": int(raw.get("observation_count") or len(observations)),
        "observation_next_cursor": str(raw.get("observation_next_cursor") or ""),
        "observation_status_counts": {
            str(key): int(count or 0)
            for key, count in _record(raw.get("observation_status_counts")).items()
            if isinstance(count, (int, float)) and not isinstance(count, bool)
        },
        "effective_skill_count": int(raw.get("effective_skill_count") or 0),
        "jobs": jobs,
        "job_count": int(raw.get("job_count") or len(jobs)),
        "job_next_cursor": str(raw.get("job_next_cursor") or ""),
        "pending_review_count": int(raw.get("pending_review_count") or 0),
        "error_count": int(raw.get("error_count") or 0),
        "attention_count": int(raw.get("attention_count") or 0),
    }


def _project_effective_state(value: Any) -> dict[str, Any]:
    raw = _record(value)
    return {
        "enabled": bool(raw.get("enabled", True)),
        "selected_version": str(raw.get("selected_version") or ""),
        "update_policy": str(raw.get("update_policy") or "follow_auto"),
        "auto_head": str(raw.get("auto_head") or ""),
    }


def project_effective_skill_target(
    value: Any,
    *,
    workspace: Path | None = None,
    ctx: str = "",
    workspace_name: str = "",
) -> dict[str, Any]:
    """Project one effective target without exposing workspace-internal paths."""

    raw = _record(value)
    candidate_query = (
        f"?project_space={quote(workspace_name)}" if workspace_name else ""
    )
    versions: list[dict[str, Any]] = []
    for item in _items(raw.get("versions")):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        revision = max(0, int(item.get("revision") or 0))
        detail_ref = ""
        if ctx and candidate_id and revision:
            detail_ref = (
                f"/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
                f"/revisions/{revision}{candidate_query}"
            )
        versions.append(
            {
                "version": str(item.get("version") or ""),
                "label": _safe_text(item.get("label"), workspace=workspace, limit=120),
                "revision": revision,
                "candidate_id": candidate_id,
                "eligible": bool(item.get("eligible")),
                "selected": bool(item.get("selected")),
                "auto_head": bool(item.get("auto_head")),
                "status": str(item.get("status") or "unavailable"),
                "recommendation": str(item.get("recommendation") or "unavailable"),
                "validation_valid": bool(item.get("validation_valid")),
                "created_at": str(item.get("created_at") or ""),
                "behavior_change": _safe_text(
                    item.get("behavior_change"),
                    workspace=workspace,
                    limit=1_500,
                ),
                "review_summary": _safe_text(
                    item.get("review_summary"),
                    workspace=workspace,
                    limit=1_200,
                ),
                "review_concerns": [
                    _safe_text(concern, workspace=workspace, limit=900)
                    for concern in _items(item.get("review_concerns"))
                    if str(concern or "").strip()
                ][:50],
                "evidence_ids": [
                    str(evidence_id)
                    for evidence_id in _items(item.get("evidence_ids"))
                    if str(evidence_id or "").strip()
                ],
                "files": [
                    str(file_name)
                    for file_name in _items(item.get("files"))
                    if str(file_name or "").strip()
                    and not str(file_name).startswith(("/", ".."))
                ],
                "detail_ref": detail_ref,
            }
        )
    history: list[dict[str, Any]] = []
    for item in _items(raw.get("history")):
        if not isinstance(item, dict):
            continue
        history.append(
            {
                "sequence": int(item.get("sequence") or 0),
                "event": str(item.get("event") or "state_changed"),
                "source": str(item.get("source") or "system"),
                "actor": _safe_text(item.get("actor"), workspace=workspace, limit=160),
                "before": _project_effective_state(item.get("before")),
                "after": _project_effective_state(item.get("after")),
                "note": _safe_text(item.get("note"), workspace=workspace, limit=1_200),
                "time": str(item.get("time") or ""),
                "version": str(item.get("version") or ""),
                "source_ref": (
                    f"message:{str(item.get('message_id') or '').strip()}"
                    if str(item.get("message_id") or "").strip()
                    else ""
                ),
            }
        )
    projected = {
        "target": str(raw.get("target") or ""),
        "group": str(raw.get("group") or ""),
        "name": str(raw.get("name") or ""),
        "description": _safe_text(raw.get("description"), workspace=workspace, limit=1_200),
        "source": str(raw.get("source") or "workspace"),
        "enabled": bool(raw.get("enabled", True)),
        "enabled_control": bool(raw.get("enabled_control", True)),
        "selected_version": str(raw.get("selected_version") or ""),
        "selected_label": _safe_text(raw.get("selected_label"), workspace=workspace, limit=120),
        "update_policy": str(raw.get("update_policy") or "follow_auto"),
        "auto_head": str(raw.get("auto_head") or ""),
        "auto_head_label": _safe_text(raw.get("auto_head_label"), workspace=workspace, limit=120),
        "latest_draft": str(raw.get("latest_draft") or ""),
        "latest_draft_label": _safe_text(
            raw.get("latest_draft_label"),
            workspace=workspace,
            limit=120,
        ),
        "revision_count": int(raw.get("revision_count") or 0),
        "eligible_revision_count": int(raw.get("eligible_revision_count") or 0),
    }
    if "versions" in raw:
        projected.update(
            {
                "versions": versions,
                "version_next_cursor": str(raw.get("version_next_cursor") or ""),
                "history": history,
                "history_next_cursor": int(raw.get("history_next_cursor") or 0),
            }
        )
    return projected


__all__ = [
    "project_effective_skill_target",
    "project_self_evolution_candidate",
    "project_self_evolution_job",
    "project_self_evolution_observation",
    "project_self_evolution_payload",
]
