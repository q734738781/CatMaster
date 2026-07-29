from __future__ import annotations

from pathlib import Path
from typing import Any

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
    recommendation = str(raw.get("recommendation") or "unavailable").strip()
    return {
        "available": bool(raw),
        "recommendation": recommendation,
        "recommendation_label": _label(recommendation, fallback="Review unavailable"),
        "summary": _safe_text(raw.get("summary"), workspace=workspace, limit=1_200),
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
    if ctx and candidate_id:
        project_query = f"?project_space={workspace_name}" if workspace_name else ""
        base = (
            f"/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
            f"/revisions/{revision}"
        )
        detail_ref = base + project_query
        diff_ref = base + "/diff" + project_query
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
        "technical_details_available": bool(diff_ref),
    }


def project_self_evolution_job(value: Any, *, workspace: Path | None = None) -> dict[str, Any]:
    raw = value.to_dict() if hasattr(value, "to_dict") else dict(value or {})
    trigger = str(raw.get("trigger_kind") or "learning")
    status = str(raw.get("status") or "unknown")
    return {
        "title": {
            "post_run": "Post-run evidence extraction",
            "explicit_learn": "Requested durable correction",
            "candidate_revision": "Requested candidate revision",
        }.get(trigger, "Learning review"),
        "status": status,
        "status_label": _label(status, fallback="Unknown"),
        "attempt_count": int(raw.get("attempt_count") or 0),
        "summary": (
            _safe_text(raw.get("error"), workspace=workspace, limit=1_200)
            if status in {"error", "recovery_review"}
            else "Evidence governance completed."
            if status == "done"
            else "Evidence is being processed."
        ),
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
        "mode": str(raw.get("mode") or "observe"),
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
        "error_count": int(raw.get("error_count") or 0),
    }


__all__ = [
    "project_self_evolution_candidate",
    "project_self_evolution_job",
    "project_self_evolution_observation",
    "project_self_evolution_payload",
]
