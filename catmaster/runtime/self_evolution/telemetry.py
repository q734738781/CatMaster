from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from catmaster.runtime.observability_store import ObservabilityStore

from .models import SkillRun
from .storage import SelfEvolutionStore


SKILL_VERSION_MANIFEST = "skill_versions.json"
_EVENT_NAMES = (
    "TOOL_RAW_INPUT",
    "TOOL_CALL_END",
    "TASK_END",
    "TASK_SUMMARY",
    "SKILL_OUTCOME",
    "SELF_EVOLUTION_SKILL_OUTCOME",
)
_SUCCESS_STATUSES = {"ok", "success", "succeeded", "completed", "done", "passed", "pass"}
_TRUSTED_OUTCOME_SOURCES = {
    "host_verifier",
    "self_evolution_verifier",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_skill_version_manifest(
    *,
    run_dir: Path | str,
    run_id: str,
    entries: list[dict[str, str]],
) -> Path:
    """Freeze the exact skills presented to one run."""

    path = Path(run_dir).expanduser().resolve() / SKILL_VERSION_MANIFEST
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise TypeError(f"skill version manifest entry {index} must be an object")
        row = {
            "skill_name": str(item.get("skill_name") or "").strip(),
            "skill_version": str(item.get("skill_version") or "").strip(),
            "virtual_path": str(item.get("virtual_path") or "").strip(),
        }
        if not row["skill_name"] or not row["skill_version"]:
            raise ValueError(
                f"skill version manifest entry {index} requires skill_name and skill_version"
            )
        normalized.append(row)
    normalized.sort(
        key=lambda item: (item["skill_name"], item["skill_version"], item["virtual_path"])
    )
    _atomic_json(path, {"run_id": str(run_id or "").strip(), "skills": normalized})
    return path


def read_skill_version_manifest(run_dir: Path | str) -> list[dict[str, str]]:
    path = Path(run_dir).expanduser().resolve() / SKILL_VERSION_MANIFEST
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"cannot read skill version manifest {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"skill version manifest {path} must contain a JSON object")
    rows = value.get("skills")
    if not isinstance(rows, list):
        raise TypeError(f"skill version manifest {path} requires an array-valued 'skills' field")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise TypeError(f"skill version manifest {path} entry {index} must be an object")
        row = {
            "skill_name": str(item.get("skill_name") or "").strip(),
            "skill_version": str(item.get("skill_version") or "").strip(),
            "virtual_path": str(item.get("virtual_path") or "").strip(),
        }
        if not row["skill_name"] or not row["skill_version"]:
            raise ValueError(
                f"skill version manifest {path} entry {index} requires "
                "skill_name and skill_version"
            )
        normalized.append(row)
    return normalized


def record_presented_skills(
    *,
    store: SelfEvolutionStore,
    run_id: str,
    entries: list[dict[str, str]],
) -> None:
    for entry in entries:
        store.upsert_skill_run(
            SkillRun(
                run_id=run_id,
                skill_name=str(entry.get("skill_name") or ""),
                skill_version=str(entry.get("skill_version") or ""),
                presented=True,
            )
        )


def _callback_id(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    return str(
        event.get("callback_run_id")
        or payload.get("callback_run_id")
        or payload.get("tool_call_id")
        or ""
    ).strip()


def _event_projection(
    run_dir: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    store = ObservabilityStore(run_dir)
    events: list[dict[str, Any]] = []
    before_id = 0
    partial = False
    errors: list[str] = []
    while True:
        try:
            page = store.read_events_page(
                limit=5_000,
                before_id=before_id,
                names=_EVENT_NAMES,
                include_legacy_trace_records=True,
            )
        except Exception as exc:
            partial = True
            errors.append(f"{type(exc).__name__}: {exc}")
            break
        page_events = [
            event
            for event in list(page.get("events") or [])
            if isinstance(event, dict)
        ]
        if not page_events:
            break
        events.extend(page_events)
        if not bool(page.get("has_more")):
            break
        next_before = int(page.get("min_id") or 0)
        if next_before <= 0 or (before_id > 0 and next_before >= before_id):
            partial = True
            errors.append(
                f"event pagination did not advance: before_id={before_id}, next_before={next_before}"
            )
            break
        before_id = next_before
    events.sort(key=lambda item: int(item.get("id") or 0))
    coverage = {
        "partial": partial,
        "errors": errors,
    }
    raw_inputs: dict[str, str] = {}
    successful_inputs: list[str] = []
    unpaired_inputs: list[str] = []
    outcome_rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        name = str(event.get("name") or event.get("event") or "").strip()
        if name == "TOOL_RAW_INPUT":
            raw: Any = payload.get("params_compact")
            if not raw:
                raw = payload.get("params_full")
            text = (
                raw
                if isinstance(raw, str)
                else json.dumps(raw, ensure_ascii=False, default=str)
                if raw
                else ""
            )
            callback_id = _callback_id(event)
            if text and callback_id:
                raw_inputs[callback_id] = text
            elif text:
                unpaired_inputs.append(text)
            continue
        if name == "TOOL_CALL_END":
            status = str(
                event.get("status")
                or payload.get("status")
                or payload.get("tool_status")
                or ""
            ).strip().lower()
            callback_id = _callback_id(event)
            if status in _SUCCESS_STATUSES and callback_id in raw_inputs:
                successful_inputs.append(raw_inputs[callback_id])
            continue
        source = str(event.get("source") or "").strip()
        rows = payload.get("skill_outcomes")
        if isinstance(rows, list):
            outcome_rows.extend(
                {**item, "_source": source}
                for item in rows
                if isinstance(item, dict)
            )
        if name in {"SKILL_OUTCOME", "SELF_EVOLUTION_SKILL_OUTCOME"}:
            outcome_rows.append({**payload, "_source": source})
    if unpaired_inputs:
        coverage["partial"] = True
        coverage["errors"].append(
            f"{len(unpaired_inputs)} skill-read inputs lacked callback IDs and were not guessed"
        )
    return successful_inputs, outcome_rows, coverage


def _exact_skill_outcomes(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    exact: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        skill_name = str(row.get("skill_name") or "").strip()
        skill_version = str(row.get("skill_version") or "").strip()
        outcome_ref = str(row.get("outcome_ref") or "").strip()
        if not skill_name or not skill_version or not outcome_ref:
            continue
        trusted = str(row.get("_source") or "") in _TRUSTED_OUTCOME_SOURCES
        outcome = str(row.get("outcome") or "").strip().lower()
        if outcome in {"success", "succeeded", "passed", "verified_success"}:
            reported_outcome = "verified_success" if trusted else "success"
        elif outcome in {"failure", "failed", "error", "verified_failure"}:
            reported_outcome = "verified_failure" if trusted else "failure"
        else:
            reported_outcome = "unknown"
        exact[(skill_name, skill_version)] = {
            "outcome": reported_outcome,
            "false_activation": bool(row.get("false_activation")),
            "outcome_ref": outcome_ref,
        }
    return exact


def finalize_skill_run_telemetry(
    *,
    store: SelfEvolutionStore,
    run_id: str,
    run_dir: Path | str,
    task_outcome: str,
    outcome_ref: str = "",
) -> list[SkillRun]:
    """Project actual reads/helper use from the raw run without storing payloads."""

    run_path = Path(run_dir).expanduser().resolve()
    manifest = read_skill_version_manifest(run_path)
    inputs, outcome_rows, coverage = _event_projection(run_path)
    exact_outcomes = _exact_skill_outcomes(outcome_rows)
    _ = task_outcome, outcome_ref
    if coverage.get("errors"):
        store.append_audit_event(
            {
                "event": "skill_telemetry_partial",
                "run_id": run_id,
                "errors": list(coverage["errors"]),
            }
        )
    records: list[SkillRun] = []
    for entry in manifest:
        skill_name = entry["skill_name"]
        version = entry["skill_version"]
        path = entry["virtual_path"].rstrip("/")
        skill_read = any(
            f"{path}/SKILL.md" in raw or (skill_name in raw and "SKILL.md" in raw)
            for raw in inputs
        )
        helper_used = any(
            (f"{path}/scripts/" in raw or f"{path}/references/" in raw)
            for raw in inputs
        )
        used = bool(skill_read or helper_used)
        exact_outcome = exact_outcomes.get((skill_name, version), {})
        if not used and not exact_outcome:
            # The manifest is an exact lookup table for skills visible to the
            # role. Availability alone is not an actual-use telemetry row.
            continue
        attributed_outcome = str(exact_outcome.get("outcome") or "unknown")
        false_activation = (
            bool(exact_outcome.get("false_activation"))
            and used
        )
        records.append(
            store.upsert_skill_run(
                SkillRun(
                    run_id=run_id,
                    skill_name=skill_name,
                    skill_version=version,
                    presented=True,
                    read=skill_read,
                    helper_used=helper_used,
                    outcome=attributed_outcome if used else "unknown",
                    false_activation=false_activation,
                    partial=bool(coverage["partial"]),
                )
            )
        )
    return records


__all__ = [
    "SKILL_VERSION_MANIFEST",
    "finalize_skill_run_telemetry",
    "read_skill_version_manifest",
    "record_presented_skills",
    "write_skill_version_manifest",
]
