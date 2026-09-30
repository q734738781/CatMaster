from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .gate import read_skill_frontmatter
from .models import LearningCandidate, SKILL_GROUPS, ValidationReport
from .settings import SelfEvolutionMode, resolve_self_evolution_mode
from .storage import SelfEvolutionStore, hash_tree, utc_now


BASE_VERSION = "base"
MEMORY_TARGET = "/memories/AGENTS.md"
_TARGET_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


class EffectiveSkillCatalogInput(BaseModel):
    """Page through the complete workspace-effective skill target catalog."""

    model_config = ConfigDict(extra="forbid")

    after: str = Field(
        "",
        description="Exact next-cursor target from the prior result; leave empty for page one.",
    )
    limit: int = Field(
        50,
        ge=1,
        le=200,
        description="Number of effective targets to return in this page.",
    )


class EffectiveSkillConflict(ValueError):
    """The requested effective-state transition no longer matches workspace state."""


def candidate_version(candidate: LearningCandidate) -> str:
    return f"{candidate.candidate_id}@r{candidate.revision:04d}"


def parse_candidate_version(value: str) -> tuple[str, int] | None:
    candidate_id, separator, revision_text = str(value or "").strip().partition("@r")
    if not separator or not candidate_id or not revision_text.isdigit():
        return None
    return candidate_id, max(1, int(revision_text))


def candidate_target(candidate: LearningCandidate) -> str:
    if candidate.action == "memory":
        return MEMORY_TARGET
    return f"{candidate.group}/{candidate.name}"


class EffectiveSkillsManager:
    """Resolve and mutate workspace-effective skill versions.

    Repository files and immutable candidate revisions remain the source content.
    This service owns only the small workspace selection state and its ordered
    control history.
    """

    def __init__(
        self,
        store: SelfEvolutionStore,
        *,
        repo_root: Path | str | None = None,
    ) -> None:
        self.store = store
        self.repo_root = Path(
            repo_root or Path(__file__).resolve().parents[3]
        ).expanduser().resolve()

    # -- workspace mode and migration ---------------------------------

    def workspace_mode(self, override: str | None = None) -> SelfEvolutionMode:
        if override is not None:
            return resolve_self_evolution_mode(override)
        payload = self.store.read_active_skills()
        saved = str(payload.get("mode") or "").strip()
        return resolve_self_evolution_mode(saved or None)

    def mode_info(self) -> dict[str, str]:
        payload = self.store.read_active_skills()
        saved = str(payload.get("mode") or "").strip()
        return {
            "mode": resolve_self_evolution_mode(saved or None),
            "source": "workspace" if saved else "deployment_default",
        }

    def ensure_migrated(self) -> dict[str, Any]:
        events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        with self.store.promotion_lock():
            payload = self.store.read_active_skills()
            skills = payload.setdefault("skills", {})
            changed = False
            for target, raw in list(skills.items()):
                if not isinstance(raw, dict):
                    continue
                repo_exists = self._repo_skill_dir(target).is_dir()
                normalized = self._normalize_target_state(
                    target,
                    raw,
                    repo_exists=repo_exists,
                )
                if normalized != raw:
                    events.append((target, dict(raw), dict(normalized)))
                    skills[target] = normalized
                    changed = True
            if changed:
                self.store.write_active_skills(payload)
        for target, before, after in events:
            self._audit_transition(
                target=target,
                action="migrate_effective_state",
                source="migration",
                actor="system",
                before=before,
                after=after,
            )
        return self.store.read_active_skills()

    def set_workspace_mode(
        self,
        mode: str,
        *,
        actor: str,
        expected_mode: str | None = None,
        note: str = "",
        source: str = "gui",
        message_id: str = "",
        thread_id: str = "",
    ) -> dict[str, Any]:
        resolved = resolve_self_evolution_mode(mode)
        actor_name = self._actor(actor)
        self._assert_control_source(source=source, message_id=message_id)
        target_events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        with self.store.promotion_lock():
            payload = self.store.read_active_skills()
            current = self.workspace_mode_from_payload(payload)
            if expected_mode is not None and current != resolve_self_evolution_mode(expected_mode):
                raise EffectiveSkillConflict(
                    f"workspace evolution mode changed from {expected_mode!r} to {current!r}"
                )
            before_mode = current
            payload["mode"] = resolved
            if resolved == "auto":
                skills = payload.setdefault("skills", {})
                for target, raw in list(skills.items()):
                    if target == MEMORY_TARGET or self._valid_skill_target(target):
                        state = self._normalize_target_state(
                            target,
                            raw if isinstance(raw, dict) else {},
                            repo_exists=self._repo_skill_dir(target).is_dir(),
                        )
                        before = dict(state)
                        auto_head = str(state.get("auto_head") or "")
                        if (
                            auto_head
                            and bool(state.get("enabled", True))
                            and state.get("update_policy") == "follow_auto"
                        ):
                            self._assert_selectable(target, auto_head)
                            if target == MEMORY_TARGET:
                                self._apply_memory_version(auto_head)
                            self._set_selected_version(state, auto_head)
                        skills[target] = state
                        if state != before:
                            target_events.append((target, before, dict(state)))
            self.store.write_active_skills(payload)
        if before_mode != resolved:
            self.store.append_audit_event(
                {
                    "event": "workspace_evolution_mode_changed",
                    "source": source,
                    "actor": actor_name,
                    "mode_before": before_mode,
                    "mode_after": resolved,
                    "note": str(note or "").strip(),
                    "message_id": str(message_id or "").strip(),
                    "thread_id": str(thread_id or "").strip(),
                }
            )
        for target, before, after in target_events:
            self._audit_transition(
                target=target,
                action="follow_auto_selected",
                source=source,
                actor=actor_name,
                before=before,
                after=after,
                note=note,
                message_id=message_id,
                thread_id=thread_id,
            )
        return {
            **self.mode_info(),
            "changes": [
                {
                    "target": target,
                    "selected_version_before": str(before.get("selected_version") or ""),
                    "selected_version_after": str(after.get("selected_version") or ""),
                }
                for target, before, after in target_events
            ],
        }

    def preview_workspace_mode(self, mode: str) -> dict[str, Any]:
        """Describe exact selections a mode change would make without mutating state."""

        resolved = resolve_self_evolution_mode(mode)
        payload = self.ensure_migrated()
        current = self.workspace_mode_from_payload(payload)
        changes: list[dict[str, str]] = []
        if resolved == "auto":
            for target, raw in sorted(payload.get("skills", {}).items()):
                if target != MEMORY_TARGET and not self._valid_skill_target(target):
                    continue
                state = self._normalize_target_state(
                    target,
                    raw if isinstance(raw, dict) else {},
                    repo_exists=self._repo_skill_dir(target).is_dir(),
                )
                auto_head = str(state.get("auto_head") or "")
                selected = str(state.get("selected_version") or "")
                if (
                    auto_head
                    and auto_head != selected
                    and bool(state.get("enabled", True))
                    and state.get("update_policy") == "follow_auto"
                ):
                    self._assert_selectable(target, auto_head)
                    changes.append(
                        {
                            "target": target,
                            "selected_version_before": selected,
                            "selected_version_after": auto_head,
                        }
                    )
        return {
            "mode_before": current,
            "mode_after": resolved,
            "changes": changes,
        }

    @staticmethod
    def workspace_mode_from_payload(payload: dict[str, Any]) -> SelfEvolutionMode:
        saved = str(payload.get("mode") or "").strip()
        return resolve_self_evolution_mode(saved or None)

    # -- semantic review to effective state ----------------------------

    def record_review_result(
        self,
        candidate: LearningCandidate,
        report: ValidationReport,
        review: dict[str, Any],
        *,
        mode: str | None = None,
    ) -> dict[str, Any]:
        recommendation = str(review.get("recommendation") or "").strip()
        result = {
            "eligible": False,
            "auto_head_advanced": False,
            "selected": False,
            "held_reason": "",
        }
        if recommendation != "approve":
            result["held_reason"] = f"reviewer recommendation is {recommendation or 'unavailable'}"
            return result
        if not report.valid or (candidate.action == "skill" and not report.loadable):
            result["held_reason"] = "candidate validation did not produce a loadable revision"
            return result
        human_checks = [
            str(item).strip()
            for item in list(review.get("human_checks") or [])
            if str(item).strip()
        ]
        if human_checks:
            result["held_reason"] = "; ".join(human_checks)
            self.store.append_audit_event(
                {
                    "event": "approved_revision_held_for_human_boundary",
                    "target": candidate_target(candidate),
                    "candidate_id": candidate.candidate_id,
                    "revision": candidate.revision,
                    "reason": result["held_reason"],
                }
            )
            return result

        target = candidate_target(candidate)
        version = candidate_version(candidate)
        self._assert_candidate_target(candidate, target)
        self._assert_selectable(target, version)
        resolved_mode = self.workspace_mode(mode)
        with self.store.promotion_lock():
            payload = self.store.read_active_skills()
            raw = payload.setdefault("skills", {}).get(target)
            repo_exists = self._repo_skill_dir(target).is_dir()
            state = self._normalize_target_state(
                target,
                raw if isinstance(raw, dict) else {},
                repo_exists=repo_exists,
            )
            before = dict(state)
            state["auto_head"] = version
            result["eligible"] = True
            result["auto_head_advanced"] = True
            if (
                resolved_mode == "auto"
                and bool(state.get("enabled", True))
                and state.get("update_policy") == "follow_auto"
            ):
                if candidate.action == "memory":
                    self._apply_memory_version(
                        version,
                        expected_base_hash=candidate.base_target_hash,
                    )
                self._set_selected_version(state, version)
                result["selected"] = True
            payload["skills"][target] = state
            self.store.write_active_skills(payload)
        self._audit_transition(
            target=target,
            action="review_approved_auto_head",
            source="auto",
            actor="semantic_reviewer",
            before=before,
            after=state,
            extra={
                "candidate_id": candidate.candidate_id,
                "revision": candidate.revision,
                "version": version,
                "selected": result["selected"],
            },
        )
        return result

    # -- authenticated exact controls ----------------------------------

    def update_target(
        self,
        target: str,
        *,
        actor: str,
        enabled: bool | None = None,
        selected_version: str | None = None,
        update_policy: str | None = None,
        expected_selected_version: str | None = None,
        note: str = "",
        source: str = "gui",
        message_id: str = "",
        thread_id: str = "",
    ) -> dict[str, Any]:
        if target != MEMORY_TARGET and not self._valid_skill_target(target):
            raise ValueError("skill target must have the form group/name")
        actor_name = self._actor(actor)
        self._assert_control_source(source=source, message_id=message_id)
        with self.store.promotion_lock():
            payload = self.store.read_active_skills()
            raw = payload.setdefault("skills", {}).get(target)
            state = self._normalize_target_state(
                target,
                raw if isinstance(raw, dict) else {},
                repo_exists=self._repo_skill_dir(target).is_dir(),
            )
            before = dict(state)
            current_selected = str(state.get("selected_version") or "")
            if (
                expected_selected_version is not None
                and expected_selected_version != current_selected
            ):
                raise EffectiveSkillConflict(
                    "selected skill version changed after the management view was opened"
                )
            if enabled is not None and target != MEMORY_TARGET:
                state["enabled"] = bool(enabled)
            if selected_version is not None:
                requested = str(selected_version or "").strip()
                if not requested:
                    raise ValueError("selected_version cannot be empty")
                self._assert_selectable(target, requested)
                if target == MEMORY_TARGET:
                    self._apply_memory_version(requested)
                self._set_selected_version(state, requested)
                state["update_policy"] = "pinned"
            if update_policy is not None:
                policy = str(update_policy or "").strip()
                if policy not in {"follow_auto", "pinned"}:
                    raise ValueError("update_policy must be follow_auto or pinned")
                state["update_policy"] = policy
                if (
                    policy == "follow_auto"
                    and bool(state.get("enabled", True))
                    and self.workspace_mode_from_payload(payload) == "auto"
                ):
                    auto_head = str(state.get("auto_head") or "")
                    if auto_head:
                        self._assert_selectable(target, auto_head)
                        if target == MEMORY_TARGET:
                            self._apply_memory_version(auto_head)
                        self._set_selected_version(state, auto_head)
            payload["skills"][target] = state
            self.store.write_active_skills(payload)
        if state != before:
            self._audit_transition(
                target=target,
                action="effective_state_changed",
                source=source,
                actor=actor_name,
                before=before,
                after=state,
                note=note,
                message_id=message_id,
                thread_id=thread_id,
            )
        return self.target_summary(target, payload=payload)

    def resolve_human_boundary(
        self,
        target: str,
        version: str,
        *,
        actor: str,
        message_id: str,
        thread_id: str,
        resolution: str,
        expected_selected_version: str,
        source: str = "chat",
    ) -> dict[str, Any]:
        """Resolve one reviewer-declared boundary from an exact user message."""

        if target != MEMORY_TARGET and not self._valid_skill_target(target):
            raise ValueError("skill target must have the form group/name")
        actor_name = self._actor(actor)
        message_ref = str(message_id or "").strip()
        thread_ref = str(thread_id or "").strip()
        answer = str(resolution or "").strip()
        self._assert_control_source(source=source, message_id=message_ref)
        if not thread_ref:
            raise ValueError("a trusted current thread is required")
        if not answer:
            raise ValueError("resolution must explain the user's exact decision")
        candidate = self._load_candidate_version(version)
        self._assert_candidate_target(candidate, target)
        review = candidate.review if isinstance(candidate.review, dict) else {}
        checks = [
            str(item).strip()
            for item in list(review.get("human_checks") or [])
            if str(item).strip()
        ]
        if not checks:
            raise ValueError("this revision has no unresolved human boundary")
        if not self._candidate_is_approved_and_readable(candidate):
            raise ValueError("the held revision is not otherwise approved and loadable")

        with self.store.promotion_lock():
            payload = self.store.read_active_skills()
            raw = payload.setdefault("skills", {}).get(target)
            state = self._normalize_target_state(
                target,
                raw if isinstance(raw, dict) else {},
                repo_exists=self._repo_skill_dir(target).is_dir(),
            )
            before = dict(state)
            current_selected = str(state.get("selected_version") or "")
            if str(expected_selected_version or "") != current_selected:
                raise EffectiveSkillConflict(
                    "selected skill version changed after the held decision was inspected"
                )
            authorizations = dict(state.get("human_authorizations") or {})
            authorizations[version] = {
                "message_id": message_ref,
                "thread_id": thread_ref,
                "actor": actor_name,
                "resolution": answer,
                "decided_at": utc_now(),
            }
            state["human_authorizations"] = authorizations
            state["auto_head"] = version
            selected = False
            if (
                self.workspace_mode_from_payload(payload) == "auto"
                and bool(state.get("enabled", True))
                and state.get("update_policy") == "follow_auto"
            ):
                if candidate.action == "memory":
                    self._apply_memory_version(
                        version,
                        expected_base_hash=candidate.base_target_hash,
                    )
                self._set_selected_version(state, version)
                selected = True
            payload["skills"][target] = state
            self.store.write_active_skills(payload)
        self._audit_transition(
            target=target,
            action="human_boundary_resolved",
            source=source,
            actor=actor_name,
            before=before,
            after=state,
            note=answer,
            message_id=message_ref,
            thread_id=thread_ref,
            extra={
                "version": version,
                "human_checks": checks,
                "selected": selected,
            },
        )
        return self.target_summary(target, payload=payload)

    def waiting_for_clarification_count(self) -> int:
        """Count complete current candidates still blocked on an actual user choice."""

        count = 0
        if not self.store.candidates_dir.is_dir():
            return count
        for candidate_dir in self.store.candidates_dir.iterdir():
            if not candidate_dir.is_dir():
                continue
            try:
                candidate = self.store.read_candidate(candidate_dir.name)
            except Exception:
                continue
            if candidate is None:
                continue
            review = candidate.review if isinstance(candidate.review, dict) else {}
            if (
                str(review.get("recommendation") or "") == "approve"
                and any(
                    str(check).strip()
                    for check in list(review.get("human_checks") or [])
                )
                and not self._candidate_is_eligible(candidate)
            ):
                count += 1
        return count

    # -- runtime resolution --------------------------------------------

    def runtime_overrides(
        self,
        *,
        run_id: str = "",
        thread_id: str = "",
        include_canary: bool = True,
    ) -> tuple[dict[str, tuple[Path, str]], set[str]]:
        payload = self.ensure_migrated()
        selected: dict[str, tuple[Path, str]] = {}
        disabled: set[str] = set()
        for target, raw in payload.get("skills", {}).items():
            if target == MEMORY_TARGET or not self._valid_skill_target(target):
                continue
            state = self._normalize_target_state(
                target,
                raw if isinstance(raw, dict) else {},
                repo_exists=self._repo_skill_dir(target).is_dir(),
            )
            if not bool(state.get("enabled", True)):
                disabled.add(target)
                continue
            version = str(state.get("selected_version") or "")
            canary = state.get("canary")
            if include_canary and self._canary_applies(
                canary,
                run_id=run_id,
                thread_id=thread_id,
            ):
                version = str(canary.get("version") or "").strip()
            if not version or version == BASE_VERSION:
                continue
            candidate = self._load_candidate_version(version)
            self._assert_candidate_target(candidate, target)
            source = (
                self.store.revision_dir(candidate.candidate_id, candidate.revision)
                / "proposed"
                / candidate.group
                / candidate.name
            )
            if not (source / "SKILL.md").is_file():
                raise EffectiveSkillConflict(
                    f"selected revision {version!r} has no readable SKILL.md"
                )
            if hash_tree(source) != candidate.bundle_hash:
                raise EffectiveSkillConflict(
                    f"selected revision bytes changed after review: {version!r}"
                )
            selected[target] = (source, version)
        return selected, disabled

    # -- catalog and history -------------------------------------------

    def list_targets(
        self,
        *,
        after: str = "",
        limit: int = 50,
        selection: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        payload = self.ensure_migrated() if selection is None else selection
        revision_records = self._revision_records()
        targets = set(self._repo_skill_targets())
        targets.add(MEMORY_TARGET)
        targets.update(
            target
            for target in payload.get("skills", {})
            if target == MEMORY_TARGET or self._valid_skill_target(target)
        )
        targets.update(
            record["target"]
            for record in revision_records
            if record.get("action") == "skill" and self._valid_skill_target(record["target"])
        )
        ordered = sorted(targets)
        if after:
            ordered = [target for target in ordered if target > after]
        capped = max(1, min(200, int(limit)))
        visible = ordered[: capped + 1]
        has_more = len(visible) > capped
        records_by_target: dict[str, list[dict[str, Any]]] = {}
        for record in revision_records:
            records_by_target.setdefault(str(record.get("target") or ""), []).append(record)
        rows = [
            self.target_summary(
                target,
                payload=payload,
                revision_records=records_by_target.get(target, []),
            )
            for target in visible[:capped]
        ]
        return rows, rows[-1]["target"] if has_more and rows else ""

    def target_count(self) -> int:
        payload = self.ensure_migrated()
        targets = set(self._repo_skill_targets())
        targets.add(MEMORY_TARGET)
        targets.update(
            target
            for target in payload.get("skills", {})
            if target == MEMORY_TARGET or self._valid_skill_target(target)
        )
        targets.update(
            record["target"]
            for record in self._revision_records()
            if (
                record.get("target") == MEMORY_TARGET
                or self._valid_skill_target(str(record.get("target") or ""))
            )
        )
        return len(targets)

    def stage_readable_context(self, destination: Path) -> list[dict[str, Any]]:
        """Copy complete selected guidance and return its model-visible catalog.

        Resolve each source from the version in the captured catalog, so later
        selection changes cannot relabel the files already staged for this phase.
        Disabled guidance remains inspectable; reading it does not enable it.
        """

        destination.mkdir(parents=True, exist_ok=True)
        tree = destination / "skills"
        tree.mkdir(exist_ok=True)
        for group in SKILL_GROUPS:
            source = self.repo_root / "skills" / group
            if source.is_dir():
                # Include shared support directories even without SKILL.md.
                shutil.copytree(source, tree / group, dirs_exist_ok=True)
        authoring = self.repo_root / "skills" / "AGENTS.MD"
        if authoring.is_file():
            shutil.copyfile(authoring, destination / "skill_authoring.md")
        (destination / "AGENTS.md").write_text(self.store.read_memory_text(), encoding="utf-8")

        summaries: list[dict[str, Any]] = []
        # Inspection normalizes legacy selections in memory through
        # target_summary; it must not migrate or modify the live selection file.
        selection = self.store.read_active_skills()
        cursor = ""
        while True:
            page, cursor = self.list_targets(after=cursor, limit=200, selection=selection)
            summaries.extend(page)
            if not cursor:
                break
        entries: list[dict[str, Any]] = []
        for summary in summaries:
            target = summary["target"]
            version = str(summary["selected_version"] or "")
            entry = {
                key: summary[key]
                for key in ("target", "description", "enabled", "selected_version")
            }
            if target == MEMORY_TARGET:
                entry["path"] = "/current/AGENTS.md"
            else:
                staged = tree / target
                if version and version != BASE_VERSION:
                    candidate = self._load_candidate_version(version)
                    self._assert_candidate_target(candidate, target)
                    source = (
                        self.store.revision_dir(candidate.candidate_id, candidate.revision)
                        / "proposed" / candidate.group / candidate.name
                    )
                    if not (source / "SKILL.md").is_file():
                        raise EffectiveSkillConflict(f"Selected skill {target} has no readable SKILL.md")
                    # Preserve the existing selected-revision check used when
                    # staging candidate context; no new integrity protocol.
                    if hash_tree(source) != candidate.bundle_hash:
                        raise EffectiveSkillConflict(f"selected revision bytes changed after review: {version!r}")
                    if staged.exists():
                        shutil.rmtree(staged)
                    shutil.copytree(source, staged)
                entry["path"] = (
                    f"/current/skills/{target}/SKILL.md"
                    if (staged / "SKILL.md").is_file() else ""
                )
                if not entry["path"]:
                    entry["unavailable_reason"] = "No readable selected skill body. Consult candidate history for unselected versions."
            entries.append(entry)
        lines = [
            "# Current guidance", "",
            "Read the path for a target and follow its relative file references.", "",
        ]
        for entry in entries:
            path = entry["path"] or entry["unavailable_reason"]
            state = "enabled" if entry["enabled"] else "disabled; available for inspection"
            lines.append(f"- `{entry['target']}` ({state}): `{path}`; {entry['description']}")
        (destination / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return entries

    def catalog_tool(self, *, entries: list[dict[str, Any]]) -> StructuredTool:
        """Page through the same guidance snapshot exposed to file tools."""

        captured = sorted((dict(row) for row in entries), key=lambda row: row["target"])

        def query_effective_skills(after: str = "", limit: int = 50) -> str:
            try:
                remaining = [row for row in captured if row["target"] > after]
                rows = remaining[:limit]
                next_cursor = rows[-1]["target"] if len(remaining) > limit else ""
                payload = {
                    "ok": True,
                    "skills": rows,
                    "next_cursor": next_cursor,
                    "total_count": len(captured),
                }
            except Exception as exc:
                payload = {
                    "ok": False,
                    "operation": "query_effective_skills",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recovery": (
                        "Reuse the exact next_cursor returned by the previous page, or restart "
                        "from an empty cursor."
                    ),
                }
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)

        return StructuredTool.from_function(
            func=query_effective_skills,
            name="query_effective_skills",
            description=(
                "List current skills and workspace guidance with directly readable paths. "
                "Copy an entry's path into read_file, then follow relative references or use "
                "glob/grep in its directory to inspect supporting files. "
                "For example, pass an entry's path as read_file.file_path, then open "
                "references/method.md relative to that skill directory if the body cites it. "
                "Disabled skills are readable for inspection; reading does not enable them. "
                "An empty path has an unavailable_reason. Follow next_cursor for more targets."
            ),
            args_schema=EffectiveSkillCatalogInput,
            infer_schema=False,
        )

    def target_summary(
        self,
        target: str,
        *,
        payload: dict[str, Any] | None = None,
        revision_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if target != MEMORY_TARGET and not self._valid_skill_target(target):
            raise ValueError("invalid effective skill target")
        state_payload = payload or self.store.read_active_skills()
        is_memory = target == MEMORY_TARGET
        repo_dir = self._repo_skill_dir(target)
        raw = state_payload.get("skills", {}).get(target)
        state = self._normalize_target_state(
            target,
            raw if isinstance(raw, dict) else {},
            repo_exists=repo_dir.is_dir(),
        )
        revisions = self._version_summaries(target, records=revision_records)
        latest = revisions[0]["version"] if revisions else ""
        selected_version = str(state.get("selected_version") or "")
        skill_md = repo_dir / "SKILL.md"
        selected_diagnostic = ""
        if selected_version and selected_version != BASE_VERSION:
            try:
                candidate = self._load_candidate_version(selected_version)
                if is_memory:
                    skill_md = (
                        self.store.revision_dir(candidate.candidate_id, candidate.revision)
                        / "memories"
                        / "AGENTS.md"
                    )
                else:
                    skill_md = (
                        self.store.revision_dir(candidate.candidate_id, candidate.revision)
                        / "proposed"
                        / candidate.group
                        / candidate.name
                        / "SKILL.md"
                    )
            except Exception as exc:
                selected_diagnostic = (
                    f"Selected version diagnostic: {type(exc).__name__}: {exc}"
                )
        description = "Workspace guidance used by future specialist runs." if is_memory else ""
        if skill_md.is_file() and not is_memory:
            try:
                description = str(read_skill_frontmatter(skill_md).get("description") or "").strip()
            except Exception as exc:
                description = f"Load diagnostic: {type(exc).__name__}: {exc}"
        if selected_diagnostic:
            description = selected_diagnostic
        group, _, name = target.partition("/")
        if is_memory:
            group, name = "workspace", "AGENTS.md"
        return {
            "target": target,
            "group": group,
            "name": name,
            "description": description,
            "source": (
                "workspace_guidance"
                if is_memory
                else
                "repository_and_workspace"
                if repo_dir.is_dir() and revisions
                else "repository"
                if repo_dir.is_dir()
                else "workspace"
            ),
            "enabled": bool(state.get("enabled", True)),
            "enabled_control": not is_memory,
            "selected_version": selected_version,
            "selected_label": self._version_label(selected_version),
            "update_policy": str(state.get("update_policy") or "follow_auto"),
            "auto_head": str(state.get("auto_head") or ""),
            "auto_head_label": self._version_label(str(state.get("auto_head") or "")),
            "latest_draft": latest,
            "latest_draft_label": self._version_label(latest),
            "revision_count": len(revisions),
            "eligible_revision_count": sum(bool(item["eligible"]) for item in revisions),
        }

    def target_detail(
        self,
        target: str,
        *,
        version_after: str = "",
        version_limit: int = 50,
        history_before: int = 0,
        history_limit: int = 50,
    ) -> dict[str, Any]:
        summary = self.target_summary(target)
        versions = self._version_summaries(target)
        if target != MEMORY_TARGET and self._repo_skill_dir(target).is_dir():
            versions.append(
                {
                    "version": BASE_VERSION,
                    "label": "Repository base",
                    "revision": 0,
                    "candidate_id": "",
                    "eligible": True,
                    "status": "base",
                    "recommendation": "base",
                    "validation_valid": True,
                    "created_at": "",
                    "behavior_change": "Repository-provided skill version.",
                }
            )
        start = 0
        if version_after:
            found = next(
                (index for index, item in enumerate(versions) if item["version"] == version_after),
                None,
            )
            if found is None:
                raise EffectiveSkillConflict("version cursor is stale")
            start = found + 1
        capped = max(1, min(100, int(version_limit)))
        visible_versions = versions[start : start + capped]
        next_version = (
            visible_versions[-1]["version"]
            if start + len(visible_versions) < len(versions) and visible_versions
            else ""
        )
        events, next_history = self.history(
            target=target,
            before=history_before,
            limit=history_limit,
        )
        selected = summary["selected_version"]
        auto_head = summary["auto_head"]
        for item in visible_versions:
            item["selected"] = item["version"] == selected
            item["auto_head"] = item["version"] == auto_head
        return {
            **summary,
            "versions": visible_versions,
            "version_next_cursor": next_version,
            "history": events,
            "history_next_cursor": next_history,
        }

    def history(
        self,
        *,
        target: str,
        before: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.store.audit_log_path.is_file():
            return [], 0
        rows: list[dict[str, Any]] = []
        for sequence, line in enumerate(
            self.store.audit_log_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if not isinstance(value, dict) or str(value.get("target") or "") != target:
                continue
            rows.append({"sequence": sequence, **value})
        rows.sort(key=lambda item: int(item["sequence"]), reverse=True)
        if before:
            rows = [item for item in rows if int(item["sequence"]) < int(before)]
        capped = max(1, min(200, int(limit)))
        visible = rows[:capped]
        next_cursor = (
            int(visible[-1]["sequence"])
            if len(rows) > len(visible) and visible
            else 0
        )
        return [self._public_history_event(item) for item in visible], next_cursor

    # -- internal helpers ----------------------------------------------

    def _normalize_target_state(
        self,
        target: str,
        raw: dict[str, Any],
        *,
        repo_exists: bool,
    ) -> dict[str, Any]:
        legacy_stable = str(raw.get("stable") or "").strip()
        has_selected = "selected_version" in raw
        selected = str(raw.get("selected_version") or "").strip()
        if not has_selected:
            selected = legacy_stable or (BASE_VERSION if repo_exists else "")
        policy = str(raw.get("update_policy") or "").strip()
        if policy not in {"follow_auto", "pinned"}:
            policy = "pinned" if legacy_stable else "follow_auto"
        auto_head = str(raw.get("auto_head") or "").strip()
        if not auto_head and legacy_stable and self._version_is_eligible(target, legacy_stable):
            auto_head = legacy_stable
        state: dict[str, Any] = {
            "enabled": bool(raw.get("enabled", True)),
            "selected_version": selected,
            "update_policy": policy,
            "auto_head": auto_head,
        }
        authorizations = raw.get("human_authorizations")
        if isinstance(authorizations, dict):
            normalized_authorizations = {
                str(version): {
                    "message_id": str(value.get("message_id") or "").strip(),
                    "thread_id": str(value.get("thread_id") or "").strip(),
                    "actor": str(value.get("actor") or "").strip(),
                    "resolution": str(value.get("resolution") or "").strip(),
                    "decided_at": str(value.get("decided_at") or "").strip(),
                }
                for version, value in authorizations.items()
                if isinstance(value, dict)
                and str(version or "").strip()
                and str(value.get("message_id") or "").strip()
            }
            if normalized_authorizations:
                state["human_authorizations"] = normalized_authorizations
        if selected and selected != BASE_VERSION:
            state["stable"] = selected
        if isinstance(raw.get("canary"), dict):
            state["canary"] = dict(raw["canary"])
        return state

    @staticmethod
    def _set_selected_version(state: dict[str, Any], version: str) -> None:
        resolved = str(version or "").strip()
        state["selected_version"] = resolved
        state.pop("canary", None)
        if resolved and resolved != BASE_VERSION:
            state["stable"] = resolved
        else:
            state.pop("stable", None)

    def _assert_selectable(self, target: str, version: str) -> None:
        if version == BASE_VERSION:
            if target == MEMORY_TARGET or not self._repo_skill_dir(target).is_dir():
                raise ValueError("this target has no repository base version")
            return
        candidate = self._load_candidate_version(version)
        self._assert_candidate_target(candidate, target)
        if not self._candidate_is_eligible(candidate):
            raise ValueError("only a reviewer-approved, valid, readable revision can be selected")

    def _load_candidate_version(self, version: str) -> LearningCandidate:
        parsed = parse_candidate_version(version)
        if parsed is None:
            raise ValueError(f"invalid immutable candidate version: {version!r}")
        candidate = self.store.read_candidate_revision(*parsed)
        if candidate is None:
            raise ValueError(f"candidate revision is unavailable: {version}")
        return candidate

    @staticmethod
    def _assert_candidate_target(candidate: LearningCandidate, target: str) -> None:
        if candidate_target(candidate) != target:
            raise EffectiveSkillConflict(
                f"candidate revision belongs to {candidate_target(candidate)!r}, not {target!r}"
            )

    def _candidate_is_approved_and_readable(
        self,
        candidate: LearningCandidate,
    ) -> bool:
        review = candidate.review if isinstance(candidate.review, dict) else {}
        validation = candidate.validation if isinstance(candidate.validation, dict) else {}
        if str(review.get("recommendation") or "") != "approve":
            return False
        if review.get("error"):
            return False
        if validation.get("valid") is not True:
            return False
        if candidate.action == "skill" and validation.get("loadable") is False:
            return False
        if candidate.action == "memory":
            source = (
                self.store.revision_dir(candidate.candidate_id, candidate.revision)
                / "memories"
                / "AGENTS.md"
            )
        else:
            source = (
                self.store.revision_dir(candidate.candidate_id, candidate.revision)
                / "proposed"
                / candidate.group
                / candidate.name
                / "SKILL.md"
            )
        return source.is_file()

    def _candidate_is_eligible(self, candidate: LearningCandidate) -> bool:
        if not self._candidate_is_approved_and_readable(candidate):
            return False
        review = candidate.review if isinstance(candidate.review, dict) else {}
        if not any(
            str(item).strip()
            for item in list(review.get("human_checks") or [])
        ):
            return True
        payload = self.store.read_active_skills()
        raw = payload.get("skills", {}).get(candidate_target(candidate))
        if not isinstance(raw, dict):
            return False
        authorizations = raw.get("human_authorizations")
        if not isinstance(authorizations, dict):
            return False
        authorization = authorizations.get(candidate_version(candidate))
        return bool(
            isinstance(authorization, dict)
            and str(authorization.get("message_id") or "").strip()
        )

    def _version_is_eligible(self, target: str, version: str) -> bool:
        try:
            candidate = self._load_candidate_version(version)
            self._assert_candidate_target(candidate, target)
        except Exception:
            return False
        return self._candidate_is_eligible(candidate)

    def _apply_memory_version(
        self,
        version: str,
        *,
        expected_base_hash: str = "",
    ) -> None:
        candidate = self._load_candidate_version(version)
        if candidate.action != "memory":
            raise ValueError("selected version is not a workspace-memory revision")
        source = (
            self.store.revision_dir(candidate.candidate_id, candidate.revision)
            / "memories"
            / "AGENTS.md"
        )
        if not source.is_file():
            raise ValueError("workspace-memory revision content is unavailable")
        current_hash = self.store.memory_hash()
        if expected_base_hash and current_hash != expected_base_hash:
            raise EffectiveSkillConflict(
                "workspace memory changed after this revision was prepared"
            )
        swapped, _observed = self.store.compare_and_swap_memory(
            expected_hash=current_hash,
            new_text=source.read_text(encoding="utf-8"),
        )
        if not swapped:
            raise EffectiveSkillConflict("workspace memory changed during selection")

    def _repo_skill_dir(self, target: str) -> Path:
        if not self._valid_skill_target(target):
            return self.repo_root / "skills" / "__invalid__"
        group, name = target.split("/", 1)
        return self.repo_root / "skills" / group / name

    def _repo_skill_targets(self) -> list[str]:
        targets: list[str] = []
        for group in SKILL_GROUPS:
            root = self.repo_root / "skills" / group
            if not root.is_dir():
                continue
            for skill_md in root.glob("*/SKILL.md"):
                targets.append(f"{group}/{skill_md.parent.name}")
        return targets

    @staticmethod
    def _valid_skill_target(target: str) -> bool:
        group, separator, name = str(target or "").partition("/")
        return bool(
            separator
            and group in SKILL_GROUPS
            and _TARGET_NAME_RE.fullmatch(name)
        )

    def _revision_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.store.candidates_dir.is_dir():
            return rows
        for candidate_dir in sorted(self.store.candidates_dir.iterdir()):
            if not candidate_dir.is_dir():
                continue
            for revision_dir in sorted(candidate_dir.glob("r[0-9][0-9][0-9][0-9]")):
                descriptor_path = revision_dir / "candidate.json"
                if not descriptor_path.is_file():
                    continue
                try:
                    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    rows.append(
                        {
                            "target": "",
                            "action": "",
                            "candidate_id": candidate_dir.name,
                            "revision": int(revision_dir.name[1:]),
                            "read_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if not isinstance(descriptor, dict):
                    continue
                action = str(descriptor.get("action") or "")
                target = (
                    MEMORY_TARGET
                    if action == "memory"
                    else f"{descriptor.get('group') or ''}/{descriptor.get('name') or ''}"
                )
                rows.append(
                    {
                        "target": target,
                        "action": action,
                        "candidate_id": str(descriptor.get("candidate_id") or candidate_dir.name),
                        "revision": max(1, int(descriptor.get("revision") or revision_dir.name[1:])),
                        "created_at": str(descriptor.get("created_at") or ""),
                    }
                )
        return rows

    def _version_summaries(
        self,
        target: str,
        *,
        records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self._revision_records() if records is None else records:
            if record.get("target") != target:
                continue
            version = f"{record['candidate_id']}@r{int(record['revision']):04d}"
            try:
                candidate = self._load_candidate_version(version)
                proposal_path = (
                    self.store.revision_dir(candidate.candidate_id, candidate.revision)
                    / "proposal.json"
                )
                proposal: dict[str, Any] = {}
                if proposal_path.is_file():
                    value = json.loads(proposal_path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        proposal = value
                recommendation = str(candidate.review.get("recommendation") or "unavailable")
                valid = candidate.validation.get("valid") is True
                eligible = self._candidate_is_eligible(candidate)
                if eligible:
                    status = "approved"
                elif recommendation == "needs_revision":
                    status = "needs_revision"
                elif recommendation == "reject":
                    status = "rejected"
                elif not valid:
                    status = "invalid"
                else:
                    status = "unavailable"
                revision_root = self.store.revision_dir(
                    candidate.candidate_id,
                    candidate.revision,
                )
                if candidate.action == "memory":
                    content_root = revision_root / "memories"
                else:
                    content_root = (
                        revision_root / "proposed" / candidate.group / candidate.name
                    )
                readable_files = [
                    path.relative_to(content_root).as_posix()
                    for path in sorted(content_root.rglob("*"))
                    if path.is_file() and not path.is_symlink()
                ] if content_root.is_dir() else []
                rows.append(
                    {
                        "version": version,
                        "label": f"r{candidate.revision:04d}",
                        "revision": candidate.revision,
                        "candidate_id": candidate.candidate_id,
                        "eligible": eligible,
                        "status": status,
                        "recommendation": recommendation,
                        "validation_valid": valid,
                        "created_at": candidate.created_at,
                        "behavior_change": str(
                            proposal.get("expected_step_change")
                            or candidate.review.get("summary")
                            or candidate.rationale
                            or ""
                        ).strip(),
                        "review_summary": str(candidate.review.get("summary") or "").strip(),
                        "review_concerns": [
                            str(item).strip()
                            for item in list(candidate.review.get("concerns") or [])
                            if str(item).strip()
                        ],
                        "evidence_ids": [str(item) for item in candidate.evidence_ids],
                        "files": readable_files,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "version": version,
                        "label": f"r{int(record['revision']):04d}",
                        "revision": int(record["revision"]),
                        "candidate_id": str(record["candidate_id"]),
                        "eligible": False,
                        "status": "read_error",
                        "recommendation": "unavailable",
                        "validation_valid": False,
                        "created_at": str(record.get("created_at") or ""),
                        "behavior_change": f"Revision cannot be read: {type(exc).__name__}: {exc}",
                    }
                )
        rows.sort(
            key=lambda item: (str(item.get("created_at") or ""), int(item["revision"])),
            reverse=True,
        )
        return rows

    @staticmethod
    def _version_label(version: str) -> str:
        if version == BASE_VERSION:
            return "Repository base"
        parsed = parse_candidate_version(version)
        return f"r{parsed[1]:04d}" if parsed else "Not selected"

    @staticmethod
    def _canary_applies(canary: Any, *, run_id: str, thread_id: str) -> bool:
        if not isinstance(canary, dict):
            return False
        run_ids = {str(item).strip() for item in canary.get("run_ids", []) if str(item).strip()}
        thread_ids = {
            str(item).strip() for item in canary.get("thread_ids", []) if str(item).strip()
        }
        return run_id in run_ids or thread_id in thread_ids

    @staticmethod
    def _actor(actor: str) -> str:
        value = str(actor or "").strip()
        if not value:
            raise ValueError("authenticated actor is required")
        return value

    @staticmethod
    def _assert_control_source(*, source: str, message_id: str) -> None:
        if str(source or "").strip() == "chat" and not str(message_id or "").strip():
            raise ValueError("a chat state change requires the current user-message handle")

    def _audit_transition(
        self,
        *,
        target: str,
        action: str,
        source: str,
        actor: str,
        before: dict[str, Any],
        after: dict[str, Any],
        note: str = "",
        message_id: str = "",
        thread_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.store.append_audit_event(
            {
                "event": action,
                "target": target,
                "source": source,
                "actor": actor,
                "state_before": dict(before),
                "state_after": dict(after),
                "note": str(note or "").strip(),
                "decided_at": utc_now(),
                "message_id": str(message_id or "").strip(),
                "thread_id": str(thread_id or "").strip(),
                **dict(extra or {}),
            }
        )

    @staticmethod
    def _public_history_event(event: dict[str, Any]) -> dict[str, Any]:
        before = event.get("state_before")
        if not isinstance(before, dict):
            before = event.get("pointer_before") if isinstance(event.get("pointer_before"), dict) else {}
        after = event.get("state_after")
        if not isinstance(after, dict):
            after = event.get("pointer_after") if isinstance(event.get("pointer_after"), dict) else {}
        return {
            "sequence": int(event.get("sequence") or 0),
            "event": str(event.get("event") or "state_changed"),
            "source": str(event.get("source") or "legacy"),
            "actor": str(event.get("actor") or "system"),
            "before": before,
            "after": after,
            "note": str(event.get("note") or event.get("rationale") or ""),
            "time": str(event.get("decided_at") or event.get("ts") or ""),
            "version": str(event.get("version") or ""),
            "message_id": str(event.get("message_id") or ""),
            "thread_id": str(event.get("thread_id") or ""),
        }


__all__ = [
    "BASE_VERSION",
    "MEMORY_TARGET",
    "EffectiveSkillConflict",
    "EffectiveSkillsManager",
    "candidate_target",
    "candidate_version",
    "parse_candidate_version",
]
