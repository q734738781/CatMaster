from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from .models import LearningCandidate, ValidationReport
from .storage import SelfEvolutionStore, hash_text, hash_tree, utc_now


class PromotionConflict(RuntimeError):
    pass


class PromotionManager:
    """Compatibility manager for legacy candidate release endpoints."""

    def __init__(self, store: SelfEvolutionStore, *, repo_root: Path | str | None = None) -> None:
        self.store = store
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()

    @staticmethod
    def _version(candidate: LearningCandidate) -> str:
        return f"{candidate.candidate_id}@r{candidate.revision:04d}"

    @staticmethod
    def _skill_key(candidate: LearningCandidate) -> str:
        return (
            "/memories/AGENTS.md"
            if candidate.action == "memory"
            else f"{candidate.group}/{candidate.name}"
        )

    def allowed_actions(self, candidate: LearningCandidate) -> list[str]:
        status = candidate.status
        if status == "review":
            actions: list[str] = []
            if not candidate.review:
                actions.append("run_review")
            if candidate.action == "skill":
                actions.append("request_revision")
                if self.promotion_readiness(candidate).get("canary_ready"):
                    actions.append("start_canary")
                actions.append("reject")
                return actions
            actions.append("request_revision")
            if self.promotion_readiness(candidate).get("ready"):
                actions.append("promote_stable")
            actions.append("reject")
            return actions
        if status in {"pending", "revision"}:
            return ["request_revision", "reject"]
        if status == "canary":
            actions = ["quarantine", "rollback"]
            if self.promotion_readiness(candidate).get("ready"):
                actions.insert(0, "promote_stable")
            return actions
        if status == "stable":
            return ["quarantine", "retire", "rollback"]
        return []

    def promotion_readiness(self, candidate: LearningCandidate) -> dict[str, Any]:
        if candidate.status not in {"review", "canary"}:
            return {
                "ready": False,
                "reason": f"Candidate status is {candidate.status or 'unknown'}, not eligible for release.",
                "bundle_unchanged": False,
                "target_unchanged": False,
            }

        revision_root = self.store.revision_dir(candidate.candidate_id, candidate.revision)
        if candidate.action == "memory":
            source = revision_root / "memories" / "AGENTS.md"
            proposed_hash = hash_text(source.read_text(encoding="utf-8")) if source.is_file() else ""
            current_hash = self.store.memory_hash()
            target_label = "workspace memory"
        else:
            source = revision_root / "proposed" / candidate.group / candidate.name
            proposed_hash = hash_tree(source)
            current_target, _had_workspace_target = self._effective_skill_path(candidate)
            current_hash = hash_tree(current_target)
            target_label = "effective target skill"

        canary = self._canary_evidence(candidate)
        bundle_unchanged = bool(candidate.bundle_hash) and proposed_hash == candidate.bundle_hash
        target_unchanged = current_hash == candidate.base_target_hash
        reasons: list[str] = []
        if not bundle_unchanged:
            reasons.append("The reviewed revision bundle changed or is missing.")
        if not target_unchanged:
            reasons.append(
                f"The {target_label} changed after this revision was created; request a rebased revision."
            )
        if candidate.action == "skill" and candidate.status == "review":
            reasons.append("Skill revisions must pass an explicitly scoped canary before stable promotion.")
        release_ready = (
            bundle_unchanged
            and target_unchanged
            and (
                candidate.action == "memory"
                or candidate.status == "canary"
            )
        )
        canary_ready = (
            candidate.action == "skill"
            and candidate.status == "review"
            and bundle_unchanged
            and target_unchanged
        )
        return {
            "ready": release_ready,
            "canary_ready": canary_ready,
            "reason": " ".join(reasons),
            "bundle_unchanged": bundle_unchanged,
            "target_unchanged": target_unchanged,
            "canary_actual_use": canary,
            "expected_target_hash": candidate.base_target_hash,
            "current_target_hash": current_hash,
        }

    def start_canary(
        self,
        candidate: LearningCandidate,
        report: ValidationReport,
        *,
        actor: str,
        thread_ids: list[str] | None = None,
        run_ids: list[str] | None = None,
        rationale: str = "",
    ) -> LearningCandidate:
        if candidate.action != "skill":
            raise ValueError("workspace preference revisions do not support thread-scoped canaries")
        if "start_canary" not in self.allowed_actions(candidate):
            raise ValueError("this revision cannot start a canary")
        if not report.valid:
            detail = (
                str(report.errors[0]).strip()
                if report.errors
                else "no validation detail was recorded"
            )
            raise ValueError(f"candidate validation failed: {detail}")
        if not report.loadable:
            detail = (
                str(report.diagnostics[0]).strip()
                if report.diagnostics
                else "the active DeepAgents loader did not discover the candidate"
            )
            raise ValueError(f"candidate cannot start a canary: {detail}")
        actor_name = self._human(actor)
        scoped_threads = sorted({str(item).strip() for item in (thread_ids or []) if str(item).strip()})
        scoped_runs = sorted({str(item).strip() for item in (run_ids or []) if str(item).strip()})
        if not scoped_threads and not scoped_runs:
            raise ValueError("canary requires at least one explicit thread or run")
        readiness = self.promotion_readiness(candidate)
        if not readiness["canary_ready"]:
            raise ValueError(str(readiness["reason"] or "candidate is not ready"))
        with self.store.promotion_lock():
            live = self._live(candidate)
            active = self.store.read_active_skills()
            key = self._skill_key(live)
            before = dict(active["skills"].get(key) or {})
            pointer = {
                "version": self._version(live),
                "thread_ids": scoped_threads,
                "run_ids": scoped_runs,
            }
            active["skills"].setdefault(key, {})["canary"] = pointer
            self.store.write_active_skills(active)
            live = self.store.update_candidate_status(live.candidate_id, "canary")
            self._audit_pointer(
                event="canary_started",
                candidate=live,
                actor=actor_name,
                rationale=rationale,
                before=before,
                after=dict(active["skills"][key]),
            )
        return live

    def _canary_evidence(self, candidate: LearningCandidate) -> dict[str, Any]:
        if candidate.action != "skill":
            return {
                "passed": None,
                "decision": "human",
                "successful_runs": 0,
                "verified_successful_runs": 0,
                "unverified_successful_runs": 0,
                "failed_runs": 0,
                "presented_runs": 0,
                "read_runs": 0,
                "helper_used_runs": 0,
            }
        version = self._version(candidate)
        records = [
            item
            for item in self.store.list_skill_runs(
                skill_name=self._skill_key(candidate),
                limit=2_000,
            )
            if item.skill_version == version
        ]
        attempted = [item for item in records if item.used]
        verified_successful = [
            item
            for item in attempted
            if item.outcome == "verified_success" and not item.false_activation
        ]
        failed = [
            item
            for item in records
            if item.false_activation
            or (item.used and item.outcome in {"failure", "verified_failure"})
        ]
        unverified_successful = [
            item
            for item in attempted
            if item.outcome == "success" and not item.false_activation
        ]
        return {
            "passed": None,
            "decision": "human",
            "successful_runs": len(verified_successful),
            "verified_successful_runs": len(verified_successful),
            "unverified_successful_runs": len(unverified_successful),
            "failed_runs": len(failed),
            "presented_runs": len(records),
            "read_runs": sum(item.read for item in records),
            "helper_used_runs": sum(item.helper_used for item in records),
        }

    def promote_stable(
        self,
        candidate: LearningCandidate,
        report: ValidationReport,
        *,
        actor: str,
        rationale: str = "",
    ) -> LearningCandidate:
        if "promote_stable" not in self.allowed_actions(candidate):
            raise ValueError("this revision cannot be promoted stable")
        if not report.valid:
            detail = (
                str(report.errors[0]).strip()
                if report.errors
                else "no validation detail was recorded"
            )
            raise ValueError(f"candidate validation failed: {detail}")
        if candidate.action == "skill" and not report.loadable:
            detail = (
                str(report.diagnostics[0]).strip()
                if report.diagnostics
                else "the active DeepAgents loader did not discover the candidate"
            )
            raise ValueError(f"candidate cannot be promoted: {detail}")
        actor_name = self._human(actor)
        readiness = self.promotion_readiness(candidate)
        if not readiness["ready"]:
            raise ValueError(str(readiness["reason"] or "candidate is not ready"))
        with self.store.promotion_lock():
            live = self._live(candidate)
            active = self.store.read_active_skills()
            key = self._skill_key(live)
            before_pointer = dict(active["skills"].get(key) or {})
            if live.action == "memory":
                self._promote_memory(live)
            else:
                self._materialize_skill(live)
            target_state = active["skills"].setdefault(key, {})
            target_state.update(
                {
                    "enabled": True,
                    "selected_version": self._version(live),
                    "update_policy": "pinned",
                    "stable": self._version(live),
                }
            )
            target_state.setdefault("auto_head", "")
            target_state.pop("canary", None)
            self.store.write_active_skills(active)
            live = self.store.update_candidate_status(live.candidate_id, "stable")
            self._audit_pointer(
                event="stable_promoted",
                candidate=live,
                actor=actor_name,
                rationale=rationale,
                before=before_pointer,
                after=dict(active["skills"][key]),
            )
        return live

    def promote(
        self,
        candidate: LearningCandidate,
        report: ValidationReport,
        *,
        decision_source: str = "human",
        actor: str = "",
        rationale: str = "",
    ) -> LearningCandidate:
        """Compatibility entry point for the legacy explicit release surface."""

        if str(decision_source or "") != "human":
            raise ValueError("legacy promotion accepts only an authenticated human decision")
        return self.promote_stable(
            candidate,
            report,
            actor=actor,
            rationale=rationale,
        )

    def request_revision(
        self,
        candidate: LearningCandidate,
        *,
        actor: str,
        rationale: str,
    ) -> LearningCandidate:
        if "request_revision" not in self.allowed_actions(candidate):
            raise ValueError("this candidate cannot be revised")
        actor_name = self._human(actor)
        if not str(rationale or "").strip():
            raise ValueError("revision guidance is required")
        with self.store.promotion_lock():
            live = self._live(candidate)
            updated = self.store.update_candidate_status(live.candidate_id, "revision")
            self.store.append_audit_event(
                {
                    "event": "revision_requested",
                    "candidate_id": updated.candidate_id,
                    "revision": updated.revision,
                    "actor": actor_name,
                    "rationale": str(rationale).strip(),
                }
            )
        return updated

    def reject(
        self,
        candidate: LearningCandidate,
        *,
        actor: str,
        rationale: str = "",
    ) -> LearningCandidate:
        if "reject" not in self.allowed_actions(candidate):
            raise ValueError("candidate decision is already final or not human-reviewable")
        actor_name = self._human(actor)
        with self.store.promotion_lock():
            live = self._live(candidate)
            live = self.store.update_candidate_status(live.candidate_id, "rejected")
            self.store.append_audit_event(
                {
                    "event": "human_decision",
                    "candidate_id": live.candidate_id,
                    "revision": live.revision,
                    "action": "reject",
                    "actor": actor_name,
                    "candidate_hash": live.bundle_hash,
                    "rationale": str(rationale or "").strip(),
                }
            )
        return live

    def quarantine(
        self,
        candidate: LearningCandidate,
        *,
        actor: str,
        rationale: str,
    ) -> LearningCandidate:
        if "quarantine" not in self.allowed_actions(candidate):
            raise ValueError("only a canary or stable revision can be quarantined")
        return self._remove_pointer(
            candidate,
            actor=actor,
            rationale=rationale,
            status="inactive",
            event="candidate_quarantined",
        )

    def retire(
        self,
        candidate: LearningCandidate,
        *,
        actor: str,
        rationale: str,
    ) -> LearningCandidate:
        if "retire" not in self.allowed_actions(candidate):
            raise ValueError("only a stable revision can be retired")
        return self._remove_pointer(
            candidate,
            actor=actor,
            rationale=rationale,
            status="inactive",
            event="candidate_retired",
        )

    def rollback(
        self,
        candidate: LearningCandidate,
        *,
        actor: str = "",
        rationale: str = "",
    ) -> LearningCandidate:
        if "rollback" not in self.allowed_actions(candidate):
            raise ValueError("only active canary or stable revisions can be rolled back")
        actor_name = self._human(actor)
        with self.store.promotion_lock():
            live = self._live(candidate)
            key = self._skill_key(live)
            active = self.store.read_active_skills()
            before = dict(active["skills"].get(key) or {})
            if live.status == "canary":
                active["skills"].setdefault(key, {}).pop("canary", None)
            elif live.action == "memory":
                self._rollback_memory(live)
                previous = self._previous_stable_pointer(live)
                if previous:
                    state = active["skills"].setdefault(key, {})
                    state.update(
                        {
                            "selected_version": previous,
                            "update_policy": "pinned",
                            "stable": previous,
                        }
                    )
                else:
                    active["skills"].pop(key, None)
            else:
                previous = self._previous_stable_pointer(live)
                if previous:
                    state = active["skills"].setdefault(key, {})
                    state.update(
                        {
                            "enabled": True,
                            "selected_version": previous,
                            "update_policy": "pinned",
                            "stable": previous,
                        }
                    )
                    state.setdefault("auto_head", "")
                    self._materialize_pointer(previous, group=live.group, name=live.name)
                else:
                    repository_target = self.repo_root / "skills" / live.group / live.name
                    if repository_target.is_dir():
                        state = active["skills"].setdefault(key, {})
                        state.update(
                            {
                                "enabled": True,
                                "selected_version": "base",
                                "update_policy": "pinned",
                            }
                        )
                        state.setdefault("auto_head", "")
                        state.pop("stable", None)
                        state.pop("canary", None)
                    else:
                        active["skills"].pop(key, None)
                    self._remove_materialized_skill(live)
            self.store.write_active_skills(active)
            live = self.store.update_candidate_status(live.candidate_id, "inactive")
            self._audit_pointer(
                event="candidate_rolled_back",
                candidate=live,
                actor=actor_name,
                rationale=rationale,
                before=before,
                after=dict(active["skills"].get(key) or {}),
            )
        return live

    def _remove_pointer(
        self,
        candidate: LearningCandidate,
        *,
        actor: str,
        rationale: str,
        status: str,
        event: str,
    ) -> LearningCandidate:
        actor_name = self._human(actor)
        with self.store.promotion_lock():
            live = self._live(candidate)
            key = self._skill_key(live)
            active = self.store.read_active_skills()
            before = dict(active["skills"].get(key) or {})
            pointers = active["skills"].get(key) or {}
            version = self._version(live)
            if str(pointers.get("stable") or "") == version:
                pointers.pop("stable", None)
                if str(pointers.get("selected_version") or "") == version:
                    repository_target = self.repo_root / "skills" / live.group / live.name
                    if live.action == "skill" and repository_target.is_dir():
                        pointers["selected_version"] = "base"
                    else:
                        pointers["selected_version"] = ""
                        pointers["enabled"] = False
                    pointers["update_policy"] = "pinned"
                if live.action == "skill":
                    self._remove_materialized_skill(live)
            canary = pointers.get("canary") if isinstance(pointers.get("canary"), dict) else {}
            if str(canary.get("version") or "") == version:
                pointers.pop("canary", None)
            if pointers:
                active["skills"][key] = pointers
            else:
                active["skills"].pop(key, None)
            self.store.write_active_skills(active)
            live = self.store.update_candidate_status(live.candidate_id, status)
            self._audit_pointer(
                event=event,
                candidate=live,
                actor=actor_name,
                rationale=rationale,
                before=before,
                after=dict(active["skills"].get(key) or {}),
            )
        return live

    @staticmethod
    def _human(actor: str) -> str:
        actor_name = str(actor or "").strip()
        if not actor_name:
            raise ValueError("authenticated human actor is required")
        return actor_name

    def _live(self, candidate: LearningCandidate) -> LearningCandidate:
        live = self.store.read_candidate(candidate.candidate_id)
        if live is None:
            raise ValueError("candidate disappeared before the decision")
        if live.revision != candidate.revision:
            raise PromotionConflict("candidate revision changed; reopen the exact revision before deciding")
        return live

    def _promote_memory(self, candidate: LearningCandidate) -> None:
        revision_root = self.store.revision_dir(candidate.candidate_id, candidate.revision)
        source = revision_root / "memories" / "AGENTS.md"
        updated = source.read_text(encoding="utf-8")
        proposed_hash = hash_text(updated)
        if not candidate.bundle_hash or proposed_hash != candidate.bundle_hash:
            raise PromotionConflict("reviewed memory file changed before stable promotion")
        current = self.store.read_memory_text()
        current_hash = hash_text(current)
        if current_hash != candidate.base_target_hash:
            raise PromotionConflict("workspace memory changed after this revision was created")
        before = revision_root / "before" / "memories" / "AGENTS.md"
        before.parent.mkdir(parents=True, exist_ok=True)
        if before.exists() and before.read_text(encoding="utf-8") != current:
            raise PromotionConflict("immutable memory rollback snapshot differs")
        if not before.exists():
            before.write_text(current, encoding="utf-8")
        swapped, observed_hash = self.store.compare_and_swap_memory(
            expected_hash=current_hash,
            new_text=updated,
        )
        if not swapped:
            raise PromotionConflict(f"workspace memory changed during promotion: current {observed_hash}")

    def _effective_skill_path(self, candidate: LearningCandidate) -> tuple[Path, bool]:
        workspace_target = self.store.self_develop_skills_dir / candidate.group / candidate.name
        if workspace_target.is_dir():
            return workspace_target, True
        return self.repo_root / "skills" / candidate.group / candidate.name, False

    def _materialize_skill(self, candidate: LearningCandidate) -> None:
        source = (
            self.store.revision_dir(candidate.candidate_id, candidate.revision)
            / "proposed"
            / candidate.group
            / candidate.name
        )
        if hash_tree(source) != candidate.bundle_hash:
            raise PromotionConflict("reviewed skill bundle changed before stable promotion")
        target = self.store.self_develop_skills_dir / candidate.group / candidate.name
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{candidate.name}.stable-", dir=str(target.parent)))
        try:
            shutil.copytree(source, temp, dirs_exist_ok=True)
            discard = target.parent / f".{candidate.name}.old-{candidate.candidate_id}"
            self._remove_path(discard)
            if target.exists():
                os.replace(target, discard)
            os.replace(temp, target)
            self._remove_path(discard)
        finally:
            self._remove_path(temp)

    def _materialize_pointer(self, version: str, *, group: str, name: str) -> None:
        parsed = self._parse_version(version)
        if parsed is None:
            raise PromotionConflict("previous stable pointer is invalid")
        candidate_id, revision = parsed
        candidate = self.store.read_candidate_revision(candidate_id, revision)
        if candidate is None:
            raise PromotionConflict("previous stable revision is unavailable")
        self._materialize_skill(candidate)

    @staticmethod
    def _parse_version(version: str) -> tuple[str, int] | None:
        candidate_id, separator, revision_text = str(version or "").partition("@r")
        if not separator or not candidate_id or not revision_text.isdigit():
            return None
        return candidate_id, int(revision_text)

    def _remove_materialized_skill(self, candidate: LearningCandidate) -> None:
        target = self.store.self_develop_skills_dir / candidate.group / candidate.name
        self._remove_path(target)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            for item in [path, *path.rglob("*")]:
                if item.is_symlink():
                    continue
                with contextlib.suppress(OSError):
                    writable = stat.S_IWUSR | (stat.S_IXUSR if item.is_dir() else 0)
                    item.chmod(item.stat().st_mode | writable)
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            with contextlib.suppress(OSError):
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
            path.unlink()

    def _rollback_memory(self, candidate: LearningCandidate) -> None:
        before = (
            self.store.revision_dir(candidate.candidate_id, candidate.revision)
            / "before"
            / "memories"
            / "AGENTS.md"
        )
        if not before.is_file():
            raise PromotionConflict("prior workspace memory snapshot is missing")
        current = self.store.read_memory_text()
        if hash_text(current) != candidate.bundle_hash:
            raise PromotionConflict("workspace memory changed after this revision became stable")
        restored = before.read_text(encoding="utf-8")
        swapped, observed_hash = self.store.compare_and_swap_memory(
            expected_hash=hash_text(current),
            new_text=restored,
        )
        if not swapped:
            raise PromotionConflict(f"workspace memory changed during rollback: current {observed_hash}")

    def _previous_stable_pointer(self, candidate: LearningCandidate) -> str:
        if not self.store.audit_log_path.is_file():
            return ""
        try:
            lines = self.store.audit_log_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PromotionConflict(
                f"cannot read release audit history {self.store.audit_log_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for line_number, line in reversed(list(enumerate(lines, start=1))):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception as exc:
                raise PromotionConflict(
                    f"cannot decode release audit history line {line_number}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise PromotionConflict(
                    f"release audit history line {line_number} is not a JSON object"
                )
            if event.get("event") != "stable_promoted":
                continue
            if event.get("candidate_id") != candidate.candidate_id:
                continue
            before = event.get("pointer_before") if isinstance(event.get("pointer_before"), dict) else {}
            return str(before.get("stable") or "")
        return ""

    def _audit_pointer(
        self,
        *,
        event: str,
        candidate: LearningCandidate,
        actor: str,
        rationale: str,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:
        self.store.append_audit_event(
            {
                "event": event,
                "candidate_id": candidate.candidate_id,
                "revision": candidate.revision,
                "target": self._skill_key(candidate),
                "version": self._version(candidate),
                "actor": actor,
                "rationale": str(rationale or "").strip(),
                "pointer_before": before,
                "pointer_after": after,
                "candidate_hash": candidate.bundle_hash,
                "decided_at": utc_now(),
            }
        )


__all__ = ["PromotionConflict", "PromotionManager"]
