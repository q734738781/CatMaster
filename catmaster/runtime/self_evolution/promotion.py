from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .models import LearningCandidate, ValidationReport
from .storage import SelfEvolutionStore, hash_text, hash_tree, utc_now


class PromotionConflict(RuntimeError):
    pass


class PromotionManager:
    def __init__(self, store: SelfEvolutionStore, *, repo_root: Path | str | None = None) -> None:
        self.store = store
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()

    def promotion_readiness(self, candidate: LearningCandidate) -> dict[str, Any]:
        if candidate.status not in {"reviewed", "approved"}:
            return {
                "ready": False,
                "reason": f"Candidate status is {candidate.status or 'unknown'}, not awaiting a human decision.",
                "bundle_unchanged": False,
                "target_unchanged": False,
            }

        candidate_dir = self.store.candidate_dir(candidate.candidate_id)
        if candidate.action == "memory":
            source = candidate_dir / "memories" / "AGENTS.md"
            proposed_hash = hash_text(source.read_text(encoding="utf-8")) if source.is_file() else ""
            current_hash = self.store.memory_hash()
            target_label = "workspace memory"
        else:
            source = candidate_dir / "proposed" / candidate.group / candidate.name
            proposed_hash = hash_tree(source)
            current_target, _had_workspace_target = self._effective_skill_path(candidate)
            current_hash = hash_tree(current_target)
            target_label = "effective workspace skill"

        bundle_unchanged = bool(candidate.bundle_hash) and proposed_hash == candidate.bundle_hash
        target_unchanged = current_hash == candidate.base_target_hash
        reasons: list[str] = []
        if not bundle_unchanged:
            reasons.append("The reviewed candidate bundle changed or is missing.")
        if not target_unchanged:
            reasons.append(f"The {target_label} changed after this candidate was proposed; regenerate it from the current version.")
        return {
            "ready": bundle_unchanged and target_unchanged,
            "reason": " ".join(reasons),
            "bundle_unchanged": bundle_unchanged,
            "target_unchanged": target_unchanged,
            "expected_target_hash": candidate.base_target_hash,
            "current_target_hash": current_hash,
        }

    def promote(
        self,
        candidate: LearningCandidate,
        report: ValidationReport,
        *,
        decision_source: str = "automatic_system",
        actor: str = "",
        rationale: str = "",
    ) -> LearningCandidate:
        if candidate.status not in {"reviewed", "approved"}:
            raise ValueError("only reviewed candidates awaiting a decision can be promoted")
        if not report.valid:
            raise ValueError("candidate validation did not pass")
        source = str(decision_source or "").strip()
        actor_name = str(actor or "").strip()
        if candidate.action == "skill" and source != "human":
            raise ValueError("workspace skill promotion requires an explicit human decision")
        if source == "human" and not actor_name:
            raise ValueError("authenticated human actor is required")
        decision = {
            "decision_source": source,
            "actor": actor_name,
            "decided_at": utc_now(),
            "candidate_hash": candidate.bundle_hash,
            "rationale": str(rationale or "").strip(),
        }
        try:
            with self.store.promotion_lock():
                live_candidate = self.store.read_candidate(candidate.candidate_id)
                if live_candidate is None:
                    raise ValueError("candidate disappeared before promotion")
                if live_candidate.status not in {"reviewed", "approved"}:
                    raise ValueError(
                        f"candidate decision is already final with status {live_candidate.status or 'unknown'}"
                    )
                candidate = live_candidate
                decision["candidate_hash"] = candidate.bundle_hash
                if source == "human":
                    self.store.append_audit_event(
                        {
                            "event": "human_decision",
                            "candidate_id": candidate.candidate_id,
                            "action": "promote",
                            **decision,
                        }
                    )
                if candidate.action == "memory":
                    self._promote_memory(candidate)
                else:
                    self._promote_skill(candidate)
                candidate.status = "promoted"
                candidate.promotion = {
                    **dict(candidate.promotion),
                    "promoted_at": utc_now(),
                    "decision": decision,
                }
                self.store.write_candidate(candidate)
                self.store.append_audit_event(
                    {
                        "event": "candidate_promoted",
                        "candidate_id": candidate.candidate_id,
                        "action": candidate.action,
                        "group": candidate.group,
                        "name": candidate.name,
                        **decision,
                        "promotion": candidate.promotion,
                    }
                )
        except PromotionConflict as exc:
            candidate.status = "conflict"
            candidate.promotion = {
                **dict(candidate.promotion),
                "error": str(exc),
                "decision": decision,
            }
            self.store.write_candidate(candidate)
            self.store.append_audit_event(
                {
                    "event": "promotion_conflict",
                    "candidate_id": candidate.candidate_id,
                    "action": candidate.action,
                    **decision,
                    "error": str(exc),
                }
            )
            raise
        return candidate

    def reject(
        self,
        candidate: LearningCandidate,
        *,
        actor: str,
        rationale: str = "",
    ) -> LearningCandidate:
        if candidate.status not in {"reviewed", "approved"}:
            raise ValueError("only reviewed candidates awaiting a decision can be rejected")
        actor_name = str(actor or "").strip()
        if not actor_name:
            raise ValueError("authenticated human actor is required")
        with self.store.promotion_lock():
            live_candidate = self.store.read_candidate(candidate.candidate_id)
            if live_candidate is None:
                raise ValueError("candidate disappeared before rejection")
            if live_candidate.status not in {"reviewed", "approved"}:
                raise ValueError(
                    f"candidate decision is already final with status {live_candidate.status or 'unknown'}"
                )
            candidate = live_candidate
            decision = {
                "decision_source": "human",
                "actor": actor_name,
                "decided_at": utc_now(),
                "candidate_hash": candidate.bundle_hash,
                "rationale": str(rationale or "").strip(),
            }
            candidate.status = "rejected"
            candidate.review = {
                **dict(candidate.review or {}),
                "human_decision": {"action": "reject", **decision},
            }
            self.store.write_candidate(candidate)
            self.store.append_audit_event(
                {
                    "event": "human_decision",
                    "candidate_id": candidate.candidate_id,
                    "action": "reject",
                    **decision,
                }
            )
        return candidate

    def _promote_memory(self, candidate: LearningCandidate) -> None:
        candidate_dir = self.store.candidate_dir(candidate.candidate_id)
        source = candidate_dir / "memories" / "AGENTS.md"
        updated = source.read_text(encoding="utf-8")
        proposed_hash = hash_text(updated)
        if not candidate.bundle_hash or proposed_hash != candidate.bundle_hash:
            raise PromotionConflict("reviewed memory file changed before promotion")

        current = self.store.read_memory_text()
        current_hash = hash_text(current)
        if current_hash != candidate.base_target_hash:
            raise PromotionConflict(
                f"workspace memory changed after proposal: expected {candidate.base_target_hash or '<missing>'}, "
                f"current {current_hash or '<missing>'}"
            )

        before = candidate_dir / "before" / "memories" / "AGENTS.md"
        before.parent.mkdir(parents=True, exist_ok=True)
        before.write_text(current, encoding="utf-8")
        swapped, observed_hash = self.store.compare_and_swap_memory(
            expected_hash=current_hash,
            new_text=updated,
        )
        if not swapped:
            raise PromotionConflict(f"workspace memory changed during promotion: current {observed_hash or '<missing>'}")
        candidate.promotion = {
            "target_path": "/memories/AGENTS.md",
            "before_hash": current_hash,
            "promoted_hash": proposed_hash,
        }

    def _effective_skill_path(self, candidate: LearningCandidate) -> tuple[Path, bool]:
        workspace_target = self.store.self_develop_skills_dir / candidate.group / candidate.name
        if workspace_target.is_dir():
            return workspace_target, True
        return self.repo_root / "skills" / candidate.group / candidate.name, False

    def _promote_skill(self, candidate: LearningCandidate) -> None:
        source = self.store.candidate_dir(candidate.candidate_id) / "proposed" / candidate.group / candidate.name
        source_hash = hash_tree(source)
        if not source_hash or source_hash != candidate.bundle_hash:
            raise PromotionConflict("reviewed skill bundle changed before promotion")

        current_effective, had_workspace_target = self._effective_skill_path(candidate)
        current_hash = hash_tree(current_effective)
        if current_hash != candidate.base_target_hash:
            raise PromotionConflict(
                f"skill target changed after proposal: expected {candidate.base_target_hash or '<missing>'}, "
                f"current {current_hash or '<missing>'}"
            )

        target = self.store.self_develop_skills_dir / candidate.group / candidate.name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = self.store.candidate_dir(candidate.candidate_id) / "before" / "workspace_bundle"
        if before.exists():
            shutil.rmtree(before)
        before.parent.mkdir(parents=True, exist_ok=True)

        temp = Path(tempfile.mkdtemp(prefix=f".{candidate.name}.candidate-", dir=str(target.parent)))
        try:
            shutil.copytree(source, temp, dirs_exist_ok=True)
            if hash_tree(temp) != source_hash:
                raise PromotionConflict("temporary skill bundle hash differs from reviewed bundle")
            if target.exists():
                os.replace(target, before)
            try:
                os.replace(temp, target)
            except Exception:
                if before.exists() and not target.exists():
                    os.replace(before, target)
                raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)

        candidate.promotion = {
            "target_path": str(target.relative_to(self.store.root)),
            "had_workspace_target": had_workspace_target,
            "before_hash": current_hash,
            "promoted_hash": hash_tree(target),
        }

    def rollback(
        self,
        candidate: LearningCandidate,
        *,
        actor: str = "",
        rationale: str = "",
    ) -> LearningCandidate:
        if candidate.status != "promoted":
            raise ValueError("only promoted candidates can be rolled back")
        with self.store.promotion_lock():
            if candidate.action == "memory":
                self._rollback_memory(candidate)
            else:
                self._rollback_skill(candidate)
            candidate.status = "rolled_back"
            candidate.promotion = {**dict(candidate.promotion), "rolled_back_at": utc_now()}
            self.store.write_candidate(candidate)
            self.store.append_audit_event(
                {
                    "event": "candidate_rolled_back",
                    "candidate_id": candidate.candidate_id,
                    "action": candidate.action,
                    "group": candidate.group,
                    "name": candidate.name,
                    "decision_source": "human" if str(actor or "").strip() else "system",
                    "actor": str(actor or "").strip(),
                    "rationale": str(rationale or "").strip(),
                    "candidate_hash": candidate.bundle_hash,
                }
            )
        return candidate

    def _rollback_memory(self, candidate: LearningCandidate) -> None:
        before = self.store.candidate_dir(candidate.candidate_id) / "before" / "memories" / "AGENTS.md"
        if not before.is_file():
            raise PromotionConflict("prior workspace memory file is missing")
        current = self.store.read_memory_text()
        current_hash = hash_text(current)
        promoted_hash = str(candidate.promotion.get("promoted_hash") or "")
        if current_hash != promoted_hash:
            raise PromotionConflict("workspace memory changed after this candidate was promoted")
        restored = before.read_text(encoding="utf-8")
        swapped, observed_hash = self.store.compare_and_swap_memory(
            expected_hash=current_hash,
            new_text=restored,
        )
        if not swapped:
            raise PromotionConflict(f"workspace memory changed during rollback: current {observed_hash or '<missing>'}")
        candidate.promotion["rollback_hash"] = hash_text(restored)

    def _rollback_skill(self, candidate: LearningCandidate) -> None:
        target = self.store.self_develop_skills_dir / candidate.group / candidate.name
        promoted_hash = str(candidate.promotion.get("promoted_hash") or "")
        if hash_tree(target) != promoted_hash:
            raise PromotionConflict("workspace skill changed after this candidate was promoted")
        before = self.store.candidate_dir(candidate.candidate_id) / "before" / "workspace_bundle"
        discard = target.parent / f".{candidate.name}.rollback-{candidate.candidate_id}"
        with contextlib.suppress(FileNotFoundError):
            if discard.is_dir():
                shutil.rmtree(discard)
            else:
                discard.unlink()
        os.replace(target, discard)
        try:
            if bool(candidate.promotion.get("had_workspace_target")):
                if not before.is_dir():
                    raise PromotionConflict("prior workspace skill bundle is missing")
                os.replace(before, target)
        except Exception:
            if discard.exists() and not target.exists():
                os.replace(discard, target)
            raise
        finally:
            if discard.exists():
                shutil.rmtree(discard)
        candidate.promotion["rollback_hash"] = hash_tree(target)


__all__ = ["PromotionConflict", "PromotionManager"]
