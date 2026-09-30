from __future__ import annotations

import logging
import stat
from pathlib import Path
from threading import RLock
from typing import Any

from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
import yaml

from .models import LearningCandidate, SKILL_GROUPS, ValidationReport
from .storage import SelfEvolutionStore, hash_text, hash_tree


_DEEPAGENTS_SKILLS_LOGGER = "deepagents.middleware.skills"
_LOAD_PROBE_LOCK = RLock()


def read_skill_frontmatter(path: Path) -> dict[str, Any]:
    """Read frontmatter for catalog display without imposing authoring style."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("SKILL.md frontmatter is not closed") from exc
    raw = "\n".join(lines[1:end])
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return dict(loaded)


_frontmatter = read_skill_frontmatter


def _is_within(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved = path.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _safe_path_component(value: str) -> bool:
    """Return whether a model-selected name stays one filesystem component."""

    text = str(value or "")
    return bool(
        text
        and text not in {".", ".."}
        and "/" not in text
        and "\\" not in text
        and "\x00" not in text
    )


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - logging must not break validation
            return
        self.messages.append(message)


def probe_skill_loadability(
    *,
    revision_root: Path,
    group: str,
    name: str,
) -> tuple[bool, list[str]]:
    """Probe a candidate through the active public DeepAgents middleware.

    This intentionally does not reproduce the loader's metadata rules. It asks
    the installed runtime to discover the candidate and captures its own
    diagnostics for the evolving agent.
    """

    proposed_root = Path(revision_root) / "proposed"
    expected_path = f"/{group}/{name}/SKILL.md"
    skill_root = proposed_root / group / name
    if not skill_root.is_dir():
        return False, [f"candidate skill directory is missing: {expected_path.rsplit('/', 1)[0]}"]
    if not (skill_root / "SKILL.md").is_file():
        return False, [f"active DeepAgents loader requires {expected_path}"]

    collector = _WarningCollector()
    loader_logger = logging.getLogger(_DEEPAGENTS_SKILLS_LOGGER)
    with _LOAD_PROBE_LOCK:
        loader_logger.addHandler(collector)
        try:
            backend = FilesystemBackend(root_dir=proposed_root, virtual_mode=True)
            middleware = SkillsMiddleware(
                backend=backend,
                sources=[f"/{group}/"],
                system_prompt=None,
            )
            update = middleware.before_agent({}, None, {}) or {}  # type: ignore[arg-type]
        except Exception as exc:
            # A loader integration failure is a host diagnostic, not evidence that
            # the candidate's SOP is bad or a reason to fabricate a replacement.
            return False, [
                "active DeepAgents load probe failed without judging the candidate: "
                f"{type(exc).__name__}: {exc}"
            ]
        finally:
            loader_logger.removeHandler(collector)

    metadata = [
        item
        for item in list(update.get("skills_metadata") or [])
        if isinstance(item, dict)
    ]
    loaded = any(str(item.get("path") or "") == expected_path for item in metadata)
    # Do not maintain a second classifier for loader failures. When discovery
    # fails, return the loader's warnings verbatim so the proposer can repair
    # against the implementation that will actually load the skill.
    relevant = [] if loaded else list(collector.messages)
    for load_error in list(update.get("skills_load_errors") or []):
        message = str(load_error or "").strip()
        if message:
            relevant.append(message)
    if not loaded and not relevant:
        relevant.append(
            f"active DeepAgents loader did not discover {expected_path}; inspect its SKILL.md load diagnostics"
        )
    return loaded, list(dict.fromkeys(relevant))


class CandidateGate:
    """Check concrete transaction boundaries and report everything else.

    ``valid`` means only that the candidate is safe to retain as an immutable
    review artifact. It is not a quality score. Loader failures and ineffective
    deltas are repairable diagnostics returned to the proposer and reviewer.
    """

    def __init__(self, store: SelfEvolutionStore) -> None:
        self.store = store

    def run(self, candidate: LearningCandidate) -> ValidationReport:
        checks: list[str] = []
        errors: list[str] = []
        diagnostics: list[str] = []
        loadable = True
        repair_required = False
        if candidate.action == "memory":
            repair_required = self._validate_memory(
                candidate,
                checks=checks,
                errors=errors,
                diagnostics=diagnostics,
            )
        elif candidate.action == "skill":
            loadable, repair_required = self._validate_skill(
                candidate,
                checks=checks,
                errors=errors,
                diagnostics=diagnostics,
            )
        else:
            errors.append(f"unsupported candidate action: {candidate.action}")
            repair_required = True
        return ValidationReport(
            candidate_id=candidate.candidate_id,
            valid=not errors,
            checks=checks,
            errors=errors,
            diagnostics=diagnostics,
            loadable=loadable,
            repair_required=repair_required or bool(errors),
        )

    def _validate_memory(
        self,
        candidate: LearningCandidate,
        *,
        checks: list[str],
        errors: list[str],
        diagnostics: list[str],
    ) -> bool:
        root = self.store.revision_dir(candidate.candidate_id, candidate.revision)
        path = root / "memories" / "AGENTS.md"
        repair_required = False
        if path.is_symlink():
            errors.append("candidate memory path must not be a symlink")
            return True
        if not path.is_file():
            diagnostics.append("memory candidate has no memories/AGENTS.md to review")
            repair_required = True
        else:
            try:
                raw_text = path.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError) as exc:
                errors.append(
                    "cannot read candidate memory /memories/AGENTS.md: "
                    f"{type(exc).__name__}: {exc}"
                )
                return True
            checks.append("candidate memory is readable")
            proposed_hash = hash_text(raw_text)
            if proposed_hash == candidate.base_target_hash:
                diagnostics.append("memory candidate does not change the current AGENTS.md")
            if candidate.bundle_hash and proposed_hash != candidate.bundle_hash:
                errors.append(
                    "reviewed memory bytes changed after the revision identity was recorded"
                )
        proposed = root / "proposed"
        if proposed.exists() and any(item.is_file() for item in proposed.rglob("*")):
            diagnostics.append(
                "candidate revision also contains skill-bundle files outside its selected memory action"
            )
        return repair_required

    def _validate_skill(
        self,
        candidate: LearningCandidate,
        *,
        checks: list[str],
        errors: list[str],
        diagnostics: list[str],
    ) -> tuple[bool, bool]:
        if not _safe_path_component(candidate.group):
            errors.append(f"skill group is not one safe path component: {candidate.group!r}")
            return False, True
        if not _safe_path_component(candidate.name):
            errors.append(f"skill name is not one safe path component: {candidate.name!r}")
            return False, True
        mounted = candidate.group in SKILL_GROUPS
        if not mounted:
            diagnostics.append(
                f"{candidate.group!r} is not mounted by the active CatMaster skill runtime; "
                f"available roots: {', '.join(SKILL_GROUPS)}"
            )

        revision_root = self.store.revision_dir(candidate.candidate_id, candidate.revision)
        root = revision_root / "proposed" / candidate.group / candidate.name
        if not _is_within(revision_root / "proposed", root):
            errors.append("candidate target escapes the authorized proposed root")
            return False, True

        repair_required = not mounted
        if not root.is_dir():
            diagnostics.append(
                f"candidate bundle is missing: /proposed/{candidate.group}/{candidate.name}"
            )
            return False, True

        memory_path = revision_root / "memories" / "AGENTS.md"
        current_memory = revision_root / "current" / "AGENTS.md"
        if memory_path.is_file() and (
            not current_memory.is_file() or memory_path.read_bytes() != current_memory.read_bytes()
        ):
            diagnostics.append(
                "candidate revision also contains a changed memory file outside its selected skill action"
            )

        for path in root.rglob("*"):
            relative = path.relative_to(root)
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                errors.append(
                    f"cannot inspect candidate path /proposed/{candidate.group}/{candidate.name}/{relative}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if path.is_symlink():
                errors.append(
                    f"candidate transaction cannot contain a symlink: {relative}"
                )
                continue
            if path.is_dir():
                continue
            if not stat.S_ISREG(mode):
                errors.append(
                    f"candidate transaction cannot contain a special file: {relative}"
                )
                continue
            if not _is_within(root, path):
                errors.append(f"candidate path escapes its authorized bundle: {relative}")
                continue
        checks.append("candidate paths stay inside the authorized bundle")

        loader_discovered, loader_diagnostics = probe_skill_loadability(
            revision_root=revision_root,
            group=candidate.group,
            name=candidate.name,
        )
        diagnostics.extend(loader_diagnostics)
        loadable = mounted and loader_discovered
        if loadable:
            checks.append("active DeepAgents middleware discovers the candidate skill")
        else:
            repair_required = True

        try:
            proposed_hash = hash_tree(root)
        except OSError as exc:
            errors.append(
                "cannot read candidate bundle for immutable revision identity: "
                f"{type(exc).__name__}: {exc}"
            )
            return loadable, True
        if proposed_hash and proposed_hash == candidate.base_target_hash:
            diagnostics.append("skill candidate does not change the current effective bundle")
        if candidate.bundle_hash and proposed_hash != candidate.bundle_hash:
            errors.append(
                "reviewed skill bytes changed after the revision identity was recorded"
            )
        return loadable, repair_required


__all__ = [
    "CandidateGate",
    "probe_skill_loadability",
    "read_skill_frontmatter",
]
