from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from .models import LearningCandidate, SKILL_GROUPS, ValidationReport
from .storage import SelfEvolutionStore, hash_text


_REQUIRED_SECTIONS = (
    "## Overview",
    "## Quick Start",
    "## Workflow",
    "## Method-critical defaults",
    "## Output Contract",
    "## References",
)
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_MAX_MEMORY_BYTES = 512 * 1024


def _frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("SKILL.md frontmatter is not closed") from exc
    raw = "\n".join(lines[1:end])
    if yaml is None:
        data: dict[str, Any] = {}
        for line in raw.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                data[key.strip()] = value.strip().strip("'\"")
        return data
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return dict(loaded)


def _is_within(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved = path.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


class CandidateGate:
    def __init__(
        self,
        store: SelfEvolutionStore,
        *,
        max_files: int = 200,
        max_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self.store = store
        self.max_files = max(1, int(max_files))
        self.max_bytes = max(1024, int(max_bytes))

    def run(self, candidate: LearningCandidate) -> ValidationReport:
        checks: list[str] = []
        errors: list[str] = []
        if candidate.action == "memory":
            self._validate_memory(candidate, checks=checks, errors=errors)
        elif candidate.action == "skill":
            self._validate_skill(candidate, checks=checks, errors=errors)
        else:
            errors.append(f"unsupported candidate action: {candidate.action}")
        return ValidationReport(
            candidate_id=candidate.candidate_id,
            valid=not errors,
            checks=checks,
            errors=errors,
        )

    def _validate_memory(self, candidate: LearningCandidate, *, checks: list[str], errors: list[str]) -> None:
        root = self.store.candidate_dir(candidate.candidate_id)
        path = root / "memories" / "AGENTS.md"
        if not path.is_file():
            errors.append("memory candidate must contain memories/AGENTS.md")
            return
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        text = raw_text.strip()
        if not text:
            errors.append("memories/AGENTS.md is empty")
        elif len(text.encode("utf-8")) > _MAX_MEMORY_BYTES:
            errors.append("memories/AGENTS.md exceeds 512 KB")
        else:
            checks.append("complete memory file is non-empty and bounded")
        proposed_hash = hash_text(raw_text)
        if proposed_hash == candidate.base_target_hash:
            errors.append("memory candidate does not change the current AGENTS.md")
        if candidate.bundle_hash and proposed_hash != candidate.bundle_hash:
            errors.append("memory candidate changed after its content hash was recorded")
        proposed = root / "proposed"
        if proposed.exists() and any(path.is_file() for path in proposed.rglob("*")):
            errors.append("memory candidate must not also contain a skill bundle")

    def _validate_skill(self, candidate: LearningCandidate, *, checks: list[str], errors: list[str]) -> None:
        if candidate.group not in SKILL_GROUPS:
            errors.append(f"unknown skill group: {candidate.group!r}")
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,119}", candidate.name):
            errors.append(f"invalid skill name: {candidate.name!r}")
            return
        root = self.store.candidate_dir(candidate.candidate_id) / "proposed" / candidate.group / candidate.name
        if not root.is_dir():
            errors.append(f"skill bundle is missing: proposed/{candidate.group}/{candidate.name}")
            return
        memory_path = self.store.candidate_dir(candidate.candidate_id) / "memories" / "AGENTS.md"
        if memory_path.is_file():
            errors.append("skill candidate must not also contain a memory edit")

        files: list[Path] = []
        total_bytes = 0
        for path in root.rglob("*"):
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                errors.append(f"cannot inspect {path.relative_to(root)}: {exc}")
                continue
            relative = path.relative_to(root)
            if path.is_symlink():
                errors.append(f"symlinks are forbidden: {relative}")
                continue
            if path.is_dir():
                continue
            if not stat.S_ISREG(mode):
                errors.append(f"special files are forbidden: {relative}")
                continue
            if not _is_within(root, path):
                errors.append(f"path escapes skill bundle: {relative}")
                continue
            files.append(path)
            total_bytes += path.stat().st_size

        if len(files) > self.max_files:
            errors.append(f"skill bundle has {len(files)} files; limit is {self.max_files}")
        if total_bytes > self.max_bytes:
            errors.append(f"skill bundle has {total_bytes} bytes; limit is {self.max_bytes}")
        if not files:
            errors.append("skill bundle is empty")
        else:
            checks.append(f"bundle containment and size passed ({len(files)} files, {total_bytes} bytes)")

        skill_md = root / "SKILL.md"
        if not skill_md.is_file():
            errors.append("skill bundle must contain SKILL.md")
            return
        try:
            frontmatter = _frontmatter(skill_md)
        except Exception as exc:
            errors.append(str(exc))
            return
        if str(frontmatter.get("name") or "").strip() != candidate.name:
            errors.append("SKILL.md frontmatter name must match the skill directory")
        if not str(frontmatter.get("description") or "").strip():
            errors.append("SKILL.md frontmatter description is required")
        workspace_group = self.store.self_develop_skills_dir / candidate.group
        for other_skill_md in workspace_group.glob("*/SKILL.md"):
            if other_skill_md.parent.name == candidate.name:
                continue
            try:
                other_name = str(_frontmatter(other_skill_md).get("name") or "").strip()
            except Exception:
                continue
            if other_name == candidate.name:
                errors.append(
                    f"workspace skill name {candidate.name!r} already exists in directory {other_skill_md.parent.name!r}"
                )
                break
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        positions = [text.find(section) for section in _REQUIRED_SECTIONS]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            errors.append("SKILL.md required sections are missing or out of order")
        else:
            checks.append("SKILL.md frontmatter and section order passed")

        for match in _MARKDOWN_LINK.finditer(text):
            raw_target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if not raw_target or raw_target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            pure = PurePosixPath(raw_target)
            if pure.is_absolute() or ".." in pure.parts:
                errors.append(f"local reference escapes bundle: {raw_target}")
                continue
            target = root / pure
            if not target.exists():
                errors.append(f"referenced local file is missing: {raw_target}")
        if not any(error.startswith(("local reference", "referenced local")) for error in errors):
            checks.append("local Markdown references resolve inside the bundle")

        for path in files:
            relative = path.relative_to(root)
            if path.suffix == ".py":
                try:
                    compile(path.read_text(encoding="utf-8"), str(relative), "exec")
                except Exception as exc:
                    errors.append(f"Python syntax failed for {relative}: {exc}")
            elif path.suffix in {".sh", ".bash"}:
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode != 0:
                    errors.append(f"shell syntax failed for {relative}: {result.stderr.strip()}")
        if not any("syntax failed" in error for error in errors):
            checks.append("code syntax smoke checks passed")


__all__ = ["CandidateGate"]
