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
_SECTION_HEADING = re.compile(r"^##\s+.+$", re.MULTILINE)
_PLACEHOLDER = re.compile(
    r"\b(?:replace this(?: with)?|todo|tbd|fill (?:this|it) in)\b",
    re.IGNORECASE,
)
_MAX_MEMORY_BYTES = 512 * 1024
_DEEPAGENT_BUILTINS = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
}
_DEEPAGENT_TASK_TOOL = {"task"}


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
        allowed_tool_names: set[str] | None = None,
    ) -> None:
        self.store = store
        self.max_files = max(1, int(max_files))
        self.max_bytes = max(1024, int(max_bytes))
        self._allowed_tool_names_override = (
            None if allowed_tool_names is None else set(allowed_tool_names)
        )
        if self._allowed_tool_names_override is None:
            from catmaster.tools.registry import get_tool_registry

            self._registered_tool_names = {
                str(name).strip()
                for name in get_tool_registry().tools
                if str(name).strip()
            }
        else:
            self._registered_tool_names = set(self._allowed_tool_names_override)

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
        root = self.store.revision_dir(candidate.candidate_id, candidate.revision)
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
        revision_root = self.store.revision_dir(candidate.candidate_id, candidate.revision)
        root = revision_root / "proposed" / candidate.group / candidate.name
        if not root.is_dir():
            errors.append(f"skill bundle is missing: proposed/{candidate.group}/{candidate.name}")
            return
        memory_path = revision_root / "memories" / "AGENTS.md"
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
        description = str(frontmatter.get("description") or "").strip()
        if not description:
            errors.append("SKILL.md frontmatter description is required")
        elif _PLACEHOLDER.search(description):
            errors.append("SKILL.md frontmatter description still contains scaffold placeholder text")
        declared_tools = self._declared_tools(frontmatter.get("allowed-tools"))
        allowed_tools = self._allowed_tools_for_group(candidate.group)
        unknown_tools = sorted(set(declared_tools) - allowed_tools)
        if unknown_tools:
            errors.append(
                f"SKILL.md declares tools absent from the final {candidate.group} "
                "specialist/worker surface: "
                + ", ".join(unknown_tools)
            )
        elif declared_tools:
            checks.append(
                f"declared allowed-tools resolve on the final {candidate.group} "
                "specialist/worker surface"
            )
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
            empty_sections = [
                section
                for section in _REQUIRED_SECTIONS
                if not self._section_content(text, section)
            ]
            if empty_sections:
                errors.append(
                    "SKILL.md scaffold sections must contain substantive content: "
                    + ", ".join(empty_sections)
                )
            elif _PLACEHOLDER.search(self._body_without_frontmatter(text)):
                errors.append("SKILL.md body still contains scaffold placeholder text")
            else:
                checks.append("required sections contain non-placeholder content")

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

    @staticmethod
    def _declared_tools(value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split() if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _body_without_frontmatter(text: str) -> str:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return text
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[index + 1 :])
        return text

    @staticmethod
    def _section_content(text: str, section: str) -> str:
        start = text.find(section)
        if start < 0:
            return ""
        body_start = start + len(section)
        next_heading = _SECTION_HEADING.search(text, body_start)
        body_end = next_heading.start() if next_heading else len(text)
        content = text[body_start:body_end]
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        return content.strip()

    def _allowed_tools_for_group(self, group: str) -> set[str]:
        if self._allowed_tool_names_override is not None:
            return (
                set(self._allowed_tool_names_override)
                | _DEEPAGENT_BUILTINS
                | _DEEPAGENT_TASK_TOOL
            )

        surfaces = self._runtime_tool_surfaces().get(group, ())
        if not surfaces:
            return set(_DEEPAGENT_BUILTINS)

        available = (
            set(self._registered_tool_names)
            | _DEEPAGENT_BUILTINS
            | _DEEPAGENT_TASK_TOOL
        )
        resolved = [set(surface) & available for surface in surfaces]
        # Shared skill groups are mounted into every listed consumer. A declared
        # tool is safe only when every consumer that may load the skill owns it.
        return set.intersection(*resolved) if resolved else set()

    @staticmethod
    def _runtime_tool_surfaces() -> dict[str, tuple[set[str], ...]]:
        """Read the specialist runtime's actual static surfaces at validation time.

        The allowlists remain owned by ``catmaster.specialists.runtime``. Importing
        them lazily avoids a second, drifting copy in the self-evolution system.
        """

        from catmaster.specialists import runtime as specialist_runtime

        builtins = set(specialist_runtime._DEEPAGENT_BUILTIN_TOOL_NAMES)
        autonomous = set(specialist_runtime._DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES)

        def worker(name: str) -> set[str]:
            return (
                set(getattr(specialist_runtime, name))
                | builtins
                | autonomous
            )

        materials = worker("_MATERIALS_WORKER_TOOL_ALLOWLIST")
        dynamics = worker("_DYNAMICS_WORKER_TOOL_ALLOWLIST")
        ml = worker("_ML_WORKER_TOOL_ALLOWLIST")
        orca_xtb = worker("_ORCA_XTB_WORKER_TOOL_ALLOWLIST")
        research = (
            set(specialist_runtime._RESEARCH_TOOL_ALLOWLIST)
            | builtins
            | autonomous
            | _DEEPAGENT_TASK_TOOL
        )
        litreview = (
            set(specialist_runtime._LITREVIEW_LOCAL_TOOL_ALLOWLIST)
            | builtins
            | autonomous
        )
        writing_entry = (
            set(specialist_runtime._WRITING_TOOL_ALLOWLIST)
            | builtins
            | autonomous
            | _DEEPAGENT_TASK_TOOL
        )
        peer_entry = (
            set(specialist_runtime._PEER_REVIEW_TOOL_ALLOWLIST)
            | builtins
            | autonomous
            | _DEEPAGENT_TASK_TOOL
        )
        experiment_entry = (
            set(specialist_runtime._EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST)
            | builtins
            | autonomous
            | _DEEPAGENT_TASK_TOOL
        )
        writing_worker = worker("_WRITING_WORKER_TOOL_ALLOWLIST")
        peer_worker = worker("_PEER_REVIEW_WORKER_TOOL_ALLOWLIST")
        return {
            "materials_worker": (materials,),
            "dynamics_worker": (dynamics,),
            "ml_worker": (ml,),
            "orca_xtb_worker": (orca_xtb,),
            "research_specialist": (research,),
            "litreview_agent": (litreview,),
            "execution": (materials, dynamics, ml, orca_xtb),
            "writing_specialist": (
                writing_entry,
                peer_entry,
                writing_worker,
                peer_worker,
            ),
            "writing_quality": (
                writing_entry,
                peer_entry,
                experiment_entry,
                litreview,
                writing_worker,
                peer_worker,
            ),
        }


__all__ = ["CandidateGate"]
