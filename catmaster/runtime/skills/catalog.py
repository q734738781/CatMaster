from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

from .models import SkillCatalogEntry, SkillMeta
from .role_skills import ROLE_SKILL_NAMES

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

_FRONTMATTER_DELIMITER = "---"
_ALLOWED_TOOLS_KEY = "allowed-tools"
_ROLES_KEY = "catmaster-roles"
_LANES_KEY = "catmaster-lanes"
_TAGS_KEY = "catmaster-tags"
_TOOL_NAME_ALIASES = {
    "bash": "execute",
    "bash_exec": "execute",
}


def _read_frontmatter_block(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if first.strip() != _FRONTMATTER_DELIMITER:
            raise ValueError("SKILL.md missing YAML frontmatter start delimiter '---'")
        buf: list[str] = []
        for line in f:
            if line.strip() == _FRONTMATTER_DELIMITER:
                return "".join(buf)
            buf.append(line)
    raise ValueError("SKILL.md frontmatter is not closed with '---'")


def _fallback_parse_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    in_metadata = False
    for raw in frontmatter_text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("metadata:"):
            in_metadata = True
            out["metadata"] = metadata
            continue
        if in_metadata:
            if raw.startswith(" ") or raw.startswith("\t"):
                key, sep, value = stripped.partition(":")
                if sep:
                    metadata[str(key).strip()] = str(value).strip().strip("'\"")
                continue
            in_metadata = False
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        out[str(key).strip()] = str(value).strip().strip("'\"")
    return out


def _parse_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(frontmatter_text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("frontmatter must be a YAML mapping")
        return dict(loaded)
    parsed = _fallback_parse_frontmatter(frontmatter_text)
    if not parsed:
        raise ValueError("frontmatter cannot be empty")
    return parsed


def _split_allowed_tools(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split() if item.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(item).strip() for item in raw if str(item).strip()]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in parts:
        token = _normalize_tool_token(item)
        if token is None or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _split_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(item).strip() for item in raw if str(item).strip()]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in parts:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _consume_frontmatter(f) -> None:
    first = f.readline()
    if first.strip() != _FRONTMATTER_DELIMITER:
        raise ValueError("SKILL.md missing YAML frontmatter start delimiter '---'")
    for line in f:
        if line.strip() == _FRONTMATTER_DELIMITER:
            return
    raise ValueError("SKILL.md frontmatter is not closed with '---'")


def _normalize_tool_token(raw: str) -> str | None:
    token = str(raw or "").strip().strip("`").strip()
    if not token:
        return None
    lowered = token.lower()
    if lowered in {"(none specified)", "none specified", "(none)", "none"}:
        return None
    return _TOOL_NAME_ALIASES.get(token, token)


def _parse_allowed_tools_body_section(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    in_section = False

    with path.open("r", encoding="utf-8") as f:
        _consume_frontmatter(f)
        for raw in f:
            stripped = raw.strip()
            lowered = stripped.lower()

            if lowered == "## allowed tools":
                in_section = True
                continue

            if not in_section:
                continue

            if stripped.startswith("## "):
                break

            if not stripped:
                continue

            candidates: list[str]
            if stripped.startswith(("- ", "* ")):
                candidates = [stripped[2:].strip()]
            else:
                candidates = [part.strip() for part in stripped.split(",")]

            for candidate in candidates:
                token = _normalize_tool_token(candidate)
                if token is None or token in seen:
                    continue
                seen.add(token)
                out.append(token)
    return out


def _skill_md_relpath(*, abs_skill_md: Path, repo_root: Path | None, source_root: Path) -> str:
    if repo_root is not None:
        try:
            return str(abs_skill_md.relative_to(repo_root)).replace("\\", "/")
        except Exception:
            pass
    try:
        return str(abs_skill_md.relative_to(source_root.parent)).replace("\\", "/")
    except Exception:
        return str(abs_skill_md).replace("\\", "/")


class SkillCatalog:
    def __init__(self, *, source_roots: list[Path], repo_root: Path | None = None) -> None:
        self.source_roots = [Path(root).expanduser().resolve() for root in source_roots]
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root is not None else None
        self._entries: list[SkillCatalogEntry] = []
        self._lock = RLock()

    @classmethod
    def create_default(cls, *, repo_root: Path | None = None) -> "SkillCatalog":
        resolved_repo_root = (repo_root or Path.cwd()).expanduser().resolve()
        roots = [
            resolved_repo_root / "skills" / "materials_worker",
            resolved_repo_root / "skills" / "dynamics_worker",
            resolved_repo_root / "skills" / "ml_worker",
            resolved_repo_root / "skills" / "orca_xtb_worker",
            resolved_repo_root / "skills" / "research_specialist",
            resolved_repo_root / "skills" / "litreview_agent",
            resolved_repo_root / "skills" / "writing_specialist",
            resolved_repo_root / "skills" / "writing_quality",
        ]
        return cls(
            source_roots=roots,
            repo_root=resolved_repo_root,
        )

    def refresh(self) -> list[SkillMeta]:
        entries: list[SkillCatalogEntry] = []
        for source_root in self.source_roots:
            if not source_root.exists() or not source_root.is_dir():
                continue
            for skill_dir in sorted(source_root.iterdir(), key=lambda p: p.name):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue
                try:
                    frontmatter = _parse_frontmatter(_read_frontmatter_block(skill_md))
                    name = str(frontmatter.get("name") or "").strip()
                    description = str(frontmatter.get("description") or "").strip()
                    if not name:
                        raise ValueError("frontmatter.name is required")
                    if not description:
                        raise ValueError("frontmatter.description is required")
                    metadata = frontmatter.get("metadata")
                    metadata_dict = metadata if isinstance(metadata, dict) else {}
                    allowed_tools = _split_allowed_tools(frontmatter.get(_ALLOWED_TOOLS_KEY))
                    roles = _split_string_list(metadata_dict.get(_ROLES_KEY))
                    lanes = _split_string_list(metadata_dict.get(_LANES_KEY))
                    tags = _split_string_list(metadata_dict.get(_TAGS_KEY))
                    if not allowed_tools:
                        allowed_tools = _parse_allowed_tools_body_section(skill_md)
                    compatibility_raw = frontmatter.get("compatibility")
                    compatibility = str(compatibility_raw).strip() if compatibility_raw is not None else None
                    if compatibility == "":
                        compatibility = None
                    source_root_name = source_root.name
                    mount_token = f"@{source_root_name}"
                    if source_root_name == "writing_specialist":
                        if not lanes:
                            lanes = ["writing"]
                        if not roles:
                            roles = ["write_director", "section_writer", "write_reviewer"]
                        if not tags:
                            tags = ["writing"]
                    elif source_root_name == "research_specialist":
                        if not lanes:
                            lanes = ["research"]
                        if not roles:
                            roles = ["research_lead", "research_state_updater", "proposal"]
                        if not tags:
                            tags = ["researcher"]
                    elif source_root_name == "litreview_agent":
                        if not lanes:
                            lanes = ["literature_review", "research"]
                        if not roles:
                            roles = [
                                "literature_deep_research",
                                "literature_worker",
                                "research_lead",
                            ]
                        if not tags:
                            tags = ["literature"]
                    elif source_root_name in {
                        "materials_worker",
                        "dynamics_worker",
                        "ml_worker",
                        "orca_xtb_worker",
                    }:
                        if not lanes:
                            lanes = ["all"]
                        if not roles:
                            roles = [
                                "proposal",
                                "director",
                                "fast_director",
                                "task_runner",
                                "research_lead",
                                "research_state_updater",
                            ]
                        if not tags:
                            tags = [source_root_name]
                    meta = SkillMeta(
                        name=name,
                        description=description,
                        file_path=_skill_md_relpath(
                            abs_skill_md=skill_md.resolve(),
                            repo_root=self.repo_root,
                            source_root=source_root,
                        ),
                        abs_skill_dir=skill_dir.resolve(),
                        abs_skill_md=skill_md.resolve(),
                        directory_name=skill_dir.name,
                        source_root_name=source_root_name,
                        mount_token=mount_token,
                        compatibility=compatibility,
                        allowed_tools=allowed_tools,
                        roles=roles,
                        lanes=lanes,
                        tags=tags,
                    )
                    entries.append(
                        SkillCatalogEntry(
                            source_root=source_root,
                            abs_skill_dir=skill_dir.resolve(),
                            abs_skill_md=skill_md.resolve(),
                            frontmatter=frontmatter,
                            meta=meta,
                        )
                    )
                except Exception as exc:
                    logger.warning("Skip invalid skill %s: %s", skill_md, exc)
                    continue
        with self._lock:
            self._entries = entries
            return [entry.meta for entry in self._entries]

    def list_skills(self) -> list[SkillMeta]:
        with self._lock:
            return [entry.meta for entry in self._entries]

    def get_skill(self, name: str) -> SkillMeta | None:
        target = str(name or "").strip()
        if not target:
            return None
        with self._lock:
            for entry in self._entries:
                if entry.meta.name == target:
                    return entry.meta
        return None

    def list_entries(self) -> list[SkillCatalogEntry]:
        with self._lock:
            return list(self._entries)


class CatMasterSkillsRuntime:
    def __init__(
        self,
        *,
        catalog: SkillCatalog,
        role_skill_names: dict[str, list[str]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.role_skill_names = role_skill_names or ROLE_SKILL_NAMES

    @classmethod
    def create_default(cls, *, repo_root: Path | None = None) -> "CatMasterSkillsRuntime":
        return cls(catalog=SkillCatalog.create_default(repo_root=repo_root))

    def refresh_catalog(self) -> list[SkillMeta]:
        return self.catalog.refresh()

    def visible_skills(self, role: str, lane: str | None = None) -> list[SkillMeta]:
        all_skills = self.catalog.list_skills()
        if not all_skills:
            return []
        role_text = str(role or "").strip()
        lane_text = str(lane or "").strip()
        visible: list[SkillMeta] = []
        legacy_names = set(self.role_skill_names.get(role_text) or [])
        for skill in all_skills:
            if skill.roles or skill.lanes:
                if skill.roles and role_text not in skill.roles:
                    continue
                if skill.lanes and lane_text and lane_text not in skill.lanes and "all" not in skill.lanes:
                    continue
                if skill.lanes and not lane_text and "all" not in skill.lanes:
                    continue
                visible.append(skill)
                continue
            if skill.name in legacy_names:
                visible.append(skill)
        return visible


__all__ = ["SkillCatalog", "CatMasterSkillsRuntime"]
