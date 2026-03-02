#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File-based project memory store and event log utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from catmaster.tools.base import ensure_project_space_layout, system_root, workspace_root

_INDEX_FILE = "MEMORY.md"
_TOPIC_DIR = "topics"
_EVENT_REL = Path("memory") / "events.jsonl"

_DEFAULT_TOPICS: dict[str, str] = {
    "GOAL.md": """# GOAL\n\n## TL;DR\n- Primary objective:\n- Definition of success:\n- Non-goals:\n\n## Objectives\n- (empty)\n\n## Success Criteria\n- (empty)\n\n## Non-goals\n- (empty)\n""",
    "FACTS.md": """# FACTS\n\n## TL;DR\n- Canonical definitions: (empty)\n- Verified facts: (empty)\n- Tentative hypotheses: (empty)\n\n## Canonical Definitions\n- (empty)\n\n## Verified Facts\n- (empty)\n\n## Tentative\n- (empty)\n\n## Decision Log\n- (empty)\n""",
    "FILES.md": """# FILES\n\n## TL;DR\n- Entry points: (empty)\n- Primary artifacts: (empty)\n\n## Index\n- (empty)\n""",
    "CONSTRAINTS.md": """# CONSTRAINTS\n\n## TL;DR\n- Hard constraints:\n  - (empty)\n\n## Hard Constraints\n- (empty)\n\n## Soft Preferences\n- (empty)\n""",
    "QUESTIONS.md": """# QUESTIONS\n\n## TL;DR\n- Active blockers: (empty)\n\n## Active\n- (empty)\n\n## Resolved\n- (empty)\n""",
    "RUNBOOK.md": """# RUNBOOK\n\n## TL;DR\n- Standard operating procedure pointer file.\n\n## Checklist\n1. Confirm goal and success criteria from task packet.\n2. Read MEMORY.md first, then locate details via rg/sed.\n3. Persist large outputs to files and cite paths.\n4. Finish by returning one structured TaskOutput response.\n""",
}

_DEFAULT_INDEX = """# MEMORY (AUTOLOADED INDEX)\n\n## What this is\n- Autoloaded memory index for this project.\n- Keep this file concise and pointer-first.\n- Put details in MEMORY/topics/*.md.\n\n## Latest State\n- Current focus: (empty)\n- Current milestone: (empty)\n- Latest run artifact: (empty)\n- Next action: (empty)\n\n## Top Constraints\n1. (empty)\n\n## Active Open Questions\n1. (empty)\n\n## Pointers\n- Goal / principles: MEMORY/topics/GOAL.md\n- Facts / decisions: MEMORY/topics/FACTS.md\n- Files / artifacts: MEMORY/topics/FILES.md\n- Constraints: MEMORY/topics/CONSTRAINTS.md\n- Open questions: MEMORY/topics/QUESTIONS.md\n- Runbook: MEMORY/topics/RUNBOOK.md\n\n## Keyword Map\n- GOAL: objective, success criteria, scope, non-goals\n- FACTS: definition, verified, result, decision\n- FILES: path, artifact, dataset, output, script\n- CONSTRAINTS: must, cannot, policy, budget\n- QUESTIONS: unknown, blocked, clarify, decide\n- RUNBOOK: steps, checklist, verify\n"""


@dataclass(frozen=True)
class MemoryStore:
    workspace: Path

    @staticmethod
    def create_default(*, workspace: Path | str | None = None) -> "MemoryStore":
        root = Path(workspace).expanduser().resolve() if workspace is not None else Path.cwd().resolve()
        ensure_project_space_layout(root, create=True)
        return MemoryStore(workspace=root)

    @property
    def memory_dir(self) -> Path:
        return workspace_root(self.workspace) / "MEMORY"

    @property
    def index_path(self) -> Path:
        return self.memory_dir / _INDEX_FILE

    @property
    def topics_dir(self) -> Path:
        return self.memory_dir / _TOPIC_DIR

    @property
    def events_path(self) -> Path:
        return system_root(self.workspace) / _EVENT_REL

    def ensure_exists(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(_DEFAULT_INDEX, encoding="utf-8")
        for name, content in _DEFAULT_TOPICS.items():
            topic_path = self.topics_dir / name
            if not topic_path.exists():
                topic_path.write_text(content, encoding="utf-8")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")

    def refresh_index_from_topics(self) -> None:
        """Regenerate MEMORY.md index from topic files.

        This keeps the autoload index aligned with topic updates even when
        agents only patch MEMORY/topics/*.md.
        """
        self.ensure_exists()
        topics: Dict[str, str] = {}
        for name in _DEFAULT_TOPICS:
            path = self.topics_dir / name
            try:
                topics[name] = path.read_text(encoding="utf-8")
            except Exception:
                topics[name] = ""

        goal_text = topics.get("GOAL.md", "")
        files_text = topics.get("FILES.md", "")
        constraints_text = topics.get("CONSTRAINTS.md", "")
        questions_text = topics.get("QUESTIONS.md", "")

        current_focus = _extract_labeled_tldr(goal_text, "Primary objective") or "(empty)"
        current_milestone = _extract_labeled_tldr(goal_text, "Definition of success") or "(empty)"
        latest_run_artifacts = _extract_bullets_from_section(files_text, "## Index", limit=None)
        latest_run_artifact = " | ".join(latest_run_artifacts) if latest_run_artifacts else "(empty)"
        next_actions = _extract_bullets_from_section(questions_text, "## Active", limit=None)
        next_action = " | ".join(next_actions) if next_actions else "(empty)"

        top_constraints = _extract_bullets_from_section(constraints_text, "## Hard Constraints", limit=None)
        if not top_constraints:
            top_constraints = _extract_bullets_from_section(constraints_text, "## TL;DR", limit=None)
        active_open_questions = _extract_bullets_from_section(questions_text, "## Active", limit=None)

        lines: List[str] = [
            "# MEMORY (AUTOLOADED INDEX)",
            "",
            "## What this is",
            "- Autoloaded memory index for this project.",
            "- Keep this file concise and pointer-first.",
            "- Put details in MEMORY/topics/*.md.",
            "",
            "## Latest State",
            f"- Current focus: {current_focus}",
            f"- Current milestone: {current_milestone}",
            f"- Latest run artifact: {latest_run_artifact}",
            f"- Next action: {next_action}",
            "",
            "## Top Constraints",
        ]
        if top_constraints:
            for i, item in enumerate(top_constraints, start=1):
                lines.append(f"{i}. {item}")
        else:
            lines.append("1. (empty)")

        lines.extend([
            "",
            "## Active Open Questions",
        ])
        if active_open_questions:
            for i, item in enumerate(active_open_questions, start=1):
                lines.append(f"{i}. {item}")
        else:
            lines.append("1. (empty)")

        lines.extend([
            "",
            "## Pointers",
            "- Goal / principles: MEMORY/topics/GOAL.md",
            "- Facts / decisions: MEMORY/topics/FACTS.md",
            "- Files / artifacts: MEMORY/topics/FILES.md",
            "- Constraints: MEMORY/topics/CONSTRAINTS.md",
            "- Open questions: MEMORY/topics/QUESTIONS.md",
            "- Runbook: MEMORY/topics/RUNBOOK.md",
            "",
            "## Keyword Map",
            "- GOAL: objective, success criteria, scope, non-goals",
            "- FACTS: definition, verified, result, decision",
            "- FILES: path, artifact, dataset, output, script",
            "- CONSTRAINTS: must, cannot, policy, budget",
            "- QUESTIONS: unknown, blocked, clarify, decide",
            "- RUNBOOK: steps, checklist, verify",
            "",
        ])

        self.index_path.write_text("\n".join(lines), encoding="utf-8")

    def read_index(self, *, max_lines: Optional[int] = None, max_chars: Optional[int] = None) -> str:
        text = self.index_path.read_text(encoding="utf-8")
        if max_lines is not None and max_lines >= 0:
            text = "\n".join(text.splitlines()[:max_lines])
        if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
            text = text[:max_chars]
        return text

    def read_topic(self, topic: str, *, max_lines: Optional[int] = None, max_chars: Optional[int] = None) -> str:
        topic_name = topic.strip().upper()
        if not topic_name.endswith(".MD"):
            topic_name = f"{topic_name}.md"
        topic_name = topic_name.upper().replace(".MD", ".md")
        path = self.topics_dir / topic_name
        if not path.exists():
            raise FileNotFoundError(f"Missing memory topic: {path}")
        text = path.read_text(encoding="utf-8")
        if max_lines is not None and max_lines >= 0:
            text = "\n".join(text.splitlines()[:max_lines])
        if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
            text = text[:max_chars]
        return text

    def read_events_tail(self, *, limit: Optional[int] = 20) -> List[Dict[str, Any]]:
        if isinstance(limit, int) and limit <= 0:
            return []
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        out: List[Dict[str, Any]] = []
        selected_lines = lines if limit is None else lines[-limit:]
        for raw in selected_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def append_event(self, event: Dict[str, Any]) -> str:
        payload = dict(event or {})
        payload.setdefault("ts", _now_iso())
        line = json.dumps(payload, ensure_ascii=False)
        with self.events_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        rel = self.events_path.relative_to(system_root(self.workspace))
        return str(rel)

    def merge_task_result(
        self,
        *,
        run_id: str,
        task_id: str,
        outcome: str,
        task_goal: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure_exists()
        warnings.warn(
            "MemoryStore.merge_task_result is deprecated and now event-only. "
            "Use Orchestrator memory patch flow for MEMORY/** updates.",
            DeprecationWarning,
            stacklevel=2,
        )
        summary = str(result.get("summary") or "").strip()
        facts = _clean_text_list(result.get("facts"))
        constraints = _clean_text_list(result.get("constraints"))
        questions = _clean_text_list(result.get("open_questions"))
        next_steps = _clean_text_list(result.get("next_steps"))
        files = _clean_file_entries(result.get("files"))
        decisions = _clean_decision_entries(result.get("decisions"))

        event = {
            "ts": _now_iso(),
            "run_id": run_id,
            "task_id": task_id,
            "goal": task_goal,
            "outcome": outcome,
            "summary": summary,
            "facts": facts,
            "constraints": constraints,
            "open_questions": questions,
            "next_steps": next_steps,
            "files": files,
            "decisions": decisions,
            "artifacts": _clean_text_list(result.get("artifacts")),
        }
        event_rel = self.append_event(event)

        return {
            "event_path": event_rel,
            "memory_index": str(self.index_path.relative_to(workspace_root(self.workspace))),
            "topics_dir": str(self.topics_dir.relative_to(workspace_root(self.workspace))),
            "mode": "event_only",
        }

    def artifact_index(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen: set[str] = set()
        pattern = re.compile(r"^-\s*PATH:\s*([^|]+?)\s*(?:\|\s*(.*))?$")
        files_path = self.topics_dir / "FILES.md"
        if files_path.exists():
            lines = files_path.read_text(encoding="utf-8").splitlines()
            for raw in reversed(lines):
                m = pattern.match(raw.strip())
                if not m:
                    continue
                path = (m.group(1) or "").strip()
                if not path or path in seen:
                    continue
                attrs = _parse_pipe_attrs((m.group(2) or "").strip())
                records.append({
                    "path": path,
                    "description": attrs.get("desc", ""),
                    "kind": attrs.get("kind", "output"),
                    "type": "dir" if path.endswith("/") else "file",
                })
                seen.add(path)
                if isinstance(limit, int) and len(records) >= limit:
                    return records

        # fallback from recent events
        for event in reversed(self.read_events_tail(limit=None)):
            for file_item in event.get("files") or []:
                if not isinstance(file_item, dict):
                    continue
                path = str(file_item.get("path") or "").strip()
                if not path or path in seen:
                    continue
                records.append({
                    "path": path,
                    "description": str(file_item.get("description") or "").strip(),
                    "kind": str(file_item.get("kind") or "output").strip() or "output",
                    "type": "dir" if path.endswith("/") else "file",
                })
                seen.add(path)
                if isinstance(limit, int) and len(records) >= limit:
                    return records
        return records


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _clean_text_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        text = " ".join(str(item or "").split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clean_file_entries(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            path = item.strip()
            description = ""
            kind = "output"
        elif isinstance(item, dict):
            path = str(item.get("path") or "").strip()
            description = str(item.get("description") or "").strip()
            kind = str(item.get("kind") or "output").strip() or "output"
        else:
            continue
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append({"path": path, "description": description, "kind": kind})
    return out


def _clean_decision_entries(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        decision = " ".join(str(item.get("decision") or "").split())
        rationale = " ".join(str(item.get("rationale") or "").split())
        if not decision:
            continue
        out.append({"decision": decision, "rationale": rationale})
    return out


def _parse_pipe_attrs(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for part in raw.split("|"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        attrs[key.strip().lower()] = value.strip()
    return attrs


def _extract_labeled_tldr(text: str, label: str) -> str:
    pattern = re.compile(rf"(?mi)^-\s*{re.escape(label)}\s*:[ \t]*(.*)$")
    m = pattern.search(text or "")
    if not m:
        return ""
    value = " ".join((m.group(1) or "").split()).strip()
    return "" if not value or value == "(empty)" else value


def _extract_bullets_from_section(
    text: str,
    heading: str,
    *,
    limit: Optional[int],
) -> List[str]:
    lines = (text or "").splitlines()
    start = -1
    for i, raw in enumerate(lines):
        if raw.strip() == heading:
            start = i + 1
            break
    if start < 0:
        return []

    out: List[str] = []
    for raw in lines[start:]:
        stripped = raw.strip()
        if stripped.startswith("## "):
            break
        m = re.match(r"^\s*[-*]\s+(.*)$", raw)
        if not m:
            continue
        value = " ".join((m.group(1) or "").split()).strip()
        if not value or value == "(empty)" or value.endswith(":"):
            continue
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def _first_non_empty(items: List[str]) -> str:
    for item in items:
        text = " ".join(str(item or "").split()).strip()
        if text and text != "(empty)":
            return text
    return ""


__all__ = ["MemoryStore"]
