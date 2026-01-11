#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context pack builder for deterministic task context assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

from catmaster.runtime.whiteboard import WhiteboardStore
from catmaster.runtime.artifact_log import ArtifactLog
from catmaster.tools.base import system_root


@dataclass(frozen=True)
class ContextPackPolicy:
    max_whiteboard_chars: int = 8000
    max_artifacts: int = 50
    include_journal: bool = False
    max_journal_chars: int = 2000


class ContextPackBuilder:
    def __init__(self, whiteboard: WhiteboardStore):
        self.whiteboard = whiteboard

    def build(self, task_goal: str, role: str, *, policy: Optional[ContextPackPolicy] = None) -> Dict[str, str]:
        policy = policy or ContextPackPolicy()
        if role == "task_runner":
            core_sections = ["Key Facts", "Key Files", "Constraints", "Open Questions"]
        else:
            core_sections = ["Goal", "Key Facts", "Key Files", "Constraints", "Open Questions"]
        core_excerpt = self.whiteboard.read_sections(core_sections, max_chars=policy.max_whiteboard_chars)

        journal_excerpt = ""
        if policy.include_journal:
            journal_excerpt = self.whiteboard.read_sections(["Journal"], max_chars=policy.max_journal_chars)

        key_files = _parse_key_files(core_excerpt)
        artifact_log = ArtifactLog(system_root() / "artifacts.csv")
        artifact_log.ensure_exists()
        artifact_log_entries = _sort_artifacts_by_time(_artifact_log_slice(artifact_log.load()))
        artifact_slice = _merge_artifacts(
            key_files,
            artifact_log_entries,
            policy.max_artifacts,
        )

        constraints = self.whiteboard.read_sections(["Constraints"])
        workspace_policy = _workspace_policy_summary(role)
        return {
            "task_goal": task_goal,
            "role": role,
            "whiteboard_excerpt": _with_current_state_header(core_excerpt, journal_excerpt),
            "artifact_slice": artifact_slice,
            "constraints": constraints,
            "workspace_policy": workspace_policy,
        }


def _with_current_state_header(core_excerpt: str, journal_excerpt: str) -> str:
    chunks = ["## Current State", core_excerpt]
    if journal_excerpt:
        chunks.append(journal_excerpt)
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def _workspace_policy_summary(role: str) -> str:
    return (
        "Workspace policy:\n"
        "- Use user workspace only (view='user').\n"
        "- Do not read or write system metadata.\n"
        "- Reuse existing artifacts; avoid scanning the full workspace.\n"
        f"- Role: {role}"
    )


def _parse_key_files(whiteboard_excerpt: str) -> List[str]:
    paths: List[str] = []
    pattern = re.compile(r"^\s*(?:-\s*)?FILE\[[^\]]+\]\s*:\s*([^|]+)")
    for raw in whiteboard_excerpt.splitlines():
        match = pattern.match(raw)
        if not match:
            continue
        path_part = match.group(1).strip()
        if path_part:
            paths.append(path_part)
    return paths


def _artifact_log_slice(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sliced: List[Dict[str, str]] = []
    for entry in entries:
        path = entry.get("path")
        if not path:
            continue
        sliced.append({
            "path": path,
            "kind": "output",
            "description": entry.get("description", ""),
            "type": entry.get("type", ""),
        })
    return sliced


def _sort_artifacts_by_time(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def _key(entry: Dict[str, str]) -> str:
        return entry.get("updated_time", "")

    return sorted(entries, key=_key, reverse=True)


def _merge_artifacts(
    key_files: List[str],
    artifact_log_entries: List[Dict[str, str]],
    limit: int,
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    seen = set()
    for path in key_files:
        if path in seen:
            continue
        merged.append({
            "path": path,
            "kind": "input",
            "description": "Whiteboard key file",
            "type": ArtifactLog.infer_type(path),
        })
        seen.add(path)
        if len(merged) >= limit:
            break
    for entry in artifact_log_entries:
        path = entry.get("path")
        if not path or path in seen:
            continue
        merged.append(entry)
        seen.add(path)
        if len(merged) >= limit:
            break
    return merged


__all__ = ["ContextPackBuilder", "ContextPackPolicy"]
