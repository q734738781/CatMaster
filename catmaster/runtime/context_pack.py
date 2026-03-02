#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context pack builder for file-based memory with progressive disclosure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from catmaster.runtime.memory_store import MemoryStore
from catmaster.tools.base import workspace_root


@dataclass(frozen=True)
class ContextPackPolicy:
    memory_head_lines: Optional[int] = None
    max_memory_chars: Optional[int] = None
    max_artifacts: int = 50
    inject_goal_for_worker: bool = False


class ContextPackBuilder:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def build(self, task_goal: str, role: str, *, policy: Optional[ContextPackPolicy] = None) -> Dict[str, object]:
        policy = policy or ContextPackPolicy()
        memory_excerpt = self.memory.read_index(
            max_lines=policy.memory_head_lines,
            max_chars=policy.max_memory_chars,
        )
        files_root = workspace_root(self.memory.workspace)

        if role == "task_runner" and not policy.inject_goal_for_worker:
            memory_excerpt = _remove_goal_pointer(memory_excerpt)

        return {
            "task_goal": task_goal,
            "role": role,
            "workspace_root": str(files_root),
            "memory_index_excerpt": memory_excerpt,
            "workspace_policy": _workspace_policy_summary(role, files_root=str(files_root)),
        }


def _workspace_policy_summary(role: str, *, files_root: str) -> str:
    return (
        "Project files policy:\n"
        f"- Current files root: {files_root}\n"
        "- Tool path params are resolved relative to this files root.\n"
        "- Metadata is internal; do not read or write metadata paths from tasks.\n"
        "- Use progressive disclosure: locate with rg, then read small excerpts with sed/head/tail.\n"
        "- Do not cat large files into context. Persist large outputs and cite paths.\n"
        f"- Role: {role}"
    )


def _remove_goal_pointer(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        if raw.strip().lower().startswith("- goal / principles:"):
            continue
        lines.append(raw)
    return "\n".join(lines)


__all__ = ["ContextPackBuilder", "ContextPackPolicy"]
