#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context pack builder for file-based memory with progressive disclosure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from catmaster.runtime.memory_store import MemoryStore


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
        if role == "task_runner" and not policy.inject_goal_for_worker:
            memory_excerpt = _remove_goal_pointer(memory_excerpt)

        return {
            "task_goal": task_goal,
            "role": role,
            "workspace_root": ".",
            "memory_index_excerpt": memory_excerpt,
            "workspace_policy": _workspace_policy_summary(role),
        }


def _workspace_policy_summary(role: str) -> str:
    return (
        "Project files policy:\n"
        '- Treat "." as the project files root.\n'
        '- All path parameters and returned paths are relative to ".".\n'
        "- Never use absolute paths or metadata paths.\n"
        "- Use search_files / list_directory / directory_tree to discover files.\n"
        "- Use read_text_file with head/tail for progressive disclosure.\n"
        "- Use bash_exec for shell commands, content grep, parser invocation, and external binaries.\n"
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
