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


class ContextPackBuilder:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def build(self, task_goal: str, role: str, *, policy: Optional[ContextPackPolicy] = None) -> Dict[str, object]:
        policy = policy or ContextPackPolicy()
        memory_excerpt = self.memory.read_index(
            max_lines=policy.memory_head_lines,
            max_chars=policy.max_memory_chars,
        )
        files_root_abs = str(workspace_root(self.memory.workspace).resolve())

        return {
            "task_goal": task_goal,
            "role": role,
            "workspace_root": ".",
            "workspace_root_abs_ref": files_root_abs,
            "memory_index_excerpt": memory_excerpt,
            "workspace_policy": _workspace_policy_summary(role, files_root_abs=files_root_abs),
        }


def _workspace_policy_summary(role: str, *, files_root_abs: str) -> str:
    return (
        "Project files policy:\n"
        '- Treat "." as the project files root.\n'
        f"- Reference absolute files root (orientation only): {files_root_abs}\n"
        '- Use relative paths for filesystem function-tool arguments; keep returned paths relative to ".".\n'
        "- Absolute paths are fallback-only references; if used for filesystem tools, they must be under project files root.\n"
        "- Never use metadata paths in filesystem function-tool arguments.\n"
        "- Use search_files / list_directory / directory_tree to discover files.\n"
        "- Use read_text_file with head/tail for progressive disclosure.\n"
        "- Use bash_exec for shell commands, content grep, parser invocation, and external binaries.\n"
        f"- Role: {role}"
    )

__all__ = ["ContextPackBuilder", "ContextPackPolicy"]
