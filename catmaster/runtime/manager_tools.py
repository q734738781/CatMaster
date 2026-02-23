#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manager-facing helpers for file-based memory reads and context pack building."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from catmaster.runtime.context_pack import ContextPackBuilder, ContextPackPolicy
from catmaster.runtime.memory_store import MemoryStore


def memory_read_index(
    max_lines: Optional[int] = 200,
    max_chars: Optional[int] = None,
    *,
    workspace: Optional[str | Path] = None,
) -> str:
    store = MemoryStore.create_default(workspace=workspace)
    store.ensure_exists()
    return store.read_index(max_lines=max_lines, max_chars=max_chars)


def memory_events_tail(
    limit: int = 20,
    *,
    workspace: Optional[str | Path] = None,
) -> Dict[str, Any]:
    store = MemoryStore.create_default(workspace=workspace)
    store.ensure_exists()
    return {"events": store.read_events_tail(limit=limit)}


def context_pack_build(
    task_goal: str,
    role: str,
    policy: Optional[Dict[str, Any]] = None,
    *,
    workspace: Optional[str | Path] = None,
) -> Dict[str, Any]:
    store = MemoryStore.create_default(workspace=workspace)
    store.ensure_exists()
    builder = ContextPackBuilder(store)
    context_policy = ContextPackPolicy(**policy) if policy else None
    return builder.build(task_goal, role, policy=context_policy)


__all__ = [
    "memory_read_index",
    "memory_events_tail",
    "context_pack_build",
]
