#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manager-facing tool interfaces for whiteboard reads and context pack building.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from catmaster.runtime.whiteboard import WhiteboardStore
from catmaster.runtime.context_pack import ContextPackBuilder, ContextPackPolicy


def whiteboard_read(sections: Optional[List[str]] = None, max_chars: Optional[int] = None) -> str:
    store = WhiteboardStore.create_default()
    store.ensure_exists()
    if sections:
        return store.read_sections(sections, max_chars=max_chars)
    text = store.read()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def whiteboard_get_hash() -> Dict[str, Any]:
    store = WhiteboardStore.create_default()
    store.ensure_exists()
    data = store.path.read_bytes()
    return {"hash": store.get_hash(), "bytes": len(data)}


def context_pack_build(task_goal: str, role: str, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    store = WhiteboardStore.create_default()
    store.ensure_exists()
    builder = ContextPackBuilder(store)
    context_policy = ContextPackPolicy(**policy) if policy else None
    return builder.build(task_goal, role, policy=context_policy)


__all__ = [
    "whiteboard_read",
    "whiteboard_get_hash",
    "context_pack_build",
]
