#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime utilities for run context, state stores, and tracing."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "RunContext": ("catmaster.runtime.run_context", "RunContext"),
    "ToolExecutor": ("catmaster.runtime.tool_executor", "ToolExecutor"),
    "ArtifactStore": ("catmaster.runtime.artifact_store", "ArtifactStore"),
    "MemoryStore": ("catmaster.runtime.memory_store", "MemoryStore"),
    "TraceStore": ("catmaster.runtime.trace_store", "TraceStore"),
    "ToolPolicy": ("catmaster.runtime.tool_policy", "ToolPolicy"),
    "ToolBackend": ("catmaster.runtime.tool_backend", "ToolBackend"),
    "LocalToolBackend": ("catmaster.runtime.local_tool_backend", "LocalToolBackend"),
    "RunControl": ("catmaster.runtime.run_control", "RunControl"),
    "ArtifactPersistenceHandler": ("catmaster.runtime.artifact_callback", "ArtifactPersistenceHandler"),
    "LLMTracingHandler": ("catmaster.runtime.artifact_callback", "LLMTracingHandler"),
    "UIEventHandler": ("catmaster.runtime.artifact_callback", "UIEventHandler"),
    "ContextPackBuilder": ("catmaster.runtime.context_pack", "ContextPackBuilder"),
    "ContextPackPolicy": ("catmaster.runtime.context_pack", "ContextPackPolicy"),
    "memory_read_index": ("catmaster.runtime.manager_tools", "memory_read_index"),
    "memory_events_tail": ("catmaster.runtime.manager_tools", "memory_events_tail"),
    "context_pack_build": ("catmaster.runtime.manager_tools", "context_pack_build"),
    "load_usage_summary": ("catmaster.runtime.usage_stats", "load_usage_summary"),
    "usage_summary_path": ("catmaster.runtime.usage_stats", "usage_summary_path"),
    "summarize_usage_from_metadata": ("catmaster.runtime.usage_stats", "summarize_usage_from_metadata"),
    "write_usage_summary_from_metadata": ("catmaster.runtime.usage_stats", "write_usage_summary_from_metadata"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)
