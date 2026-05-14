from __future__ import annotations

from collections.abc import Callable

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .config import CAMPAIGNS_ROOT, MEMORIES_ROOT, SCRATCH_ROOT


def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/campaigns/": FilesystemBackend(root_dir=CAMPAIGNS_ROOT, virtual_mode=True),
            "/scratch/": FilesystemBackend(root_dir=SCRATCH_ROOT, virtual_mode=True),
            "/memories/": StoreBackend(runtime),
        },
    )


def build_checkpointer() -> InMemorySaver:
    return InMemorySaver()


def build_store() -> InMemoryStore:
    return InMemoryStore()


def build_default_middleware(tool_limits: dict[str, int] | None = None, model_limit: int = 30) -> list:
    middleware = [ModelCallLimitMiddleware(run_limit=model_limit)]
    for tool_name, limit in (tool_limits or {}).items():
        middleware.append(ToolCallLimitMiddleware(tool_name=tool_name, run_limit=limit))
    return middleware
