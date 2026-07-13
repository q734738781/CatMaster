from __future__ import annotations

from typing import Any, Sequence

from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillSource, SkillsMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime


class ReloadDeepAgentContextMiddleware(AgentMiddleware):
    """Reload checkpointed DeepAgents skills and memory once per invocation."""

    def __init__(
        self,
        *,
        backend: Any,
        skills: Sequence[SkillSource],
        memory: Sequence[str],
    ) -> None:
        self._skills = SkillsMiddleware(backend=backend, sources=list(skills), system_prompt=None)
        self._memory = MemoryMiddleware(backend=backend, sources=list(memory), system_prompt=None)

    @staticmethod
    def _without_cached_context(state: Any) -> dict[str, Any]:
        refreshed = dict(state)
        refreshed.pop("skills_metadata", None)
        refreshed.pop("skills_load_errors", None)
        refreshed.pop("memory_contents", None)
        return refreshed

    @staticmethod
    def _combine_updates(
        skill_update: dict[str, Any] | None,
        memory_update: dict[str, Any] | None,
    ) -> dict[str, Any]:
        update = dict(skill_update or {})
        update.setdefault("skills_load_errors", [])
        update.update(dict(memory_update or {}))
        return update

    def before_agent(
        self,
        state: Any,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        refreshed = self._without_cached_context(state)
        skill_update = self._skills.before_agent(refreshed, runtime, config)
        memory_update = self._memory.before_agent(refreshed, runtime, config)
        return self._combine_updates(skill_update, memory_update)

    async def abefore_agent(
        self,
        state: Any,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        refreshed = self._without_cached_context(state)
        skill_update = await self._skills.abefore_agent(refreshed, runtime, config)
        memory_update = await self._memory.abefore_agent(refreshed, runtime, config)
        return self._combine_updates(skill_update, memory_update)


__all__ = ["ReloadDeepAgentContextMiddleware"]
