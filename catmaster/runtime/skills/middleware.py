from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from .catalog import CatMasterSkillsRuntime
from .models import SkillMeta
from .prompt_addendum import render_skills_addendum


class CatMasterSkillsMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        role: str,
        lane: str | None,
        skills_runtime: CatMasterSkillsRuntime,
        mounted_skill_tokens: Sequence[str] | None,
    ) -> None:
        super().__init__()
        self.role = str(role or "").strip()
        self.lane = str(lane or "").strip()
        self.skills_runtime = skills_runtime
        self.mounted_skill_tokens = {str(item).strip() for item in list(mounted_skill_tokens or []) if str(item).strip()}
        self._visible_skills: list[SkillMeta] = []
        self.tools = []

    def _list_visible_skills(self) -> list[SkillMeta]:
        return list(self._visible_skills)

    def _refresh_visible_skills(self) -> None:
        self.skills_runtime.refresh_catalog()
        self._visible_skills = self.skills_runtime.visible_skills(self.role, self.lane)

    def before_agent(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        _ = (state, runtime)
        self._refresh_visible_skills()
        return None

    async def abefore_agent(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        _ = (state, runtime)
        self._refresh_visible_skills()
        return None

    @staticmethod
    def _system_message_text(message: Any) -> str:
        content_blocks = getattr(message, "content_blocks", None)
        if isinstance(content_blocks, list):
            parts: list[str] = []
            for block in content_blocks:
                block_type = ""
                block_text = ""
                if isinstance(block, dict):
                    block_type = str(block.get("type") or "")
                    block_text = str(block.get("text") or "")
                else:
                    block_type = str(getattr(block, "type", "") or "")
                    block_text = str(getattr(block, "text", "") or "")
                if block_type == "text" and block_text:
                    parts.append(block_text)
            return "\n".join(part for part in parts if part).strip()
        return str(getattr(message, "content", message) or "").strip()

    @classmethod
    def _append_addendum_message(cls, *, current_system: Any, addendum: str) -> SystemMessage:
        current_text = cls._system_message_text(current_system)
        if "## Skills" in current_text and "## Available Skills" in current_text:
            if isinstance(current_system, SystemMessage):
                return current_system
            return SystemMessage(content=current_text)

        current_blocks = getattr(current_system, "content_blocks", None)
        if isinstance(current_blocks, list):
            merged_blocks = list(current_blocks)
        else:
            merged_blocks = []
            if current_text:
                merged_blocks.append({"type": "text", "text": current_text})
        merged_blocks.append({"type": "text", "text": addendum})

        if isinstance(current_system, SystemMessage):
            return SystemMessage(
                content_blocks=merged_blocks,
                additional_kwargs=dict(getattr(current_system, "additional_kwargs", {}) or {}),
                response_metadata=dict(getattr(current_system, "response_metadata", {}) or {}),
                name=getattr(current_system, "name", None),
                id=getattr(current_system, "id", None),
            )
        return SystemMessage(content_blocks=merged_blocks)

    def _override_request_with_skills_addendum(self, request: Any) -> Any:
        addendum = render_skills_addendum(
            role=self.role,
            skills=self._visible_skills,
            mounted_skill_tokens=self.mounted_skill_tokens,
        )
        current_system = getattr(request, "system_message", None)
        merged = self._append_addendum_message(current_system=current_system, addendum=addendum)
        return request.override(system_message=merged)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._override_request_with_skills_addendum(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._override_request_with_skills_addendum(request))


__all__ = ["CatMasterSkillsMiddleware"]
