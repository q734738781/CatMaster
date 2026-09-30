from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import current_tool_context
from catmaster.tools.base import project_space_root


class ManageEffectiveSkillsInput(BaseModel):
    """[workspace/skills] Inspect effective Skills or apply an exact user-directed mode, version, enable, pin, rollback, or held-authorization change."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "list",
        "detail",
        "set_state",
        "set_mode",
        "resolve_boundary",
    ] = Field(
        ...,
        description=(
            "Use list/detail to inspect first. set_state changes one exact target; "
            "set_mode changes workspace evolution; resolve_boundary is only for a "
            "current user message that unambiguously answers every reviewer human_check."
        ),
    )
    target: str = Field(
        "",
        description="Exact group/name target, e.g. research_execution/research-graph-writeback, or /memories/AGENTS.md. A sec_...@r0001 value is a version, not a target; leave empty only for list or set_mode.",
    )
    cursor: str = Field(
        "",
        description="Exact next cursor returned by list; leave empty for page one.",
    )
    limit: int = Field(
        50,
        ge=1,
        le=100,
        description="Catalog or detail page size.",
    )
    version_cursor: str = Field(
        "",
        description="Exact version_next_cursor returned by detail; leave empty for its first page.",
    )
    history_cursor: int = Field(
        0,
        ge=0,
        description="Exact history_next_cursor returned by detail; pass 0 for its first page.",
    )
    enabled_action: Literal["keep", "enable", "disable"] = Field(
        "keep",
        description="set_state enable control; use keep when it must not change.",
    )
    selected_version: str = Field(
        "",
        description=(
            "Exact eligible version from detail, including base, for set_state or "
            "resolve_boundary. Leave empty when no version selection is requested."
        ),
    )
    update_policy: Literal["keep", "follow_auto", "pinned"] = Field(
        "keep",
        description="set_state policy control; exact version selection pins automatically.",
    )
    expected_selected_version: str = Field(
        "",
        description=(
            "Selected version observed in detail before any state mutation. Pass the "
            "empty string when the inspected target had no selected version."
        ),
    )
    mode: Literal["", "off", "observe", "auto"] = Field(
        "",
        description="New workspace mode for set_mode; leave empty for other operations.",
    )
    expected_mode: Literal["", "off", "observe", "auto"] = Field(
        "",
        description="Workspace mode observed before set_mode; leave empty for other operations.",
    )
    resolution: str = Field(
        "",
        description=(
            "For resolve_boundary, concise interpretation of the current user's exact "
            "authorization or choice. Generic assent, silence, or task success is not a resolution."
        ),
    )


def _artifact(data: dict[str, Any]) -> dict[str, Any]:
    return {"tool_name": "manage_effective_skills", "data": data}


def _trusted_chat_context() -> tuple[str, str]:
    context = current_tool_context()
    thread_id = str(context.get("thread_id") or "").strip()
    message_id = str(context.get("user_message_id") or "").strip()
    if not thread_id or not message_id:
        raise ValueError(
            "Skill state changes require a trusted current thread and user-message handle."
        )
    return thread_id, message_id


def manage_effective_skills(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Inspect or update the workspace-effective Skills state from ordinary chat."""

    tool_name = "manage_effective_skills"
    try:
        params = ManageEffectiveSkillsInput.model_validate(payload)
        # Keep these imports invocation-local: the global tool registry is also
        # used while the self-evolution package initializes.
        from catmaster.runtime.self_evolution.effective import EffectiveSkillsManager
        from catmaster.runtime.self_evolution.storage import SelfEvolutionStore

        workspace = project_space_root()
        manager = EffectiveSkillsManager(
            SelfEvolutionStore(workspace, project_id=workspace.name)
        )
        if params.operation == "list":
            rows, next_cursor = manager.list_targets(
                after=params.cursor,
                limit=params.limit,
            )
            result = {
                **manager.mode_info(),
                "skills": rows,
                "next_cursor": next_cursor,
                "total_count": manager.target_count(),
            }
        elif params.operation == "detail":
            if not params.target:
                raise ValueError("detail requires target")
            result = {
                "skill": manager.target_detail(
                    params.target,
                    version_after=params.version_cursor,
                    version_limit=params.limit,
                    history_before=params.history_cursor,
                    history_limit=params.limit,
                )
            }
        elif params.operation == "set_state":
            if not params.target:
                raise ValueError("set_state requires target")
            enabled = {
                "keep": None,
                "enable": True,
                "disable": False,
            }[params.enabled_action]
            policy = None if params.update_policy == "keep" else params.update_policy
            selected = params.selected_version or None
            if enabled is None and policy is None and selected is None:
                raise ValueError("set_state requires one exact state change")
            thread_id, message_id = _trusted_chat_context()
            result = {
                "skill": manager.update_target(
                    params.target,
                    actor=f"chat:{thread_id}",
                    enabled=enabled,
                    selected_version=selected,
                    update_policy=policy,
                    expected_selected_version=params.expected_selected_version,
                    note="User-directed effective Skills state change.",
                    source="chat",
                    message_id=message_id,
                    thread_id=thread_id,
                ),
                "source_ref": f"message:{message_id}",
            }
        elif params.operation == "set_mode":
            if not params.mode or not params.expected_mode:
                raise ValueError("set_mode requires mode and expected_mode from inspection")
            thread_id, message_id = _trusted_chat_context()
            result = {
                **manager.set_workspace_mode(
                    params.mode,
                    actor=f"chat:{thread_id}",
                    expected_mode=params.expected_mode,
                    note="User-directed workspace evolution mode change.",
                    source="chat",
                    message_id=message_id,
                    thread_id=thread_id,
                ),
                "source_ref": f"message:{message_id}",
            }
        else:
            if not params.target or not params.selected_version:
                raise ValueError(
                    "resolve_boundary requires target and the exact held selected_version"
                )
            thread_id, message_id = _trusted_chat_context()
            result = {
                "skill": manager.resolve_human_boundary(
                    params.target,
                    params.selected_version,
                    actor=f"chat:{thread_id}",
                    message_id=message_id,
                    thread_id=thread_id,
                    resolution=params.resolution,
                    expected_selected_version=params.expected_selected_version,
                ),
                "source_ref": f"message:{message_id}",
            }
        content = json.dumps(result, ensure_ascii=False, sort_keys=True)
        return content, _artifact(result)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {}},
            error_code="effective_skills_error",
        ) from exc


__all__ = ["ManageEffectiveSkillsInput", "manage_effective_skills"]
