"""
LangGraph-based orchestration graph for CatMaster.

Replaces the monolithic Orchestrator class with a composable StateGraph
that routes between proposal, director, task runner, and finalize nodes.
Proposal / director / task runner are built with LangChain v1
``create_agent`` and produce typed structured responses.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from catmaster.agents.nodes import (
    run_proposal,
    run_director,
    run_fast_director,
    run_memory_patch,
    run_task,
    summarize_node,
)
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.trace_store import TraceStore
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.artifact_callback import build_callbacks
from catmaster.runtime.tool_backend import ToolBackend
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.tool_output_adapter import (
    tool_error_to_message,
)
from catmaster.tools.registry import ToolRegistry, get_tool_registry
from catmaster.runtime.tool_surface import RuntimeToolSurface, build_runtime_tool_surface
from catmaster.runtime.skills import (
    CatMasterSkillsMiddleware,
    CatMasterSkillsRuntime,
    render_director_skill_guide,
    render_fast_director_skill_guide,
    render_proposal_skill_guide,
)
from catmaster.tools.base import workspace_scope, system_root
from catmaster.ui.reporters import Reporter, NullReporter
from catmaster.ui import make_event
from catmaster.agents.response_schemas import ProposalOutput, DirectorOutput, FastDirectorOutput, TaskOutput, MemoryPatchOutput
from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    DIRECTOR_SYSTEM_PROMPT,
    FAST_DIRECTOR_SYSTEM_PROMPT,
    TASK_RUNNER_SYSTEM_PROMPT,
    MEMORY_PATCHER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_TASK_RUNNER_TOOL_DENYLIST: set[str] = set()
def _dedupe_tools_by_name(tools: Sequence[BaseTool]) -> list[BaseTool]:
    out: list[BaseTool] = []
    seen: set[str] = set()
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(tool)
    return out


def _make_memory_scoped_apply_tool(tool: BaseTool) -> BaseTool:
    """Force memory patch apply tool writes to MEMORY/** only."""
    if str(getattr(tool, "name", "") or "") != "apply_aider_edits":
        return tool

    func = getattr(tool, "func", None)
    coroutine = getattr(tool, "coroutine", None)
    if func is None and coroutine is None:
        return tool

    raw_schema = getattr(tool, "args_schema", None)
    args_schema = copy.deepcopy(raw_schema) if isinstance(raw_schema, dict) else {}
    properties = args_schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("allowed_paths", None)
    required = args_schema.get("required")
    if isinstance(required, list):
        args_schema["required"] = [item for item in required if item != "allowed_paths"]
    if isinstance(args_schema, dict):
        args_schema.setdefault("additionalProperties", False)

    description = str(getattr(tool, "description", "") or "").strip()
    if description:
        description = f"{description}\nEdits are strictly limited to MEMORY/**."
    else:
        description = "Apply Aider SEARCH/REPLACE edit blocks. Edits are strictly limited to MEMORY/**."

    def _wrapper(runtime: Any | None = None, **kwargs: Any):
        forced_kwargs = dict(kwargs)
        forced_kwargs["allowed_paths"] = ["MEMORY/"]
        return func(runtime=runtime, **forced_kwargs)

    async def _awrapper(runtime: Any | None = None, **kwargs: Any):
        forced_kwargs = dict(kwargs)
        forced_kwargs["allowed_paths"] = ["MEMORY/"]
        return await coroutine(runtime=runtime, **forced_kwargs)

    return StructuredTool.from_function(
        func=_wrapper if func is not None else None,
        coroutine=_awrapper if coroutine is not None else None,
        name="apply_aider_edits",
        description=description,
        args_schema=args_schema,
        infer_schema=False,
        response_format="content_and_artifact",
    )


class ToolCallBudgetExceededError(RuntimeError):
    """Raised when an agent exceeds its per-invocation tool-call budget."""


@dataclass(frozen=True)
class GraphRunPolicy:
    allow_memory_patch: bool = True
    allow_human_intervention: bool = True
    enable_literature_tool: bool = True
    enable_history_prefetch: bool = True


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class CatMasterState(TypedDict, total=False):
    user_request: str
    lane: str
    chat_session_id: str
    session_context_text: str
    entry_context_tokens_estimate: int

    # Per-agent message histories
    proposal_messages: Annotated[list[AnyMessage], add_messages]
    director_messages: Annotated[list[AnyMessage], add_messages]
    runner_messages: Annotated[list[AnyMessage], add_messages]

    # Proposal
    proposal_md: str
    work_packages: list[str]
    proposal_approved: bool
    proposal_feedback: str
    proposal_review_enabled: bool

    # Director
    director_decision: dict
    next_action: str
    pending_memory_updates: list[dict]
    memory_patch_result: dict

    # Task tracking
    tasks: list[dict]
    observations: list[dict]
    current_task_id: str
    current_task_packet: dict
    task_result: dict

    # Run status
    status: str
    summary: str
    contract_violation: dict

    # HITL
    hitl_history: list[dict]


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def _snippet(text: Any, limit: int = 140) -> str:
    if text is None:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _load_create_agent():
    try:
        from langchain.agents import create_agent as _create_agent
    except Exception as exc:
        raise RuntimeError(
            "LangChain v1 create_agent is unavailable. Install 'langchain>=1.0'."
        ) from exc
    return _create_agent


def _load_tool_strategy():
    try:
        from langchain.agents.structured_output import ToolStrategy as _ToolStrategy
    except Exception as exc:
        raise RuntimeError(
            "LangChain ToolStrategy is unavailable. Install 'langchain>=1.0'."
        ) from exc
    return _ToolStrategy


def _load_llm_tool_selector_middleware():
    try:
        from langchain.agents.middleware.tool_selection import (
            LLMToolSelectorMiddleware as _LLMToolSelectorMiddleware,
        )
    except Exception as exc:
        raise RuntimeError(
            "LangChain LLMToolSelectorMiddleware is unavailable. Install 'langchain>=1.0'."
        ) from exc
    return _LLMToolSelectorMiddleware


def _make_tool_call_budget_middleware(*, role: str, max_tool_calls: int) -> list[Any]:
    """Enforce per-invocation tool-call cap and map recoverable errors to ToolMessage."""
    try:
        from langchain.agents.middleware import AgentMiddleware
    except Exception as exc:
        raise RuntimeError(
            "LangChain middleware is unavailable. Install 'langchain>=1.0'."
        ) from exc

    tool_limit = max(1, int(max_tool_calls))
    counter: dict[str, int] = {"used": 0}

    class _ResetToolCallCounterMiddleware(AgentMiddleware):
        def before_agent(self, state: dict, runtime: Any) -> dict[str, Any] | None:
            _ = (state, runtime)
            counter["used"] = 0
            return None

        async def abefore_agent(self, state: dict, runtime: Any) -> dict[str, Any] | None:
            _ = (state, runtime)
            counter["used"] = 0
            return None

    class _ToolCallBudgetMiddleware(AgentMiddleware):
        @staticmethod
        def _request_info(request: Any) -> tuple[int, str, str]:
            used = int(counter.get("used", 0))
            tool_call = getattr(request, "tool_call", None) or {}
            tool_name = str(tool_call.get("name") or "unknown")
            tool_call_id = str(tool_call.get("id") or "")
            return used, tool_name, tool_call_id

        @staticmethod
        def _budget_exceeded_message(*, used: int, tool_name: str, tool_call_id: str) -> ToolMessage:
            exc = ToolCallBudgetExceededError(
                f"{role} tool-call budget exceeded: executed={used}, limit={tool_limit}, next_tool={tool_name}"
            )
            return tool_error_to_message(
                exc=exc,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        @staticmethod
        def _error_message(*, exc: Exception, tool_name: str, tool_call_id: str) -> ToolMessage:
            return tool_error_to_message(
                exc=exc,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            used, tool_name, tool_call_id = self._request_info(request)
            if used >= tool_limit:
                return self._budget_exceeded_message(
                    used=used,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            try:
                result = handler(request)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise RuntimeError(
                        "Synchronous tool middleware cannot execute awaitable handlers. "
                        "Invoke the agent asynchronously (ainvoke/arun)."
                    )
                return result
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception as exc:
                return self._error_message(
                    exc=exc,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            finally:
                counter["used"] = used + 1

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            used, tool_name, tool_call_id = self._request_info(request)
            if used >= tool_limit:
                return self._budget_exceeded_message(
                    used=used,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            try:
                result = handler(request)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception as exc:
                return self._error_message(
                    exc=exc,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            finally:
                counter["used"] = used + 1

    return [_ResetToolCallCounterMiddleware(), _ToolCallBudgetMiddleware()]


_SKILL_FILESYSTEM_ALWAYS_INCLUDE: list[str] = []


def _build_role_middleware(
    *,
    role: str,
    lane: str | None,
    max_tool_calls: int,
    skills_runtime: CatMasterSkillsRuntime | None,
    mounted_skill_tokens: Sequence[str] | None,
    selector_model: BaseChatModel | None,
    enable_selector: bool,
) -> list[Any]:
    middleware: list[Any] = []
    if role != "memory_patch" and skills_runtime is not None:
        middleware.append(
            CatMasterSkillsMiddleware(
                role=role,
                lane=lane,
                skills_runtime=skills_runtime,
                mounted_skill_tokens=mounted_skill_tokens,
            )
        )

    if enable_selector:
        LLMToolSelectorMiddleware = _load_llm_tool_selector_middleware()
        # TODO: Revisit skill/tool coupling after the tool surface expansion stabilizes.
        # Official LLMToolSelectorMiddleware currently selects from the active tool list
        # independently of any matched skill `allowed_tools`. If we later want hard
        # skill-aware routing, add a matched-skill tool filter before this middleware.
        middleware.append(
            LLMToolSelectorMiddleware(
                model=selector_model,
                max_tools=20,
                always_include=list(_SKILL_FILESYSTEM_ALWAYS_INCLUDE),
            )
        )

    middleware.extend(
        _make_tool_call_budget_middleware(
            role=role,
            max_tool_calls=max_tool_calls,
        )
    )
    return middleware


def _build_proposal_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 60,
    middleware: Sequence[Any] | None = None,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_proposal_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    middleware_chain = list(middleware) if middleware is not None else _make_tool_call_budget_middleware(
        role="proposal",
        max_tool_calls=max_steps,
    )
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=PROPOSAL_SYSTEM_PROMPT,
        response_format=ToolStrategy(ProposalOutput, handle_errors=False),
        middleware=middleware_chain,
    )


def _build_director_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 60,
    middleware: Sequence[Any] | None = None,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_director_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    middleware_chain = list(middleware) if middleware is not None else _make_tool_call_budget_middleware(
        role="director",
        max_tool_calls=max_steps,
    )
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        response_format=ToolStrategy(DirectorOutput, handle_errors=False),
        middleware=middleware_chain,
    )


def _build_fast_director_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 60,
    middleware: Sequence[Any] | None = None,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_fast_director_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    middleware_chain = list(middleware) if middleware is not None else _make_tool_call_budget_middleware(
        role="fast_director",
        max_tool_calls=max_steps,
    )
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=FAST_DIRECTOR_SYSTEM_PROMPT,
        response_format=ToolStrategy(FastDirectorOutput, handle_errors=False),
        middleware=middleware_chain,
    )


def _build_task_runner_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    memory_store: MemoryStore,
    *,
    max_steps: int = 60,
    middleware: Sequence[Any] | None = None,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_task_runner_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    _ = memory_store
    middleware_chain = list(middleware) if middleware is not None else _make_tool_call_budget_middleware(
        role="task_runner",
        max_tool_calls=max_steps,
    )
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=TASK_RUNNER_SYSTEM_PROMPT,
        response_format=ToolStrategy(TaskOutput, handle_errors=False),
        middleware=middleware_chain,
    )


def _build_memory_patcher_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 30,
    middleware: Sequence[Any] | None = None,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_memory_patcher_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    middleware_chain = list(middleware) if middleware is not None else _make_tool_call_budget_middleware(
        role="memory_patch",
        max_tool_calls=max_steps,
    )
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=MEMORY_PATCHER_SYSTEM_PROMPT,
        response_format=ToolStrategy(MemoryPatchOutput, handle_errors=False),
        middleware=middleware_chain,
    )


# ---------------------------------------------------------------------------
# Outer graph node wrappers (delegate to nodes.py functions)
# ---------------------------------------------------------------------------

async def _run_proposal_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    run_dir: Path,
    max_steps: int,
) -> Command:
    return await run_proposal(
        state,
        agent=agent,
        memory_store=memory_store,
        execution_context_guide=execution_context_guide,
        run_dir=run_dir,
        max_steps=max_steps,
    )


async def _run_director_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    max_steps: int,
    allow_memory_patch: bool,
) -> Command:
    return await run_director(
        state,
        agent=agent,
        memory_store=memory_store,
        execution_context_guide=execution_context_guide,
        max_steps=max_steps,
        allow_memory_patch=allow_memory_patch,
    )


async def _run_fast_director_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    max_steps: int,
    allow_memory_patch: bool,
) -> Command:
    return await run_fast_director(
        state,
        agent=agent,
        memory_store=memory_store,
        execution_context_guide=execution_context_guide,
        max_steps=max_steps,
        allow_memory_patch=allow_memory_patch,
    )


async def _run_task_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    max_steps: int,
    continue_goto: str,
    intervention_goto: str,
    run_dir: Path,
    reporter: Optional[Reporter] = None,
    run_id: str = "",
) -> Command:
    task_id = str(state.get("current_task_id", "") or "").strip()
    task_packet = state.get("current_task_packet") if isinstance(state.get("current_task_packet"), dict) else {}
    task_goal = str(task_packet.get("goal") or "").strip()
    if reporter is not None and task_id:
        try:
            reporter.emit(
                make_event(
                    "TASK_START",
                    category="task",
                    run_id=run_id or None,
                    task_id=task_id,
                    payload={"goal": task_goal},
                )
            )
        except Exception:
            pass

    command = await run_task(
        state,
        agent=agent,
        memory_store=memory_store,
        max_steps=max_steps,
        continue_goto=continue_goto,
        intervention_goto=intervention_goto,
    )
    update = command.update or {}
    task_result = update.get("task_result") or {}
    if task_result:
        if reporter is not None and task_id:
            task_summary = str(task_result.get("task_summary") or "").strip()
            task_outcome = str(task_result.get("task_outcome") or "").strip() or "failure"
            key_artifacts = list(task_result.get("key_artifacts") or [])
            try:
                reporter.emit(
                    make_event(
                        "TASK_SUMMARY",
                        category="task",
                        run_id=run_id or None,
                        task_id=task_id,
                        payload={
                            "outcome": task_outcome,
                            "summary_snippet": _snippet(task_summary, 320),
                            "artifacts": key_artifacts,
                        },
                    )
                )
                reporter.emit(
                    make_event(
                        "TASK_END",
                        category="task",
                        run_id=run_id or None,
                        task_id=task_id,
                        payload={"outcome": task_outcome},
                    )
                )
            except Exception:
                pass
        _write_observation_file(
            run_dir=run_dir,
            task_id=task_id,
            outcome=str(task_result.get("task_outcome", "") or ""),
            summary=str(task_result.get("task_summary", "") or ""),
            key_artifacts=list(task_result.get("key_artifacts") or []),
        )
    return command


async def _run_memory_patch_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    run_id: str,
    max_steps: int,
    continue_goto: str,
) -> Command:
    return await run_memory_patch(
        state,
        agent=agent,
        memory_store=memory_store,
        run_id=run_id,
        max_steps=max_steps,
        continue_goto=continue_goto,
    )


def _summarize_node_wrapper(
    state: CatMasterState,
    *,
    memory_store: MemoryStore,
) -> Dict[str, Any]:
    # Finalization is now a no-extra-LLM pass-through node.
    return summarize_node(state, memory_store=memory_store)


def _write_observation_file(
    *,
    run_dir: Path,
    task_id: str,
    outcome: str,
    summary: str,
    key_artifacts: list,
) -> None:
    """Write an observation markdown file for a completed task."""
    try:
        obs_dir = run_dir / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(obs_dir.glob("obs_*.md"))) + 1
        fname = f"obs_{index:03d}_{task_id}.md"
        path = obs_dir / fname
        lines = [
            f"# Observation {index}",
            f"- Task: {task_id}",
            f"- Outcome: {outcome}",
            "",
            "## Summary",
            summary or "",
            "",
            "## Key Artifacts",
        ]
        if key_artifacts:
            for item in key_artifacts:
                kpath = item.get("path", "")
                desc = item.get("description", "")
                kind = item.get("kind", "")
                lines.append(f"- {kpath} ({kind}): {desc}")
        else:
            lines.append("- (none)")
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _derive_graph_task_state_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    current_task_id = str(state.get("current_task_id") or "").strip()
    current_task_packet = state.get("current_task_packet") if isinstance(state.get("current_task_packet"), dict) else {}
    current_work_label = str(current_task_packet.get("goal") or "").strip()
    status = str(state.get("status") or "running").strip().lower()
    current_phase = "executing" if current_task_id else ("planning" if status in {"running", "starting", ""} else status)
    return {
        "current_task_id": current_task_id,
        "current_task_packet": current_task_packet,
        "current_work_label": current_work_label,
        "current_phase": current_phase,
    }


def _proposal_review_node(state: CatMasterState) -> Command:
    """HITL checkpoint for proposal review.

    Uses LangGraph ``interrupt()`` to pause execution and wait for human
    feedback.  The human can reply with "approved" / "approve" / "ok" to
    accept, or provide textual feedback to trigger a revision loop.

    If proposal_approved is already True (e.g. auto-approve or resume
    after approval), this is a no-op pass-through.
    """
    if not bool(state.get("proposal_review_enabled", True)):
        return Command(goto="run_director", update={"proposal_approved": True, "proposal_feedback": ""})

    if state.get("proposal_approved"):
        return Command(goto="run_director", update={})

    proposal_md = state.get("proposal_md", "")
    work_packages = state.get("work_packages", [])
    logger.info(
        "[proposal_review] proposal_md_len=%d, work_packages_count=%d",
        len(proposal_md), len(work_packages),
    )
    if not proposal_md:
        logger.warning("[proposal_review] proposal_md is EMPTY - state keys: %s", list(state.keys()))

    feedback = interrupt({
        "type": "proposal_review",
        "proposal_md": proposal_md,
        "work_packages": work_packages,
        "message": "Review the proposal. Reply 'approve' to accept, or provide feedback.",
    })

    feedback_text = str(feedback or "").strip()
    existing = list(state.get("hitl_history") or [])
    existing.append({
        "interrupt_type": "proposal_review",
        "feedback": feedback_text,
        "approved": feedback_text.lower() in ("approved", "approve", "ok", "yes", "y", "lgtm"),
        "work_packages": list(work_packages or []),
    })
    if feedback_text.lower() in ("approved", "approve", "ok", "yes", "y", "lgtm"):
        return Command(
            goto="run_director",
            update={
                "proposal_approved": True,
                "proposal_feedback": "",
                "hitl_history": existing,
            },
        )

    return Command(
        goto="run_proposal",
        update={
            "proposal_approved": False,
            "proposal_feedback": feedback_text,
            "hitl_history": existing,
        },
    )


def _needs_intervention_node(state: CatMasterState) -> Command:
    """HITL checkpoint when a task fails and needs human guidance."""
    result = state.get("task_result") or {}
    feedback = interrupt({
        "type": "task_intervention",
        "task_id": state.get("current_task_id", ""),
        "task_outcome": result.get("task_outcome", ""),
        "task_summary": result.get("task_summary", ""),
        "message": "A task needs intervention. Provide guidance or type 'skip' to continue.",
    })

    feedback_text = str(feedback or "").strip()
    existing = list(state.get("hitl_history") or [])
    existing.append({
        "task_id": state.get("current_task_id", ""),
        "feedback": feedback_text,
    })
    return Command(goto="run_director", update={"hitl_history": existing})


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_standard_graph(
    *,
    task_runner_model: BaseChatModel,
    proposal_model: BaseChatModel,
    director_model: BaseChatModel,
    memory_patch_model: BaseChatModel,
    memory_store: MemoryStore,
    proposal_tools: Sequence[BaseTool],
    director_tools: Sequence[BaseTool],
    task_tools: Sequence[BaseTool],
    memory_tools: Sequence[BaseTool],
    proposal_execution_context_guide: str,
    director_execution_context_guide: Optional[str] = None,
    reporter: Reporter | None = None,
    run_id: str = "",
    run_dir: Optional[Path] = None,
    patch_repair_attempts: int = 1,
    tool_backend: Optional[ToolBackend] = None,
    max_task_steps: int = 60,
    max_plan_steps: int = 60,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_control: Optional[RunControl] = None,
    skills_runtime: Optional[CatMasterSkillsRuntime] = None,
    mounted_skill_tokens: Sequence[str] | None = None,
    tool_selector_model: Optional[BaseChatModel] = None,
    allow_human_intervention: bool = True,
    allow_memory_patch: bool = True,
) -> Any:
    """Build and compile the standard-lane LangGraph."""
    effective_run_dir = run_dir or Path(".")
    effective_director_execution_context_guide = (
        director_execution_context_guide or proposal_execution_context_guide
    )

    proposal_middleware = _build_role_middleware(
        role="proposal",
        lane="standard",
        max_tool_calls=max_plan_steps,
        skills_runtime=skills_runtime,
        mounted_skill_tokens=mounted_skill_tokens,
        selector_model=None,
        enable_selector=False,
    )
    director_middleware = _build_role_middleware(
        role="director",
        lane="standard",
        max_tool_calls=max_plan_steps,
        skills_runtime=skills_runtime,
        mounted_skill_tokens=mounted_skill_tokens,
        selector_model=None,
        enable_selector=False,
    )
    task_middleware = _build_role_middleware(
        role="task_runner",
        lane="standard",
        max_tool_calls=max_task_steps,
        skills_runtime=skills_runtime,
        mounted_skill_tokens=mounted_skill_tokens,
        selector_model=tool_selector_model,
        enable_selector=True,
    )
    memory_middleware = _build_role_middleware(
        role="memory_patch",
        lane="standard",
        max_tool_calls=max_plan_steps,
        skills_runtime=None,
        mounted_skill_tokens=(),
        selector_model=None,
        enable_selector=False,
    )

    proposal_agent = _build_proposal_agent(
        proposal_model,
        list(proposal_tools),
        max_steps=max_plan_steps,
        middleware=proposal_middleware,
    )
    director_agent = _build_director_agent(
        director_model,
        list(director_tools),
        max_steps=max_plan_steps,
        middleware=director_middleware,
    )
    task_agent = _build_task_runner_agent(
        task_runner_model,
        list(task_tools),
        memory_store,
        max_steps=max_task_steps,
        middleware=task_middleware,
    )
    memory_patch_agent = _build_memory_patcher_agent(
        memory_patch_model,
        list(memory_tools),
        max_steps=max_plan_steps,
        middleware=memory_middleware,
    )

    graph = StateGraph(CatMasterState)

    graph.add_node("run_proposal", partial(
        _run_proposal_wrapper,
        agent=proposal_agent,
        memory_store=memory_store,
        execution_context_guide=proposal_execution_context_guide,
        run_dir=effective_run_dir,
        max_steps=max_plan_steps,
    ))

    graph.add_node("proposal_review", _proposal_review_node)

    graph.add_node("run_director", partial(
        _run_director_wrapper,
        agent=director_agent,
        memory_store=memory_store,
        execution_context_guide=effective_director_execution_context_guide,
        max_steps=max_plan_steps,
        allow_memory_patch=allow_memory_patch,
    ))

    graph.add_node("run_task", partial(
        _run_task_wrapper,
        agent=task_agent,
        memory_store=memory_store,
        max_steps=max_task_steps,
        continue_goto="run_director",
        intervention_goto="needs_intervention" if allow_human_intervention else "run_director",
        run_dir=effective_run_dir,
        reporter=reporter,
        run_id=run_id,
    ))

    graph.add_node("run_memory_patch", partial(
        _run_memory_patch_wrapper,
        agent=memory_patch_agent,
        memory_store=memory_store,
        run_id=run_id,
        max_steps=max_plan_steps,
        continue_goto="summarize",
    ))

    graph.add_node("summarize", partial(
        _summarize_node_wrapper,
        memory_store=memory_store,
    ))

    graph.add_node("needs_intervention", _needs_intervention_node)

    graph.set_entry_point("run_proposal")
    graph.add_edge("summarize", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def build_fast_graph(
    *,
    task_runner_model: BaseChatModel,
    director_model: BaseChatModel,
    memory_patch_model: BaseChatModel,
    memory_store: MemoryStore,
    director_tools: Sequence[BaseTool],
    task_tools: Sequence[BaseTool],
    memory_tools: Sequence[BaseTool],
    fast_director_execution_context_guide: str,
    reporter: Reporter | None = None,
    run_id: str = "",
    run_dir: Optional[Path] = None,
    patch_repair_attempts: int = 1,
    tool_backend: Optional[ToolBackend] = None,
    max_task_steps: int = 60,
    max_plan_steps: int = 60,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_control: Optional[RunControl] = None,
    skills_runtime: Optional[CatMasterSkillsRuntime] = None,
    mounted_skill_tokens: Sequence[str] | None = None,
    tool_selector_model: Optional[BaseChatModel] = None,
    allow_memory_patch: bool = True,
) -> Any:
    """Build and compile the fast-lane LangGraph (proposal-free director loop)."""
    effective_run_dir = run_dir or Path(".")
    fast_director_middleware = _build_role_middleware(
        role="fast_director",
        lane="fast",
        max_tool_calls=max_plan_steps,
        skills_runtime=skills_runtime,
        mounted_skill_tokens=mounted_skill_tokens,
        selector_model=None,
        enable_selector=False,
    )
    task_middleware = _build_role_middleware(
        role="task_runner",
        lane="fast",
        max_tool_calls=max_task_steps,
        skills_runtime=skills_runtime,
        mounted_skill_tokens=mounted_skill_tokens,
        selector_model=tool_selector_model,
        enable_selector=True,
    )
    memory_middleware = _build_role_middleware(
        role="memory_patch",
        lane="fast",
        max_tool_calls=max_plan_steps,
        skills_runtime=None,
        mounted_skill_tokens=(),
        selector_model=None,
        enable_selector=False,
    )

    fast_director_agent = _build_fast_director_agent(
        director_model,
        list(director_tools),
        max_steps=max_plan_steps,
        middleware=fast_director_middleware,
    )
    task_agent = _build_task_runner_agent(
        task_runner_model,
        list(task_tools),
        memory_store,
        max_steps=max_task_steps,
        middleware=task_middleware,
    )
    memory_patch_agent = _build_memory_patcher_agent(
        memory_patch_model,
        list(memory_tools),
        max_steps=max_plan_steps,
        middleware=memory_middleware,
    )

    graph = StateGraph(CatMasterState)

    graph.add_node("run_fast_director", partial(
        _run_fast_director_wrapper,
        agent=fast_director_agent,
        memory_store=memory_store,
        execution_context_guide=fast_director_execution_context_guide,
        max_steps=max_plan_steps,
        allow_memory_patch=allow_memory_patch,
    ))

    graph.add_node("run_task", partial(
        _run_task_wrapper,
        agent=task_agent,
        memory_store=memory_store,
        max_steps=max_task_steps,
        continue_goto="run_fast_director",
        intervention_goto="run_fast_director",
        run_dir=effective_run_dir,
        reporter=reporter,
        run_id=run_id,
    ))

    graph.add_node("run_memory_patch", partial(
        _run_memory_patch_wrapper,
        agent=memory_patch_agent,
        memory_store=memory_store,
        run_id=run_id,
        max_steps=max_plan_steps,
        continue_goto="summarize",
    ))

    graph.add_node("summarize", partial(
        _summarize_node_wrapper,
        memory_store=memory_store,
    ))

    graph.set_entry_point("run_fast_director")
    graph.add_edge("summarize", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


class GraphRunner:
    """Convenience wrapper that builds the graph, sets up callbacks, and runs.

    Handles the interrupt/resume cycle required by HITL nodes
    (``proposal_review`` and ``needs_intervention``).  When a node calls
    ``interrupt()``, the runner collects human feedback via the ``Reporter``
    protocol and resumes the graph automatically.
    """

    MAX_INTERRUPT_ROUNDS = 20
    DEPRECATED = True
    DEPRECATION_NOTICE = (
        "GraphRunner is legacy and deprecated. "
        "Prefer SpecialistRunner for active development."
    )

    def __init__(
        self,
        *,
        task_runner_model: BaseChatModel,
        proposal_model: Optional[BaseChatModel] = None,
        director_model: Optional[BaseChatModel] = None,
        memory_patch_model: BaseChatModel,
        registry: Optional[ToolRegistry] = None,
        memory_store: MemoryStore,
        run_context: RunContext,
        reporter: Optional[Reporter] = None,
        tool_backend: Optional[ToolBackend] = None,
        run_control: Optional[RunControl] = None,
        max_task_steps: int = 60,
        max_plan_steps: int = 60,
        recursion_limit: int = 300,
        patch_repair_attempts: int = 1,
        stream_debug_console: bool = False,
        print_state_messages: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        skills_runtime: Optional[CatMasterSkillsRuntime] = None,
        tool_selector_model: Optional[BaseChatModel] = None,
        run_policy: GraphRunPolicy | None = None,
    ) -> None:
        self.task_runner_model = task_runner_model
        self.proposal_model = proposal_model or task_runner_model
        self.director_model = director_model or task_runner_model
        self.memory_patch_model = memory_patch_model
        self.registry = registry or get_tool_registry()
        self.memory_store = memory_store
        self.run_context = run_context
        self.reporter = reporter or NullReporter()
        self.tool_backend = tool_backend
        self.run_control = run_control
        self.max_task_steps = max_task_steps
        self.max_plan_steps = max_plan_steps
        try:
            parsed_limit = int(recursion_limit)
        except Exception:
            parsed_limit = 300
        self.recursion_limit = parsed_limit if parsed_limit > 0 else 1_000_000
        self.patch_repair_attempts = patch_repair_attempts
        self.stream_debug_console = bool(stream_debug_console)
        self.print_state_messages = bool(print_state_messages)
        self.checkpointer = checkpointer or MemorySaver()
        self.skills_runtime = skills_runtime
        self.tool_selector_model = tool_selector_model
        self.run_policy = run_policy or GraphRunPolicy()
        self.is_deprecated = True
        self.deprecation_notice = self.DEPRECATION_NOTICE

        self.memory_store.ensure_exists()
        self.artifact_store = ArtifactStore(run_context.run_dir)
        self.trace_store = TraceStore(run_context.run_dir)

    def _emit(self, name: str, *, category: str = "run", payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.reporter.emit(
                make_event(
                    name,
                    category=category,
                    run_id=self.run_context.run_id,
                    payload=payload or {},
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Interrupt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_interrupt_value(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract the interrupt payload from a graph result, if any.

        LangGraph stores pending interrupts under the ``__interrupt__``
        key.  Each entry has a ``value`` dict produced by the node's
        ``interrupt(...)`` call.
        """
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return None
        if isinstance(interrupts, (list, tuple)) and len(interrupts) > 0:
            entry = interrupts[0]
            if hasattr(entry, "value"):
                return entry.value
            if isinstance(entry, dict):
                return entry.get("value")
        return None

    @staticmethod
    def _stream_preview(value: Any, *, limit: int = 180) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return _snippet(value, limit)
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        return _snippet(text, limit)

    def _log_stream_update(self, update: dict[str, Any]) -> None:
        for node, payload in (update or {}).items():
            if node == "__interrupt__":
                logger.info("[graph.stream] interrupt=%s", self._stream_preview(payload))
                continue
            if not isinstance(payload, dict):
                logger.info("[graph.stream] node=%s payload=%s", node, self._stream_preview(payload))
                continue
            keys = list(payload.keys())
            if not keys:
                logger.info("[graph.stream] node=%s keys=(empty update)", node)
                continue
            message_fields: list[str] = []
            for key, value in payload.items():
                if not isinstance(value, list):
                    continue
                if key == "messages" or key.endswith("_messages"):
                    message_fields.append(str(key))
            msg_count = (
                sum(len(payload.get(field, []) or []) for field in message_fields)
                if message_fields
                else None
            )
            tool_summary = ""
            if message_fields:
                # Prefer canonical `messages`; otherwise use the first message field.
                selected_field = "messages" if "messages" in message_fields else message_fields[0]
                selected_messages = payload.get(selected_field, []) or []
                if selected_messages:
                    tail = selected_messages[-1]
                    tool_calls = getattr(tail, "tool_calls", None)
                    if isinstance(tool_calls, list) and tool_calls:
                        names = [str(call.get("name") or "") for call in tool_calls if isinstance(call, dict)]
                        names = [name for name in names if name]
                        if names:
                            tool_summary = f" tool_calls={','.join(names[:6])}"
            logger.info(
                "[graph.stream] node=%s keys=%s%s%s",
                node,
                ",".join(keys[:8]),
                f" messages={msg_count}" if msg_count is not None else "",
                tool_summary,
            )

    def _emit_stream_update(self, update: dict[str, Any]) -> None:
        for node, payload in (update or {}).items():
            if node == "__interrupt__":
                continue
            if not isinstance(payload, dict):
                self._emit(
                    "GRAPH_NODE_UPDATE",
                    category="graph",
                    payload={
                        "node": str(node or ""),
                        "keys": [],
                        "message_count": 0,
                        "tool_calls": [],
                        "text_preview": self._stream_preview(payload),
                    },
                )
                continue

            keys = [str(key) for key in payload.keys()]
            message_fields = [
                str(key)
                for key, value in payload.items()
                if isinstance(value, list) and (key == "messages" or str(key).endswith("_messages"))
            ]
            message_count = sum(len(payload.get(field, []) or []) for field in message_fields)
            text_preview = ""
            tool_calls: list[str] = []

            selected_field = "messages" if "messages" in message_fields else (message_fields[0] if message_fields else "")
            if selected_field:
                selected_messages = payload.get(selected_field, []) or []
                if selected_messages:
                    tail = selected_messages[-1]
                    content = getattr(tail, "content", None)
                    if isinstance(content, str):
                        text_preview = _snippet(content, 180)
                    elif isinstance(content, list):
                        fragments: list[str] = []
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get("text"), str):
                                fragments.append(str(item.get("text") or ""))
                        if fragments:
                            text_preview = _snippet("\n".join(fragments), 180)
                    raw_tool_calls = getattr(tail, "tool_calls", None)
                    if isinstance(raw_tool_calls, list):
                        tool_calls = [
                            str(call.get("name") or "").strip()
                            for call in raw_tool_calls
                            if isinstance(call, dict) and str(call.get("name") or "").strip()
                        ]

            self._emit(
                "GRAPH_NODE_UPDATE",
                category="graph",
                payload={
                    "node": str(node or ""),
                    "keys": keys[:12],
                    "message_count": message_count,
                    "tool_calls": tool_calls[:8],
                    "text_preview": text_preview,
                },
            )

    async def _ainvoke_graph_once(
        self,
        compiled: Any,
        graph_input: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        live_stream = self.reporter.is_live()
        if not (live_stream or self.stream_debug_console or self.print_state_messages):
            return await compiled.ainvoke(graph_input, config=config)

        logger.info(
            "[graph.stream] enabled stream_mode=updates (live=%s, stream_debug_console=%s, print_state_messages=%s)",
            live_stream,
            self.stream_debug_console,
            self.print_state_messages,
        )
        streamed_result: Dict[str, Any] = {}
        async for update in compiled.astream(graph_input, config=config, stream_mode="updates"):
            if isinstance(update, dict):
                if self.stream_debug_console or self.print_state_messages:
                    self._log_stream_update(update)
                if live_stream:
                    self._emit_stream_update(update)
                if "__interrupt__" in update:
                    streamed_result["__interrupt__"] = update["__interrupt__"]

        try:
            if hasattr(compiled, "aget_state"):
                snapshot = await compiled.aget_state(config)
            else:
                snapshot = compiled.get_state(config)
            values = getattr(snapshot, "values", None)
            if isinstance(values, dict):
                result = dict(values)
                if "__interrupt__" in streamed_result and "__interrupt__" not in result:
                    result["__interrupt__"] = streamed_result["__interrupt__"]
                return result
        except Exception as exc:
            logger.debug("stream snapshot fallback failed: %s", exc)

        return streamed_result

    def _collect_human_feedback(self, interrupt_payload: Dict[str, Any]) -> str:
        """Collect feedback from the human via the Reporter protocol."""
        interrupt_type = interrupt_payload.get("type", "")
        logger.info(
            "[_collect_human_feedback] type=%s, payload_keys=%s",
            interrupt_type, list(interrupt_payload.keys()),
        )

        if interrupt_type == "proposal_review":
            proposal_md = interrupt_payload.get("proposal_md", "")
            work_packages = interrupt_payload.get("work_packages", [])
            logger.info(
                "[_collect_human_feedback] proposal_md_len=%d, work_packages=%d",
                len(proposal_md), len(work_packages),
            )
            return self.reporter.prompt_proposal_feedback(
                todo=work_packages,
                proposal_description=proposal_md,
            )

        if interrupt_type == "task_intervention":
            task_summary = interrupt_payload.get("task_summary", "")
            task_id = interrupt_payload.get("task_id", "")
            return self.reporter.prompt_hitl_feedback(
                report_text=task_summary,
                report_path=task_id,
            )

        return self.reporter.prompt_interrupt_feedback(
            guidance=interrupt_payload.get("message", "Provide feedback."),
            run_id=self.run_context.run_id,
            phase=interrupt_type,
        )

    # ------------------------------------------------------------------
    # Lifecycle side-effects (ported from old Orchestrator)
    # ------------------------------------------------------------------

    def _initialize_memory_goal(self, user_request: str) -> None:
        """Deprecated no-op kept for compatibility with older tests/patches."""
        _ = user_request

    def _write_task_state(self, state: Dict[str, Any], lane: str) -> None:
        """Persist task_state.json for resume and WebUI inspection."""
        path = self.run_context.run_dir / "task_state.json"
        body: Dict[str, Any] = {
            "schema_version": 2,
            "lane": lane,
            "user_request": state.get("user_request", ""),
            "proposal_review_enabled": bool(state.get("proposal_review_enabled", True)),
            "tasks": state.get("tasks", []),
            "observations": state.get("observations", []),
            "status": state.get("status", "running"),
            "hitl_history": state.get("hitl_history", []),
        }
        body.update(_derive_graph_task_state_fields(state))
        memory_patch_result = state.get("memory_patch_result")
        if isinstance(memory_patch_result, dict) and memory_patch_result:
            body["memory_patch_result"] = memory_patch_result
        contract_violation = state.get("contract_violation")
        if isinstance(contract_violation, dict) and contract_violation:
            body["contract_violation"] = contract_violation
        if lane == "standard":
            body["proposal"] = {
                "proposal_path": "proposal.md",
                "work_packages": state.get("work_packages", []),
            }
        session_context = str(state.get("session_context_text") or "").strip()
        if session_context:
            body["session_context_text"] = session_context
        chat_session_id = str(state.get("chat_session_id") or "").strip()
        if chat_session_id:
            body["chat_session_id"] = chat_session_id
        entry_context_tokens_estimate = int(state.get("entry_context_tokens_estimate") or 0)
        if entry_context_tokens_estimate > 0:
            body["entry_context_tokens_estimate"] = entry_context_tokens_estimate
        last_interrupt = state.get("last_interrupt")
        if isinstance(last_interrupt, dict) and last_interrupt:
            body["last_interrupt"] = last_interrupt
        resume_checkpoint = state.get("task_resume_checkpoint")
        if resume_checkpoint not in (None, ""):
            body["task_resume_checkpoint"] = resume_checkpoint
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                pass
        status = str(body.get("status") or "")
        if "interrupt_history" not in body and "interrupt_history" in existing:
            body["interrupt_history"] = existing.get("interrupt_history")
        if status in {"interrupted_paused", "awaiting_human_feedback"}:
            for key in ("task_resume_checkpoint", "last_interrupt"):
                if key not in body and key in existing:
                    body[key] = existing.get(key)
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        user_request: str,
        *,
        lane: str = "standard",
        proposal_review: bool = True,
        session_context_text: str = "",
        chat_session_id: str = "",
        entry_context_tokens_estimate: int = 0,
    ) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("GraphRunner.run() cannot be called inside a running event loop; use GraphRunner.arun().")
        return asyncio.run(
            self.arun(
                user_request,
                lane=lane,
                proposal_review=proposal_review,
                session_context_text=session_context_text,
                chat_session_id=chat_session_id,
                entry_context_tokens_estimate=entry_context_tokens_estimate,
            )
        )

    async def arun(
        self,
        user_request: str,
        *,
        lane: str = "standard",
        proposal_review: bool = True,
        session_context_text: str = "",
        chat_session_id: str = "",
        entry_context_tokens_estimate: int = 0,
    ) -> Dict[str, Any]:
        workspace = self.run_context.workspace
        run_dir = self.run_context.run_dir

        callbacks = build_callbacks(
            artifact_store=self.artifact_store,
            trace_store=self.trace_store,
            reporter=self.reporter,
            run_id=self.run_context.run_id,
            enable_step_logs=self.print_state_messages,
        )

        try:
            surface: RuntimeToolSurface = build_runtime_tool_surface(
                registry=self.registry,
                run_context=self.run_context,
                run_dir=run_dir,
                task_runner_denylist=_TASK_RUNNER_TOOL_DENYLIST,
                include_literature_tool=self.run_policy.enable_literature_tool,
            )
            local_pool = _dedupe_tools_by_name(
                list(surface.proposal_tools) + list(surface.director_tools) + list(surface.task_tools)
            )
            apply_tool = next(
                (tool for tool in local_pool if str(getattr(tool, "name", "") or "") == "apply_aider_edits"),
                None,
            )
            if apply_tool is not None:
                apply_tool = _make_memory_scoped_apply_tool(apply_tool)
            memory_tools = _dedupe_tools_by_name(
                [apply_tool] if apply_tool is not None else []
            )
            if apply_tool is None:
                logger.warning("apply_aider_edits is unavailable for memory patcher; updates may fail.")

            if self.skills_runtime is not None:
                self.skills_runtime.refresh_catalog()
                proposal_execution_context_guide = render_proposal_skill_guide(
                    self.skills_runtime.visible_skills("proposal")
                )
                director_execution_context_guide = render_director_skill_guide(
                    self.skills_runtime.visible_skills("director")
                )
                fast_director_execution_context_guide = render_fast_director_skill_guide(
                    self.skills_runtime.visible_skills("fast_director")
                )
            else:
                proposal_execution_context_guide = render_proposal_skill_guide([])
                director_execution_context_guide = render_director_skill_guide([])
                fast_director_execution_context_guide = render_fast_director_skill_guide([])
            mounted_skill_tokens: tuple[str, ...] = ()

            if lane == "fast":
                fast_director_tools = [
                    tool
                    for tool in surface.fast_director_tools
                    if str(getattr(tool, "name", "") or "") != "apply_aider_edits"
                ]
                compiled = build_fast_graph(
                    task_runner_model=self.task_runner_model,
                    director_model=self.director_model,
                    memory_patch_model=self.memory_patch_model,
                    memory_store=self.memory_store,
                    director_tools=fast_director_tools,
                    task_tools=surface.task_tools,
                    memory_tools=memory_tools,
                    fast_director_execution_context_guide=fast_director_execution_context_guide,
                    reporter=self.reporter,
                    run_id=self.run_context.run_id,
                    run_dir=run_dir,
                    patch_repair_attempts=self.patch_repair_attempts,
                    tool_backend=self.tool_backend,
                    max_task_steps=self.max_task_steps,
                    max_plan_steps=self.max_plan_steps,
                    checkpointer=self.checkpointer,
                    run_control=self.run_control,
                    skills_runtime=self.skills_runtime,
                    mounted_skill_tokens=mounted_skill_tokens,
                    tool_selector_model=self.tool_selector_model,
                    allow_memory_patch=self.run_policy.allow_memory_patch,
                )
            else:
                compiled = build_standard_graph(
                    task_runner_model=self.task_runner_model,
                    proposal_model=self.proposal_model,
                    director_model=self.director_model,
                    memory_patch_model=self.memory_patch_model,
                    memory_store=self.memory_store,
                    proposal_tools=surface.proposal_tools,
                    director_tools=surface.director_tools,
                    task_tools=surface.task_tools,
                    memory_tools=memory_tools,
                    proposal_execution_context_guide=proposal_execution_context_guide,
                    director_execution_context_guide=director_execution_context_guide,
                    reporter=self.reporter,
                    run_id=self.run_context.run_id,
                    run_dir=run_dir,
                    patch_repair_attempts=self.patch_repair_attempts,
                    tool_backend=self.tool_backend,
                    max_task_steps=self.max_task_steps,
                    max_plan_steps=self.max_plan_steps,
                    checkpointer=self.checkpointer,
                    run_control=self.run_control,
                    skills_runtime=self.skills_runtime,
                    mounted_skill_tokens=mounted_skill_tokens,
                    tool_selector_model=self.tool_selector_model,
                    allow_human_intervention=self.run_policy.allow_human_intervention,
                    allow_memory_patch=self.run_policy.allow_memory_patch,
                )

            initial_state: CatMasterState = {
                "user_request": user_request,
                "lane": lane,
                "chat_session_id": str(chat_session_id or "").strip(),
                "session_context_text": str(session_context_text or "").strip(),
                "entry_context_tokens_estimate": int(entry_context_tokens_estimate or 0),
                "proposal_messages": [],
                "director_messages": [],
                "runner_messages": [],
                "proposal_md": "",
                "work_packages": [],
                "proposal_approved": False,
                "proposal_feedback": "",
                "proposal_review_enabled": bool(proposal_review),
                "director_decision": {},
                "next_action": "",
                "tasks": [],
                "observations": [],
                "current_task_id": "task_01",
                "current_task_packet": {"goal": user_request},
                "task_result": {},
                "pending_memory_updates": [],
                "memory_patch_result": {},
                "status": "running",
                "summary": "",
                "contract_violation": {},
                "hitl_history": [],
            }

            self._write_task_state(initial_state, lane)

            thread_id = self.run_context.run_id
            config: Dict[str, Any] = {
                "callbacks": callbacks,
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.recursion_limit,
            }
            return await self._ainvoke_loop(compiled, initial_state, config, workspace, lane)
        finally:
            pass

    async def _ainvoke_loop(
        self,
        compiled: Any,
        initial_state: CatMasterState,
        config: Dict[str, Any],
        workspace: Optional[Path],
        lane: str = "standard",
    ) -> Dict[str, Any]:
        """Run the graph with workspace scope, handling HITL interrupt/resume."""
        user_request = initial_state.get("user_request", "")

        async def _do_invoke() -> Dict[str, Any]:
            result = await self._ainvoke_graph_once(compiled, initial_state, config)

            for _ in range(self.MAX_INTERRUPT_ROUNDS):
                interrupt_payload = self._get_interrupt_value(result)
                if interrupt_payload is None:
                    break

                if not self.run_policy.allow_human_intervention:
                    interrupt_type = str(interrupt_payload.get("type") or "unknown")
                    result = {
                        **result,
                        "status": "failure",
                        "summary": f"Run aborted because human intervention is disabled but interrupt '{interrupt_type}' was raised.",
                    }
                    break

                if self.run_control and self.run_control.is_interrupt_requested():
                    self.run_control.ack_interrupt(phase="hitl_paused")
                    self._write_task_state(
                        {**result, "status": "interrupted_paused"}, lane,
                    )
                    self._emit("RUN_PAUSED", payload={"phase": "hitl_paused", "status": "interrupted_paused"})
                    return {
                        "tasks": result.get("tasks", []),
                        "observations": result.get("observations", []),
                        "summary": result.get("summary", ""),
                        "final_answer": "",
                        "status": "interrupted_paused",
                    }

                interrupt_type = str(interrupt_payload.get("type") or "")
                self._write_task_state(
                    {
                        **result,
                        "status": "awaiting_human_feedback",
                        "last_interrupt": interrupt_payload,
                    },
                    lane,
                )
                self._emit(
                    "RUN_WAITING_INPUT",
                    payload={
                        "status": "awaiting_human_feedback",
                        "interrupt_type": interrupt_type,
                        "message": str(interrupt_payload.get("message") or ""),
                    },
                )
                feedback = self._collect_human_feedback(interrupt_payload)
                if not feedback and not self.reporter.is_live():
                    logger.warning("HITL interrupt with no live reporter; auto-approving")
                    feedback = "approve"
                self._emit(
                    "RUN_INPUT_RECEIVED",
                    payload={
                        "interrupt_type": interrupt_type,
                        "feedback_len": len(str(feedback or "")),
                    },
                )
                # Clear stale HITL snapshot state before resume so WebUI does not
                # reconstruct the same prompt after approval/submission.
                self._write_task_state(
                    {
                        **result,
                        "status": "running",
                    },
                    lane,
                )

                result = await self._ainvoke_graph_once(
                    compiled,
                    Command(resume=feedback),
                    config,
                )

            if self.run_control and self.run_control.is_interrupt_requested():
                self.run_control.ack_interrupt(phase="completed_with_interrupt")

            status = result.get("status", "done")
            summary = result.get("summary", "")
            self._write_task_state({**result, "status": status}, lane)

            self._emit("RUN_END", payload={"status": status, "summary_snippet": _snippet(summary, 320)})

            return {
                "tasks": result.get("tasks", []),
                "observations": result.get("observations", []),
                "summary": summary,
                "final_answer": summary,
                "status": status,
                "pending_memory_updates": result.get("pending_memory_updates", []),
                "run_id": self.run_context.run_id,
                "run_dir": str(self.run_context.run_dir),
                "run_export_path": "",
            }

        if workspace is not None:
            with workspace_scope(workspace):
                return await _do_invoke()
        return await _do_invoke()


__all__ = [
    "CatMasterState",
    "GraphRunPolicy",
    "build_standard_graph",
    "build_fast_graph",
    "GraphRunner",
]
