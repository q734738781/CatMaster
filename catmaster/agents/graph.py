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
import re
import shutil
from contextlib import asynccontextmanager
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
from catmaster.runtime.mcp_filesystem import MCPFilesystemRuntime
from catmaster.runtime.tool_surface import RuntimeToolSurface, build_runtime_tool_surface
from catmaster.runtime.usage_stats import write_usage_summary
from catmaster.runtime.run_ledger.blob_builder import build_run_search_blob
from catmaster.runtime.run_ledger.models import RunLedgerEntry
from catmaster.runtime.run_ledger.store import RunLedgerStore
from catmaster.runtime.run_ledger.history_reader import HistoryReader
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
from catmaster.llm.config import MCPConfig
from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    DIRECTOR_SYSTEM_PROMPT,
    FAST_DIRECTOR_SYSTEM_PROMPT,
    TASK_RUNNER_SYSTEM_PROMPT,
    MEMORY_PATCHER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_TASK_RUNNER_TOOL_DENYLIST: set[str] = set()
_MEMORY_PATCH_READONLY_TOOL_ALLOWLIST: set[str] = {
    "search_files",
    "list_directory",
    "read_text_file",
    "read_multiple_files",
}


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


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class CatMasterState(TypedDict, total=False):
    user_request: str
    lane: str
    historical_runs_context_text: str
    historical_runs_citations: list[dict]

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
        from catmaster.runtime.safe_tool_selector import SafeLLMToolSelectorMiddleware as _LLMToolSelectorMiddleware
    except Exception as exc:
        raise RuntimeError(
            "SafeLLMToolSelectorMiddleware is unavailable. Verify the CatMaster runtime install."
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


_SKILL_FILESYSTEM_ALWAYS_INCLUDE = [
    "read_text_file",
    "read_multiple_files",
    "search_files",
    "list_directory",
    "directory_tree",
]


def _build_role_middleware(
    *,
    role: str,
    max_tool_calls: int,
    skills_runtime: CatMasterSkillsRuntime | None,
    skills_mount_available: bool,
    selector_model: BaseChatModel | None,
    enable_selector: bool,
) -> list[Any]:
    middleware: list[Any] = []
    if role != "memory_patch" and skills_runtime is not None:
        middleware.append(
            CatMasterSkillsMiddleware(
                role=role,
                skills_runtime=skills_runtime,
                skills_mount_available=skills_mount_available,
            )
        )

    if enable_selector:
        LLMToolSelectorMiddleware = _load_llm_tool_selector_middleware()
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
) -> Command:
    return await run_director(
        state,
        agent=agent,
        memory_store=memory_store,
        execution_context_guide=execution_context_guide,
        max_steps=max_steps,
    )


async def _run_fast_director_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    max_steps: int,
) -> Command:
    return await run_fast_director(
        state,
        agent=agent,
        memory_store=memory_store,
        execution_context_guide=execution_context_guide,
        max_steps=max_steps,
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
) -> Command:
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
        _write_observation_file(
            run_dir=run_dir,
            task_id=str(state.get("current_task_id", "") or ""),
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
    if feedback_text.lower() in ("approved", "approve", "ok", "yes", "y", "lgtm"):
        return Command(goto="run_director", update={"proposal_approved": True, "proposal_feedback": ""})

    return Command(
        goto="run_proposal",
        update={"proposal_approved": False, "proposal_feedback": feedback_text},
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
    run_id: str = "",
    run_dir: Optional[Path] = None,
    patch_repair_attempts: int = 1,
    tool_backend: Optional[ToolBackend] = None,
    max_task_steps: int = 60,
    max_plan_steps: int = 60,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_control: Optional[RunControl] = None,
    skills_runtime: Optional[CatMasterSkillsRuntime] = None,
    skills_mount_available: bool = False,
    tool_selector_model: Optional[BaseChatModel] = None,
) -> Any:
    """Build and compile the standard-lane LangGraph."""
    effective_run_dir = run_dir or Path(".")
    effective_director_execution_context_guide = (
        director_execution_context_guide or proposal_execution_context_guide
    )

    proposal_middleware = _build_role_middleware(
        role="proposal",
        max_tool_calls=max_plan_steps,
        skills_runtime=skills_runtime,
        skills_mount_available=skills_mount_available,
        selector_model=None,
        enable_selector=False,
    )
    director_middleware = _build_role_middleware(
        role="director",
        max_tool_calls=max_plan_steps,
        skills_runtime=skills_runtime,
        skills_mount_available=skills_mount_available,
        selector_model=None,
        enable_selector=False,
    )
    task_middleware = _build_role_middleware(
        role="task_runner",
        max_tool_calls=max_task_steps,
        skills_runtime=skills_runtime,
        skills_mount_available=skills_mount_available,
        selector_model=tool_selector_model,
        enable_selector=True,
    )
    memory_middleware = _build_role_middleware(
        role="memory_patch",
        max_tool_calls=max_plan_steps,
        skills_runtime=None,
        skills_mount_available=False,
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
    ))

    graph.add_node("run_task", partial(
        _run_task_wrapper,
        agent=task_agent,
        memory_store=memory_store,
        max_steps=max_task_steps,
        continue_goto="run_director",
        intervention_goto="needs_intervention",
        run_dir=effective_run_dir,
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
    run_id: str = "",
    run_dir: Optional[Path] = None,
    patch_repair_attempts: int = 1,
    tool_backend: Optional[ToolBackend] = None,
    max_task_steps: int = 60,
    max_plan_steps: int = 60,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_control: Optional[RunControl] = None,
    skills_runtime: Optional[CatMasterSkillsRuntime] = None,
    skills_mount_available: bool = False,
    tool_selector_model: Optional[BaseChatModel] = None,
) -> Any:
    """Build and compile the fast-lane LangGraph (proposal-free director loop)."""
    effective_run_dir = run_dir or Path(".")
    fast_director_middleware = _build_role_middleware(
        role="fast_director",
        max_tool_calls=max_plan_steps,
        skills_runtime=skills_runtime,
        skills_mount_available=skills_mount_available,
        selector_model=None,
        enable_selector=False,
    )
    task_middleware = _build_role_middleware(
        role="task_runner",
        max_tool_calls=max_task_steps,
        skills_runtime=skills_runtime,
        skills_mount_available=skills_mount_available,
        selector_model=tool_selector_model,
        enable_selector=True,
    )
    memory_middleware = _build_role_middleware(
        role="memory_patch",
        max_tool_calls=max_plan_steps,
        skills_runtime=None,
        skills_mount_available=False,
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
    ))

    graph.add_node("run_task", partial(
        _run_task_wrapper,
        agent=task_agent,
        memory_store=memory_store,
        max_steps=max_task_steps,
        continue_goto="run_fast_director",
        intervention_goto="run_fast_director",
        run_dir=effective_run_dir,
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
        mcp_config: Optional[MCPConfig] = None,
        max_task_steps: int = 60,
        max_plan_steps: int = 60,
        recursion_limit: int = 300,
        patch_repair_attempts: int = 1,
        stream_debug_console: bool = False,
        print_state_messages: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        run_ledger_store: Optional[RunLedgerStore] = None,
        history_reader: Optional[HistoryReader] = None,
        skills_runtime: Optional[CatMasterSkillsRuntime] = None,
        tool_selector_model: Optional[BaseChatModel] = None,
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
        self.mcp_config = mcp_config or MCPConfig()
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
        self.run_ledger_store = run_ledger_store
        self.history_reader = history_reader
        self.skills_runtime = skills_runtime
        self.tool_selector_model = tool_selector_model

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
                if self.print_state_messages:
                    for field in message_fields:
                        field_messages = payload.get(field, []) or []
                        for idx, msg in enumerate(field_messages):
                            try:
                                dumped = msg.model_dump() if hasattr(msg, "model_dump") else str(msg)
                                msg_text = json.dumps(dumped, ensure_ascii=False, default=str)
                            except Exception:
                                dumped = {}
                                msg_text = str(msg)
                            logger.info(
                                "[graph.stream.messages] node=%s field=%s idx=%d payload=%s",
                                node,
                                field,
                                idx,
                                msg_text,
                            )
                            if isinstance(dumped, dict):
                                msg_type = str(dumped.get("type") or "").strip().lower()
                                if msg_type == "ai":
                                    parsed_calls = list(dumped.get("tool_calls") or [])
                                    for call_idx, call in enumerate(parsed_calls):
                                        name = str(call.get("name") or "")
                                        args = call.get("args")
                                        if isinstance(args, str):
                                            args_json = args
                                        else:
                                            try:
                                                args_json = json.dumps(args, ensure_ascii=False, default=str)
                                            except Exception:
                                                args_json = str(args)
                                        logger.info(
                                            "[graph.stream.tool_input] node=%s field=%s msg_idx=%d call_idx=%d name=%s args_json=%s",
                                            node,
                                            field,
                                            idx,
                                            call_idx,
                                            name,
                                            args_json,
                                        )
                                    raw_calls = list(
                                        (dumped.get("additional_kwargs") or {}).get("tool_calls") or []
                                    )
                                    for call_idx, call in enumerate(raw_calls):
                                        fn = call.get("function") if isinstance(call, dict) else {}
                                        if not isinstance(fn, dict):
                                            fn = {}
                                        logger.info(
                                            "[graph.stream.tool_input.raw] node=%s field=%s msg_idx=%d call_idx=%d name=%s arguments_raw=%s",
                                            node,
                                            field,
                                            idx,
                                            call_idx,
                                            str(fn.get("name") or ""),
                                            str(fn.get("arguments") or ""),
                                        )
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
                        # Debug mode: print full toolcall payload per turn without truncation.
                        for idx, call in enumerate(tool_calls):
                            try:
                                payload_text = json.dumps(call, ensure_ascii=False, default=str)
                            except Exception:
                                payload_text = str(call)
                            logger.info(
                                "[graph.stream.toolcall] node=%s field=%s idx=%d payload=%s",
                                node,
                                selected_field,
                                idx,
                                payload_text,
                            )
            logger.info(
                "[graph.stream] node=%s keys=%s%s%s",
                node,
                ",".join(keys[:8]),
                f" messages={msg_count}" if msg_count is not None else "",
                tool_summary,
            )

    async def _ainvoke_graph_once(
        self,
        compiled: Any,
        graph_input: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not (self.stream_debug_console or self.print_state_messages):
            return await compiled.ainvoke(graph_input, config=config)

        logger.info(
            "[graph.stream] enabled stream_mode=updates (stream_debug_console=%s, print_state_messages=%s)",
            self.stream_debug_console,
            self.print_state_messages,
        )
        streamed_result: Dict[str, Any] = {}
        async for update in compiled.astream(graph_input, config=config, stream_mode="updates"):
            if isinstance(update, dict):
                self._log_stream_update(update)
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
        """Write the user objective into MEMORY/topics/GOAL.md.

        Mirrors old Orchestrator._initialize_memory_goal.
        """
        goal = " ".join((user_request or "").split()).strip()
        if not goal:
            return
        self.memory_store.ensure_exists()
        goal_path = self.memory_store.topics_dir / "GOAL.md"
        original = goal_path.read_text(encoding="utf-8") if goal_path.exists() else ""
        updated = original

        primary_match = re.search(r"(?m)^- Primary objective:\s*(.*)$", updated)
        current_primary = (primary_match.group(1) if primary_match else "").strip()
        primary_changed = current_primary != goal

        if primary_match:
            updated = re.sub(
                r"(?m)^- Primary objective:.*$",
                f"- Primary objective: {goal}",
                updated,
                count=1,
            )
        else:
            line = f"- Primary objective: {goal}\n"
            marker = "## TL;DR\n"
            if marker in updated:
                updated = updated.replace(marker, marker + line, 1)
            else:
                updated = updated.rstrip()
                if updated:
                    updated += "\n\n"
                updated += "## TL;DR\n" + line
            primary_changed = True

        lines = updated.splitlines()
        compacted: List[str] = []
        seen_change_log = False
        for raw in lines:
            if raw.strip() == "## Change log":
                if seen_change_log:
                    continue
                seen_change_log = True
            compacted.append(raw)
        updated = "\n".join(compacted)

        if primary_changed:
            ts = datetime.utcnow().strftime("%Y-%m-%d")
            entry = f"- [{ts}] {goal}"
            if "## Change log" not in updated:
                updated = updated.rstrip()
                if updated:
                    updated += "\n\n"
                updated += "## Change log\n"
            if entry not in updated:
                updated = updated.rstrip() + "\n" + entry + "\n"

        normalized = updated.rstrip() + "\n"
        if normalized != (original.rstrip() + "\n"):
            goal_path.write_text(normalized, encoding="utf-8")
            refresh_fn = getattr(self.memory_store, "refresh_index_from_topics", None)
            if callable(refresh_fn):
                try:
                    refresh_fn()
                except Exception:
                    pass

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
        historical_runs_context = str(state.get("historical_runs_context_text") or "").strip()
        if historical_runs_context:
            body["historical_runs_context_text"] = historical_runs_context
        historical_runs_citations = state.get("historical_runs_citations")
        if isinstance(historical_runs_citations, list) and historical_runs_citations:
            body["historical_runs_citations"] = historical_runs_citations
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

    def _publish_report(self, user_request: str, final_answer: str) -> Dict[str, str]:
        """Generate FINAL_REPORT.md and copy memory into the current run reports."""
        run_reports = self.run_context.run_dir / "reports"
        run_reports.mkdir(parents=True, exist_ok=True)

        final_report = run_reports / "FINAL_REPORT.md"
        final_report.write_text("\n".join([
            "# Final Report",
            "",
            "## User Query",
            user_request,
            "",
            "## Final Answer",
            final_answer,
            "",
        ]), encoding="utf-8")

        memory_src = self.memory_store.index_path
        memory_dst = run_reports / "MEMORY.md"
        try:
            shutil.copy2(memory_src, memory_dst)
        except Exception:
            if memory_src.exists():
                memory_dst.write_text(
                    memory_src.read_text(encoding="utf-8"), encoding="utf-8",
                )

        return {
            "run_dir": str(self.run_context.run_dir),
            "final_report": str(final_report),
            "memory_report": str(memory_dst),
        }

    def _relpath_to_system_root(self, path: Path) -> str:
        sys_root = system_root(workspace=self.run_context.workspace).resolve()
        try:
            return str(path.resolve().relative_to(sys_root)).replace("\\", "/")
        except Exception:
            return str(path.resolve())

    @staticmethod
    def _task_goals_for_export(state: Dict[str, Any], fallback_goals: List[str]) -> List[str]:
        tasks = state.get("tasks")
        if isinstance(tasks, list):
            out: List[str] = []
            for item in tasks:
                if not isinstance(item, dict):
                    continue
                for key in ("goal", "task_detail", "title", "task"):
                    value = " ".join(str(item.get(key) or "").split()).strip()
                    if value:
                        out.append(value)
                        break
                if len(out) >= 10:
                    break
            if out:
                return out
        return list(fallback_goals[:10])

    @staticmethod
    def _top_observations_for_export(state: Dict[str, Any], *, limit: int = 10) -> List[str]:
        observations = state.get("observations")
        if not isinstance(observations, list):
            return []
        out: List[str] = []
        for item in observations:
            if isinstance(item, dict):
                text = (
                    str(item.get("summary") or item.get("observation") or item.get("value") or "")
                    .replace("\n", " ")
                    .strip()
                )
            else:
                text = str(item).replace("\n", " ").strip()
            if not text:
                continue
            out.append(text)
            if len(out) >= limit:
                break
        return out

    def _publish_run_export(
        self,
        *,
        state: Dict[str, Any],
        user_request: str,
        final_answer: str,
        lane: str,
        status: str,
        report_paths: Dict[str, str],
    ) -> Dict[str, str]:
        run_reports = self.run_context.run_dir / "reports"
        run_reports.mkdir(parents=True, exist_ok=True)
        export_path = run_reports / "RUN_EXPORT.json"

        blob = build_run_search_blob(self.run_context.run_dir)
        task_goals = self._task_goals_for_export(state, blob.task_goals)
        top_observations = self._top_observations_for_export(state)
        payload: Dict[str, Any] = {
            "request": user_request,
            "answer_summary": final_answer,
            "lane": lane,
            "status": status,
            "task_goals": task_goals,
            "top_observations": top_observations,
            "tool_names": blob.tool_names,
            "artifact_paths": blob.artifact_paths,
            "final_report_path": self._relpath_to_system_root(Path(report_paths.get("final_report", ""))),
            "run_dir": self._relpath_to_system_root(self.run_context.run_dir),
            "run_id": self.run_context.run_id,
            "project_id": self.run_context.project_id,
        }
        export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "run_export": str(export_path),
            "run_export_relpath": self._relpath_to_system_root(export_path),
        }

    async def _upsert_run_ledger(
        self,
        *,
        lane: str,
        status: str,
        user_request: str,
        final_answer: str,
        report_paths: Dict[str, str],
        export_paths: Dict[str, str],
    ) -> None:
        if self.run_ledger_store is None:
            return
        blob = build_run_search_blob(self.run_context.run_dir)
        final_report_relpath = self._relpath_to_system_root(Path(report_paths.get("final_report", "")))
        run_export_relpath = str(export_paths.get("run_export_relpath") or "").strip()
        entry = RunLedgerEntry(
            project_id=self.run_context.project_id,
            run_id=self.run_context.run_id,
            lane=lane,
            status=status,
            request=user_request,
            answer_summary=final_answer,
            search_blob_text=blob.search_blob_text,
            final_report_relpath=final_report_relpath,
            run_export_relpath=run_export_relpath,
            ts_start=self.run_context.start_time,
            ts_end=datetime.now(timezone.utc).isoformat(),
            model_name=self.run_context.model_name,
            provider=str(self.run_context.provider or ""),
        )
        self.run_ledger_store.upsert_entry(entry)
        if self.history_reader is not None:
            try:
                await self.history_reader.aindex_entry(entry)
            except Exception as exc:
                logger.warning("run ledger dense index update failed: %s", exc)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _open_mcp_filesystem_runtime(self):
        fs_cfg = self.mcp_config.filesystem if self.mcp_config is not None else None
        if fs_cfg is None or not fs_cfg.enabled:
            yield None
            return
        runtime = MCPFilesystemRuntime(
            config=fs_cfg,
            run_context=self.run_context,
            reporter=self.reporter,
        )
        async with runtime as active:
            yield active

    def run(
        self,
        user_request: str,
        *,
        lane: str = "standard",
        proposal_review: bool = True,
    ) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("GraphRunner.run() cannot be called inside a running event loop; use GraphRunner.arun().")
        return asyncio.run(self.arun(user_request, lane=lane, proposal_review=proposal_review))

    async def arun(
        self,
        user_request: str,
        *,
        lane: str = "standard",
        proposal_review: bool = True,
    ) -> Dict[str, Any]:
        workspace = self.run_context.workspace
        run_dir = self.run_context.run_dir

        self._initialize_memory_goal(user_request)

        callbacks = build_callbacks(
            artifact_store=self.artifact_store,
            trace_store=self.trace_store,
            reporter=self.reporter,
            run_id=self.run_context.run_id,
            print_raw_tool_calls=self.print_state_messages,
            print_llm_context_messages=self.print_state_messages,
        )

        try:
            async with self._open_mcp_filesystem_runtime() as mcp_fs_runtime:
                surface: RuntimeToolSurface = build_runtime_tool_surface(
                    registry=self.registry,
                    run_context=self.run_context,
                    run_dir=run_dir,
                    mcp_fs_runtime=mcp_fs_runtime,
                    task_runner_denylist=_TASK_RUNNER_TOOL_DENYLIST,
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
                memory_read_tools: list[BaseTool] = []
                if mcp_fs_runtime is not None:
                    memory_read_tools = [
                        tool
                        for tool in mcp_fs_runtime.role_filtered_tools(role="memory_patch")
                        if str(getattr(tool, "name", "") or "") in _MEMORY_PATCH_READONLY_TOOL_ALLOWLIST
                    ]
                memory_tools = _dedupe_tools_by_name(
                    memory_read_tools + ([apply_tool] if apply_tool is not None else [])
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
                        run_id=self.run_context.run_id,
                        run_dir=run_dir,
                        patch_repair_attempts=self.patch_repair_attempts,
                        tool_backend=self.tool_backend,
                        max_task_steps=self.max_task_steps,
                        max_plan_steps=self.max_plan_steps,
                        checkpointer=self.checkpointer,
                        run_control=self.run_control,
                        skills_runtime=self.skills_runtime,
                        skills_mount_available=mcp_fs_runtime is not None and mcp_fs_runtime.skills_root is not None,
                        tool_selector_model=self.tool_selector_model,
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
                        run_id=self.run_context.run_id,
                        run_dir=run_dir,
                        patch_repair_attempts=self.patch_repair_attempts,
                        tool_backend=self.tool_backend,
                        max_task_steps=self.max_task_steps,
                        max_plan_steps=self.max_plan_steps,
                        checkpointer=self.checkpointer,
                        run_control=self.run_control,
                        skills_runtime=self.skills_runtime,
                        skills_mount_available=mcp_fs_runtime is not None and mcp_fs_runtime.skills_root is not None,
                        tool_selector_model=self.tool_selector_model,
                    )

                historical_runs_context_text = ""
                historical_runs_citations: list[dict[str, Any]] = []
                if self.history_reader is not None:
                    try:
                        history_pack = await self.history_reader.aload_context(
                            query=user_request,
                            project_id=self.run_context.project_id,
                            lane=lane,
                        )
                        historical_runs_context_text = str(history_pack.context_text or "").strip()
                        raw_citations = history_pack.citations if isinstance(history_pack.citations, list) else []
                        historical_runs_citations = [c for c in raw_citations if isinstance(c, dict)]
                    except Exception as exc:
                        logger.warning("historical runs prefetch failed: %s", exc)

                initial_state: CatMasterState = {
                    "user_request": user_request,
                    "lane": lane,
                    "historical_runs_context_text": historical_runs_context_text,
                    "historical_runs_citations": historical_runs_citations,
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
            try:
                write_usage_summary(self.run_context.run_dir)
            except Exception as exc:
                logger.debug("usage summary write failed: %s", exc)


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

            if status in ("done", "failure"):
                try:
                    report_paths = self._publish_report(user_request, summary)
                    export_paths = self._publish_run_export(
                        state=result,
                        user_request=user_request,
                        final_answer=summary,
                        lane=lane,
                        status=status,
                        report_paths=report_paths,
                    )
                    await self._upsert_run_ledger(
                        lane=lane,
                        status=status,
                        user_request=user_request,
                        final_answer=summary,
                        report_paths=report_paths,
                        export_paths=export_paths,
                    )
                except Exception as exc:
                    logger.warning("finalization side-effects failed: %s", exc)

            self._emit("RUN_END", payload={"status": status, "summary_snippet": _snippet(summary, 320)})

            return {
                "tasks": result.get("tasks", []),
                "observations": result.get("observations", []),
                "summary": summary,
                "final_answer": summary,
                "status": status,
            }

        if workspace is not None:
            with workspace_scope(workspace):
                return await _do_invoke()
        return await _do_invoke()


__all__ = [
    "CatMasterState",
    "build_standard_graph",
    "build_fast_graph",
    "GraphRunner",
]
