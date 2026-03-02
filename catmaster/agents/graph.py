"""
LangGraph-based orchestration graph for CatMaster.

Replaces the monolithic Orchestrator class with a composable StateGraph
that routes between proposal, director, task runner, memory patcher,
and summarizer nodes. Proposal / director / task runner are built with
LangChain v1 ``create_agent`` and produce typed structured responses.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from catmaster.agents.nodes import (
    run_proposal,
    run_director,
    run_task,
    memory_patch_node,
    finalize_memory_patch_node,
    plan_commit_node,
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
    CatMasterToolExecutionError,
    tool_error_to_message,
)
from catmaster.tools.registry import ToolRegistry, get_tool_registry
from catmaster.runtime.mcp_filesystem import MCPFilesystemRuntime
from catmaster.runtime.tool_surface import RuntimeToolSurface, build_runtime_tool_surface
from catmaster.runtime.usage_stats import write_usage_summary
from catmaster.tools.base import workspace_scope
from catmaster.ui.reporters import Reporter, NullReporter
from catmaster.ui import make_event
from catmaster.agents.response_schemas import ProposalOutput, DirectorOutput, TaskOutput
from catmaster.llm.config import MCPConfig
from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    DIRECTOR_SYSTEM_PROMPT,
    TASK_RUNNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_TASK_RUNNER_TOOL_DENYLIST = {"memory_apply_aider_edits"}


class ToolCallBudgetExceededError(RuntimeError):
    """Raised when an agent exceeds its per-invocation tool-call budget."""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class CatMasterState(TypedDict, total=False):
    user_request: str
    lane: str

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
                    result = asyncio.run(result)
            except (CatMasterToolExecutionError, ValidationError, ValueError, KeyError) as exc:
                counter["used"] = used + 1
                return self._error_message(
                    exc=exc,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                counter["used"] = used + 1
                raise
            except Exception as exc:
                counter["used"] = used + 1
                return self._error_message(
                    exc=exc,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            counter["used"] = used + 1
            return result

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
            except (CatMasterToolExecutionError, ValidationError, ValueError, KeyError) as exc:
                counter["used"] = used + 1
                return self._error_message(
                    exc=exc,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                counter["used"] = used + 1
                raise
            except Exception as exc:
                counter["used"] = used + 1
                return self._error_message(
                    exc=exc,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            counter["used"] = used + 1
            return result

    return [_ResetToolCallCounterMiddleware(), _ToolCallBudgetMiddleware()]


def _build_proposal_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 60,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_proposal_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=PROPOSAL_SYSTEM_PROMPT,
        response_format=ToolStrategy(ProposalOutput, handle_errors=False),
        middleware=_make_tool_call_budget_middleware(role="proposal", max_tool_calls=max_steps),
    )


def _build_director_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 60,
) -> Any:
    role_tools = list(tools)
    logger.info(
        "[build_director_agent] response_format=%s, tools=%s",
        True,
        [t.name for t in role_tools],
    )
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        response_format=ToolStrategy(DirectorOutput, handle_errors=False),
        middleware=_make_tool_call_budget_middleware(role="director", max_tool_calls=max_steps),
    )


def _build_task_runner_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    memory_store: MemoryStore,
    *,
    max_steps: int = 60,
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
    middleware = _make_tool_call_budget_middleware(
        role="task_runner",
        max_tool_calls=max_steps,
    )
    return create_agent(
        model=model,
        tools=role_tools,
        system_prompt=TASK_RUNNER_SYSTEM_PROMPT,
        response_format=ToolStrategy(TaskOutput, handle_errors=False),
        middleware=middleware,
    )


# ---------------------------------------------------------------------------
# Outer graph node wrappers (delegate to nodes.py functions)
# ---------------------------------------------------------------------------

async def _run_proposal_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    tools_description: str,
    run_dir: Path,
    max_steps: int,
) -> Command:
    return await run_proposal(
        state,
        agent=agent,
        memory_store=memory_store,
        tools_description=tools_description,
        run_dir=run_dir,
        max_steps=max_steps,
    )


async def _run_director_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    tools_description: str,
    max_steps: int,
) -> Command:
    return await run_director(
        state,
        agent=agent,
        memory_store=memory_store,
        tools_description=tools_description,
        max_steps=max_steps,
    )


async def _run_task_wrapper(
    state: CatMasterState,
    *,
    agent: Any,
    memory_store: MemoryStore,
    max_steps: int,
) -> Command:
    return await run_task(
        state,
        agent=agent,
        memory_store=memory_store,
        max_steps=max_steps,
    )


def _memory_patch_node_wrapper(
    state: CatMasterState,
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str,
    patch_repair_attempts: int,
    tool_backend: Optional[ToolBackend],
    run_dir: Path,
) -> Command:
    result = memory_patch_node(
        state,
        model=model,
        memory_store=memory_store,
        run_id=run_id,
        patch_repair_attempts=patch_repair_attempts,
        tool_backend=tool_backend,
    )
    task_result = result.get("task_result") or {}
    _write_observation_file(
        run_dir=run_dir,
        task_id=state.get("current_task_id", ""),
        outcome=task_result.get("task_outcome", ""),
        summary=task_result.get("task_summary", ""),
        key_artifacts=task_result.get("key_artifacts", []),
    )
    lane = str(state.get("lane") or "").strip().lower()
    if lane == "fast":
        return Command(goto="finalize_memory_patch", update=result)
    if str(task_result.get("task_outcome") or "") == "needs_intervention":
        return Command(goto="needs_intervention", update=result)
    return Command(goto="run_director", update=result)


def _finalize_memory_patch_node_wrapper(
    state: CatMasterState,
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str,
    patch_repair_attempts: int,
    tool_backend: Optional[ToolBackend],
) -> Command:
    result = finalize_memory_patch_node(
        state,
        model=model,
        memory_store=memory_store,
        run_id=run_id,
        patch_repair_attempts=patch_repair_attempts,
        tool_backend=tool_backend,
    )
    return Command(goto="summarize", update=result)


def _summarize_node_wrapper(
    state: CatMasterState,
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
) -> Dict[str, Any]:
    return summarize_node(state, model=model, memory_store=memory_store)


def _plan_commit_node_wrapper(
    state: CatMasterState,
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str,
    tool_backend: Optional[ToolBackend],
) -> Command:
    result = plan_commit_node(
        state,
        model=model,
        memory_store=memory_store,
        run_id=run_id,
        tool_backend=tool_backend,
    )
    return Command(goto="run_director", update=result)


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
        return Command(goto="plan_commit", update={"proposal_approved": True, "proposal_feedback": ""})

    if state.get("proposal_approved"):
        return Command(goto="plan_commit", update={})

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
        return Command(goto="plan_commit", update={"proposal_approved": True, "proposal_feedback": ""})

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
    summary_model: BaseChatModel,
    memory_store: MemoryStore,
    proposal_tools: Sequence[BaseTool],
    director_tools: Sequence[BaseTool],
    task_tools: Sequence[BaseTool],
    tools_description: str,
    director_tools_description: Optional[str] = None,
    run_id: str = "",
    run_dir: Optional[Path] = None,
    patch_repair_attempts: int = 1,
    tool_backend: Optional[ToolBackend] = None,
    max_task_steps: int = 60,
    max_plan_steps: int = 60,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_control: Optional[RunControl] = None,
) -> Any:
    """Build and compile the standard-lane LangGraph."""
    effective_run_dir = run_dir or Path(".")
    effective_director_tools_description = director_tools_description or tools_description

    proposal_agent = _build_proposal_agent(
        proposal_model,
        list(proposal_tools),
        max_steps=max_plan_steps,
    )
    director_agent = _build_director_agent(
        director_model,
        list(director_tools),
        max_steps=max_plan_steps,
    )
    task_agent = _build_task_runner_agent(
        task_runner_model,
        list(task_tools),
        memory_store,
        max_steps=max_task_steps,
    )

    graph = StateGraph(CatMasterState)

    graph.add_node("run_proposal", partial(
        _run_proposal_wrapper,
        agent=proposal_agent,
        memory_store=memory_store,
        tools_description=tools_description,
        run_dir=effective_run_dir,
        max_steps=max_plan_steps,
    ))

    graph.add_node("proposal_review", _proposal_review_node)

    graph.add_node("plan_commit", partial(
        _plan_commit_node_wrapper,
        model=memory_patch_model,
        memory_store=memory_store,
        run_id=run_id,
        tool_backend=tool_backend,
    ))

    graph.add_node("run_director", partial(
        _run_director_wrapper,
        agent=director_agent,
        memory_store=memory_store,
        tools_description=effective_director_tools_description,
        max_steps=max_plan_steps,
    ))

    graph.add_node("run_task", partial(
        _run_task_wrapper,
        agent=task_agent,
        memory_store=memory_store,
        max_steps=max_task_steps,
    ))

    graph.add_node("memory_patch", partial(
        _memory_patch_node_wrapper,
        model=memory_patch_model,
        memory_store=memory_store,
        run_id=run_id,
        patch_repair_attempts=patch_repair_attempts,
        tool_backend=tool_backend,
        run_dir=effective_run_dir,
    ))

    graph.add_node("finalize_memory_patch", partial(
        _finalize_memory_patch_node_wrapper,
        model=memory_patch_model,
        memory_store=memory_store,
        run_id=run_id,
        patch_repair_attempts=patch_repair_attempts,
        tool_backend=tool_backend,
    ))

    graph.add_node("summarize", partial(
        _summarize_node_wrapper,
        model=summary_model,
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
    memory_patch_model: BaseChatModel,
    summary_model: BaseChatModel,
    memory_store: MemoryStore,
    task_tools: Sequence[BaseTool],
    run_id: str = "",
    run_dir: Optional[Path] = None,
    patch_repair_attempts: int = 1,
    tool_backend: Optional[ToolBackend] = None,
    max_task_steps: int = 60,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    run_control: Optional[RunControl] = None,
) -> Any:
    """Build and compile the fast-lane LangGraph (single task, no director)."""
    effective_run_dir = run_dir or Path(".")
    task_agent = _build_task_runner_agent(
        task_runner_model,
        list(task_tools),
        memory_store,
        max_steps=max_task_steps,
    )

    graph = StateGraph(CatMasterState)

    graph.add_node("run_task", partial(
        _run_task_wrapper,
        agent=task_agent,
        memory_store=memory_store,
        max_steps=max_task_steps,
    ))

    graph.add_node("memory_patch", partial(
        _memory_patch_node_wrapper,
        model=memory_patch_model,
        memory_store=memory_store,
        run_id=run_id,
        patch_repair_attempts=patch_repair_attempts,
        tool_backend=tool_backend,
        run_dir=effective_run_dir,
    ))

    graph.add_node("finalize_memory_patch", partial(
        _finalize_memory_patch_node_wrapper,
        model=memory_patch_model,
        memory_store=memory_store,
        run_id=run_id,
        patch_repair_attempts=patch_repair_attempts,
        tool_backend=tool_backend,
    ))

    graph.add_node("summarize", partial(
        _summarize_node_wrapper,
        model=summary_model,
        memory_store=memory_store,
    ))

    graph.set_entry_point("run_task")
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
        summary_model: BaseChatModel,
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
    ) -> None:
        self.task_runner_model = task_runner_model
        self.proposal_model = proposal_model or task_runner_model
        self.director_model = director_model or task_runner_model
        self.memory_patch_model = memory_patch_model
        self.summary_model = summary_model
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
        contract_violation = state.get("contract_violation")
        if isinstance(contract_violation, dict) and contract_violation:
            body["contract_violation"] = contract_violation
        if lane == "standard":
            body["proposal"] = {
                "proposal_path": "proposal.md",
                "work_packages": state.get("work_packages", []),
            }
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
                surface: RuntimeToolSurface = await build_runtime_tool_surface(
                    registry=self.registry,
                    run_context=self.run_context,
                    run_dir=run_dir,
                    mcp_fs_runtime=mcp_fs_runtime,
                    task_runner_denylist=_TASK_RUNNER_TOOL_DENYLIST,
                )

                if lane == "fast":
                    compiled = build_fast_graph(
                        task_runner_model=self.task_runner_model,
                        memory_patch_model=self.memory_patch_model,
                        summary_model=self.summary_model,
                        memory_store=self.memory_store,
                        task_tools=surface.task_tools,
                        run_id=self.run_context.run_id,
                        run_dir=run_dir,
                        patch_repair_attempts=self.patch_repair_attempts,
                        tool_backend=self.tool_backend,
                        max_task_steps=self.max_task_steps,
                        checkpointer=self.checkpointer,
                        run_control=self.run_control,
                    )
                else:
                    compiled = build_standard_graph(
                        task_runner_model=self.task_runner_model,
                        proposal_model=self.proposal_model,
                        director_model=self.director_model,
                        memory_patch_model=self.memory_patch_model,
                        summary_model=self.summary_model,
                        memory_store=self.memory_store,
                        proposal_tools=surface.proposal_tools,
                        director_tools=surface.director_tools,
                        task_tools=surface.task_tools,
                        tools_description=surface.task_runner_capability_guide_full,
                        director_tools_description=surface.task_runner_capability_guide_short,
                        run_id=self.run_context.run_id,
                        run_dir=run_dir,
                        patch_repair_attempts=self.patch_repair_attempts,
                        tool_backend=self.tool_backend,
                        max_task_steps=self.max_task_steps,
                        max_plan_steps=self.max_plan_steps,
                        checkpointer=self.checkpointer,
                        run_control=self.run_control,
                    )

                initial_state: CatMasterState = {
                    "user_request": user_request,
                    "lane": lane,
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
                self._publish_report(user_request, summary)

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
