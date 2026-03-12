"""
LangGraph node functions for CatMaster multi-agent pipeline.

Contains:
  - Context builders (_build_*_context) for each agent
  - Wrapper nodes (run_proposal, run_director, run_task) that invoke
    ``create_agent`` subgraphs and map results back to parent state
  - Downstream summary node + legacy memory patch helpers
  - Helper utilities for task result normalization
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_CONTEXT_TEMPLATE,
    PROPOSAL_REVISION_CONTEXT_TEMPLATE,
    PROPOSAL_NO_REVIEW_CONTEXT_APPENDIX,
    DIRECTOR_CONTEXT_TEMPLATE,
    FAST_DIRECTOR_CONTEXT_TEMPLATE,
    TASK_CONTEXT_TEMPLATE,
    MEMORY_PATCH_CONTEXT_TEMPLATE,
)
from catmaster.agents.response_schemas import (
    ProposalOutput,
    DirectorOutput,
    FastDirectorOutput,
    MemoryPatchOutput,
    TaskOutput,
)
from catmaster.agents.llm_utils import llm_text
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.context_pack import ContextPackBuilder, ContextPackPolicy
logger = logging.getLogger(__name__)


_TOOL_NAME_ALIASES = {
    "bash_exec": "bash",
}


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _session_context_section(state: Dict[str, Any]) -> str:
    text = str(state.get("session_context_text") or "").strip()
    if not text:
        text = "(none)"
    return (
        "=== Relevant chat session context ===\n"
        f"{text}"
    )


def _build_proposal_context(
    state: Dict[str, Any],
    memory_store: MemoryStore,
    execution_context_guide: str,
) -> str:
    """Build the HumanMessage text for the proposal agent."""
    user_request = state["user_request"]
    memory_index = memory_store.read_index()
    artifacts_index = json.dumps(
        memory_store.artifact_index(), ensure_ascii=False,
    )
    logger.info(
        "[_build_proposal_context] user_request_len=%d, memory_index_len=%d, "
        "artifacts_index_len=%d, execution_context_guide_len=%d",
        len(user_request), len(memory_index), len(artifacts_index), len(execution_context_guide),
    )
    if not execution_context_guide:
        logger.warning("[_build_proposal_context] execution_context_guide is EMPTY")

    feedback = state.get("proposal_feedback", "")
    review_enabled = bool(state.get("proposal_review_enabled", True))

    if feedback and state.get("proposal_md"):
        ctx = PROPOSAL_REVISION_CONTEXT_TEMPLATE.format(
            user_request=user_request,
            proposal_md=state["proposal_md"],
            work_packages_json=json.dumps(
                state.get("work_packages", []), ensure_ascii=False,
            ),
            memory_index_excerpt=memory_index,
            artifacts_index=artifacts_index,
            execution_context_guide=execution_context_guide,
            feedback=feedback,
        )
        ctx = f"{ctx}\n\n{_session_context_section(state)}"
        if not review_enabled:
            ctx = f"{ctx}\n\n{PROPOSAL_NO_REVIEW_CONTEXT_APPENDIX}"
        logger.info("[_build_proposal_context] revision context total_len=%d", len(ctx))
        return ctx

    ctx = PROPOSAL_CONTEXT_TEMPLATE.format(
        user_request=user_request,
        memory_index_excerpt=memory_index,
        artifacts_index=artifacts_index,
        execution_context_guide=execution_context_guide,
    )
    ctx = f"{ctx}\n\n{_session_context_section(state)}"
    if not review_enabled:
        ctx = f"{ctx}\n\n{PROPOSAL_NO_REVIEW_CONTEXT_APPENDIX}"
    logger.info("[_build_proposal_context] fresh context total_len=%d", len(ctx))
    return ctx


def _build_director_context(
    state: Dict[str, Any],
    memory_store: MemoryStore,
    execution_context_guide: str,
) -> str:
    """Build the HumanMessage text for the director agent."""
    observations = state.get("observations", [])
    director_observations = _director_observations_view(observations)
    task_outcomes_history = _director_task_outcomes_history(
        state.get("tasks", []),
        director_observations,
    )

    ctx = DIRECTOR_CONTEXT_TEMPLATE.format(
        user_request=state["user_request"],
        proposal_md=state.get("proposal_md", ""),
        work_packages_json=json.dumps(
            state.get("work_packages", []), ensure_ascii=False,
        ),
        memory_index_excerpt=memory_store.read_index(),
        task_outcomes_history_text=_render_task_outcomes_history_lines(task_outcomes_history),
        already_done_json=json.dumps(director_observations, ensure_ascii=False),
        execution_context_guide=execution_context_guide,
    )
    return f"{ctx}\n\n{_session_context_section(state)}"


def _build_fast_director_context(
    state: Dict[str, Any],
    memory_store: MemoryStore,
    execution_context_guide: str,
) -> str:
    """Build the HumanMessage text for the fast-lane director agent."""
    observations = state.get("observations", [])
    task_outcomes_history = _director_task_outcomes_history(
        state.get("tasks", []),
        observations,
    )

    ctx = FAST_DIRECTOR_CONTEXT_TEMPLATE.format(
        user_request=state["user_request"],
        memory_index_excerpt=memory_store.read_index(),
        task_outcomes_history_text=_render_task_outcomes_history_lines(task_outcomes_history),
        execution_context_guide=execution_context_guide,
    )
    return f"{ctx}\n\n{_session_context_section(state)}"


def _build_task_context(
    state: Dict[str, Any],
    memory_store: MemoryStore,
) -> str:
    """Build the HumanMessage text for the task runner agent."""
    task_packet = state.get("current_task_packet") or {}
    task_goal = str(
        task_packet.get("goal") or state.get("user_request", ""),
    ).strip()

    context_builder = ContextPackBuilder(memory_store)
    context_pack = context_builder.build(
        task_goal,
        role="task_runner",
        policy=ContextPackPolicy(
            memory_head_lines=None,
            max_memory_chars=None,
        ),
    )

    task_detail = str(task_packet.get("task_detail") or "(none)").strip()
    expected_outputs = task_packet.get("expected_outputs")
    suggested_tools = task_packet.get("suggested_tools")
    reference_hint = task_packet.get("reference_hint")

    def _normalize_tool_hint(value: Any) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        return _TOOL_NAME_ALIASES.get(token, token)

    def _bullet_lines(value: Any) -> str:
        if isinstance(value, str):
            item = value.strip()
            return f"- {item}" if item else "(none)"
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            return "\n".join(f"- {item}" for item in items) if items else "(none)"
        if value is None:
            return "(none)"
        item = str(value).strip()
        return f"- {item}" if item else "(none)"

    def _csv_items(value: Any) -> str:
        if isinstance(value, str):
            item = _normalize_tool_hint(value)
            return item if item else "(none)"
        if isinstance(value, list):
            items = [_normalize_tool_hint(v) for v in value]
            items = [item for item in items if item]
            return ", ".join(items) if items else "(none)"
        if value is None:
            return "(none)"
        item = _normalize_tool_hint(value)
        return item if item else "(none)"

    return TASK_CONTEXT_TEMPLATE.format(
        goal=task_goal,
        task_detail=task_detail,
        expected_outputs=_bullet_lines(expected_outputs),
        suggested_tools=_csv_items(suggested_tools),
        reference_hint=_bullet_lines(reference_hint),
        workspace_policy=context_pack.get("workspace_policy", ""),
        workspace_root_abs_ref=context_pack.get("workspace_root_abs_ref", ""),
        memory_index_excerpt=context_pack.get("memory_index_excerpt", ""),
    )


# ---------------------------------------------------------------------------
# Wrapper nodes for ReAct agents
# ---------------------------------------------------------------------------

def _last_message_snippet(messages: list[AnyMessage], limit: int = 280) -> str:
    if not messages:
        return ""
    text = str(getattr(messages[-1], "content", "") or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _is_need_more_steps(messages: list[AnyMessage]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return False
    text = str(last.content or "").strip().lower()
    return "need more steps" in text


def _tool_messages_seen(messages: list[AnyMessage]) -> list[str]:
    names: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "").strip()
        if name:
            names.append(name)
    return names


def _structured_contract_violation(
    *,
    role: str,
    reason: str,
    messages: list[AnyMessage],
    max_steps: int,
    error: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": role,
        "reason": reason,
        "observed_tool_messages": _tool_messages_seen(messages),
        "remaining_steps_exhausted": _is_need_more_steps(messages),
        "message_count": len(messages or []),
        "last_message": _last_message_snippet(messages),
        "max_steps": int(max_steps),
    }
    if error:
        payload["error"] = error
    if schema_name:
        payload["expected_schema"] = schema_name
    return payload


def _structured_contract_violation_summary(role: str, violation: dict[str, Any]) -> str:
    observed = ", ".join(violation.get("observed_tool_messages") or []) or "(none)"
    exhausted = "yes" if bool(violation.get("remaining_steps_exhausted")) else "no"
    reason = str(violation.get("reason") or "unknown")
    return (
        f"{role} agent structured contract violation: {reason} "
        f"(observed tool messages: {observed}; remaining_steps exhausted: {exhausted})."
    )


def _require_structured_response(
    result: dict[str, Any],
    *,
    schema_cls: type[ProposalOutput] | type[DirectorOutput] | type[FastDirectorOutput] | type[TaskOutput] | type[MemoryPatchOutput],
    role: str,
    messages: list[AnyMessage],
    max_steps: int,
) -> tuple[ProposalOutput | DirectorOutput | FastDirectorOutput | TaskOutput | MemoryPatchOutput | None, dict[str, Any] | None]:
    raw = result.get("structured_response")
    if raw is None:
        violation = _structured_contract_violation(
            role=role,
            reason="missing_structured_response",
            messages=messages,
            max_steps=max_steps,
            schema_name=schema_cls.__name__,
        )
        return None, violation
    if isinstance(raw, schema_cls):
        return raw, None
    try:
        return schema_cls.model_validate(raw), None
    except Exception as exc:
        violation = _structured_contract_violation(
            role=role,
            reason="invalid_structured_response",
            messages=messages,
            max_steps=max_steps,
            error=str(exc),
            schema_name=schema_cls.__name__,
        )
        return None, violation


def _supports_remaining_steps_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "remaining_steps" not in text:
        return False
    markers = (
        "extra_forbidden",
        "extra inputs are not permitted",
        "unexpected keyword",
        "validationerror",
        "unknown field",
    )
    return any(marker in text for marker in markers)


async def _invoke_agent_with_step_budget(
    *,
    agent: Any,
    messages: list[AnyMessage],
    max_steps: int,
    role: str,
) -> dict[str, Any]:
    async def _invoke(payload: dict[str, Any]) -> dict[str, Any]:
        ainvoke = getattr(agent, "ainvoke", None)
        if callable(ainvoke):
            return await ainvoke(payload)
        invoke = getattr(agent, "invoke", None)
        if callable(invoke):
            return invoke(payload)
        raise TypeError(f"{type(agent).__name__} exposes neither invoke nor ainvoke")

    payload = {
        "messages": list(messages or []),
        "remaining_steps": max(1, int(max_steps)),
    }
    try:
        return await _invoke(payload)
    except Exception as exc:
        if not _supports_remaining_steps_error(exc):
            raise
        logger.info(
            "[%s] agent invoke rejected remaining_steps; retrying with messages-only payload",
            role,
        )
        return await _invoke({"messages": list(messages or [])})


async def run_proposal(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    run_dir: Path,
    max_steps: int = 30,
) -> Command:
    """Invoke the proposal ReAct agent and map results to parent state."""
    ctx_text = _build_proposal_context(state, memory_store, execution_context_guide)
    ctx_msg = HumanMessage(content=ctx_text)
    input_messages: list[AnyMessage] = [ctx_msg]
    try:
        result = await _invoke_agent_with_step_budget(
            agent=agent,
            messages=input_messages,
            max_steps=max_steps,
            role="proposal",
        )
    except Exception as exc:
        logger.exception("proposal agent invoke failed: %s", exc)
        violation = _structured_contract_violation(
            role="proposal",
            reason="invoke_exception",
            messages=input_messages,
            max_steps=max_steps,
            error=str(exc),
            schema_name=ProposalOutput.__name__,
        )
        summary = f"proposal agent invoke failed before structured response: {exc}"
        return Command(
            goto="summarize",
            update={
                "proposal_messages": input_messages,
                "proposal_md": "",
                "work_packages": [],
                "proposal_approved": False,
                "proposal_feedback": "",
                "status": "failure",
                "summary": summary,
                "contract_violation": violation,
            },
        )

    msgs = list(result.get("messages", []) or [])
    parsed, violation = _require_structured_response(
        result,
        schema_cls=ProposalOutput,
        role="proposal",
        messages=msgs,
        max_steps=max_steps,
    )
    if not isinstance(parsed, ProposalOutput):
        violation_payload = violation or _structured_contract_violation(
            role="proposal",
            reason="missing_structured_response",
            messages=msgs,
            max_steps=max_steps,
            schema_name=ProposalOutput.__name__,
        )
        if _is_need_more_steps(msgs):
            logger.warning("[run_proposal] hit remaining_steps limit (max_steps=%d)", max_steps)
        logger.warning(
            "[run_proposal] missing/invalid structured response. message_count=%d, last=%s",
            len(msgs), _last_message_snippet(msgs),
        )
        return Command(
            goto="summarize",
            update={
                "proposal_messages": _cap_messages(msgs),
                "proposal_md": "",
                "work_packages": [],
                "proposal_approved": False,
                "proposal_feedback": "",
                "status": "failure",
                "summary": _structured_contract_violation_summary("proposal", violation_payload),
                "contract_violation": violation_payload,
            },
        )

    resp = parsed
    if resp.status == "fail":
        logger.warning(
            "[run_proposal] proposal agent reported FAIL: error=%r, needs_human=%s",
            resp.error, resp.needs_human,
        )
        summary = str(resp.error or "Proposal agent reported failure.").strip()
        if not summary:
            summary = "Proposal agent reported failure."
        return Command(
            goto="summarize",
            update={
                "proposal_messages": _cap_messages(msgs),
                "proposal_md": "",
                "work_packages": [],
                "proposal_approved": False,
                "proposal_feedback": "",
                "status": "failure",
                "summary": summary,
                "contract_violation": {},
            },
        )

    proposal_md = resp.proposal_md
    work_packages = resp.work_packages

    if proposal_md:
        try:
            (run_dir / "proposal.md").write_text(proposal_md, encoding="utf-8")
        except Exception:
            pass

    return Command(
        goto="proposal_review",
        update={
            "proposal_messages": _cap_messages(msgs),
            "proposal_md": proposal_md,
            "work_packages": work_packages,
            "proposal_approved": False,
            "proposal_feedback": "",
            "contract_violation": {},
        },
    )


async def run_director(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    max_steps: int = 30,
    allow_memory_patch: bool = True,
) -> Command:
    """Invoke the director ReAct agent and route via Command."""
    ctx_text = _build_director_context(state, memory_store, execution_context_guide)
    ctx_msg = HumanMessage(content=ctx_text)
    input_messages: list[AnyMessage] = [ctx_msg]
    try:
        result = await _invoke_agent_with_step_budget(
            agent=agent,
            messages=input_messages,
            max_steps=max_steps,
            role="director",
        )
    except Exception as exc:
        logger.exception("director agent invoke failed: %s", exc)
        violation = _structured_contract_violation(
            role="director",
            reason="invoke_exception",
            messages=input_messages,
            max_steps=max_steps,
            error=str(exc),
            schema_name=DirectorOutput.__name__,
        )
        summary = f"director agent invoke failed before structured response: {exc}"
        return Command(
            goto="summarize",
            update={
                "director_messages": input_messages,
                "director_decision": {},
                "next_action": "ContractViolation",
                "status": "failure",
                "summary": summary,
                "contract_violation": violation,
            },
        )

    msgs = list(result.get("messages", []) or [])
    parsed, violation = _require_structured_response(
        result,
        schema_cls=DirectorOutput,
        role="director",
        messages=msgs,
        max_steps=max_steps,
    )
    if not isinstance(parsed, DirectorOutput):
        if _is_need_more_steps(msgs):
            logger.warning("[run_director] hit remaining_steps limit (max_steps=%d)", max_steps)
        logger.warning("director agent returned missing/invalid structured response")
        violation_payload = violation or _structured_contract_violation(
            role="director",
            reason="missing_structured_response",
            messages=msgs,
            max_steps=max_steps,
            schema_name=DirectorOutput.__name__,
        )
        return Command(
            goto="summarize",
            update={
                "director_messages": _cap_messages(msgs),
                "director_decision": {},
                "next_action": "ContractViolation",
                "status": "failure",
                "summary": _structured_contract_violation_summary("director", violation_payload),
                "contract_violation": violation_payload,
            },
        )

    resp = parsed
    update: Dict[str, Any] = {
        "director_messages": _cap_messages(msgs),
        "director_decision": resp.model_dump(),
        "next_action": resp.state,
        "pending_memory_updates": [],
        "contract_violation": {},
    }

    if resp.state == "PerformNextTask":
        branch = resp.perform_next_task
        task_packet = branch.task_packet if branch is not None else None
        task_id = f"task_{_next_task_index(state.get('tasks', [])):02d}"
        update["current_task_id"] = task_id
        update["current_task_packet"] = task_packet.model_dump() if task_packet else {}
        new_task = {
            "task_id": task_id,
            "goal": task_packet.goal if task_packet else "",
            "task_packet": task_packet.model_dump() if task_packet else {},
            "status": "pending",
        }
        update["tasks"] = list(state.get("tasks") or []) + [new_task]
        update["runner_messages"] = []
        return Command(goto="run_task", update=update)

    if resp.state == "MajorReviseProposal":
        branch = resp.major_revise_proposal
        if branch is not None:
            if isinstance(branch.updated_proposal_md, str):
                update["proposal_md"] = branch.updated_proposal_md
            if isinstance(branch.updated_work_packages, list):
                update["work_packages"] = branch.updated_work_packages
        update["proposal_approved"] = False
        update["proposal_feedback"] = resp.rationale
        return Command(goto="proposal_review", update=update)

    if resp.state == "MinorReviseProposal":
        branch = resp.minor_revise_proposal
        if branch is not None:
            if isinstance(branch.updated_proposal_md, str):
                update["proposal_md"] = branch.updated_proposal_md
            if isinstance(branch.updated_work_packages, list):
                update["work_packages"] = branch.updated_work_packages
        return Command(goto="run_director", update=update)

    # StopAndSynthesize
    stop_payload = resp.stop_and_synthesize
    final_answer_md = ""
    if stop_payload is not None:
        final_answer_md = str(stop_payload.final_answer_md or "").strip()
    updates = [item.model_dump() for item in list(resp.update_memory or [])]
    update["pending_memory_updates"] = updates
    if final_answer_md:
        update["summary"] = final_answer_md
    if updates and allow_memory_patch:
        return Command(goto="run_memory_patch", update=update)
    return Command(goto="summarize", update=update)


async def run_fast_director(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    execution_context_guide: str,
    max_steps: int = 30,
    allow_memory_patch: bool = True,
) -> Command:
    """Invoke the fast-lane director agent and route via Command."""
    ctx_text = _build_fast_director_context(state, memory_store, execution_context_guide)
    ctx_msg = HumanMessage(content=ctx_text)
    input_messages: list[AnyMessage] = [ctx_msg]
    try:
        result = await _invoke_agent_with_step_budget(
            agent=agent,
            messages=input_messages,
            max_steps=max_steps,
            role="fast_director",
        )
    except Exception as exc:
        logger.exception("fast director agent invoke failed: %s", exc)
        violation = _structured_contract_violation(
            role="fast_director",
            reason="invoke_exception",
            messages=input_messages,
            max_steps=max_steps,
            error=str(exc),
            schema_name=FastDirectorOutput.__name__,
        )
        summary = f"fast director agent invoke failed before structured response: {exc}"
        return Command(
            goto="summarize",
            update={
                "director_messages": input_messages,
                "director_decision": {},
                "next_action": "ContractViolation",
                "status": "failure",
                "summary": summary,
                "contract_violation": violation,
            },
        )

    msgs = list(result.get("messages", []) or [])
    parsed, violation = _require_structured_response(
        result,
        schema_cls=FastDirectorOutput,
        role="fast_director",
        messages=msgs,
        max_steps=max_steps,
    )
    if not isinstance(parsed, FastDirectorOutput):
        if _is_need_more_steps(msgs):
            logger.warning("[run_fast_director] hit remaining_steps limit (max_steps=%d)", max_steps)
        logger.warning("fast director agent returned missing/invalid structured response")
        violation_payload = violation or _structured_contract_violation(
            role="fast_director",
            reason="missing_structured_response",
            messages=msgs,
            max_steps=max_steps,
            schema_name=FastDirectorOutput.__name__,
        )
        return Command(
            goto="summarize",
            update={
                "director_messages": _cap_messages(msgs),
                "director_decision": {},
                "next_action": "ContractViolation",
                "status": "failure",
                "summary": _structured_contract_violation_summary("fast_director", violation_payload),
                "contract_violation": violation_payload,
            },
        )

    resp = parsed
    update: Dict[str, Any] = {
        "director_messages": _cap_messages(msgs),
        "director_decision": resp.model_dump(),
        "next_action": resp.state,
        "pending_memory_updates": [],
        "contract_violation": {},
    }

    if resp.state == "PerformNextTask":
        branch = resp.perform_next_task
        task_packet = branch.task_packet if branch is not None else None
        task_id = f"task_{_next_task_index(state.get('tasks', [])):02d}"
        update["current_task_id"] = task_id
        update["current_task_packet"] = task_packet.model_dump() if task_packet else {}
        new_task = {
            "task_id": task_id,
            "goal": task_packet.goal if task_packet else "",
            "task_packet": task_packet.model_dump() if task_packet else {},
            "status": "pending",
        }
        update["tasks"] = list(state.get("tasks") or []) + [new_task]
        update["runner_messages"] = []
        return Command(goto="run_task", update=update)

    stop_payload = resp.stop_and_synthesize
    final_answer_md = ""
    if stop_payload is not None:
        final_answer_md = str(stop_payload.final_answer_md or "").strip()
    updates = [item.model_dump() for item in list(resp.update_memory or [])]
    update["pending_memory_updates"] = updates
    if final_answer_md:
        update["summary"] = final_answer_md
    if updates and allow_memory_patch:
        return Command(goto="run_memory_patch", update=update)
    return Command(goto="summarize", update=update)


async def run_task(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    max_steps: int = 40,
    continue_goto: str = "run_director",
    intervention_goto: str = "needs_intervention",
) -> Command:
    """Invoke the task runner ReAct agent and map results to parent state."""
    ctx_text = _build_task_context(state, memory_store)
    ctx_msg = HumanMessage(content=ctx_text)

    try:
        result = await _invoke_agent_with_step_budget(
            agent=agent,
            messages=[ctx_msg],
            max_steps=max_steps,
            role="task_runner",
        )
    except Exception as exc:
        logger.exception("task runner invoke failed: %s", exc)
        violation = _structured_contract_violation(
            role="task_runner",
            reason="invoke_exception",
            messages=[ctx_msg],
            max_steps=max_steps,
            error=str(exc),
            schema_name=TaskOutput.__name__,
        )
        summary = f"task runner invoke failed before structured response: {exc}"
        task_result = {
            "task_outcome": "failure",
            "task_summary": summary,
            "key_artifacts": [],
            "structured_result": {
                "summary": summary,
                "facts": [],
                "files": [],
                "constraints": [],
                "open_questions": [],
                "decisions": [],
                "next_steps": [],
                "artifacts": [],
            },
        }
        task_state_update = _task_state_update_from_result(
            state=state,
            task_result=task_result,
        )
        target = continue_goto
        update_payload: Dict[str, Any] = {
            "runner_messages": [ctx_msg],
            **task_state_update,
            "summary": summary,
            "contract_violation": violation,
        }
        if target == "summarize":
            update_payload["status"] = "failure"
        return Command(
            goto=target,
            update=update_payload,
        )

    msgs = list(result.get("messages", []) or [])
    parsed, violation = _require_structured_response(
        result,
        schema_cls=TaskOutput,
        role="task_runner",
        messages=msgs,
        max_steps=max_steps,
    )
    if not isinstance(parsed, TaskOutput):
        summary = _structured_contract_violation_summary(
            "task_runner",
            violation
            or _structured_contract_violation(
                role="task_runner",
                reason="missing_structured_response",
                messages=msgs,
                max_steps=max_steps,
                schema_name=TaskOutput.__name__,
            ),
        )
        task_result = {
            "task_outcome": "failure",
            "task_summary": summary,
            "key_artifacts": [],
            "structured_result": {
                "summary": summary,
                "facts": [],
                "files": [],
                "constraints": [],
                "open_questions": [],
                "decisions": [],
                "next_steps": [],
                "artifacts": [],
            },
        }
        task_state_update = _task_state_update_from_result(
            state=state,
            task_result=task_result,
        )
        target = continue_goto
        update_payload: Dict[str, Any] = {
            "runner_messages": _cap_messages(msgs),
            **task_state_update,
            "summary": summary,
            "contract_violation": violation
            or _structured_contract_violation(
                role="task_runner",
                reason="missing_structured_response",
                messages=msgs,
                max_steps=max_steps,
                schema_name=TaskOutput.__name__,
            ),
        }
        if target == "summarize":
            update_payload["status"] = "failure"
        return Command(
            goto=target,
            update=update_payload,
        )

    task_result = _normalize_task_output(parsed)
    task_state_update = _task_state_update_from_result(
        state=state,
        task_result=task_result,
    )
    outcome = str(task_result.get("task_outcome") or "").strip().lower()
    target = intervention_goto if outcome == "needs_intervention" else continue_goto
    update_payload: Dict[str, Any] = {
        "runner_messages": _cap_messages(msgs),
        **task_state_update,
        "contract_violation": {},
    }
    if target == "summarize":
        if outcome == "success":
            update_payload["status"] = "done"
        elif outcome == "needs_intervention":
            update_payload["status"] = "needs_intervention"
        else:
            update_payload["status"] = "failure"
    return Command(
        goto=target,
        update=update_payload,
    )


# ---------------------------------------------------------------------------
# Task result normalization (from TaskOutput schema)
# ---------------------------------------------------------------------------

def _normalize_task_output(resp: TaskOutput) -> Dict[str, Any]:
    """Convert a TaskOutput structured response to the canonical task_result dict."""
    files = [
        {
            "path": f.path,
            "description": f.description or "",
            "kind": f.kind or "output",
        }
        for f in resp.files
    ]
    decisions = [
        {"decision": d.decision, "rationale": d.rationale}
        for d in resp.decisions
    ]
    structured = {
        "summary": resp.summary,
        "facts": list(resp.facts),
        "files": files,
        "constraints": list(resp.constraints),
        "open_questions": list(resp.open_questions),
        "decisions": decisions,
        "next_steps": list(resp.next_steps),
        "artifacts": list(resp.artifacts),
    }
    key_artifacts = [
        {"path": f["path"], "description": f["description"], "kind": f["kind"]}
        for f in files
    ]

    if resp.status == "done":
        return {
            "task_outcome": "success",
            "task_summary": resp.summary,
            "key_artifacts": key_artifacts,
            "structured_result": structured,
        }

    # status == "blocked"
    return {
        "task_outcome": "needs_intervention" if resp.needs_human else "failure",
        "task_summary": resp.error or resp.summary,
        "key_artifacts": key_artifacts,
        "structured_result": structured,
    }


def _task_state_update_from_result(
    *,
    state: Dict[str, Any],
    task_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Update observations/tasks directly from task result without auto memory patching."""
    task_id = str(state.get("current_task_id") or "").strip()
    structured_result = task_result.get("structured_result") or {}
    outcome = str(task_result.get("task_outcome") or "").strip() or "failure"

    observation = {
        "task_id": task_id,
        "outcome": outcome,
        "summary": task_result.get("task_summary", ""),
        "key_artifacts": task_result.get("key_artifacts", []),
    }
    failure_kind = str(task_result.get("failure_kind") or "").strip()
    if failure_kind:
        observation["failure_kind"] = failure_kind

    def _compact_text_list(items: Any, *, limit: int) -> list[str]:
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for item in items:
            text = _trim_text(" ".join(str(item).split()), limit=limit)
            if text:
                out.append(text)
        return out

    decision_items = structured_result.get("decisions")
    if isinstance(decision_items, list):
        compact_decisions: list[str] = []
        for item in decision_items:
            if not isinstance(item, dict):
                continue
            decision = _trim_text(" ".join(str(item.get("decision") or "").split()), limit=120)
            rationale = _trim_text(" ".join(str(item.get("rationale") or "").split()), limit=160)
            if decision and rationale:
                compact_decisions.append(f"{decision} ({rationale})")
            elif decision:
                compact_decisions.append(decision)
        if compact_decisions:
            observation["decisions"] = compact_decisions

    next_steps = _compact_text_list(structured_result.get("next_steps"), limit=140)
    if next_steps:
        observation["next_steps"] = next_steps

    open_questions = _compact_text_list(structured_result.get("open_questions"), limit=140)
    if open_questions:
        observation["open_questions"] = open_questions

    facts = _compact_text_list(structured_result.get("facts"), limit=180)
    if facts:
        observation["facts"] = facts

    existing_observations = list(state.get("observations") or [])
    existing_observations.append(observation)

    existing_tasks = list(state.get("tasks") or [])
    for task in existing_tasks:
        if str(task.get("task_id") or "") == task_id:
            task["status"] = outcome

    return {
        "observations": existing_observations,
        "tasks": existing_tasks,
        "task_result": task_result,
    }


# ---------------------------------------------------------------------------
# Memory patch node (hot-path, stop-stage only)
# ---------------------------------------------------------------------------

_MEMORY_TOPICS_ALLOWED: tuple[str, ...] = (
    "MEMORY/MEMORY.md",
    "MEMORY/topics/FACTS.md",
    "MEMORY/topics/FILES.md",
    "MEMORY/topics/CONSTRAINTS.md",
    "MEMORY/topics/QUESTIONS.md",
    "MEMORY/topics/RUNBOOK.md",
)


def _normalize_pending_memory_updates(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        content = str(item.get("content") or "").strip()
        if topic not in _MEMORY_TOPICS_ALLOWED:
            continue
        if not content:
            continue
        out.append({"topic": topic, "content": content})
    return out


def _memory_topic_path(memory_store: MemoryStore, topic: str) -> Path:
    rel = str(topic or "").strip()
    if rel == "MEMORY/MEMORY.md":
        return memory_store.index_path
    if rel.startswith("MEMORY/topics/"):
        return memory_store.memory_dir / rel.replace("MEMORY/", "", 1)
    raise ValueError(f"unsupported memory topic: {rel}")


def _render_editable_file_snapshots(memory_store: MemoryStore, topics: list[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        normalized = str(topic or "").strip()
        if normalized in _MEMORY_TOPICS_ALLOWED and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    blocks: list[str] = []
    for topic in ordered:
        path = _memory_topic_path(memory_store, topic)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        blocks.append(f'<editable_file path="{topic}">')
        blocks.append(text)
        blocks.append("</editable_file>")
    return "\n".join(blocks).strip() or "(none)"


def _append_memory_patch_event(
    *,
    memory_store: MemoryStore,
    run_id: str,
    patch_status: str,
    summary: str,
    error: str,
    requested_topics: list[str],
    applied_topics: list[str],
) -> None:
    memory_store.append_event({
        "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "run_id": run_id,
        "task_id": "final_memory_update",
        "outcome": patch_status,
        "summary": summary,
        "memory_patch_status": patch_status,
        "memory_patch_error": error,
        "requested_topics": requested_topics,
        "applied_topics": applied_topics,
    })


async def run_memory_patch(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    run_id: str = "",
    max_steps: int = 20,
    continue_goto: str = "summarize",
) -> Command:
    """Apply director-emitted memory updates at stop stage."""
    memory_store.ensure_exists()
    updates = _normalize_pending_memory_updates(state.get("pending_memory_updates"))
    requested_topics = [row["topic"] for row in updates]

    if not updates:
        result = {
            "status": "done",
            "summary": "No memory updates requested.",
            "applied_topics": [],
            "error": "",
            "needs_human": False,
        }
        return Command(
            goto=continue_goto,
            update={
                "pending_memory_updates": [],
                "memory_patch_result": result,
            },
        )

    ctx_text = MEMORY_PATCH_CONTEXT_TEMPLATE.format(
        pending_memory_updates_json=json.dumps(updates, ensure_ascii=False, indent=2),
        editable_file_snapshots=_render_editable_file_snapshots(memory_store, requested_topics),
    )
    ctx_msg = HumanMessage(content=ctx_text)
    input_messages: list[AnyMessage] = [ctx_msg]

    try:
        result = await _invoke_agent_with_step_budget(
            agent=agent,
            messages=input_messages,
            max_steps=max_steps,
            role="memory_patch",
        )
    except Exception as exc:
        logger.exception("memory patch agent invoke failed: %s", exc)
        summary = f"memory patch agent invoke failed before structured response: {exc}"
        violation = _structured_contract_violation(
            role="memory_patch",
            reason="invoke_exception",
            messages=input_messages,
            max_steps=max_steps,
            error=str(exc),
            schema_name=MemoryPatchOutput.__name__,
        )
        patch_result = {
            "status": "blocked",
            "summary": summary,
            "applied_topics": [],
            "error": summary,
            "needs_human": False,
        }
        _append_memory_patch_event(
            memory_store=memory_store,
            run_id=run_id,
            patch_status="blocked",
            summary=summary,
            error=summary,
            requested_topics=requested_topics,
            applied_topics=[],
        )
        return Command(
            goto=continue_goto,
            update={
                "pending_memory_updates": [],
                "memory_patch_result": patch_result,
                "contract_violation": violation,
            },
        )

    msgs = list(result.get("messages", []) or [])
    parsed, violation = _require_structured_response(
        result,
        schema_cls=MemoryPatchOutput,
        role="memory_patch",
        messages=msgs,
        max_steps=max_steps,
    )
    if not isinstance(parsed, MemoryPatchOutput):
        violation_payload = violation or _structured_contract_violation(
            role="memory_patch",
            reason="missing_structured_response",
            messages=msgs,
            max_steps=max_steps,
            schema_name=MemoryPatchOutput.__name__,
        )
        summary = _structured_contract_violation_summary("memory_patch", violation_payload)
        patch_result = {
            "status": "blocked",
            "summary": summary,
            "applied_topics": [],
            "error": summary,
            "needs_human": False,
        }
        _append_memory_patch_event(
            memory_store=memory_store,
            run_id=run_id,
            patch_status="blocked",
            summary=summary,
            error=summary,
            requested_topics=requested_topics,
            applied_topics=[],
        )
        return Command(
            goto=continue_goto,
            update={
                "pending_memory_updates": [],
                "memory_patch_result": patch_result,
                "contract_violation": violation_payload,
            },
        )

    payload = parsed.model_dump()
    applied_topics = [
        topic for topic in (parsed.applied_topics or []) if str(topic or "").strip() in _MEMORY_TOPICS_ALLOWED
    ]
    if parsed.status == "done" and not applied_topics:
        applied_topics = requested_topics
        payload["applied_topics"] = list(applied_topics)
    if parsed.status == "done" and "MEMORY/MEMORY.md" not in set(applied_topics):
        _refresh_memory_index(memory_store)

    _append_memory_patch_event(
        memory_store=memory_store,
        run_id=run_id,
        patch_status=parsed.status,
        summary=str(parsed.summary or "").strip(),
        error=str(parsed.error or "").strip(),
        requested_topics=requested_topics,
        applied_topics=applied_topics,
    )
    return Command(
        goto=continue_goto,
        update={
            "pending_memory_updates": [],
            "memory_patch_result": payload,
            "contract_violation": {},
        },
    )


# ---------------------------------------------------------------------------
# Summarize node
# ---------------------------------------------------------------------------

def summarize_node(
    state: Dict[str, Any],
    *,
    memory_store: MemoryStore,
) -> Dict[str, Any]:
    """Finalize output without an extra summarizer LLM pass."""
    _ = memory_store
    summary = str(state.get("summary") or "").strip()

    if not summary:
        director_decision = state.get("director_decision") or {}
        if isinstance(director_decision, dict):
            stop_payload = director_decision.get("stop_and_synthesize") or {}
            if isinstance(stop_payload, dict):
                summary = str(stop_payload.get("final_answer_md") or "").strip()

    if not summary:
        observations = list(state.get("observations") or [])
        if observations and isinstance(observations[-1], dict):
            summary = str(observations[-1].get("summary") or "").strip()

    if not summary:
        summary = "No final answer was produced."

    incoming_status = str(state.get("status") or "done")
    final_status = incoming_status if incoming_status in {"failure", "needs_intervention"} else "done"
    return {"summary": summary, "status": final_status}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _director_observations_view(
    observations: list[dict],
    *,
    max_items: int | None = None,
) -> list[dict]:
    sanitized: list[dict] = []
    for item in observations or []:
        if not isinstance(item, dict):
            continue
        row: dict = {}
        for key in ("task_id", "outcome", "summary", "failure_kind"):
            value = item.get(key)
            if value is None:
                continue
            limit = 320 if key == "summary" else 140
            text = _trim_text(" ".join(str(value).split()), limit=limit)
            if text:
                row[key] = text
        if bool(item.get("auto_replan", False)):
            row["auto_replan"] = True

        interrupted = item.get("interrupted_toolcall")
        if isinstance(interrupted, dict):
            safe_interrupt: dict[str, Any] = {}
            for key in ("tool", "status", "highlights", "cancel_accepted"):
                if key in interrupted:
                    value = interrupted.get(key)
                    if isinstance(value, str):
                        safe_interrupt[key] = _trim_text(" ".join(value.split()), limit=180)
                    else:
                        safe_interrupt[key] = value
            if safe_interrupt:
                row["interrupted_toolcall"] = safe_interrupt

        decisions = item.get("decisions")
        if isinstance(decisions, list):
            compact_decisions: list[str] = []
            for entry in decisions:
                text = _trim_text(" ".join(str(entry).split()), limit=220)
                if text:
                    compact_decisions.append(text)
            if compact_decisions:
                row["decisions"] = compact_decisions

        next_steps = item.get("next_steps")
        if isinstance(next_steps, list):
            compact_next: list[str] = []
            for entry in next_steps:
                text = _trim_text(" ".join(str(entry).split()), limit=180)
                if text:
                    compact_next.append(text)
            if compact_next:
                row["next_steps"] = compact_next

        open_questions = item.get("open_questions")
        if isinstance(open_questions, list):
            compact_questions: list[str] = []
            for entry in open_questions:
                text = _trim_text(" ".join(str(entry).split()), limit=180)
                if text:
                    compact_questions.append(text)
            if compact_questions:
                row["open_questions"] = compact_questions

        facts = item.get("facts")
        if isinstance(facts, list):
            compact_facts: list[str] = []
            for entry in facts:
                text = _trim_text(" ".join(str(entry).split()), limit=180)
                if text:
                    compact_facts.append(text)
            if compact_facts:
                row["facts"] = compact_facts
        if row:
            sanitized.append(row)
    if max_items is not None and len(sanitized) > max_items:
        return sanitized[-max_items:]
    return sanitized


def _director_task_outcomes_history(
    tasks: list[dict],
    observations: list[dict],
) -> list[dict[str, Any]]:
    obs_map: dict[str, dict] = {}
    for row in observations or []:
        if isinstance(row, dict):
            tid = str(row.get("task_id") or "").strip()
            if tid:
                obs_map[tid] = row

    history: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()

    for item in tasks or []:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("task_id") or "").strip()
        if not tid:
            continue
        seen_task_ids.add(tid)
        row: dict[str, Any] = {
            "task_id": tid,
            "status": _trim_text(str(item.get("status") or "").strip(), limit=32),
        }
        goal = _trim_text(" ".join(str(item.get("goal") or "").split()), limit=180)
        if goal:
            row["goal"] = goal
        obs = obs_map.get(tid)
        if isinstance(obs, dict):
            outcome = _trim_text(str(obs.get("outcome") or "").strip(), limit=32)
            if outcome:
                row["outcome"] = outcome
            summary = _trim_text(" ".join(str(obs.get("summary") or "").split()), limit=220)
            if summary:
                row["summary"] = summary
            artifacts = obs.get("key_artifacts")
            if isinstance(artifacts, list):
                row["artifact_count"] = len(artifacts)
            decisions = obs.get("decisions")
            if isinstance(decisions, list):
                row["decision_count"] = len(decisions)
            next_steps = obs.get("next_steps")
            if isinstance(next_steps, list) and next_steps:
                compact_steps: list[str] = []
                for step in next_steps:
                    hint = _trim_text(" ".join(str(step).split()), limit=140)
                    if hint:
                        compact_steps.append(hint)
                if compact_steps:
                    row["next_steps"] = compact_steps
        history.append(row)

    for obs in observations or []:
        if not isinstance(obs, dict):
            continue
        tid = str(obs.get("task_id") or "").strip()
        if not tid or tid in seen_task_ids:
            continue
        row: dict[str, Any] = {"task_id": tid}
        for key in (
            "outcome",
            "summary",
            "key_artifacts",
            "open_questions",
            "facts",
            "decisions",
            "next_steps",
        ):
            if key in obs:
                row[key] = obs.get(key)
        artifacts = obs.get("key_artifacts")
        if isinstance(artifacts, list):
            row["artifact_count"] = len(artifacts)
        decisions = obs.get("decisions")
        if isinstance(decisions, list):
            row["decision_count"] = len(decisions)
        history.append(row)

    return history


def _render_task_outcomes_history_lines(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(none)"

    def _fmt_list(value: Any, *, limit: int = 3) -> list[str]:
        if not isinstance(value, list):
            return []
        items = []
        for item in value:
            text = _trim_text(" ".join(str(item).split()), limit=120)
            if text:
                items.append(text)
            if len(items) >= limit:
                break
        return items

    lines: list[str] = []
    record_idx = 0
    for row in history:
        if not isinstance(row, dict):
            continue
        fields: list[str] = []
        for key in ("task_id", "status", "outcome", "goal", "summary"):
            value = row.get(key)
            if value is None:
                continue
            text = _trim_text(" ".join(str(value).split()), limit=220 if key == "summary" else 140)
            if text:
                fields.append(f"{key}: {text}")

        for key in ("artifact_count", "decision_count"):
            if key in row:
                fields.append(f"{key}: {row.get(key)}")

        list_fields = ("next_steps", "open_questions", "facts", "decisions")
        for field_key in list_fields:
            items = _fmt_list(row.get(field_key))
            if not items:
                continue
            fields.append(f"{field_key}:")
            for item in items:
                fields.append(f"  - {item}")

        if not fields:
            continue

        record_idx += 1
        lines.append(f"## Record {record_idx}")
        lines.append("")
        lines.append("```md")
        lines.extend(fields)
        lines.append("```")
        lines.append("")

    if not lines:
        return "(none)"
    return "\n".join(lines).rstrip()


def _cap_messages(messages: list[AnyMessage], *, max_messages: int | None = None) -> list[AnyMessage]:
    if max_messages is None or len(messages or []) <= max_messages:
        return list(messages or [])
    return list(messages[-max_messages:])


def _trim_text(text: str, *, limit: int) -> str:
    data = str(text or "").strip()
    return data


def _refresh_memory_index(memory_store: MemoryStore) -> None:
    refresh_fn = getattr(memory_store, "refresh_index_from_topics", None)
    if not callable(refresh_fn):
        return
    try:
        refresh_fn()
    except Exception as exc:
        logger.warning("memory index refresh failed: %s", exc)


def _next_task_index(tasks: list[dict]) -> int:
    max_idx = 0
    for t in tasks:
        tid = str(t.get("task_id", ""))
        m = re.search(r"(\d+)$", tid)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


__all__ = [
    "run_proposal",
    "run_director",
    "run_fast_director",
    "run_task",
    "run_memory_patch",
    "summarize_node",
]
