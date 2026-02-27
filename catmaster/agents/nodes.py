"""
LangGraph node functions for CatMaster multi-agent pipeline.

Contains:
  - Context builders (_build_*_context) for each ReAct agent
  - Wrapper nodes (run_proposal, run_director, run_task) that invoke
    ``create_react_agent`` subgraphs and map results back to parent state
  - Downstream nodes (memory_patch_node, plan_commit_node, summarize_node)
  - Helper utilities for task result normalization
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from catmaster.agents.orchestrator_prompts import (
    PROPOSAL_CONTEXT_TEMPLATE,
    PROPOSAL_REVISION_CONTEXT_TEMPLATE,
    DIRECTOR_CONTEXT_TEMPLATE,
    TASK_CONTEXT_TEMPLATE,
    build_memory_patch_prompt,
    build_memory_patch_repair_prompt,
    build_summary_prompt,
)
from catmaster.agents.response_schemas import (
    ProposalOutput,
    DirectorOutput,
    TaskOutput,
)
from catmaster.agents.llm_utils import llm_text
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.context_pack import ContextPackBuilder, ContextPackPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_proposal_context(
    state: Dict[str, Any],
    memory_store: MemoryStore,
    tools_description: str,
) -> str:
    """Build the HumanMessage text for the proposal agent."""
    user_request = state["user_request"]
    memory_index = memory_store.read_index(max_lines=200, max_chars=12000)
    artifacts_index = json.dumps(
        memory_store.artifact_index(limit=500), ensure_ascii=False,
    )
    logger.info(
        "[_build_proposal_context] user_request_len=%d, memory_index_len=%d, "
        "artifacts_index_len=%d, tools_description_len=%d",
        len(user_request), len(memory_index), len(artifacts_index), len(tools_description),
    )
    if not tools_description:
        logger.warning("[_build_proposal_context] tools_description is EMPTY")

    feedback = state.get("proposal_feedback", "")
    if feedback and state.get("proposal_md"):
        ctx = PROPOSAL_REVISION_CONTEXT_TEMPLATE.format(
            user_request=user_request,
            proposal_md=state["proposal_md"],
            work_packages_json=json.dumps(
                state.get("work_packages", []), ensure_ascii=False,
            ),
            memory_index_excerpt=memory_index,
            artifacts_index=artifacts_index,
            tools=tools_description,
            feedback=feedback,
        )
        logger.info("[_build_proposal_context] revision context total_len=%d", len(ctx))
        return ctx

    ctx = PROPOSAL_CONTEXT_TEMPLATE.format(
        user_request=user_request,
        memory_index_excerpt=memory_index,
        artifacts_index=artifacts_index,
        tools=tools_description,
    )
    logger.info("[_build_proposal_context] fresh context total_len=%d", len(ctx))
    return ctx


def _build_director_context(
    state: Dict[str, Any],
    memory_store: MemoryStore,
    tools_description: str,
) -> str:
    """Build the HumanMessage text for the director agent."""
    observations = state.get("observations", [])
    director_observations = _director_observations_view(observations)
    task_status_board = _director_task_status_board(
        state.get("tasks", []),
        director_observations,
    )

    return DIRECTOR_CONTEXT_TEMPLATE.format(
        user_request=state["user_request"],
        proposal_md=state.get("proposal_md", ""),
        work_packages_json=json.dumps(
            state.get("work_packages", []), ensure_ascii=False,
        ),
        memory_index_excerpt=memory_store.read_index(max_lines=160, max_chars=8000),
        already_done_json=json.dumps(director_observations, ensure_ascii=False),
        task_status_board_json=json.dumps(task_status_board, ensure_ascii=False),
        tools=tools_description,
    )


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
            memory_head_lines=200,
            max_memory_chars=12000,
            inject_goal_for_worker=False,
        ),
    )

    task_detail = str(task_packet.get("task_detail") or "(none)").strip()
    expected_outputs = task_packet.get("expected_outputs") or []
    suggested_tools = task_packet.get("suggested_tools") or []
    reference_hint = task_packet.get("reference_hint") or []

    return TASK_CONTEXT_TEMPLATE.format(
        goal=task_goal,
        task_detail=task_detail,
        expected_outputs=(
            "\n".join(f"- {o}" for o in expected_outputs) if expected_outputs else "(none)"
        ),
        suggested_tools=", ".join(suggested_tools) if suggested_tools else "(none)",
        reference_hint=(
            "\n".join(f"- {h}" for h in reference_hint) if reference_hint else "(none)"
        ),
        workspace_policy=context_pack.get("workspace_policy", ""),
        memory_index_excerpt=context_pack.get("memory_index_excerpt", ""),
    )


# ---------------------------------------------------------------------------
# Wrapper nodes for ReAct agents
# ---------------------------------------------------------------------------


def _parse_json_content(content: Any) -> Any:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {"raw_text": text}
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        joined = "\n".join(part for part in text_parts if part.strip()).strip()
        if joined:
            try:
                return json.loads(joined)
            except Exception:
                return {"raw_text": joined}
    return {"raw_text": str(content)}


def _extract_control_payload(
    messages: list[AnyMessage],
    control_tools: set[str],
) -> tuple[str, dict[str, Any]] | None:
    for msg in reversed(messages or []):
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "").strip()
        if name not in control_tools:
            continue
        parsed = _parse_json_content(msg.content)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("payload"), dict):
                return name, parsed["payload"]
            return name, parsed
        return name, {"raw": parsed}
    return None


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


def _control_tools_seen(messages: list[AnyMessage]) -> list[str]:
    names: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "").strip()
        if name:
            names.append(name)
    return names


def _contract_violation_payload(
    *,
    role: str,
    messages: list[AnyMessage],
    max_steps: int,
    expected_terminal: list[str],
) -> dict[str, Any]:
    return {
        "role": role,
        "reason": "missing_terminal_control_payload",
        "expected_terminal_tools": list(expected_terminal),
        "observed_tool_messages": _control_tools_seen(messages),
        "remaining_steps_exhausted": _is_need_more_steps(messages),
        "message_count": len(messages or []),
        "last_message": _last_message_snippet(messages),
        "max_steps": int(max_steps),
    }


def _contract_violation_summary(role: str, violation: dict[str, Any]) -> str:
    expected = ", ".join(violation.get("expected_terminal_tools") or []) or "(none)"
    observed = ", ".join(violation.get("observed_tool_messages") or []) or "(none)"
    exhausted = "yes" if bool(violation.get("remaining_steps_exhausted")) else "no"
    return (
        f"{role} agent control contract violation: no valid terminal payload "
        f"(expected: {expected}; observed tool messages: {observed}; "
        f"remaining_steps exhausted: {exhausted})."
    )


def _task_output_from_control(
    tool_name: str,
    payload: dict[str, Any],
) -> TaskOutput:
    if tool_name == "task_finish":
        return TaskOutput(
            status="done",
            summary=str(payload.get("summary") or ""),
            facts=list(payload.get("facts") or []),
            files=list(payload.get("files") or []),
            constraints=list(payload.get("constraints") or []),
            open_questions=list(payload.get("open_questions") or []),
            decisions=list(payload.get("decisions") or []),
            next_steps=list(payload.get("next_steps") or []),
            artifacts=list(payload.get("artifacts") or []),
            error="",
            needs_human=False,
            hint="",
        )

    partial = payload.get("partial_result") if isinstance(payload.get("partial_result"), dict) else {}
    summary = str(partial.get("summary") or payload.get("error") or "Task failed.").strip()
    return TaskOutput(
        status="blocked",
        summary=summary,
        facts=list(partial.get("facts") or []),
        files=list(partial.get("files") or []),
        constraints=list(partial.get("constraints") or []),
        open_questions=list(partial.get("open_questions") or []),
        decisions=list(partial.get("decisions") or []),
        next_steps=list(partial.get("next_steps") or []),
        artifacts=list(partial.get("artifacts") or []),
        error=str(payload.get("error") or summary),
        needs_human=bool(payload.get("needs_human", True)),
        hint=str(payload.get("hint") or ""),
    )

def run_proposal(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    tools_description: str,
    run_dir: Path,
    max_steps: int = 30,
    termination_mode: Literal["control_tools", "response_format"] = "control_tools",
    strict_control_contract: bool = True,
) -> Dict[str, Any]:
    """Invoke the proposal ReAct agent and map results to parent state."""
    ctx_text = _build_proposal_context(state, memory_store, tools_description)
    ctx_msg = HumanMessage(content=ctx_text)
    input_messages: list[AnyMessage] = [ctx_msg]
    try:
        result = agent.invoke({
            "messages": input_messages,
            "remaining_steps": max(1, int(max_steps)),
        })
    except Exception as exc:
        logger.exception("proposal agent invoke failed: %s", exc)
        if strict_control_contract and termination_mode == "control_tools":
            summary = f"proposal agent invoke failed before terminal control output: {exc}"
            return {
                "proposal_messages": input_messages,
                "proposal_md": "",
                "work_packages": [],
                "proposal_approved": False,
                "proposal_feedback": "",
                "status": "failure",
                "summary": summary,
                "contract_violation": {
                    "role": "proposal",
                    "reason": "invoke_exception",
                    "error": str(exc),
                    "expected_terminal_tools": ["proposal_finish", "proposal_fail"],
                },
            }
        return {
            "proposal_messages": input_messages,
            "proposal_md": "",
            "work_packages": [],
            "proposal_approved": False,
            "proposal_feedback": "",
        }

    msgs = list(result.get("messages", []) or [])
    resp: ProposalOutput | None = None
    if termination_mode == "response_format":
        resp = result.get("structured_response")
    else:
        control = _extract_control_payload(msgs, {"proposal_finish", "proposal_fail"})
        if control:
            tool_name, payload = control
            if tool_name == "proposal_finish":
                try:
                    resp = ProposalOutput(
                        status="success",
                        proposal_md=str(payload.get("proposal_md") or ""),
                        work_packages=list(payload.get("work_packages") or []),
                        error="",
                        needs_human=False,
                    )
                except Exception:
                    resp = None
            elif tool_name == "proposal_fail":
                resp = ProposalOutput(
                    status="fail",
                    proposal_md="",
                    work_packages=[],
                    error=str(payload.get("error") or "proposal_fail"),
                    needs_human=bool(payload.get("needs_human", True)),
                )

    if resp is None:
        if _is_need_more_steps(msgs):
            logger.warning("[run_proposal] hit remaining_steps limit (max_steps=%d)", max_steps)
        logger.warning(
            "[run_proposal] no terminal payload. message_count=%d, last=%s",
            len(msgs), _last_message_snippet(msgs),
        )
        if strict_control_contract and termination_mode == "control_tools":
            violation = _contract_violation_payload(
                role="proposal",
                messages=msgs,
                max_steps=max_steps,
                expected_terminal=["proposal_finish", "proposal_fail"],
            )
            return {
                "proposal_messages": _cap_messages(msgs),
                "proposal_md": "",
                "work_packages": [],
                "proposal_approved": False,
                "proposal_feedback": "",
                "status": "failure",
                "summary": _contract_violation_summary("proposal", violation),
                "contract_violation": violation,
            }
        return {
            "proposal_messages": _cap_messages(msgs),
            "proposal_md": "",
            "work_packages": [],
            "proposal_approved": False,
            "proposal_feedback": "",
        }

    if resp.status == "fail":
        logger.warning(
            "[run_proposal] proposal agent reported FAIL: error=%r, needs_human=%s",
            resp.error, resp.needs_human,
        )

    proposal_md = resp.proposal_md
    work_packages = resp.work_packages

    if proposal_md:
        try:
            (run_dir / "proposal.md").write_text(proposal_md, encoding="utf-8")
        except Exception:
            pass

    return {
        "proposal_messages": _cap_messages(msgs),
        "proposal_md": proposal_md,
        "work_packages": work_packages,
        "proposal_approved": False,
        "proposal_feedback": "",
    }


def run_director(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    tools_description: str,
    max_steps: int = 30,
    termination_mode: Literal["control_tools", "response_format"] = "control_tools",
    strict_control_contract: bool = True,
) -> Command:
    """Invoke the director ReAct agent and route via Command."""
    ctx_text = _build_director_context(state, memory_store, tools_description)
    ctx_msg = HumanMessage(content=ctx_text)
    input_messages: list[AnyMessage] = [ctx_msg]
    try:
        result = agent.invoke({
            "messages": input_messages,
            "remaining_steps": max(1, int(max_steps)),
        })
    except Exception as exc:
        logger.exception("director agent invoke failed: %s", exc)
        if strict_control_contract and termination_mode == "control_tools":
            summary = f"director agent invoke failed before terminal control output: {exc}"
            return Command(
                goto="finalize_memory_patch",
                update={
                    "director_messages": input_messages,
                    "director_decision": {},
                    "next_action": "ContractViolation",
                    "status": "failure",
                    "summary": summary,
                    "contract_violation": {
                        "role": "director",
                        "reason": "invoke_exception",
                        "error": str(exc),
                        "expected_terminal_tools": ["director_decide"],
                    },
                },
            )
        return Command(
            goto="finalize_memory_patch",
            update={
                "director_messages": input_messages,
                "director_decision": {},
                "next_action": "StopAndSynthesize",
            },
        )

    msgs = list(result.get("messages", []) or [])
    resp: DirectorOutput | None = None
    if termination_mode == "response_format":
        resp = result.get("structured_response")
    else:
        control = _extract_control_payload(msgs, {"director_decide"})
        if control:
            _, payload = control
            try:
                resp = DirectorOutput(**payload)
            except Exception as exc:
                logger.warning("Invalid director_decide payload: %s", exc)

    if resp is None:
        if _is_need_more_steps(msgs):
            logger.warning("[run_director] hit remaining_steps limit (max_steps=%d)", max_steps)
        logger.warning("director agent returned no terminal payload")
        if strict_control_contract and termination_mode == "control_tools":
            violation = _contract_violation_payload(
                role="director",
                messages=msgs,
                max_steps=max_steps,
                expected_terminal=["director_decide"],
            )
            return Command(
                goto="finalize_memory_patch",
                update={
                    "director_messages": _cap_messages(msgs),
                    "director_decision": {},
                    "next_action": "ContractViolation",
                    "status": "failure",
                    "summary": _contract_violation_summary("director", violation),
                    "contract_violation": violation,
                },
            )
        logger.warning("defaulting to StopAndSynthesize")
        return Command(
            goto="finalize_memory_patch",
            update={
                "director_messages": _cap_messages(msgs),
                "director_decision": {},
                "next_action": "StopAndSynthesize",
            },
        )

    update: Dict[str, Any] = {
        "director_messages": _cap_messages(msgs),
        "director_decision": resp.model_dump(),
        "next_action": resp.state,
    }

    if resp.state == "PerformNextTask":
        task_packet = resp.task_packet
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
        if isinstance(resp.updated_proposal_md, str):
            update["proposal_md"] = resp.updated_proposal_md
        if isinstance(resp.updated_work_packages, list):
            update["work_packages"] = resp.updated_work_packages
        update["proposal_approved"] = False
        update["proposal_feedback"] = resp.rationale
        return Command(goto="proposal_review", update=update)

    if resp.state == "MinorReviseProposal":
        if isinstance(resp.updated_proposal_md, str):
            update["proposal_md"] = resp.updated_proposal_md
        if isinstance(resp.updated_work_packages, list):
            update["work_packages"] = resp.updated_work_packages
        return Command(goto="run_director", update=update)

    # StopAndSynthesize
    return Command(goto="finalize_memory_patch", update=update)


def run_task(
    state: Dict[str, Any],
    *,
    agent: Any,
    memory_store: MemoryStore,
    max_steps: int = 40,
    termination_mode: Literal["control_tools", "response_format"] = "control_tools",
    strict_control_contract: bool = True,
) -> Dict[str, Any]:
    """Invoke the task runner ReAct agent and map results to parent state."""
    ctx_text = _build_task_context(state, memory_store)
    ctx_msg = HumanMessage(content=ctx_text)

    try:
        result = agent.invoke({
            "messages": [ctx_msg],
            "remaining_steps": max(1, int(max_steps)),
        })
    except Exception as exc:
        logger.exception("task runner invoke failed: %s", exc)
        if strict_control_contract and termination_mode == "control_tools":
            summary = f"task runner invoke failed before terminal control output: {exc}"
            return {
                "runner_messages": [ctx_msg],
                "task_result": {
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
                },
                "status": "failure",
                "summary": summary,
                "contract_violation": {
                    "role": "task_runner",
                    "reason": "invoke_exception",
                    "error": str(exc),
                    "expected_terminal_tools": ["task_finish", "task_fail"],
                },
            }
        result = {"messages": [ctx_msg]}

    msgs = list(result.get("messages", []) or [])
    resp: TaskOutput | None = None
    if termination_mode == "response_format":
        resp = result.get("structured_response")
    else:
        control = _extract_control_payload(msgs, {"task_finish", "task_fail"})
        if control:
            tool_name, payload = control
            try:
                resp = _task_output_from_control(tool_name, payload)
            except Exception as exc:
                logger.warning("Invalid %s payload: %s", tool_name, exc)

    if resp is None:
        base_summary = (
            "Task runner returned no terminal payload."
            + (" (remaining_steps exhausted)" if _is_need_more_steps(msgs) else "")
        )
        task_result = {
            "task_outcome": "failure",
            "task_summary": base_summary,
            "key_artifacts": [],
            "structured_result": {
                "summary": "Task runner returned no terminal payload.",
                "facts": [],
                "files": [],
                "constraints": [],
                "open_questions": [],
                "decisions": [],
                "next_steps": [],
                "artifacts": [],
            },
        }
        if strict_control_contract and termination_mode == "control_tools":
            violation = _contract_violation_payload(
                role="task_runner",
                messages=msgs,
                max_steps=max_steps,
                expected_terminal=["task_finish", "task_fail"],
            )
            summary = _contract_violation_summary("task_runner", violation)
            task_result["task_summary"] = summary
            task_result["structured_result"]["summary"] = summary
            return {
                "runner_messages": msgs,
                "task_result": task_result,
                "status": "failure",
                "summary": summary,
                "contract_violation": violation,
            }
    else:
        task_result = _normalize_task_output(resp)

    return {
        "runner_messages": msgs,
        "task_result": task_result,
    }


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


# ---------------------------------------------------------------------------
# Memory patch node
# ---------------------------------------------------------------------------

def memory_patch_node(
    state: Dict[str, Any],
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str = "",
    patch_repair_attempts: int = 1,
    tool_backend: Any = None,
) -> Dict[str, Any]:
    """Apply memory patches after a task completes."""
    task_result = state.get("task_result") or {}
    task_id = state.get("current_task_id", "")

    structured_result = task_result.get("structured_result") or {}
    outcome = task_result.get("task_outcome", "failure")

    memory_store.ensure_exists()
    patch_status = "skipped"
    last_error = ""
    refresh_needed = False
    max_attempts = max(1, patch_repair_attempts + 1)
    if tool_backend is None:
        logger.info("[memory_patch_node] skip patch: tool_backend unavailable")
        refresh_needed = True
    else:
        memory_index_text = memory_store.read_index(max_lines=2000, max_chars=200000)
        topic_texts = _read_memory_topics(memory_store)
        previous_edit_text = ""
        patch_status = "failed"

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                prompt = build_memory_patch_prompt()
                msgs = prompt.format_messages(
                    run_id=run_id,
                    task_id=task_id,
                    task_goal=str(state.get("current_task_packet", {}).get("goal", "")),
                    outcome=outcome,
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    **{f"topic_{k.lower().replace('.md', '')}_text": v for k, v in topic_texts.items()},
                )
            else:
                prompt = build_memory_patch_repair_prompt()
                msgs = prompt.format_messages(
                    previous_edit_text=previous_edit_text,
                    apply_error=last_error or "(none)",
                    apply_error_context_json="{}",
                    run_id=run_id,
                    task_id=task_id,
                    task_goal=str(state.get("current_task_packet", {}).get("goal", "")),
                    outcome=outcome,
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    **{f"topic_{k.lower().replace('.md', '')}_text": v for k, v in topic_texts.items()},
                )

            try:
                resp = model.invoke(msgs)
                patch_raw = llm_text(resp).strip()
                edit_text = _normalize_patch_text(patch_raw)
                previous_edit_text = edit_text
                tool_out = tool_backend.call(
                    "memory_apply_aider_edits",
                    json.dumps({"edits_text": edit_text, "allowed_paths": ["MEMORY/"], "emit_diff": True}, ensure_ascii=False),
                    toolcall_key=f"{task_id}_memory_patch_a{attempt}",
                )
                status = str(tool_out.get("status") or "").strip().lower()
                if status == "success":
                    patch_status = "success"
                    last_error = ""
                    break
                last_error = str(tool_out.get("error") or "patch apply failed")
            except Exception as exc:
                last_error = str(exc)
                logger.warning("[memory_patch_node] attempt %d failed: %s", attempt, exc)

        if patch_status != "success":
            refresh_needed = True

    if refresh_needed:
        _refresh_memory_index(memory_store)
    memory_store.append_event({
        "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "run_id": run_id,
        "task_id": task_id,
        "outcome": outcome,
        "summary": str(structured_result.get("summary") or "").strip(),
        "memory_patch_status": patch_status,
        "memory_patch_error": last_error,
    })

    observation = {
        "task_id": task_id,
        "outcome": outcome,
        "summary": task_result.get("task_summary", ""),
        "key_artifacts": task_result.get("key_artifacts", []),
    }
    failure_kind = str(task_result.get("failure_kind") or "").strip()
    if failure_kind:
        observation["failure_kind"] = failure_kind

    decision_items = structured_result.get("decisions")
    if isinstance(decision_items, list):
        compact_decisions: list[str] = []
        for item in decision_items[:5]:
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

    next_steps = structured_result.get("next_steps")
    if isinstance(next_steps, list):
        compact_next = [
            _trim_text(" ".join(str(step).split()), limit=140)
            for step in next_steps[:5]
            if str(step).strip()
        ]
        compact_next = [step for step in compact_next if step]
        if compact_next:
            observation["next_steps"] = compact_next

    open_questions = structured_result.get("open_questions")
    if isinstance(open_questions, list):
        compact_questions = [
            _trim_text(" ".join(str(question).split()), limit=140)
            for question in open_questions[:5]
            if str(question).strip()
        ]
        compact_questions = [question for question in compact_questions if question]
        if compact_questions:
            observation["open_questions"] = compact_questions

    facts = structured_result.get("facts")
    if isinstance(facts, list):
        compact_facts = [
            _trim_text(" ".join(str(fact).split()), limit=180)
            for fact in facts[:8]
            if str(fact).strip()
        ]
        compact_facts = [fact for fact in compact_facts if fact]
        if compact_facts:
            observation["facts"] = compact_facts

    existing_observations = list(state.get("observations") or [])
    existing_observations.append(observation)

    existing_tasks = list(state.get("tasks") or [])
    for t in existing_tasks:
        if t.get("task_id") == task_id:
            t["status"] = outcome

    return {
        "observations": existing_observations,
        "tasks": existing_tasks,
        "task_result": task_result,
    }


def finalize_memory_patch_node(
    state: Dict[str, Any],
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str = "",
    patch_repair_attempts: int = 1,
    tool_backend: Any = None,
) -> Dict[str, Any]:
    """Apply one final memory reconciliation patch before summarize."""
    memory_store.ensure_exists()
    final_status = str(state.get("status") or "done").strip().lower()
    if final_status not in {"done", "failure", "needs_intervention"}:
        final_status = "done"

    observations = list(state.get("observations") or [])
    tasks = list(state.get("tasks") or [])
    task_result = state.get("task_result") or {}
    structured_source = task_result.get("structured_result")
    if not isinstance(structured_source, dict):
        structured_source = {}

    summary_candidates = [
        state.get("summary"),
        task_result.get("task_summary") if isinstance(task_result, dict) else None,
    ]
    if observations and isinstance(observations[-1], dict):
        summary_candidates.append(observations[-1].get("summary"))
    summary_text = ""
    for item in summary_candidates:
        text = str(item or "").strip()
        if text:
            summary_text = text
            break
    if not summary_text:
        summary_text = f"Run finalized with status: {final_status}."

    artifact_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        task_id = str(obs.get("task_id") or "")
        for art in obs.get("key_artifacts") or []:
            if not isinstance(art, dict):
                continue
            path = str(art.get("path") or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            row = {
                "path": path,
                "description": str(art.get("description") or "").strip(),
                "source": task_id,
            }
            artifact_rows.append(row)
            if len(artifact_rows) >= 30:
                break
        if len(artifact_rows) >= 30:
            break

    facts = structured_source.get("facts")
    facts_payload = facts if isinstance(facts, list) else []

    files_payload: list[Any] = []
    src_files = structured_source.get("files")
    if isinstance(src_files, list):
        files_payload = list(src_files[:30])
    if not files_payload:
        files_payload = [row.get("path") for row in artifact_rows if row.get("path")]

    constraints = structured_source.get("constraints")
    constraints_payload = constraints if isinstance(constraints, list) else []

    open_questions = structured_source.get("open_questions")
    open_questions_payload = open_questions if isinstance(open_questions, list) else []
    if final_status == "done":
        open_questions_payload = []

    decisions = structured_source.get("decisions")
    decisions_payload = decisions if isinstance(decisions, list) else []

    next_steps = structured_source.get("next_steps")
    next_steps_payload = next_steps if isinstance(next_steps, list) else []
    if final_status == "done" and not next_steps_payload:
        next_steps_payload = ["None (goal scope complete)."]

    structured_result = {
        "summary": summary_text,
        "facts": facts_payload,
        "files": files_payload,
        "constraints": constraints_payload,
        "open_questions": open_questions_payload,
        "decisions": decisions_payload,
        "next_steps": next_steps_payload,
        "artifacts": artifact_rows,
        "final_status": final_status,
        "task_count": len(tasks),
        "completed_task_count": sum(1 for task in tasks if str(task.get("status") or "") == "success"),
    }

    patch_status = "skipped"
    last_error = ""
    refresh_needed = False
    max_attempts = max(1, patch_repair_attempts + 1)
    if tool_backend is None:
        logger.info("[finalize_memory_patch_node] skip patch: tool_backend unavailable")
        refresh_needed = True
    else:
        memory_index_text = memory_store.read_index(max_lines=2000, max_chars=200000)
        topic_texts = _read_memory_topics(memory_store)
        previous_edit_text = ""
        patch_status = "failed"

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                prompt = build_memory_patch_prompt()
                msgs = prompt.format_messages(
                    run_id=run_id,
                    task_id="finalize_memory",
                    task_goal="Reconcile memory to latest run state before summary",
                    outcome=final_status,
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    **{f"topic_{k.lower().replace('.md', '')}_text": v for k, v in topic_texts.items()},
                )
            else:
                prompt = build_memory_patch_repair_prompt()
                msgs = prompt.format_messages(
                    previous_edit_text=previous_edit_text,
                    apply_error=last_error or "(none)",
                    apply_error_context_json="{}",
                    run_id=run_id,
                    task_id="finalize_memory",
                    task_goal="Reconcile memory to latest run state before summary",
                    outcome=final_status,
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    **{f"topic_{k.lower().replace('.md', '')}_text": v for k, v in topic_texts.items()},
                )

            try:
                resp = model.invoke(msgs)
                patch_raw = llm_text(resp).strip()
                edit_text = _normalize_patch_text(patch_raw)
                previous_edit_text = edit_text
                tool_out = tool_backend.call(
                    "memory_apply_aider_edits",
                    json.dumps({"edits_text": edit_text, "allowed_paths": ["MEMORY/"], "emit_diff": True}, ensure_ascii=False),
                    toolcall_key=f"finalize_memory_patch_a{attempt}",
                )
                status = str(tool_out.get("status") or "").strip().lower()
                if status == "success":
                    patch_status = "success"
                    last_error = ""
                    break
                last_error = str(tool_out.get("error") or "patch apply failed")
            except Exception as exc:
                last_error = str(exc)
                logger.warning("[finalize_memory_patch_node] attempt %d failed: %s", attempt, exc)

        if patch_status != "success":
            refresh_needed = True

    if refresh_needed:
        _refresh_memory_index(memory_store)
    memory_store.append_event({
        "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "run_id": run_id,
        "task_id": "finalize_memory",
        "outcome": final_status,
        "summary": summary_text,
        "memory_patch_status": patch_status,
        "memory_patch_error": last_error,
    })
    return {}


# ---------------------------------------------------------------------------
# Summarize node
# ---------------------------------------------------------------------------

def summarize_node(
    state: Dict[str, Any],
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
) -> Dict[str, Any]:
    """Generate the final project summary."""
    user_request = state["user_request"]
    observations = state.get("observations", [])
    memory_index = memory_store.read_index(max_lines=200, max_chars=12000)
    artifacts = memory_store.artifact_index(limit=200)

    prompt = build_summary_prompt()
    messages = prompt.format_messages(
        user_request=user_request,
        status=state.get("status", "done"),
        memory_index_excerpt=memory_index,
        observations=json.dumps(observations, ensure_ascii=False, default=str),
        artifacts=json.dumps(artifacts, ensure_ascii=False, default=str),
    )
    resp = model.invoke(messages)
    summary = llm_text(resp).strip()
    incoming_status = str(state.get("status") or "done")
    final_status = incoming_status if incoming_status in {"failure", "needs_intervention"} else "done"
    return {"summary": summary, "status": final_status}


# ---------------------------------------------------------------------------
# Plan commit node
# ---------------------------------------------------------------------------

def plan_commit_node(
    state: Dict[str, Any],
    *,
    model: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str = "",
    tool_backend: Any = None,
) -> Dict[str, Any]:
    """Commit the approved plan to memory via LLM-generated patch."""
    proposal_md = state.get("proposal_md", "")
    work_packages = state.get("work_packages", [])
    if not proposal_md:
        return {}

    _snippet = lambda text, limit=600: (" ".join(str(text or "").split()))[:limit]

    structured_result = {
        "summary": "Plan committed for execution.",
        "facts": [
            f"Run focus: {_snippet(proposal_md)}",
            f"Work package count: {len(work_packages)}",
            "Work packages: " + " | ".join(work_packages[:8]),
        ],
        "files": [],
        "constraints": [],
        "open_questions": [],
        "decisions": [{"decision": "Plan committed", "rationale": "Approved for execution"}],
        "next_steps": work_packages[:5],
        "artifacts": [],
    }

    memory_store.ensure_exists()
    event = {
        "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "run_id": run_id,
        "task_id": "plan_commit",
        "outcome": "success",
        "summary": "Plan committed for execution.",
    }
    memory_store.append_event(event)

    refresh_needed = False
    if tool_backend is None:
        logger.info("[plan_commit_node] skip memory patch: tool_backend unavailable")
        refresh_needed = True
    else:
        memory_index_text = memory_store.read_index(max_lines=2000, max_chars=200000)
        topic_texts = _read_memory_topics(memory_store)

        prompt = build_memory_patch_prompt()
        msgs = prompt.format_messages(
            run_id=run_id,
            task_id="plan_commit",
            task_goal="Commit approved plan to memory",
            outcome="success",
            structured_result_json=json.dumps(structured_result, ensure_ascii=False),
            memory_index_text=memory_index_text,
            **{f"topic_{k.lower().replace('.md', '')}_text": v for k, v in topic_texts.items()},
        )

        try:
            resp = model.invoke(msgs)
            patch_raw = llm_text(resp).strip()
            edit_text = _normalize_patch_text(patch_raw)
            tool_out = tool_backend.call(
                "memory_apply_aider_edits",
                json.dumps({
                    "edits_text": edit_text,
                    "allowed_paths": ["MEMORY/"],
                    "emit_diff": True,
                }, ensure_ascii=False),
                toolcall_key="plan_commit_memory_patch",
            )
            status = str(tool_out.get("status") or "").strip().lower()
            if status != "success":
                refresh_needed = True
        except Exception as exc:
            logger.warning("Plan commit memory patch failed: %s", exc)
            refresh_needed = True

    if refresh_needed:
        _refresh_memory_index(memory_store)

    return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _director_observations_view(observations: list[dict], *, max_items: int = 60) -> list[dict]:
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
            for entry in decisions[:3]:
                text = _trim_text(" ".join(str(entry).split()), limit=220)
                if text:
                    compact_decisions.append(text)
            if compact_decisions:
                row["decisions"] = compact_decisions

        next_steps = item.get("next_steps")
        if isinstance(next_steps, list):
            compact_next: list[str] = []
            for entry in next_steps[:3]:
                text = _trim_text(" ".join(str(entry).split()), limit=180)
                if text:
                    compact_next.append(text)
            if compact_next:
                row["next_steps"] = compact_next

        open_questions = item.get("open_questions")
        if isinstance(open_questions, list):
            compact_questions: list[str] = []
            for entry in open_questions[:2]:
                text = _trim_text(" ".join(str(entry).split()), limit=180)
                if text:
                    compact_questions.append(text)
            if compact_questions:
                row["open_questions"] = compact_questions

        facts = item.get("facts")
        if isinstance(facts, list):
            compact_facts: list[str] = []
            for entry in facts[:3]:
                text = _trim_text(" ".join(str(entry).split()), limit=180)
                if text:
                    compact_facts.append(text)
            if compact_facts:
                row["facts"] = compact_facts
        if row:
            sanitized.append(row)
    if len(sanitized) > max_items:
        return sanitized[-max_items:]
    return sanitized


def _director_task_status_board(
    tasks: list[dict],
    observations: list[dict],
    *,
    max_items: int = 80,
) -> list[dict]:
    obs_map: dict[str, dict] = {}
    for row in observations or []:
        if isinstance(row, dict):
            tid = str(row.get("task_id") or "").strip()
            if tid:
                obs_map[tid] = row

    board: list[dict] = []
    for item in tasks or []:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("task_id") or "").strip()
        if not tid:
            continue
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
                hint = _trim_text(" ".join(str(next_steps[0]).split()), limit=140)
                if hint:
                    row["next_step_hint"] = hint
        board.append(row)

    if len(board) > max_items:
        return board[-max_items:]
    return board


def _cap_messages(messages: list[AnyMessage], *, max_messages: int = 40) -> list[AnyMessage]:
    if len(messages or []) <= max_messages:
        return list(messages or [])
    return list(messages[-max_messages:])


def _trim_text(text: str, *, limit: int) -> str:
    data = str(text or "").strip()
    if len(data) <= limit:
        return data
    return data[: max(0, limit - 3)] + "..."


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


def _read_memory_topics(memory_store: MemoryStore) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("GOAL.md", "FACTS.md", "FILES.md", "CONSTRAINTS.md", "QUESTIONS.md", "RUNBOOK.md"):
        topic_path = memory_store.topics_dir / name
        try:
            out[name] = topic_path.read_text(encoding="utf-8")
        except Exception:
            out[name] = ""
    return out


def _normalize_patch_text(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        m = re.match(r"^```[^\n]*\n(.*?)\n```$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


__all__ = [
    "run_proposal",
    "run_director",
    "run_task",
    "memory_patch_node",
    "finalize_memory_patch_node",
    "plan_commit_node",
    "summarize_node",
]
