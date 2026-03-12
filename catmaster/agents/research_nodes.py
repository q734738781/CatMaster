from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from catmaster.agents.nodes import _invoke_agent_with_step_budget
from catmaster.runtime.literature.models import LiteratureContextPack
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import (
    ConclusionRecord,
    ExperimentLaneRunner,
    ExperimentRunPack,
    HypothesisRecord,
    ResearchActionRef,
    ResearchBoard,
    ResearchContextBuilder,
    ResearchDossier,
    ResearchPlannerContextPack,
    ResearchStore,
    ResearchLiteratureRunner,
)
from catmaster.runtime.research.dossier import build_research_dossier
from catmaster.runtime.skills import CatMasterSkillsRuntime, render_research_lead_skill_guide

from .research_prompts import (
    RESEARCH_LEAD_SYSTEM_PROMPT,
    RESEARCH_STATE_UPDATER_SYSTEM_PROMPT,
    build_research_lead_context,
    build_research_state_sync_context,
)
from .research_schemas import (
    ResearchLeadOutput,
    ResearchRequest,
    ResearchStateSyncOutput,
)


def validate_research_action(
    *,
    action: ResearchLeadOutput,
    request: ResearchRequest,
    board: ResearchBoard,
) -> None:
    if action.state in {"RunLiterature", "RunExperiment"} and board.cycle_index >= board.max_cycles:
        raise ValueError("cycle budget exhausted; choose Conclude or AskHuman")
    if action.state == "RunLiterature":
        if board.used_literature_queries >= board.max_literature_queries:
            raise ValueError("literature budget exhausted")
        if action.run_literature is None:
            raise ValueError("missing literature payload")
        if action.run_literature.depth == "deep_report" and not request.allow_deep_report:
            raise ValueError("deep_report is disabled for this request")
    if action.state == "RunExperiment":
        if action.run_experiment is None:
            raise ValueError("missing experiment payload")
        if action.run_experiment.lane == "fast" and board.used_fast_runs >= board.max_fast_runs:
            raise ValueError("fast-run budget exhausted")
        if action.run_experiment.lane == "standard" and board.used_standard_runs >= board.max_standard_runs:
            raise ValueError("standard-run budget exhausted")
        known_ids = {item.hypothesis_id for item in board.hypotheses}
        missing = [hid for hid in action.run_experiment.hypothesis_ids if hid not in known_ids]
        if missing:
            raise ValueError(f"unknown hypothesis ids: {', '.join(missing)}")
    if action.state == "RunWriter":
        if action.run_writer is None:
            raise ValueError("missing writer payload")
        if not str(action.run_writer.request or "").strip():
            raise ValueError("RunWriter requires non-empty request")
        if str(action.run_writer.writing_mode or "").strip() == "none":
            raise ValueError("RunWriter requires concrete writing_mode")
def validate_research_state_sync(
    *,
    sync: ResearchStateSyncOutput,
    request: ResearchRequest,
) -> None:
    if request.exploration_policy == "anchored" and sync.new_hypotheses:
        raise ValueError("anchored policy forbids new hypotheses")
    if request.exploration_policy == "local_expand":
        for item in sync.new_hypotheses:
            if not item.parent_hypothesis_ids:
                raise ValueError("local_expand hypotheses require parent_hypothesis_ids")


def init_research_board(*, request: ResearchRequest, campaign_id: str) -> ResearchBoard:
    hypotheses = [
        HypothesisRecord(
            hypothesis_id=f"H{idx}",
            text=text,
            source="user_seed",
            status="seed",
        )
        for idx, text in enumerate([item for item in request.seed_hypotheses if str(item).strip()], start=1)
    ]
    return ResearchBoard(
        campaign_id=campaign_id,
        question=request.question,
        exploration_policy=request.exploration_policy,
        max_cycles=request.max_cycles,
        max_literature_queries=request.max_literature_queries,
        max_fast_runs=request.max_fast_runs,
        max_standard_runs=request.max_standard_runs,
        hypotheses=hypotheses,
    )


def _next_hypothesis_id(board: ResearchBoard) -> str:
    existing = {item.hypothesis_id for item in board.hypotheses}
    idx = 1
    while True:
        candidate = f"H{idx}"
        if candidate not in existing:
            return candidate
        idx += 1


def _board_with_updates(board: ResearchBoard, **updates: Any) -> ResearchBoard:
    payload = board.model_dump()
    payload.update(updates)
    return ResearchBoard.model_validate(payload)


def _dedupe_clean(items: list[str], *, limit: int = 20) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = " ".join(str(raw or "").split()).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _apply_hypothesis_updates(
    *,
    board: ResearchBoard,
    sync: ResearchStateSyncOutput,
    request: ResearchRequest,
) -> ResearchBoard:
    hypotheses = [item.model_copy(deep=True) for item in board.hypotheses]
    by_id = {item.hypothesis_id: item for item in hypotheses}
    for update in sync.hypothesis_updates:
        record = by_id.get(update.hypothesis_id)
        if record is None:
            continue
        record.status = update.status
        record.note = update.note
    for proposal in sync.new_hypotheses:
        new_id = _next_hypothesis_id(
            ResearchBoard.model_validate({**board.model_dump(), "hypotheses": [item.model_dump() for item in hypotheses]})
        )
        hypotheses.append(
            HypothesisRecord(
                hypothesis_id=new_id,
                text=proposal.text,
                source="agent_local_expand" if request.exploration_policy == "local_expand" else "agent_refinement",
                status="open",
                note=proposal.rationale,
            )
        )
    by_id = {item.hypothesis_id: item for item in hypotheses}
    for link in sync.evidence_links:
        record = by_id.get(link.hypothesis_id)
        if record is None:
            continue
        refs = list(record.evidence_refs)
        if link.ref_path and link.ref_path not in refs:
            refs.append(link.ref_path)
        record.evidence_refs = refs
    return board.model_copy(update={"hypotheses": hypotheses})


def _latest_action_markdown(
    *,
    board: ResearchBoard,
    latest_literature: LiteratureContextPack | None,
    latest_experiment: ExperimentRunPack | None,
) -> str:
    if not board.action_refs:
        return "No completed action yet. Initialize the campaign state from the question, session context, memory, and seeded hypotheses only."
    latest_ref = board.action_refs[-1]
    lines = [
        f"- action_id: {latest_ref.action_id}",
        f"- kind: {latest_ref.kind}",
        f"- status: {latest_ref.status}",
        f"- ref_path: {latest_ref.ref_path}",
        f"- summary: {latest_ref.summary}",
    ]
    if latest_ref.kind == "literature" and latest_literature is not None:
        lines.extend(
            [
                "",
                "Latest literature pack:",
                f"- query: {latest_literature.query}",
                f"- depth: {latest_literature.depth}",
                f"- summary: {latest_literature.summary}",
                "Follow-up questions:",
                *([f"- {item}" for item in latest_literature.followup_questions] or ["- (none)"]),
            ]
        )
    if latest_ref.kind == "experiment" and latest_experiment is not None:
        lines.extend(
            [
                "",
                "Latest experiment pack:",
                f"- title: {latest_experiment.brief.title}",
                f"- lane: {latest_experiment.lane}",
                f"- summary: {latest_experiment.summary}",
                f"- linked hypotheses: {', '.join(latest_experiment.brief.hypothesis_ids) or '(none)'}",
                "Top observations:",
                *([f"- {item}" for item in latest_experiment.top_observations] or ["- (none)"]),
                "Open questions:",
                *([f"- {item}" for item in latest_experiment.open_questions] or ["- (none)"]),
            ]
        )
    return "\n".join(lines).strip()


def _apply_state_sync_output(
    *,
    board: ResearchBoard,
    sync: ResearchStateSyncOutput,
    request: ResearchRequest,
) -> ResearchBoard:
    updated = _apply_hypothesis_updates(board=board, sync=sync, request=request)
    return _board_with_updates(
        updated,
        current_best_answer_md=str(sync.current_best_answer_md or "").strip(),
        supported_claims=_dedupe_clean(list(sync.supported_claims), limit=20),
        open_questions=_dedupe_clean(list(sync.open_questions), limit=20),
    )


async def init_campaign_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    progress_callback=None,
) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="planning", current_work_label="Initialize research campaign")
    if state.get("resume_mode"):
        resume_goto = str(state.get("resume_goto") or "plan_research").strip() or "plan_research"
        if resume_goto == "summarize_research":
            return Command(
                goto="summarize_research",
                update={
                    "status": state.get("status", "done"),
                    "summary": state.get("summary", ""),
                    "final_answer": state.get("final_answer", state.get("summary", "")),
                },
            )
        return Command(goto=resume_goto, update={"status": state.get("status", "running")})
    request = ResearchRequest.model_validate(state["request"])
    board = init_research_board(request=request, campaign_id=store.campaign_id)
    store.write_request(request)
    store.save_board(board)
    return Command(
        goto="initialize_research_state",
        update={"board": board, "status": "running", "sync_reason": "initialize"},
    )


async def _run_state_sync(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    state_updater_agent: Any,
    memory_store: MemoryStore,
    skills_runtime: CatMasterSkillsRuntime | None,
    progress_callback=None,
) -> Command:
    sync_reason = str(state.get("sync_reason") or "").strip().lower()
    label = "Sync research state"
    if sync_reason == "initialize":
        label = "Initialize research state"
    elif sync_reason == "literature":
        label = "Sync after literature review"
    elif sync_reason == "experiment":
        label = "Sync after experiment"
    elif sync_reason == "writer":
        label = "Sync after writing handoff"
    elif sync_reason == "human_feedback":
        label = "Sync after human feedback"
    if progress_callback is not None:
        progress_callback(current_phase="syncing", current_work_label=label)
    request = ResearchRequest.model_validate(state["request"])
    board = ResearchBoard.model_validate(
        state["board"].model_dump() if hasattr(state.get("board"), "model_dump") else state["board"]
    )
    latest_literature = state.get("latest_literature")
    latest_experiment = state.get("latest_experiment")
    if isinstance(latest_literature, dict):
        latest_literature = LiteratureContextPack.model_validate(latest_literature)
    if isinstance(latest_experiment, dict):
        latest_experiment = ExperimentRunPack.model_validate(latest_experiment)
    skill_guide = render_research_lead_skill_guide(
        skills_runtime.visible_skills("research_state_updater") if skills_runtime is not None else []
    )
    context_pack = ResearchContextBuilder(store=store, memory_store=memory_store).build_planner_context(
        board=board,
        latest_literature=latest_literature if isinstance(latest_literature, LiteratureContextPack) else None,
        latest_experiment=latest_experiment if isinstance(latest_experiment, ExperimentRunPack) else None,
        session_context=str(request.session_context_text or ""),
    )
    context = build_research_state_sync_context(
        pack=context_pack,
        latest_action_md=_latest_action_markdown(
            board=board,
            latest_literature=latest_literature if isinstance(latest_literature, LiteratureContextPack) else None,
            latest_experiment=latest_experiment if isinstance(latest_experiment, ExperimentRunPack) else None,
        ),
        research_skill_guide=skill_guide,
    )
    messages = [HumanMessage(content=context)]
    raw = (
        await _invoke_agent_with_step_budget(
            agent=state_updater_agent,
            messages=messages,
            max_steps=12,
            role="research_state_updater",
        )
    ).get("structured_response")
    sync = raw if isinstance(raw, ResearchStateSyncOutput) else ResearchStateSyncOutput.model_validate(raw)
    validate_research_state_sync(sync=sync, request=request)
    board = _apply_state_sync_output(board=board, sync=sync, request=request)
    store.save_board(board)
    return Command(
        goto="plan_research",
        update={
            "board": board,
            "planner_context": context_pack,
            "latest_state_sync": sync,
            "sync_reason": "",
        },
    )


async def initialize_research_state_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    state_updater_agent: Any,
    memory_store: MemoryStore,
    skills_runtime: CatMasterSkillsRuntime | None,
    progress_callback=None,
) -> Command:
    return await _run_state_sync(
        state,
        store=store,
        state_updater_agent=state_updater_agent,
        memory_store=memory_store,
        skills_runtime=skills_runtime,
        progress_callback=progress_callback,
    )


async def sync_research_state_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    state_updater_agent: Any,
    memory_store: MemoryStore,
    skills_runtime: CatMasterSkillsRuntime | None,
    progress_callback=None,
) -> Command:
    return await _run_state_sync(
        state,
        store=store,
        state_updater_agent=state_updater_agent,
        memory_store=memory_store,
        skills_runtime=skills_runtime,
        progress_callback=progress_callback,
    )


async def plan_research_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    planner_agent: Any,
    memory_store: MemoryStore,
    skills_runtime: CatMasterSkillsRuntime | None,
    progress_callback=None,
) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="planning", current_work_label="Plan next research action")
    request = ResearchRequest.model_validate(state["request"])
    board = ResearchBoard.model_validate(
        state["board"].model_dump() if hasattr(state.get("board"), "model_dump") else state["board"]
    )
    latest_literature = state.get("latest_literature")
    latest_experiment = state.get("latest_experiment")
    if isinstance(latest_literature, dict):
        latest_literature = LiteratureContextPack.model_validate(latest_literature)
    if isinstance(latest_experiment, dict):
        latest_experiment = ExperimentRunPack.model_validate(latest_experiment)
    skill_guide = render_research_lead_skill_guide(
        skills_runtime.visible_skills("research_lead") if skills_runtime is not None else []
    )
    context_pack = ResearchContextBuilder(store=store, memory_store=memory_store).build_planner_context(
        board=board,
        latest_literature=latest_literature if isinstance(latest_literature, LiteratureContextPack) else None,
        latest_experiment=latest_experiment if isinstance(latest_experiment, ExperimentRunPack) else None,
        session_context=str(request.session_context_text or ""),
    )
    context = build_research_lead_context(
        pack=context_pack,
        research_skill_guide=skill_guide,
    )
    messages = [HumanMessage(content=context)]
    last_error = ""
    action: ResearchLeadOutput | None = None
    for attempt in range(2):
        result = await _invoke_agent_with_step_budget(
            agent=planner_agent,
            messages=messages,
            max_steps=12,
            role="research_lead",
        )
        raw = result.get("structured_response")
        action = raw if isinstance(raw, ResearchLeadOutput) else ResearchLeadOutput.model_validate(raw)
        try:
            validate_research_action(action=action, request=request, board=board)
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt == 1:
                raise
            messages.append(
                HumanMessage(
                    content=(
                        "Repair the previous structured action. "
                        f"Validation error: {last_error}. "
                        "Return one valid ResearchLeadOutput only."
                    )
                )
            )
    if action is None:
        raise RuntimeError(f"research lead failed to produce action: {last_error}")
    return Command(
        goto=action.state,
        update={
            "lead_action": action,
            "board": board,
            "planner_context": context_pack,
        },
    )


async def execute_literature_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    literature_runner: ResearchLiteratureRunner,
    progress_callback=None,
) -> Command:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.run_literature
    if payload is None:
        raise ValueError("execute_literature_node missing payload")
    if progress_callback is not None:
        progress_callback(
            current_phase="executing",
            current_work_label=f"Literature review: {str(payload.query or '').strip() or 'Research literature'}",
        )
    pack = await literature_runner.arun(payload)
    action_id = f"lit_{board.used_literature_queries + 1:03d}"
    ref_path = store.persist_literature_pack(pack, action_id=action_id)
    board = _board_with_updates(
        board,
        cycle_index=board.cycle_index + 1,
        used_literature_queries=board.used_literature_queries + 1,
        latest_literature_ref=ref_path,
        action_refs=list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=action_id,
                kind="literature",
                status="done",
                summary=pack.summary[:240],
                ref_path=ref_path,
                run_id=None,
            )
        ],
    )
    store.save_board(board)
    store.append_action_log(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action_id": action_id,
            "kind": "literature",
            "status": "done",
            "summary": pack.summary,
            "ref_path": ref_path,
        }
    )
    packs = list(state.get("literature_packs") or [])
    packs.append(pack)
    return Command(
        goto="sync_research_state",
        update={
            "board": board,
            "latest_literature": pack,
            "literature_packs": packs,
            "sync_reason": "literature",
        },
    )


async def execute_experiment_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    experiment_runner: ExperimentLaneRunner,
    progress_callback=None,
) -> Command:
    request = ResearchRequest.model_validate(state["request"])
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    brief = action.run_experiment
    if brief is None:
        raise ValueError("execute_experiment_node missing brief")
    if progress_callback is not None:
        progress_callback(
            current_phase="executing",
            current_work_label=f"Experiment: {str(brief.title or brief.goal or '').strip() or 'Run experiment'}",
        )
    pack = await experiment_runner.arun(brief=brief, research_request=request, board=board)
    action_id = pack.experiment_id
    ref_path = store.persist_experiment_pack(pack, action_id=action_id)
    board_update = {
        "cycle_index": board.cycle_index + 1,
        "latest_experiment_ref": ref_path,
        "action_refs": list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=action_id,
                kind="experiment",
                status=pack.status,
                summary=pack.summary[:240],
                ref_path=ref_path,
                run_id=pack.run_id,
            )
        ],
    }
    if pack.lane == "fast":
        board_update["used_fast_runs"] = board.used_fast_runs + 1
    else:
        board_update["used_standard_runs"] = board.used_standard_runs + 1
    board = _board_with_updates(board, **board_update)
    store.save_board(board)
    store.append_action_log(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action_id": action_id,
            "kind": "experiment",
            "status": pack.status,
            "summary": pack.summary,
            "ref_path": ref_path,
            "run_id": pack.run_id,
        }
    )
    packs = list(state.get("experiment_packs") or [])
    packs.append(pack)
    return Command(
        goto="sync_research_state",
        update={
            "board": board,
            "latest_experiment": pack,
            "experiment_packs": packs,
            "sync_reason": "experiment",
        },
    )


async def execute_writer_handoff_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    writer_runner=None,
    progress_callback=None,
) -> dict[str, Any]:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.run_writer
    if payload is None:
        raise ValueError("RunWriter payload is missing")
    if writer_runner is None:
        raise ValueError("RunWriter executor is unavailable")
    if progress_callback is not None:
        progress_callback(current_phase="executing", current_work_label="Run writer request")
    action_id = f"writer_request_{len(board.action_refs) + 1:03d}"
    writing_result = await writer_runner(payload)
    writing_summary = str((writing_result or {}).get("summary") or "").strip()
    writing_run_id = str((writing_result or {}).get("run_id") or "").strip()
    writing_output_path = (
        str((writing_result or {}).get("final_output_path") or "").strip()
        or str((writing_result or {}).get("final_report_path") or "").strip()
    )
    summary = "\n".join(
        line
        for line in [
            board.current_best_answer_md or "",
            writing_summary,
            f"Writer run id: {writing_run_id}" if writing_run_id else "",
            f"Writer output: {writing_output_path}" if writing_output_path else "",
        ]
        if str(line or "").strip()
    ).strip()
    board = _board_with_updates(
        board,
        status="done",
        latest_writer_ref=(writing_output_path or board.latest_writer_ref),
        action_refs=list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=action_id,
                kind="writer",
                status="done",
                summary=(writing_summary or payload.request)[:240],
                ref_path=writing_output_path or "request.json",
                run_id=writing_run_id or None,
            )
        ],
    )
    store.save_board(board)
    store.append_action_log(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action_id": action_id,
            "kind": "writer",
            "status": "done",
            "summary": writing_summary or payload.request,
            "ref_path": writing_output_path or "request.json",
            "run_id": writing_run_id or None,
        }
    )
    return {
        "board": board,
        "status": "done",
        "summary": summary,
        "final_answer": summary,
        "writing_run_id": writing_run_id,
        "writing_result": writing_result or {},
    }


async def finalize_ask_human_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    progress_callback=None,
) -> dict[str, Any]:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.ask_human
    if payload is None:
        raise ValueError("AskHuman payload is missing")
    if progress_callback is not None:
        progress_callback(current_phase="waiting_human", current_work_label="Await human feedback")
    action_id = f"ask_{len(board.action_refs) + 1:03d}"
    record = {
        "action_id": action_id,
        "kind": "ask_human",
        "status": "needs_human",
        "summary": payload.blocking_reason,
        "ref_path": "request.json",
        "run_id": None,
    }
    board = _board_with_updates(
        board,
        status="needs_human",
        open_questions=_dedupe_clean([*board.open_questions, *payload.questions]),
        latest_human_questions=_dedupe_clean(list(payload.questions)),
        human_feedback_summary="",
        action_refs=list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=record["action_id"],
                kind="ask_human",
                status="needs_human",
                summary=payload.blocking_reason,
                ref_path="request.json",
                run_id=None,
            )
        ],
    )
    store.save_board(board)
    store.append_action_log({"ts": datetime.now(timezone.utc).isoformat(), **record, "questions": payload.questions})
    summary_lines = [
        board.current_best_answer_md or "Research campaign needs human input.",
        "",
        "Questions:",
        *[f"- {item}" for item in payload.questions],
        "",
        f"Context: {payload.context}",
    ]
    return {
        "board": board,
        "status": "needs_human",
        "summary": "\n".join(summary_lines).strip(),
        "final_answer": "\n".join(summary_lines).strip(),
    }


async def persist_conclusion_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    progress_callback=None,
) -> Command:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.conclude
    if payload is None:
        raise ValueError("Conclude payload is missing")
    if progress_callback is not None:
        progress_callback(current_phase="finalizing", current_work_label="Conclude research campaign")
    conclusion = ConclusionRecord(
        final_answer_md=board.current_best_answer_md,
        supported_claims=list(board.supported_claims),
        open_questions=list(board.open_questions),
        recommended_next_steps=list(payload.recommended_next_steps),
        confidence=payload.confidence,
        memory_promotion_candidates=list(payload.memory_promotion_candidates),
    )
    ref_path = store.persist_conclusion(conclusion)
    action_id = f"conclusion_{len(board.action_refs) + 1:03d}"
    board = _board_with_updates(
        board,
        status="done",
        memory_promotion_candidates=[item.model_dump() for item in payload.memory_promotion_candidates],
        action_refs=list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=action_id,
                kind="conclusion",
                status="done",
                summary=(board.current_best_answer_md or payload.why_now)[:240],
                ref_path=ref_path,
                run_id=None,
            )
        ],
    )
    store.save_board(board)
    return Command(goto="build_dossier", update={"board": board, "conclusion": conclusion})


async def build_dossier_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    progress_callback=None,
) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="finalizing", current_work_label="Build research dossier")
    board = ResearchBoard.model_validate(state["board"])
    raw_conclusion = state.get("conclusion")
    conclusion = (
        ConclusionRecord.model_validate(raw_conclusion)
        if raw_conclusion is not None
        else store.load_conclusion()
    )
    if conclusion is None:
        raise ValueError("build_dossier requires conclusion")
    literature_packs = [
        item if isinstance(item, LiteratureContextPack) else LiteratureContextPack.model_validate(item)
        for item in list(state.get("literature_packs") or [])
    ]
    if not literature_packs:
        literature_packs = store.load_literature_packs()
    experiment_packs = [
        item if isinstance(item, ExperimentRunPack) else ExperimentRunPack.model_validate(item)
        for item in list(state.get("experiment_packs") or [])
    ]
    if not experiment_packs:
        experiment_packs = store.load_experiment_packs()
    dossier = build_research_dossier(
        board=board,
        conclusion=conclusion,
        literature_packs=literature_packs,
        experiment_packs=experiment_packs,
    )
    dossier_json_path, dossier_file_path = store.persist_dossier(dossier)
    summary = "\n".join(
        [
            conclusion.final_answer_md,
            "",
            f"Dossier: {dossier_file_path}",
            f"Dossier metadata: {dossier_json_path}",
        ]
    ).strip()
    return Command(
        goto="summarize_research",
        update={
            "dossier": dossier,
            "status": "done",
            "summary": summary,
            "final_answer": summary,
        },
    )
def summarize_research_node(state: dict[str, Any]) -> dict[str, Any]:
    summary = str(state.get("summary") or "").strip() or "Research lane finished."
    status = str(state.get("status") or "done")
    return {"summary": summary, "status": status, "final_answer": str(state.get("final_answer") or summary)}
__all__ = [
    "build_dossier_node",
    "execute_experiment_node",
    "execute_literature_node",
    "execute_writer_handoff_node",
    "finalize_ask_human_node",
    "initialize_research_state_node",
    "init_campaign_node",
    "init_research_board",
    "persist_conclusion_node",
    "plan_research_node",
    "sync_research_state_node",
    "summarize_research_node",
    "validate_research_action",
    "validate_research_state_sync",
]
