from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

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
    ResearchContextReviewer,
    ResearchDossier,
    ResearchPlannerContextPack,
    ResearchStore,
    ResearchLiteratureRunner,
)
from catmaster.runtime.research.dossier import build_research_dossier
from catmaster.runtime.skills import CatMasterSkillsRuntime, render_research_lead_skill_guide

from .research_prompts import (
    RESEARCH_LEAD_SYSTEM_PROMPT,
    build_research_lead_context,
)
from .research_schemas import (
    ResearchLeadOutput,
    ResearchRequest,
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
        if request.writing_mode == "none":
            raise ValueError("RunWriter requires writing_mode != none")
        if action.run_writer is None:
            raise ValueError("missing writer payload")
    if request.exploration_policy == "anchored" and action.new_hypotheses:
        raise ValueError("anchored policy forbids new hypotheses")
    if request.exploration_policy == "local_expand":
        for item in action.new_hypotheses:
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
    action: ResearchLeadOutput,
    request: ResearchRequest,
) -> ResearchBoard:
    hypotheses = [item.model_copy(deep=True) for item in board.hypotheses]
    by_id = {item.hypothesis_id: item for item in hypotheses}
    for update in action.hypothesis_updates:
        record = by_id.get(update.hypothesis_id)
        if record is None:
            continue
        record.status = update.status
        record.note = update.note
    for proposal in action.new_hypotheses:
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
    return board.model_copy(update={"hypotheses": hypotheses})


async def init_campaign_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
) -> Command:
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
    history_context_summary = str(state.get("history_context_summary") or "").strip()
    if history_context_summary:
        board = _board_with_updates(board, history_context_summary=history_context_summary)
    store.write_request(request)
    store.save_board(board)
    return Command(goto="plan_research", update={"board": board, "status": "running"})


async def plan_research_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    planner_model: Any,
    memory_store: MemoryStore,
    history_reader: Any,
    project_id: str,
    skills_runtime: CatMasterSkillsRuntime | None,
) -> Command:
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
        history_summary=str(state.get("history_context_summary") or board.history_context_summary or ""),
    )
    review_pack = None
    if history_reader is not None:
        review_pack = await ResearchContextReviewer(
            history_reader=history_reader,
            store=store,
            memory_store=memory_store,
            project_id=project_id,
        ).areview(board=board)
        context_pack = context_pack.model_copy(update={"context_review_md": ResearchContextReviewer.render(review_pack)})
    context = build_research_lead_context(
        pack=context_pack,
        research_skill_guide=skill_guide,
    )
    structured = planner_model.with_structured_output(ResearchLeadOutput)
    messages = [
        SystemMessage(content=RESEARCH_LEAD_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]
    last_error = ""
    action: ResearchLeadOutput | None = None
    for attempt in range(2):
        response = await structured.ainvoke(messages)
        action = response if isinstance(response, ResearchLeadOutput) else ResearchLeadOutput.model_validate(response)
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
    board = _apply_hypothesis_updates(board=board, action=action, request=request)
    if action.current_best_answer_md:
        board = _board_with_updates(board, current_best_answer_md=action.current_best_answer_md)
    return Command(
        goto=action.state,
        update={
            "lead_action": action,
            "board": board,
            "planner_context": context_pack,
            "context_review": review_pack,
        },
    )


async def execute_literature_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    literature_runner: ResearchLiteratureRunner,
) -> Command:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.run_literature
    if payload is None:
        raise ValueError("execute_literature_node missing payload")
    pack = await literature_runner.arun(payload)
    action_id = f"lit_{board.used_literature_queries + 1:03d}"
    ref_path = store.persist_literature_pack(pack, action_id=action_id)
    board = _board_with_updates(
        board,
        cycle_index=board.cycle_index + 1,
        used_literature_queries=board.used_literature_queries + 1,
        latest_literature_ref=ref_path,
        current_best_answer_md=action.current_best_answer_md or board.current_best_answer_md,
        open_questions=list(dict.fromkeys([*board.open_questions, *pack.followup_questions]))[:20],
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
    return Command(goto="plan_research", update={"board": board, "latest_literature": pack, "literature_packs": packs})


async def execute_experiment_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
    experiment_runner: ExperimentLaneRunner,
) -> Command:
    request = ResearchRequest.model_validate(state["request"])
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    brief = action.run_experiment
    if brief is None:
        raise ValueError("execute_experiment_node missing brief")
    pack = await experiment_runner.arun(brief=brief, research_request=request, board=board)
    action_id = pack.experiment_id
    ref_path = store.persist_experiment_pack(pack, action_id=action_id)
    board_update = {
        "cycle_index": board.cycle_index + 1,
        "latest_experiment_ref": ref_path,
        "current_best_answer_md": action.current_best_answer_md or board.current_best_answer_md,
        "open_questions": list(dict.fromkeys([*board.open_questions, *pack.open_questions]))[:20],
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
    return Command(goto="plan_research", update={"board": board, "latest_experiment": pack, "experiment_packs": packs})


async def execute_writer_handoff_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
) -> dict[str, Any]:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.run_writer
    if payload is None:
        raise ValueError("RunWriter payload is missing")
    action_id = f"writer_request_{len(board.action_refs) + 1:03d}"
    summary = "\n".join(
        [
            action.current_best_answer_md or board.current_best_answer_md or "Writer handoff requested.",
            "",
            "Writer handoff requested from research lead.",
            f"Reason: {payload.why_now}",
            "Scope: write from existing evidence; do not launch new expensive calculations.",
        ]
    ).strip()
    board = _board_with_updates(
        board,
        status="done",
        current_best_answer_md=action.current_best_answer_md or board.current_best_answer_md,
        action_refs=list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=action_id,
                kind="writer",
                status="done",
                summary=payload.why_now[:240],
                ref_path="request.json",
                run_id=None,
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
            "summary": payload.why_now,
            "ref_path": "request.json",
        }
    )
    return {"board": board, "status": "done", "summary": summary, "final_answer": summary}


async def finalize_ask_human_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
) -> dict[str, Any]:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.ask_human
    if payload is None:
        raise ValueError("AskHuman payload is missing")
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
        action.current_best_answer_md or board.current_best_answer_md or "Research campaign needs human input.",
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
) -> Command:
    board = ResearchBoard.model_validate(state["board"])
    action = ResearchLeadOutput.model_validate(state["lead_action"])
    payload = action.conclude
    if payload is None:
        raise ValueError("Conclude payload is missing")
    ref_path = store.persist_conclusion(payload)
    action_id = f"conclusion_{len(board.action_refs) + 1:03d}"
    board = _board_with_updates(
        board,
        status="done",
        current_best_answer_md=payload.final_answer_md,
        supported_claims=list(payload.supported_claims),
        open_questions=list(payload.open_questions),
        memory_promotion_candidates=[item.model_dump() for item in payload.memory_promotion_candidates],
        action_refs=list(board.action_refs)
        + [
            ResearchActionRef(
                action_id=action_id,
                kind="conclusion",
                status="done",
                summary=payload.final_answer_md[:240],
                ref_path=ref_path,
                run_id=None,
            )
        ],
    )
    store.save_board(board)
    conclusion = ConclusionRecord.model_validate(payload.model_dump())
    return Command(goto="build_dossier", update={"board": board, "conclusion": conclusion})


async def build_dossier_node(
    state: dict[str, Any],
    *,
    store: ResearchStore,
) -> Command:
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
    if ResearchRequest.model_validate(state["request"]).writing_mode == "none":
        return Command(
            goto="summarize_research",
            update={"dossier": dossier, "status": "done", "summary": summary, "final_answer": summary},
        )
    return Command(
        goto="summarize_research",
        update={
            "dossier": dossier,
            "status": "done",
            "summary": summary,
            "final_answer": summary,
            "writing_requested": True,
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
    "init_campaign_node",
    "init_research_board",
    "persist_conclusion_node",
    "plan_research_node",
    "summarize_research_node",
    "validate_research_action",
]
