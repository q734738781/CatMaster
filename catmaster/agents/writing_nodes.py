from __future__ import annotations

import shutil
from pathlib import Path
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from catmaster.agents.nodes import _invoke_agent_with_step_budget
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.skills import (
    CatMasterSkillsRuntime,
    render_section_writer_skill_guide,
    render_write_director_skill_guide,
    render_write_reviewer_skill_guide,
)
from catmaster.runtime.research import ResearchActionRef
from catmaster.runtime.writing import (
    ManuscriptBundleModel,
    SectionDraftModel,
    SectionReviewModel,
    WritingBoard,
    WritingPlanModel,
)
from catmaster.tools.base import workspace_relpath
from catmaster.tools.analysis.polish_academic_prose import polish_academic_prose

from .writing_prompts import (
    SECTION_WRITER_SYSTEM_PROMPT,
    WRITE_DIRECTOR_SYSTEM_PROMPT,
    WRITE_FINALIZER_SYSTEM_PROMPT,
    WRITE_REVIEWER_SYSTEM_PROMPT,
    build_section_writer_context,
    build_write_finalizer_context,
    build_write_director_context,
    build_write_reviewer_context,
)
from .writing_schemas import SectionDraftOutput, SectionReviewOutput, WritingFinalizeOutput, WritingPlanOutput, WritingRequest

_CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{")
_CITE_KEYS_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
_BIB_ENTRY_RE = re.compile(r"@\w+\{([^,]+),")
_MAX_SECTION_REVISIONS = 5


def _load_memory_index_excerpt(*, writing_store) -> str:
    store = MemoryStore.create_default(workspace=writing_store.workspace)
    store.ensure_exists()
    return store.read_index(max_chars=1800)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _achemso_asset_path(name: str) -> Path:
    return _repo_root() / "writing_skills" / "achemso-latex-manuscript" / "assets" / name


def _slug(text: str, *, fallback: str = "section") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or fallback


def _files_rel(*, writing_store, path: Path) -> str:
    return str(path.resolve().relative_to((writing_store.workspace / "files").resolve())).replace("\\", "/")


def _next_pending_index(plan: WritingPlanModel, reviews: dict[str, SectionReviewModel]) -> int:
    for idx, spec in enumerate(plan.section_specs):
        review = reviews.get(spec.section_id)
        if review is None or review.status != "approved":
            return idx
    return len(plan.section_specs)


def _coerce_plan(raw: Any) -> WritingPlanModel:
    if isinstance(raw, WritingPlanModel):
        return raw
    if hasattr(raw, "model_dump"):
        return WritingPlanModel.model_validate(raw.model_dump())
    return WritingPlanModel.model_validate(raw)


def _coerce_draft(raw: Any) -> SectionDraftModel:
    if isinstance(raw, SectionDraftModel):
        return raw
    if hasattr(raw, "model_dump"):
        draft = SectionDraftModel.model_validate(raw.model_dump())
    else:
        draft = SectionDraftModel.model_validate(raw)
    planned = [str(item).strip() for item in draft.planned_figure_ids if str(item).strip()]
    realized = [str(item).strip() for item in draft.realized_figure_refs if str(item).strip()]
    legacy = [str(item).strip() for item in draft.figure_refs if str(item).strip()]
    if not planned and legacy:
        planned = [item for item in legacy if "/" not in item and "." not in Path(item).name]
    if not realized and legacy:
        realized = [item for item in legacy if "/" in item or "." in Path(item).name]
    return draft.model_copy(update={"planned_figure_ids": planned, "realized_figure_refs": realized})


def _build_figure_registry(*, drafts: dict[str, SectionDraftModel]) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for section_id, draft in drafts.items():
        realized = [str(item).strip() for item in draft.realized_figure_refs if str(item).strip()]
        planned = [str(item).strip() for item in draft.planned_figure_ids if str(item).strip()]
        if not realized and not planned:
            continue
        registry.append(
            {
                "section_id": section_id,
                "heading": draft.heading,
                "planned_figure_ids": planned,
                "realized_figure_refs": realized,
            }
        )
    return registry


def _extract_citation_keys(section_tex: str) -> list[str]:
    keys: list[str] = []
    for raw_group in _CITE_KEYS_RE.findall(str(section_tex or "")):
        for key in raw_group.split(","):
            cleaned = str(key).strip()
            if cleaned:
                keys.append(cleaned)
    return list(dict.fromkeys(keys))


def _load_active_bibliography_keys(*, writing_store) -> set[str]:
    bib_path = writing_store.files_root / "references.bib"
    if not bib_path.exists():
        return set()
    text = bib_path.read_text(encoding="utf-8")
    return {str(match).strip() for match in _BIB_ENTRY_RE.findall(text) if str(match).strip()}


def _deterministic_citation_issues(*, writing_store, draft: SectionDraftModel) -> tuple[list[str], list[str]]:
    cited_keys = _extract_citation_keys(draft.section_tex)
    if not cited_keys:
        return [], []
    available_keys = _load_active_bibliography_keys(writing_store=writing_store)
    missing = [key for key in cited_keys if key not in available_keys]
    if not missing:
        return [], []
    bib_path = workspace_relpath(writing_store.files_root / "references.bib")
    notes = [
        "Deterministic citation check failed before reviewer.",
        f"Active bibliography source checked: {bib_path}.",
        "The section cites keys that are not present in the active bibliography and must be fixed before review can approve the section.",
        "Either restore the missing bibliography entries or remove/replace the unsupported citation commands in the section text.",
    ]
    notes.extend(f"Missing bibliography key: {key}" for key in missing)
    return notes, missing


def _finalize_review_after_attempt(
    *,
    board: WritingBoard,
    spec,
    review: SectionReviewOutput,
) -> tuple[WritingBoard, SectionReviewOutput]:
    counts = dict(board.revision_counts)
    attempts = int(counts.get(spec.section_id, 0)) + 1
    counts[spec.section_id] = attempts
    if review.status != "needs_revision":
        return board.model_copy(update={"revision_counts": counts, "status": "drafting"}), review
    if attempts >= _MAX_SECTION_REVISIONS:
        forced_notes = list(review.revision_notes)
        forced_notes.append(
            f"Revision limit reached for `{spec.heading}` after {_MAX_SECTION_REVISIONS} attempts. "
            "Proceed with the current draft and carry the remaining issues forward as unresolved editorial debt."
        )
        forced_review = review.model_copy(update={"status": "approved", "revision_notes": forced_notes})
        return board.model_copy(update={"revision_counts": counts, "status": "drafting"}), forced_review
    return board.model_copy(update={"revision_counts": counts, "status": "drafting"}), review


async def init_writing_node(state: dict[str, Any], *, writing_store, source_store, progress_callback=None) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="planning", current_work_label="Initialize writing run")
    if state.get("resume_mode"):
        goto = str(state.get("resume_goto") or "plan_writing").strip() or "plan_writing"
        return Command(goto=goto, update={"status": state.get("status", "planning")})
    request = WritingRequest.model_validate(state["request"])
    dossier = source_store.load_dossier() if source_store is not None else None
    board = WritingBoard(
        run_id=writing_store.run_id,
        source_campaign_id=request.source_campaign_id,
        status="planning",
        title="",
    )
    writing_store.write_request(request.model_dump())
    writing_store.save_board(board)
    return Command(goto="plan_writing", update={"board": board, "dossier": dossier, "status": "planning"})


async def plan_writing_node(state: dict[str, Any], *, writing_store, source_store, write_director_agent, skills_runtime: CatMasterSkillsRuntime | None, progress_callback=None) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="planning", current_work_label="Plan manuscript structure")
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    dossier = state.get("dossier") or (source_store.load_dossier() if source_store is not None else None)
    dossier_json = dossier.model_dump() if hasattr(dossier, "model_dump") else dossier or {}
    source_board = source_store.load_board() if source_store is not None else None
    skill_guide = render_write_director_skill_guide(
        skills_runtime.visible_skills("write_director", "writing") if skills_runtime is not None else []
    )
    memory_index_excerpt = _load_memory_index_excerpt(writing_store=writing_store)
    context = build_write_director_context(
        request=request.model_dump(),
        dossier=dossier_json,
        board=source_board.model_dump() if source_board is not None else None,
        memory_index_excerpt=memory_index_excerpt,
        latest_literature=[item.model_dump() for item in (source_store.load_literature_packs() if source_store is not None else [])],
        latest_experiments=[item.model_dump() for item in (source_store.load_experiment_packs() if source_store is not None else [])],
        assembly_feedback=str(state.get("assembly_feedback") or "").strip() or None,
        skill_guide=skill_guide,
    )
    result = await _invoke_agent_with_step_budget(
        agent=write_director_agent,
        messages=[HumanMessage(content=context)],
        max_steps=20,
        role="write_director",
    )
    raw = result.get("structured_response")
    if isinstance(raw, WritingPlanOutput):
        plan = raw
    elif hasattr(raw, "model_dump"):
        plan = WritingPlanOutput.model_validate(raw.model_dump())
    else:
        plan = WritingPlanOutput.model_validate(raw)
    board = board.model_copy(update={"title": plan.title, "writing_mode": plan.writing_mode, "status": "drafting", "current_section_index": 0})
    writing_store.persist_plan(plan)
    writing_store.save_board(board)
    if plan.writing_mode == "paper_outline":
        return Command(goto="assemble_manuscript", update={"board": board, "plan": plan, "dossier": dossier, "status": "drafting", "assembly_feedback": None})
    return Command(goto="write_section", update={"board": board, "plan": plan, "dossier": dossier, "status": "drafting", "assembly_feedback": None})


async def write_section_node(state: dict[str, Any], *, writing_store, source_store, section_writer_agent, skills_runtime: CatMasterSkillsRuntime | None, progress_callback=None) -> Command:
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    plan = state.get("plan") or writing_store.load_plan()
    plan = _coerce_plan(plan)
    drafts = {item.section_id: item for item in writing_store.load_section_drafts()}
    reviews = {item.section_id: item for item in writing_store.load_section_reviews()}
    next_index = _next_pending_index(plan, reviews)
    if next_index >= len(plan.section_specs):
        board = board.model_copy(update={"current_section_index": next_index, "status": "reviewing"})
        writing_store.save_board(board)
        return Command(goto="assemble_manuscript", update={"board": board, "plan": plan, "section_drafts": list(drafts.values()), "section_reviews": list(reviews.values())})
    spec = plan.section_specs[next_index]
    if progress_callback is not None:
        progress_callback(current_phase="drafting", current_work_label=f"Write section: {str(spec.heading or spec.section_id).strip()}")
    prior_draft = drafts.get(spec.section_id)
    prior_review = reviews.get(spec.section_id)
    review_notes = list(prior_review.revision_notes) if prior_review is not None else []
    figure_registry = _build_figure_registry(drafts=drafts)
    dossier = state.get("dossier") or (source_store.load_dossier() if source_store is not None else None)
    dossier_json = dossier.model_dump() if hasattr(dossier, "model_dump") else dossier or {}
    skill_guide = render_section_writer_skill_guide(
        skills_runtime.visible_skills("section_writer", "writing") if skills_runtime is not None else []
    )
    memory_index_excerpt = _load_memory_index_excerpt(writing_store=writing_store)
    context = build_section_writer_context(
        request=request.model_dump(),
        plan=plan,
        spec=spec,
        dossier=dossier_json,
        memory_index_excerpt=memory_index_excerpt,
        working_manuscript_root=_files_rel(writing_store=writing_store, path=writing_store.files_root),
        working_sections_dir=_files_rel(writing_store=writing_store, path=writing_store.manuscript_sections_dir),
        working_figures_dir=_files_rel(writing_store=writing_store, path=writing_store.manuscript_figures_dir),
        prior_draft=prior_draft,
        review_notes=review_notes,
        figure_registry=figure_registry,
        skill_guide=skill_guide,
    )
    try:
        result = await _invoke_agent_with_step_budget(
            agent=section_writer_agent,
            messages=[HumanMessage(content=context)],
            max_steps=30,
            role="section_writer",
        )
        raw = result.get("structured_response")
        if isinstance(raw, SectionDraftOutput):
            draft = raw
        elif hasattr(raw, "model_dump"):
            draft = SectionDraftOutput.model_validate(raw.model_dump())
        else:
            draft = SectionDraftOutput.model_validate(raw)
    except Exception as exc:
        feedback = (
            f"Section writing failed for `{spec.heading}`. "
            "Re-plan or retry with a narrower, more robust local brief and avoid repeating the failed path.\n"
            f"Failure detail: {exc.__class__.__name__}: {exc}"
        )
        board = board.model_copy(update={"status": "planning"})
        writing_store.save_board(board)
        return Command(
            goto="plan_writing",
            update={
                "board": board,
                "plan": plan,
                "dossier": dossier,
                "status": "planning",
                "summary": feedback,
                "final_answer": feedback,
                "assembly_feedback": feedback,
            },
        )
    if draft.section_id != spec.section_id:
        draft = draft.model_copy(update={"section_id": spec.section_id, "heading": spec.heading})
    if not list(draft.planned_figure_ids):
        draft = draft.model_copy(update={"planned_figure_ids": list(spec.required_figures)})
    writing_store.persist_section_draft(draft)
    board = board.model_copy(update={"current_section_index": next_index, "status": "reviewing"})
    writing_store.save_board(board)
    return Command(goto="review_section", update={"board": board, "plan": plan, "latest_draft": draft, "dossier": dossier})


async def review_section_node(state: dict[str, Any], *, writing_store, write_reviewer_model: Any, skills_runtime: CatMasterSkillsRuntime | None, progress_callback=None) -> Command:
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    plan = state.get("plan") or writing_store.load_plan()
    plan = _coerce_plan(plan)
    draft = state.get("latest_draft")
    draft = _coerce_draft(draft)
    spec = next((item for item in plan.section_specs if item.section_id == draft.section_id), None)
    if spec is None:
        raise ValueError(f"missing section spec for {draft.section_id}")
    if progress_callback is not None:
        progress_callback(current_phase="reviewing", current_work_label=f"Review section: {str(spec.heading or spec.section_id).strip()}")
    deterministic_notes, deterministic_missing = _deterministic_citation_issues(writing_store=writing_store, draft=draft)
    if deterministic_missing:
        review = SectionReviewOutput(
            section_id=spec.section_id,
            status="needs_revision",
            revision_notes=deterministic_notes,
            unsupported_claims=[],
            missing_citations=deterministic_missing,
        )
        board, review = _finalize_review_after_attempt(board=board, spec=spec, review=review)
        writing_store.persist_section_review(review)
        writing_store.save_board(board)
        return Command(goto="write_section", update={"board": board, "plan": plan, "latest_review": review})
    figure_registry = _build_figure_registry(drafts={item.section_id: item for item in writing_store.load_section_drafts()})
    skill_guide = render_write_reviewer_skill_guide(
        skills_runtime.visible_skills("write_reviewer", "writing") if skills_runtime is not None else []
    )
    context = build_write_reviewer_context(
        request=request.model_dump(),
        spec=spec,
        draft=draft,
        figure_registry=figure_registry,
        skill_guide=skill_guide,
    )
    structured = write_reviewer_model.with_structured_output(SectionReviewOutput)
    raw = await structured.ainvoke(
        [
            SystemMessage(content=WRITE_REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    )
    if isinstance(raw, SectionReviewOutput):
        review = raw
    elif hasattr(raw, "model_dump"):
        review = SectionReviewOutput.model_validate(raw.model_dump())
    else:
        review = SectionReviewOutput.model_validate(raw)
    if review.section_id != spec.section_id:
        review = review.model_copy(update={"section_id": spec.section_id})
    board, review = _finalize_review_after_attempt(board=board, spec=spec, review=review)
    writing_store.persist_section_review(review)
    writing_store.save_board(board)
    return Command(goto="write_section", update={"board": board, "plan": plan, "latest_review": review})


async def assemble_manuscript_node(state: dict[str, Any], *, writing_store, source_store, writing_config, progress_callback=None) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="finalizing", current_work_label="Assemble manuscript")
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    plan = state.get("plan") or writing_store.load_plan()
    plan = _coerce_plan(plan)
    drafts = {item.section_id: item for item in writing_store.load_section_drafts()}
    reviews = {item.section_id: item for item in writing_store.load_section_reviews()}
    bibliography: list[str] = []
    figure_manifest: list[dict[str, Any]] = []
    section_inputs: list[str] = []
    section_graphics_usage: dict[str, list[str]] = {}
    include_bibliography = False
    if plan.writing_mode == "paper_outline":
        section_inputs = _materialize_outline_sections(
            writing_store=writing_store,
            plan=plan,
        )
    else:
        for spec in plan.section_specs:
            draft = drafts.get(spec.section_id)
            if draft is None:
                continue
            section_tex = _resolve_section_tex(writing_store=writing_store, draft=draft)
            if section_tex is not None:
                section_ref, rewritten_section_tex = _materialize_section_tex(
                    writing_store=writing_store,
                    spec=spec,
                    draft=draft,
                    section_tex=section_tex,
                )
                section_inputs.append(section_ref)
                if _section_uses_citations(rewritten_section_tex):
                    include_bibliography = True
                for graphic_ref in _extract_graphics_targets(rewritten_section_tex):
                    section_graphics_usage.setdefault(graphic_ref, []).append(spec.section_id)
            bibliography.extend(draft.citations)
            for ref in draft.realized_figure_refs:
                copied_ref = _materialize_figure_ref(writing_store=writing_store, ref=ref)
                record: dict[str, Any] = {
                    "section_id": draft.section_id,
                    "realized_ref": ref,
                    "planned_figure_ids": list(draft.planned_figure_ids),
                }
                if copied_ref is not None:
                    record["copied_ref"] = copied_ref
                figure_manifest.append(record)
    duplicate_graphics = {
        graphic_ref: section_ids
        for graphic_ref, section_ids in section_graphics_usage.items()
        if len(set(section_ids)) > 1
    }
    if duplicate_graphics:
        details = "; ".join(
            f"{graphic_ref} used in {', '.join(sorted(set(section_ids)))}"
            for graphic_ref, section_ids in sorted(duplicate_graphics.items())
        )
        feedback = (
            "Assembly failed because the same realized figure image was inserted as separate local figures across sections. "
            "Re-plan or revise sections so each realized image is used as one manuscript-level figure object.\n"
            f"Duplicate inclusions: {details}"
        )
        board = board.model_copy(update={"status": "planning"})
        writing_store.save_board(board)
        return Command(
            goto="plan_writing",
            update={
                "board": board,
                "plan": plan,
                "dossier": state.get("dossier"),
                "status": "planning",
                "summary": feedback,
                "final_answer": feedback,
                "assembly_feedback": feedback,
            },
        )
    latex_path = None
    if section_inputs or plan.writing_mode == "paper_outline":
        latex_path = _write_achemso_manuscript(
            writing_store=writing_store,
            plan=plan,
            section_inputs=section_inputs,
            author_name=str(getattr(writing_config, "author_name", "") or "CatMaster"),
            include_bibliography=include_bibliography,
        )
    if latex_path is None:
        raise ValueError("writing assembly produced no TeX manuscript output")
    preferred_path = latex_path
    polish_note = "Academic polish: skipped."
    try:
        _, polish_artifact = polish_academic_prose(
            {
                "source_path": preferred_path,
                "focus": request.request,
            }
        )
        polish_data = polish_artifact.get("data") if isinstance(polish_artifact, dict) else {}
        model_name = str(polish_data.get("model_name") or "").strip()
        polish_note = f"Academic polish: applied in place to {preferred_path}" + (f" via {model_name}" if model_name else "")
    except Exception as exc:
        polish_note = f"Academic polish: skipped ({exc})"
    bundle = ManuscriptBundleModel(
        source_campaign_id=request.source_campaign_id,
        writing_mode=plan.writing_mode,
        title=plan.title,
        ordered_sections=[spec.heading for spec in plan.section_specs],
        bibliography_shortlist=list(dict.fromkeys(bibliography))[:20] if include_bibliography else [],
        figure_manifest=figure_manifest,
        final_manuscript_path=preferred_path,
        final_latex_path=latex_path,
    )
    bundle_path = writing_store.persist_bundle(bundle)
    board = board.model_copy(update={"status": "finalizing", "latest_manuscript_ref": preferred_path, "latest_bundle_ref": bundle_path})
    writing_store.save_board(board)
    summary = "\n".join(
        [
            f"Writing source campaign: {request.source_campaign_id or '(none)'}",
            f"Title: {plan.title}",
            f"Primary manuscript: {preferred_path}",
            f"LaTeX manuscript: {latex_path or '(none)'}",
            polish_note,
            f"Bundle: {bundle_path}",
        ]
    ).strip()
    return Command(
        goto="finalize_writing",
        update={
            "board": board,
            "status": "finalizing",
            "summary": summary,
            "final_answer": summary,
            "bundle": bundle,
        },
    )


async def finalize_writing_node(
    state: dict[str, Any],
    *,
    writing_store,
    source_store,
    write_finalizer_agent,
    skills_runtime: CatMasterSkillsRuntime | None,
    progress_callback=None,
) -> Command:
    if progress_callback is not None:
        progress_callback(current_phase="finalizing", current_work_label="Compile and finalize manuscript")
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    plan = state.get("plan") or writing_store.load_plan()
    plan = _coerce_plan(plan)
    bundle = state.get("bundle") or writing_store.load_bundle()
    if bundle is None:
        raise ValueError("missing manuscript bundle for finalization")
    bundle_json = bundle.model_dump() if hasattr(bundle, "model_dump") else bundle
    skill_guide = render_write_director_skill_guide(
        skills_runtime.visible_skills("write_director", "writing") if skills_runtime is not None else []
    )
    context = build_write_finalizer_context(
        request=request.model_dump(),
        plan=plan,
        bundle=bundle_json,
        skill_guide=skill_guide,
    )
    result = await _invoke_agent_with_step_budget(
        agent=write_finalizer_agent,
        messages=[HumanMessage(content=context)],
        max_steps=12,
        role="write_director",
    )
    raw = result.get("structured_response")
    if isinstance(raw, WritingFinalizeOutput):
        finalized = raw
    elif hasattr(raw, "model_dump"):
        finalized = WritingFinalizeOutput.model_validate(raw.model_dump())
    else:
        finalized = WritingFinalizeOutput.model_validate(raw)
    final_path = str(finalized.final_latex_path or getattr(bundle, "final_latex_path", None) or bundle.final_manuscript_path).strip()
    board = board.model_copy(update={"status": "done", "latest_manuscript_ref": final_path or board.latest_manuscript_ref})
    writing_store.save_board(board)
    try:
        source_board = source_store.load_board() if source_store is not None else None
        if source_board is not None:
            source_board.latest_writer_ref = final_path or source_board.latest_writer_ref
            source_board.action_refs = list(source_board.action_refs) + [
                ResearchActionRef(
                    action_id=f"writer_{len(source_board.action_refs) + 1:03d}",
                    kind="writer",
                    status="done",
                    summary=plan.title,
                    ref_path=final_path or bundle.final_manuscript_path,
                    run_id=None,
                )
            ]
            source_store.save_board(source_board)
    except Exception:
        pass
    summary = "\n".join(
        [
            str(state.get("summary") or "").strip(),
            str(finalized.summary or "").strip(),
            *[f"- {item}" for item in finalized.compile_notes if str(item).strip()],
        ]
    ).strip()
    return Command(
        goto="summarize_writing",
        update={
            "board": board,
            "status": "done",
            "summary": summary,
            "final_answer": summary,
        },
    )
def _resolve_section_tex(*, writing_store, draft) -> str | None:
    explicit = _read_first_explicit_latex_artifact(
        writing_store=writing_store,
        refs=[str(item) for item in draft.latex_artifact_refs if str(item).strip()],
    )
    if explicit is not None:
        return explicit
    inline = str(draft.section_tex or "").strip()
    return inline or None


def _materialize_section_tex(*, writing_store, spec, draft=None, section_tex: str) -> tuple[str, str]:
    manuscript_sections_dir = writing_store.manuscript_sections_dir
    manuscript_sections_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(spec.section_id, fallback='section')}_{_slug(spec.heading, fallback='body')}.tex"
    path = manuscript_sections_dir / filename
    rewritten = _rewrite_section_graphics(
        writing_store=writing_store,
        section_tex=str(section_tex or "").strip(),
        realized_figure_refs=[str(item) for item in getattr(draft, "realized_figure_refs", []) if str(item).strip()],
    )
    path.write_text(rewritten + "\n", encoding="utf-8")
    return f"sections/{filename}", rewritten


def _materialize_outline_sections(*, writing_store, plan) -> list[str]:
    refs: list[str] = []
    for spec in plan.section_specs:
        body = "\n".join(
            [
                f"\\section*{{{spec.heading}}}",
                f"% Purpose: {spec.purpose}",
                "",
            ]
        ).strip()
        section_ref, _ = _materialize_section_tex(writing_store=writing_store, spec=spec, section_tex=body)
        refs.append(section_ref)
    return refs


def _section_uses_citations(section_tex: str) -> bool:
    return bool(_CITE_RE.search(str(section_tex or "")))


def _write_achemso_manuscript(*, writing_store, plan, section_inputs: list[str], author_name: str, include_bibliography: bool) -> str:
    manuscript_dir = writing_store.files_root
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    if include_bibliography:
        bib_template_path = _achemso_asset_path("achemso-demo.bib")
        bibliography_path = manuscript_dir / "references.bib"
        if not bibliography_path.exists():
            shutil.copy2(bib_template_path, bibliography_path)
    title = str(plan.title or "").strip() or "Untitled manuscript"
    body_inputs = "\n".join(f"\\input{{{ref}}}" for ref in section_inputs).strip() or "% No sections assembled"
    abstract_text = str(plan.abstract_md or "").strip()
    abstract_block = ""
    if plan.writing_mode != "section_draft" and abstract_text:
        abstract_block = "\n".join(
            [
                "\\begin{abstract}",
                abstract_text,
                "\\end{abstract}",
                "",
            ]
        )
    rendered = "\n".join(
        [
            "% Auto-generated by CatMaster writing lane.",
            "\\documentclass[journal=jacsat,manuscript=article]{achemso}",
            "",
            "\\usepackage{chemformula}",
            "\\usepackage[T1]{fontenc}",
            "",
            f"\\author{{{author_name}}}",
            "\\affiliation{CatMaster}",
            f"\\title{{{title}}}",
            "\\abbreviations{}",
            "\\keywords{}",
            "",
            "\\begin{document}",
            "",
            abstract_block.rstrip(),
            body_inputs,
            "",
            "\\begin{acknowledgement}",
            "\\end{acknowledgement}",
            "",
            "\\begin{suppinfo}",
            "\\end{suppinfo}",
            "",
            "\\bibliography{references}" if include_bibliography else "",
            "\\end{document}",
        ]
    ).strip()
    return writing_store.write_manuscript("MANUSCRIPT.tex", rendered + "\n")


def _read_first_explicit_latex_artifact(*, writing_store, refs: list[str]) -> str | None:
    seen: set[str] = set()
    for ref in refs:
        cleaned = str(ref or "").strip()
        if not cleaned or cleaned in seen or not cleaned.endswith(".tex"):
            continue
        seen.add(cleaned)
        path = writing_store.workspace / "files" / cleaned
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def _rewrite_section_graphics(*, writing_store, section_tex: str, realized_figure_refs: list[str]) -> str:
    manuscript_dir = writing_store.files_root
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    def _replace(match: re.Match[str]) -> str:
        options = match.group(1) or ""
        raw_ref = str(match.group(2) or "").strip()
        copied_name = _copy_workspace_graphic_for_manuscript(
            writing_store=writing_store,
            ref=raw_ref,
            fallback_refs=realized_figure_refs,
        )
        target_ref = copied_name or raw_ref
        if options:
            return f"\\includegraphics[{options}]{{{target_ref}}}"
        return f"\\includegraphics{{{target_ref}}}"

    return re.sub(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", _replace, section_tex)


def _extract_graphics_targets(section_tex: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", str(section_tex or "")):
        ref = str(match.group(1) or "").strip()
        if ref:
            refs.append(ref)
    return refs


def _copy_workspace_graphic_for_manuscript(*, writing_store, ref: str, fallback_refs: list[str] | None = None) -> str | None:
    cleaned = str(ref or "").strip()
    if not cleaned:
        return None
    source = Path(cleaned)
    if not source.is_absolute():
        candidates = [
            writing_store.workspace / "files" / cleaned,
            writing_store.files_root / cleaned,
        ]
        if cleaned.startswith("manuscript/"):
            candidates.append(writing_store.workspace / "files" / cleaned)
        else:
            candidates.append(writing_store.workspace / "files" / "manuscript" / cleaned)
        source = next((path for path in candidates if path.exists()), None)
        if source is None and fallback_refs:
            raw_name = Path(cleaned).name
            for fallback in fallback_refs:
                candidate = writing_store.workspace / "files" / str(fallback).strip()
                if candidate.exists() and candidate.is_file() and candidate.name == raw_name:
                    source = candidate
                    break
        if source is None:
            source = candidates[0]
    if not source.exists() or not source.is_file():
        return None
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".pdf", ".eps"}:
        return None
    dest = writing_store.manuscript_figures_dir / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return f"figures/{source.name}"


def _materialize_figure_ref(*, writing_store, ref: str) -> str | None:
    cleaned = str(ref or "").strip()
    if not cleaned:
        return None
    copied_name = _copy_workspace_graphic_for_manuscript(writing_store=writing_store, ref=cleaned)
    if copied_name is None:
        return None
    dest_name = Path(copied_name).name
    dest = writing_store.manuscript_figures_dir / dest_name
    return str(dest.resolve().relative_to((writing_store.workspace / "files").resolve())).replace("\\", "/")


def summarize_writing_node(state: dict[str, Any]) -> dict[str, Any]:
    summary = str(state.get("summary") or "").strip() or "Writing lane finished."
    status = str(state.get("status") or "done")
    return {"summary": summary, "status": status, "final_answer": str(state.get("final_answer") or summary)}


__all__ = [
    "assemble_manuscript_node",
    "finalize_writing_node",
    "init_writing_node",
    "plan_writing_node",
    "review_section_node",
    "summarize_writing_node",
    "write_section_node",
]
