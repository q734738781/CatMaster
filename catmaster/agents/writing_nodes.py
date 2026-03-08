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
from catmaster.tools.analysis.polish_academic_prose import polish_academic_prose

from .writing_prompts import (
    SECTION_WRITER_SYSTEM_PROMPT,
    WRITE_DIRECTOR_SYSTEM_PROMPT,
    WRITE_REVIEWER_SYSTEM_PROMPT,
    build_section_writer_context,
    build_write_director_context,
    build_write_reviewer_context,
)
from .writing_schemas import SectionDraftOutput, SectionReviewOutput, WritingPlanOutput, WritingRequest


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
        return SectionDraftModel.model_validate(raw.model_dump())
    return SectionDraftModel.model_validate(raw)


async def init_writing_node(state: dict[str, Any], *, writing_store, source_store) -> Command:
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


async def plan_writing_node(state: dict[str, Any], *, writing_store, source_store, write_director_agent, skills_runtime: CatMasterSkillsRuntime | None) -> Command:
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
        skill_guide=skill_guide,
    )
    result = await _invoke_agent_with_step_budget(
        agent=write_director_agent,
        messages=[SystemMessage(content=WRITE_DIRECTOR_SYSTEM_PROMPT), HumanMessage(content=context)],
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
        return Command(goto="assemble_manuscript", update={"board": board, "plan": plan, "dossier": dossier, "status": "drafting"})
    return Command(goto="write_section", update={"board": board, "plan": plan, "dossier": dossier, "status": "drafting"})


async def write_section_node(state: dict[str, Any], *, writing_store, source_store, section_writer_agent, skills_runtime: CatMasterSkillsRuntime | None) -> Command:
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
    prior_draft = drafts.get(spec.section_id)
    prior_review = reviews.get(spec.section_id)
    review_notes = list(prior_review.revision_notes) if prior_review is not None else []
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
        prior_draft=prior_draft,
        review_notes=review_notes,
        skill_guide=skill_guide,
    )
    result = await _invoke_agent_with_step_budget(
        agent=section_writer_agent,
        messages=[SystemMessage(content=SECTION_WRITER_SYSTEM_PROMPT), HumanMessage(content=context)],
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
    if draft.section_id != spec.section_id:
        draft = draft.model_copy(update={"section_id": spec.section_id, "heading": spec.heading})
    writing_store.persist_section_draft(draft)
    board = board.model_copy(update={"current_section_index": next_index, "status": "reviewing"})
    writing_store.save_board(board)
    return Command(goto="review_section", update={"board": board, "plan": plan, "latest_draft": draft, "dossier": dossier})


async def review_section_node(state: dict[str, Any], *, writing_store, write_reviewer_model: Any, skills_runtime: CatMasterSkillsRuntime | None) -> Command:
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    plan = state.get("plan") or writing_store.load_plan()
    plan = _coerce_plan(plan)
    draft = state.get("latest_draft")
    draft = _coerce_draft(draft)
    spec = next((item for item in plan.section_specs if item.section_id == draft.section_id), None)
    if spec is None:
        raise ValueError(f"missing section spec for {draft.section_id}")
    skill_guide = render_write_reviewer_skill_guide(
        skills_runtime.visible_skills("write_reviewer", "writing") if skills_runtime is not None else []
    )
    context = build_write_reviewer_context(
        request=request.model_dump(),
        spec=spec,
        draft=draft,
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
    writing_store.persist_section_review(review)
    counts = dict(board.revision_counts)
    if review.status == "needs_revision":
        attempts = int(counts.get(spec.section_id, 0)) + 1
        counts[spec.section_id] = attempts
        board = board.model_copy(update={"revision_counts": counts, "status": "drafting"})
        writing_store.save_board(board)
        if attempts < 2:
            return Command(goto="write_section", update={"board": board, "plan": plan, "latest_review": review})
    board = board.model_copy(update={"status": "drafting"})
    writing_store.save_board(board)
    return Command(goto="write_section", update={"board": board, "plan": plan, "latest_review": review})


async def assemble_manuscript_node(state: dict[str, Any], *, writing_store, source_store, writing_config) -> dict[str, Any]:
    request = WritingRequest.model_validate(state["request"])
    board = WritingBoard.model_validate(state["board"])
    plan = state.get("plan") or writing_store.load_plan()
    plan = _coerce_plan(plan)
    drafts = {item.section_id: item for item in writing_store.load_section_drafts()}
    reviews = {item.section_id: item for item in writing_store.load_section_reviews()}
    bibliography: list[str] = []
    figure_manifest: list[dict[str, Any]] = []
    section_inputs: list[str] = []
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
                section_inputs.append(
                    _materialize_section_tex(
                        writing_store=writing_store,
                        spec=spec,
                        section_tex=section_tex,
                    )
                )
            bibliography.extend(draft.citations)
            for ref in draft.figure_refs:
                copied_ref = _materialize_figure_ref(writing_store=writing_store, ref=ref)
                record: dict[str, Any] = {"section_id": draft.section_id, "ref": ref}
                if copied_ref is not None:
                    record["copied_ref"] = copied_ref
                figure_manifest.append(record)
    latex_path = None
    if section_inputs or plan.writing_mode == "paper_outline":
        latex_path = _write_achemso_manuscript(
            writing_store=writing_store,
            plan=plan,
            section_inputs=section_inputs,
            author_name=str(getattr(writing_config, "author_name", "") or "CatMaster"),
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
        bibliography_shortlist=list(dict.fromkeys(bibliography))[:20],
        figure_manifest=figure_manifest,
        final_manuscript_path=preferred_path,
        final_latex_path=latex_path,
    )
    bundle_path = writing_store.persist_bundle(bundle)
    board = board.model_copy(update={"status": "done", "latest_manuscript_ref": preferred_path, "latest_bundle_ref": bundle_path})
    writing_store.save_board(board)
    try:
        source_board = source_store.load_board() if source_store is not None else None
        if source_board is not None:
            source_board.latest_writer_ref = preferred_path
            source_board.action_refs = list(source_board.action_refs) + [
                ResearchActionRef(
                    action_id=f"writer_{len(source_board.action_refs) + 1:03d}",
                    kind="writer",
                    status="done",
                    summary=plan.title,
                    ref_path=preferred_path,
                    run_id=None,
                )
            ]
            source_store.save_board(source_board)
    except Exception:
        pass
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
    return {
        "board": board,
        "status": "done",
        "summary": summary,
        "final_answer": summary,
}
def _resolve_section_tex(*, writing_store, draft) -> str | None:
    explicit = _read_first_explicit_latex_artifact(
        writing_store=writing_store,
        refs=[str(item) for item in draft.latex_artifact_refs if str(item).strip()],
    )
    if explicit is not None:
        return explicit
    inline = str(draft.section_tex or "").strip()
    return inline or None


def _materialize_section_tex(*, writing_store, spec, section_tex: str) -> str:
    manuscript_sections_dir = writing_store.files_root / "manuscript" / "sections"
    manuscript_sections_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(spec.section_id, fallback='section')}_{_slug(spec.heading, fallback='body')}.tex"
    path = manuscript_sections_dir / filename
    path.write_text(str(section_tex).strip() + "\n", encoding="utf-8")
    return f"sections/{filename}"


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
        refs.append(_materialize_section_tex(writing_store=writing_store, spec=spec, section_tex=body))
    return refs


def _write_achemso_manuscript(*, writing_store, plan, section_inputs: list[str], author_name: str) -> str:
    template_path = _achemso_asset_path("achemso-demo.tex")
    bib_template_path = _achemso_asset_path("achemso-demo.bib")
    template = template_path.read_text(encoding="utf-8")
    manuscript_dir = writing_store.files_root / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    bibliography_path = manuscript_dir / "references.bib"
    shutil.copy2(bib_template_path, bibliography_path)

    abstract_text = str(plan.abstract_md or "Abstract not yet provided.").strip()
    main_body = "\n\n".join([f"\\input{{{ref}}}" for ref in section_inputs]) or "% No sections assembled"
    author_block = "\n".join(
        [
            f"\\author{{{author_name}}}",
            "\\affiliation{CatMaster}",
            "",
        ]
    )
    rendered = template
    rendered = re.sub(
        r"\\author\{.*?\\title",
        lambda _: author_block + "\\title",
        rendered,
        count=1,
        flags=re.DOTALL,
    )
    rendered = re.sub(
        r"\\title(?:\[.*?\])?\s*\{.*?\}",
        lambda _: f"\\title{{{plan.title}}}",
        rendered,
        count=1,
        flags=re.DOTALL,
    )
    rendered = re.sub(r"\\abbreviations\{.*?\}", lambda _: "\\abbreviations{}", rendered, count=1, flags=re.DOTALL)
    rendered = re.sub(r"\\keywords\{.*?\}", lambda _: "\\keywords{}", rendered, count=1, flags=re.DOTALL)
    rendered = _replace_environment_body(rendered, "tocentry", "Graphical TOC entry not provided.")
    rendered = _replace_environment_body(rendered, "abstract", abstract_text)
    rendered = re.sub(
        r"\\section\{Introduction\}.*?\\begin\{acknowledgement\}",
        lambda _: main_body + "\n\n\\begin{acknowledgement}",
        rendered,
        count=1,
        flags=re.DOTALL,
    )
    rendered = _replace_environment_body(rendered, "acknowledgement", "")
    rendered = _replace_environment_body(rendered, "suppinfo", "")
    rendered = re.sub(
        r"\\bibliography\{.*?\}",
        lambda _: "\\bibliography{references}",
        rendered,
        count=1,
        flags=re.DOTALL,
    )
    return writing_store.write_manuscript("MANUSCRIPT.tex", rendered.strip() + "\n")


def _replace_environment_body(text: str, env: str, body: str) -> str:
    replacement = f"\\\\begin{{{env}}}\n{body}\n\\\\end{{{env}}}"
    pattern = rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}"
    return re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)


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


def _materialize_figure_ref(*, writing_store, ref: str) -> str | None:
    cleaned = str(ref or "").strip()
    if not cleaned:
        return None
    source = writing_store.workspace / "files" / cleaned
    if not source.exists() or not source.is_file():
        return None
    dest = writing_store.files_root / "manuscript" / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return str(dest.resolve().relative_to((writing_store.workspace / "files").resolve())).replace("\\", "/")


def summarize_writing_node(state: dict[str, Any]) -> dict[str, Any]:
    summary = str(state.get("summary") or "").strip() or "Writing lane finished."
    status = str(state.get("status") or "done")
    return {"summary": summary, "status": status, "final_answer": str(state.get("final_answer") or summary)}


__all__ = [
    "assemble_manuscript_node",
    "init_writing_node",
    "plan_writing_node",
    "review_section_node",
    "summarize_writing_node",
    "write_section_node",
]
