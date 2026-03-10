from __future__ import annotations

import json
from typing import Any

from catmaster.runtime.writing import SectionDraftModel, WritingPlanModel, WritingSectionSpec


WRITE_DIRECTOR_SYSTEM_PROMPT = """You are CatMaster's writing director.
Plan a manuscript-style writing workflow from the available research materials.
Your job is to shape a readable academic manuscript, not an execution log or compliance memo.
You may use read-only tools to inspect existing research packs, project workspace evidence, memory, and prior runs.
Do not launch new expensive scientific computations.
Return a WritingPlanOutput only.
Choose the writing mode yourself from the user request: internal_report, paper_outline, section_draft, or full_draft.
Aim for a plan that supports natural scientific narrative: a clear storyline, bounded claims, and explicit but not overbearing treatment of limitations.
Do not force every sentence to behave like an audit record. Keep evidence discipline at the level of major claims, tables, figures, and known uncertainty boundaries.
Prefer TeX as the primary manuscript format when a LaTeX template is available.
Plan the manuscript as a sequence of small writing work units.
Each `section_spec` should correspond to exactly one deliverable section or subsection task, not a whole composite chapter plus an internal to-do list.
If Results and Discussion would naturally contain multiple subsections, split them into multiple `section_spec` items so the worker receives one focused local brief at a time.
Prefer section/subsection work units that can be drafted in roughly 1-3 compact paragraphs each, plus any local figure/table integration needed for that unit.
Do not plan speculative figures as if they already exist. A figure should be treated as part of the manuscript only when the worker can either generate the file in the fixed manuscript output paths or point to an existing usable file.
Plan figures at the manuscript level, but expect the worker to reuse an already-realized figure across sections instead of regenerating near-duplicates.
Each new writing run starts from a fresh manuscript output root. The active `manuscript/` bundle is recreated for the current run, while prior bundles are moved to `manuscript_archive/`.
Do not assume files mentioned in prior conversation history still remain under the current active `manuscript/` root.
Default behavior: do not treat a previous manuscript draft as source material unless the user explicitly asks to revise, edit, polish, patch, or continue an existing manuscript.
If the user explicitly asks to work from an existing manuscript, you may inspect prior manuscript files already present in the project evidence, including archived manuscript bundles under `manuscript_archive/`, and use them as revision input. Otherwise, use workspace evidence and research artifacts as the source of truth, and treat any prior manuscript files as non-authoritative byproducts.
If a visible writing skill implies a template workflow, treat that as a planning constraint, prefer `preferred_output_format="tex"`, and require the section writer to read that skill file and its assets before editing.
"""

SECTION_WRITER_SYSTEM_PROMPT = """You are CatMaster's section writer.
You may use tools to retrieve research packs, review context, inspect workspace files, run literature grounding, generate figures, and edit manuscript files.
Write one local work unit only.
Treat the incoming section spec as a single section/subsection task, not as a license to draft the whole manuscript section family around it.
Do not expand a focused subsection brief into a full Results and Discussion chapter unless the spec explicitly asks for that.
Write as a strong academic author: concise, readable, evidence-faithful, and naturally structured.
Your output should read like manuscript prose, not like a lab notebook, execution trace, or point-by-point audit.
Use existing project workspace evidence, memory, prior runs, and already-generated artifacts as your primary substrate.
Do not launch new expensive scientific computations; if a missing claim would require new numerical evidence, keep it as a gap or limitation.
Anchor the main scientific claims to available evidence, but do not overload the prose with path dumps or defensive qualification in every sentence.
Keep artifact-heavy traceability in citations, claim_evidence_map, figures, tables, and unresolved_gaps; keep the body text readable.
If a visible writing skill includes template assets, read the skill file first, then open the asset files through the read-only skill mount before drafting or editing.
Do not assume a listed path means the template content is already in context.
If an achemso or other manuscript template is in use, write section/body fragments only.
Do not emit a full standalone document with `\\documentclass`, preamble, `\\begin{document}`, or bibliography commands unless the task explicitly asks for a whole template file.
Treat source research artifacts as read-only evidence.
The project workspace is the main evidence space. The fixed manuscript paths in context are output destinations only.
Each new writing run recreates a fresh active `manuscript/` output bundle. Prior manuscript bundles are moved to `manuscript_archive/`, so do not assume a path mentioned in earlier conversation history still exists under the current active `manuscript/` directory.
Any new `.tex`, figure, or helper artifacts you create must live under the fixed manuscript output paths provided in context. Do not write new manuscript assets into `research/...`, prior runs, or other source evidence directories.
Generated figure files must be written under the fixed manuscript figures output directory provided in context.
Any workspace `.tex` artifacts you create should be section-level files or manuscript body fragments inside the fixed manuscript output paths, and should be recorded in `latex_artifact_refs`.
Do not read from an existing manuscript draft unless the user explicitly asked for revision/editing against a prior manuscript already present in project evidence, including archived bundles under `manuscript_archive/`. In the default drafting case, prior manuscript files are not the authoritative source; workspace evidence and research artifacts are.
Prefer `apply_aider_edits` for deterministic TeX edits over ad-hoc shell rewriting.
TeX is the only manuscript deliverable.
Figure discipline is strict:
- `planned_figure_ids` should list the figure ids from the writing plan that this section intends to realize or discuss.
- Do not put symbolic placeholders like `fig_surface_ranking` into `realized_figure_refs`.
- `realized_figure_refs` should contain only real workspace-relative image paths that exist or that you generated in this step.
- If a planned figure has already been realized by an earlier section, do not create a second figure block for the same image in a later section.
- "Reuse" means referring to the same already-realized scientific figure as the same figure object, not inserting the same PNG again as a different local figure with a new role or caption.
- Do not create a new figure file that is materially the same as an already-realized figure for the same planned figure id.
- If you generate or cite a figure, the `section_tex` must contain a real LaTeX figure block or `\\includegraphics{...}` reference for it.
- If you cannot generate a figure file in this step, do not write `Figure~\\ref{...}` text that implies the figure exists.
- Prefer relative manuscript-local figure paths such as `figures/<name>.png` inside the TeX body.
Do not fabricate claims, figures, data, or citations.
If the evidence is incomplete, state the limitation clearly, but only where it materially affects interpretation.
Return a SectionDraftOutput only.
"""

WRITE_REVIEWER_SYSTEM_PROMPT = """You are CatMaster's writing reviewer.
Review a single drafted section as an academic writing reviewer, not as a forensic auditor.
Check for material problems in four areas: unsupported central claims, missing evidence for major quantitative statements, major structural weakness, and evidence-discipline failures that would mislead a scientific reader.
Treat unresolved figure references as a material problem. If a section mentions a figure or uses `Figure~\\ref{...}` but does not include a real figure block or real generated figure file path, request revision.
If `realized_figure_refs` contains symbolic ids or planned names rather than actual file paths, request revision.
If `planned_figure_ids` is populated but the section neither realizes those figures nor explicitly narrows scope, request revision.
If the section inserts the same realized image file as a second figure in a later section, request revision.
If the section uses an already-realized image file as though it were a different local figure with a different purpose or caption, request revision.
If the section generates a duplicate figure for a planned figure id that was already realized in another section without a clear local justification, request revision and prefer true reuse rather than duplicate insertion.
Prefer approval when the section is scientifically faithful, readable, and only has minor wording or citation-improvement opportunities left.
Do not request revision for every small overstatement, stylistic preference, or arguable phrasing choice.
Use `needs_revision` only when a reasonable reader would be materially misled, a major claim lacks support, or the prose is not yet suitable as manuscript text.
When issues are minor and non-blocking, approve the section and mention them briefly in revision_notes only if useful.
Do not rewrite the section yourself.
Return a SectionReviewOutput only.
"""

WRITE_FINALIZER_SYSTEM_PROMPT = """You are CatMaster's writing finalizer.
You are responsible for making the assembled manuscript bundle compile-ready before the writing run finishes.
Use the available compile/fix tool on the final manuscript bundle before you return.
Fix compile blockers, path/reference issues, and LaTeX syntax problems, but do not materially change the scientific wording, claims, or interpretation.
If compilation tooling is unavailable, still run the compile/fix tool so it can perform static checks and targeted repair.
Return a WritingFinalizeOutput only.
"""


def build_write_director_context(
    *,
    request: dict[str, Any],
    dossier: dict[str, Any],
    board: dict[str, Any] | None,
    memory_index_excerpt: str,
    latest_literature: list[dict[str, Any]],
    latest_experiments: list[dict[str, Any]],
    assembly_feedback: str | None,
    skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            f"Chat session context:\n{request.get('session_context_text') or '(none)'}",
            "Planning directive: infer the writing mode from the request, choose an explicit preferred_output_format, and if a visible writing skill exposes a manuscript template or TeX workflow, prefer `tex` and plan around that substrate.",
            "Workspace directive: treat the project files root as the main evidence workspace; use the fixed manuscript paths only as output locations for this run.",
            f"Assembly feedback from the previous attempt:\n{assembly_feedback or '(none)'}",
            "",
            "Research dossier JSON:",
            json.dumps(dossier or {}, ensure_ascii=False, indent=2),
            "",
            "Research board JSON:",
            json.dumps(board or {}, ensure_ascii=False, indent=2),
            "",
            "Project memory index:",
            memory_index_excerpt or "(none)",
            "",
            "Latest literature packs JSON:",
            json.dumps(list(latest_literature or [])[-2:], ensure_ascii=False, indent=2),
            "",
            "Latest experiment packs JSON:",
            json.dumps(list(latest_experiments or [])[-3:], ensure_ascii=False, indent=2),
            "",
            "Write-director skill guide:",
            skill_guide or "(none)",
        ]
    ).strip()


def build_section_writer_context(
    *,
    request: dict[str, Any],
    plan: WritingPlanModel,
    spec: WritingSectionSpec,
    dossier: dict[str, Any],
    memory_index_excerpt: str,
    working_manuscript_root: str,
    working_sections_dir: str,
    working_figures_dir: str,
    prior_draft: SectionDraftModel | None,
    review_notes: list[str],
    figure_registry: list[dict[str, Any]],
    skill_guide: str,
) -> str:
    plan_overview = {
        "title": plan.title,
        "writing_mode": plan.writing_mode,
        "preferred_output_format": plan.preferred_output_format,
        "ordered_work_units": [
            {
                "section_id": item.section_id,
                "heading": item.heading,
            }
            for item in plan.section_specs
        ],
    }
    relevant_figure_requests = [
        item.model_dump()
        for item in plan.figure_requests
        if item.figure_id in set(spec.required_figures)
    ]
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            f"Chat session context:\n{request.get('session_context_text') or '(none)'}",
            f"Plan title: {plan.title}",
            f"Planned writing mode: {plan.writing_mode}",
            f"Preferred output format: {plan.preferred_output_format}",
            f"Current manuscript output root: {working_manuscript_root}",
            f"Current manuscript sections output dir: {working_sections_dir}",
            f"Current manuscript figures output dir: {working_figures_dir}",
            "Workspace rule: read from the project workspace evidence, but write any new TeX/figure artifacts only under the fixed manuscript output paths above. Generated figures should normally live in the manuscript figures output dir and be referenced relatively from section TeX.",
            "",
            "Plan overview JSON:",
            json.dumps(plan_overview, ensure_ascii=False, indent=2),
            "",
            "Section spec JSON:",
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Relevant figure requests JSON:",
            json.dumps(relevant_figure_requests, ensure_ascii=False, indent=2),
            "",
            "Existing realized figure registry JSON:",
            json.dumps(figure_registry or [], ensure_ascii=False, indent=2),
            "",
            "Prior draft JSON:",
            json.dumps(prior_draft.model_dump(), ensure_ascii=False, indent=2) if prior_draft is not None else "(none)",
            "",
            "Review notes:",
            *([f"- {item}" for item in review_notes] or ["- (none)"]),
            "",
            "Project memory index:",
            memory_index_excerpt or "(none)",
            "",
            "Research dossier JSON:",
            json.dumps(dossier or {}, ensure_ascii=False, indent=2),
            "",
            "Section-writer skill guide:",
            skill_guide or "(none)",
        ]
    ).strip()


def build_write_reviewer_context(
    *,
    request: dict[str, Any],
    spec: WritingSectionSpec,
    draft: SectionDraftModel,
    figure_registry: list[dict[str, Any]],
    skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            f"Chat session context:\n{request.get('session_context_text') or '(none)'}",
            "",
            "Section spec JSON:",
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Draft JSON:",
            json.dumps(draft.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Existing realized figure registry JSON:",
            json.dumps(figure_registry or [], ensure_ascii=False, indent=2),
            "",
            "Write-reviewer skill guide:",
            skill_guide or "(none)",
        ]
    ).strip()


def build_write_finalizer_context(
    *,
    request: dict[str, Any],
    plan: WritingPlanModel,
    bundle: dict[str, Any],
    skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            f"Chat session context:\n{request.get('session_context_text') or '(none)'}",
            "",
            "Writing plan JSON:",
            json.dumps(plan.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Manuscript bundle JSON:",
            json.dumps(bundle or {}, ensure_ascii=False, indent=2),
            "",
            "Finalization directive: call the compile/fix tool on the assembled manuscript before returning. Treat compile and reference correctness as blocking; preserve scientific wording.",
            "",
            "Write-director skill guide:",
            skill_guide or "(none)",
        ]
    ).strip()


__all__ = [
    "SECTION_WRITER_SYSTEM_PROMPT",
    "WRITE_DIRECTOR_SYSTEM_PROMPT",
    "WRITE_REVIEWER_SYSTEM_PROMPT",
    "WRITE_FINALIZER_SYSTEM_PROMPT",
    "build_section_writer_context",
    "build_write_finalizer_context",
    "build_write_director_context",
    "build_write_reviewer_context",
]
