from __future__ import annotations

import json
from typing import Any

from catmaster.runtime.writing import SectionDraftModel, WritingPlanModel, WritingSectionSpec


def get_write_director_system_prompt(output_format: str) -> str:
    fmt = str(output_format or "tex").strip().lower() or "tex"
    format_guidance = (
        "Primary output format is markdown. Plan a markdown-first workflow. "
        "Do not route the work into TeX-only assembly, cite syntax, or compile/fix workflow."
        if fmt == "md"
        else "Primary output format is TeX. Prefer TeX-ready scientific sections and compile-safe assembly."
    )
    return """You are CatMaster's writing director.
Plan a manuscript-style or report-style writing workflow from the available research materials.
Your job is to shape a readable scientific deliverable, not an execution log or compliance memo.
You may use read-only tools to inspect existing research packs, project workspace evidence, memory, and prior runs.
Do not launch new expensive scientific computations.
Return a WritingPlanOutput only.
Respect the requested writing mode and output format. """ + format_guidance + """
Aim for a plan that supports natural scientific narrative: a clear storyline, bounded claims, and explicit but not overbearing treatment of limitations.
Do not force every sentence to behave like an audit record. Keep evidence discipline at the level of major claims, tables, figures, and known uncertainty boundaries.
Plan the manuscript as a sequence of small writing work units.
Each `section_spec` should correspond to exactly one deliverable section or subsection task, not a whole composite chapter plus an internal to-do list.
If Results and Discussion would naturally contain multiple subsections, split them into multiple `section_spec` items so the worker receives one focused local brief at a time.
Prefer section/subsection work units that can be drafted in roughly 1-3 compact paragraphs each, plus any local figure/table integration needed for that unit.
Do not plan speculative figures as if they already exist. A figure should be treated as part of the deliverable only when the worker can either generate the file in the fixed output paths or point to an existing usable file.
Plan figures at the deliverable level, but expect the worker to reuse an already-realized figure across sections instead of regenerating near-duplicates.
Do not plan figures whose main payload is paragraph text, long bullet lists, or caption-like explanation embedded inside the graphic itself.
When a point can be communicated more cleanly in prose or a caption, prefer prose/caption over text-heavy visualization.
Each new writing run starts from a fresh output root. The active bundle is recreated for the current run, while prior bundles are moved to `manuscript_archive/`.
Do not assume files mentioned in prior conversation history still remain under the current active output root.
Default behavior: do not treat a previous manuscript draft as source material unless the user explicitly asks to revise, edit, polish, patch, or continue an existing manuscript.
If the user explicitly asks to work from an existing manuscript, you may inspect prior manuscript files already present in the project evidence, including archived manuscript bundles under `manuscript_archive/`, and use them as revision input. Otherwise, use workspace evidence and research artifacts as the source of truth, and treat any prior manuscript files as non-authoritative byproducts.
If a visible writing skill implies a template workflow, treat that as a planning constraint, prefer `preferred_output_format="tex"`, and require the section writer to read that skill file and its assets before editing.
"""


def get_section_writer_system_prompt(output_format: str) -> str:
    fmt = str(output_format or "tex").strip().lower() or "tex"
    format_guidance = (
        """Markdown is the required output format.
Write Markdown sections only.
Do not emit LaTeX commands such as `\\section`, `\\cite`, `\\begin{document}`, `\\includegraphics`, or bibliography blocks.
Use standard Markdown headings, lists, tables, and fenced blocks only when materially useful."""
        if fmt == "md"
        else """TeX is the required output format.
If an achemso or other manuscript template is in use, write section/body fragments only.
Do not emit a full standalone document with `\\documentclass`, preamble, `\\begin{document}`, or bibliography commands unless the task explicitly asks for a whole template file."""
    )
    return """You are CatMaster's section writer.
You may use tools to retrieve research packs, review context, inspect workspace files, run literature grounding, generate figures, and edit deliverable files.
Write one local work unit only.
Treat the incoming section spec as a single section/subsection task, not as a license to draft the whole manuscript section family around it.
Do not expand a focused subsection brief into a full Results and Discussion chapter unless the spec explicitly asks for that.
Write as a strong scientific author: concise, readable, evidence-faithful, and naturally structured.
Your output should read like scientific prose, not like a lab notebook, execution trace, or point-by-point audit.
Use existing project workspace evidence, memory, prior runs, and already-generated artifacts as your primary substrate.
Do not launch new expensive scientific computations; if a missing claim would require new numerical evidence, keep it as a gap or limitation.
Anchor the main scientific claims to available evidence, but do not overload the prose with path dumps or defensive qualification in every sentence.
Keep artifact-heavy traceability in citations, claim_evidence_map, figures, tables, and unresolved_gaps; keep the body text readable.
If a visible writing skill includes template assets, read the skill file first, then open the asset files through the read-only skill mount before drafting or editing.
Do not assume a listed path means the template content is already in context.
Treat source research artifacts as read-only evidence.
The project workspace is the main evidence space. The fixed output paths in context are output destinations only.
Each new writing run recreates a fresh active output bundle. Prior bundles are moved to `manuscript_archive/`, so do not assume a path mentioned in earlier conversation history still exists under the current active directory.
Any new output artifacts you create must live under the fixed output paths provided in context. Do not write new manuscript assets into `research/...`, prior runs, or other source evidence directories.
Generated figure files must be written under the fixed figures output directory provided in context.
""" + format_guidance + """
Figure discipline is strict:
- `planned_figure_ids` should list the figure ids from the writing plan that this section intends to realize or discuss.
- Do not put symbolic placeholders like `fig_surface_ranking` into `realized_figure_refs`.
- `realized_figure_refs` should contain only real workspace-relative image paths that exist or that you generated in this step.
- If a planned figure has already been realized by an earlier section, do not create a second figure block for the same image in a later section.
- "Reuse" means referring to the same already-realized scientific figure as the same figure object, not inserting the same PNG again as a different local figure with a new role or caption.
- Do not create a new figure file that is materially the same as an already-realized figure for the same planned figure id.
- If you generate or cite a figure, make sure the section body references it using syntax appropriate for the requested output format.
- If you cannot generate a figure file in this step, do not imply the figure already exists.
- For conceptual, mechanistic, or workflow figures, prefer `generate_nanobanana_figure` over hand-built matplotlib diagrams.
- Reserve matplotlib/seaborn/plotly for quantitative plots, statistical graphics, and other data-native visualizations.
- Do not use matplotlib or any plotting library to render long prose, paragraph annotations, bullet lists, or caption-like explanation inside the figure canvas.
- Keep in-figure text minimal: short axis labels, short legends, short panel labels, short callouts, and concise identifiers only.
- If a label or annotation would exceed a short phrase, move that explanation into the manuscript body or caption instead of plotting it.
- Avoid text-dense schematic/plot hybrids that depend on large blocks of rendered text to be understandable.
Do not fabricate claims, figures, data, or citations.
If the evidence is incomplete, state the limitation clearly, but only where it materially affects interpretation.
Return a SectionDraftOutput only.
"""


def get_write_reviewer_system_prompt(output_format: str) -> str:
    fmt = str(output_format or "tex").strip().lower() or "tex"
    format_guidance = (
        "Review markdown structure, heading hierarchy, figure/file references, and evidence discipline. Do not require TeX syntax."
        if fmt == "md"
        else "Review TeX section/body fragments, citation syntax, figure references, and compile-facing evidence discipline."
    )
    return """You are CatMaster's writing reviewer.
Review a single drafted section as a scientific writing reviewer, not as a forensic auditor.
Check for material problems in four areas: unsupported central claims, missing evidence for major quantitative statements, major structural weakness, and evidence-discipline failures that would mislead a scientific reader.
Treat invented or obviously unresolvable citation keys as a material problem. If the prose would be acceptable without a citation, prefer deleting the bad citation over preserving a fake key.
Treat unresolved figure references as a material problem. If a section mentions a figure but does not include a real figure block or real generated figure file path, request revision.
If `realized_figure_refs` contains symbolic ids or planned names rather than actual file paths, request revision.
If `planned_figure_ids` is populated but the section neither realizes those figures nor explicitly narrows scope, request revision.
If the section inserts the same realized image file as a second figure in a later section, request revision.
If the section uses an already-realized image file as though it were a different local figure with a different purpose or caption, request revision.
If the section generates a duplicate figure for a planned figure id that was already realized in another section without a clear local justification, request revision and prefer true reuse rather than duplicate insertion.
If a matplotlib-style figure is mainly carrying prose, long annotation blocks, or caption-like text inside the plotted image, request revision.
If labels, legends, or annotations are so long that they are likely to wrap, overlap, or misalign at manuscript scale, request revision.
Prefer figures with short labels and manuscript/caption-based explanation over text-heavy figures.
Prefer approval when the section is scientifically faithful, readable, and only has minor wording or citation-improvement opportunities left.
Do not request revision for every small overstatement, stylistic preference, or arguable phrasing choice.
Use `needs_revision` only when a reasonable reader would be materially misled, a major claim lacks support, or the prose is not yet suitable as manuscript text.
When issues are minor and non-blocking, approve the section and mention them briefly in revision_notes only if useful.
Do not rewrite the section yourself.
""" + format_guidance + """
Return a SectionReviewOutput only.
"""


def get_write_finalizer_system_prompt(output_format: str) -> str:
    fmt = str(output_format or "tex").strip().lower() or "tex"
    if fmt == "md":
        return """You are CatMaster's writing finalizer.
You are responsible for making the assembled markdown bundle publication-ready before the writing run finishes.
Do not call or rely on TeX compilation workflow.
Fix markdown structure issues, broken references, heading consistency, and light editorial problems, but do not materially change the scientific wording, claims, or interpretation.
Return a WritingFinalizeOutput only.
"""
    return """You are CatMaster's writing finalizer.
You are responsible for making the assembled manuscript bundle compile-ready before the writing run finishes.
Use the available compile/fix tool on the final manuscript bundle before you return.
Fix compile blockers, path/reference issues, and LaTeX syntax problems, but do not materially change the scientific wording, claims, or interpretation.
If compilation tooling is unavailable, still run the compile/fix tool so it can perform static checks and targeted repair.
Return a WritingFinalizeOutput only.
"""

WRITE_DIRECTOR_SYSTEM_PROMPT = get_write_director_system_prompt("tex")
SECTION_WRITER_SYSTEM_PROMPT = get_section_writer_system_prompt("tex")
WRITE_REVIEWER_SYSTEM_PROMPT = get_write_reviewer_system_prompt("tex")
WRITE_FINALIZER_SYSTEM_PROMPT = get_write_finalizer_system_prompt("tex")


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
            f"Recovery feedback from the previous attempt:\n{assembly_feedback or '(none)'}",
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
    "get_section_writer_system_prompt",
    "get_write_director_system_prompt",
    "get_write_finalizer_system_prompt",
    "get_write_reviewer_system_prompt",
    "build_section_writer_context",
    "build_write_finalizer_context",
    "build_write_director_context",
    "build_write_reviewer_context",
]
