from __future__ import annotations

import json
from typing import Any

from catmaster.runtime.writing import SectionDraftModel, WritingPlanModel, WritingSectionSpec


WRITE_DIRECTOR_SYSTEM_PROMPT = """You are CatMaster's writing director.
Plan a manuscript-style writing workflow from the available research materials.
Your job is to shape a readable academic manuscript, not an execution log or compliance memo.
You may use read-only tools to inspect existing research packs, workspace evidence, memory, and prior runs.
Do not launch new expensive scientific computations.
Return a WritingPlanOutput only.
Choose the writing mode yourself from the user request: internal_report, paper_outline, section_draft, or full_draft.
Aim for a plan that supports natural scientific narrative: a clear storyline, bounded claims, and explicit but not overbearing treatment of limitations.
Do not force every sentence to behave like an audit record. Keep evidence discipline at the level of major claims, tables, figures, and known uncertainty boundaries.
Prefer TeX as the primary manuscript format when a LaTeX template is available.
If a visible writing skill implies a template workflow, treat that as a planning constraint, prefer `preferred_output_format="tex"`, and require the section writer to read that skill file and its assets before editing.
"""

SECTION_WRITER_SYSTEM_PROMPT = """You are CatMaster's section writer.
You may use tools to retrieve research packs, review context, inspect workspace files, run literature grounding, generate figures, and edit manuscript files.
Write one section only.
Write as a strong academic author: concise, readable, evidence-faithful, and naturally structured.
Your output should read like manuscript prose, not like a lab notebook, execution trace, or point-by-point audit.
Use existing workspace evidence, memory, prior runs, and already-generated artifacts as your primary substrate.
Do not launch new expensive scientific computations; if a missing claim would require new numerical evidence, keep it as a gap or limitation.
Anchor the main scientific claims to available evidence, but do not overload the prose with path dumps or defensive qualification in every sentence.
Keep artifact-heavy traceability in citations, claim_evidence_map, figures, tables, and unresolved_gaps; keep the body text readable.
If a visible writing skill includes template assets, read the skill file first, then open the asset files through the read-only skill mount before drafting or editing.
Do not assume a listed path means the template content is already in context.
If an achemso or other manuscript template is in use, write section/body fragments only.
Do not emit a full standalone document with `\\documentclass`, preamble, `\\begin{document}`, or bibliography commands unless the task explicitly asks for a whole template file.
Any workspace `.tex` artifacts you create should be section-level files or manuscript body fragments, and should be recorded in `latex_artifact_refs`.
Prefer `apply_aider_edits` for deterministic TeX edits over ad-hoc shell rewriting.
TeX is the only manuscript deliverable.
Do not fabricate claims, figures, data, or citations.
If the evidence is incomplete, state the limitation clearly, but only where it materially affects interpretation.
Return a SectionDraftOutput only.
"""

WRITE_REVIEWER_SYSTEM_PROMPT = """You are CatMaster's writing reviewer.
Review a single drafted section as an academic writing reviewer, not as a forensic auditor.
Check for material problems in four areas: unsupported central claims, missing evidence for major quantitative statements, major structural weakness, and evidence-discipline failures that would mislead a scientific reader.
Prefer approval when the section is scientifically faithful, readable, and only has minor wording or citation-improvement opportunities left.
Do not request revision for every small overstatement, stylistic preference, or arguable phrasing choice.
Use `needs_revision` only when a reasonable reader would be materially misled, a major claim lacks support, or the prose is not yet suitable as manuscript text.
When issues are minor and non-blocking, approve the section and mention them briefly in revision_notes only if useful.
Do not rewrite the section yourself.
Return a SectionReviewOutput only.
"""


def build_write_director_context(
    *,
    request: dict[str, Any],
    dossier: dict[str, Any],
    board: dict[str, Any] | None,
    memory_index_excerpt: str,
    latest_literature: list[dict[str, Any]],
    latest_experiments: list[dict[str, Any]],
    skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            "Planning directive: infer the writing mode from the request, choose an explicit preferred_output_format, and if a visible writing skill exposes a manuscript template or TeX workflow, prefer `tex` and plan around that substrate.",
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
    prior_draft: SectionDraftModel | None,
    review_notes: list[str],
    skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            f"Plan title: {plan.title}",
            f"Planned writing mode: {plan.writing_mode}",
            f"Preferred output format: {plan.preferred_output_format}",
            "",
            "Writing plan JSON:",
            json.dumps(plan.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Section spec JSON:",
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2),
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
    skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"User writing request: {request.get('request', '')}",
            f"Source campaign: {request.get('source_campaign_id') or '(none)'}",
            "",
            "Section spec JSON:",
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Draft JSON:",
            json.dumps(draft.model_dump(), ensure_ascii=False, indent=2),
            "",
            "Write-reviewer skill guide:",
            skill_guide or "(none)",
        ]
    ).strip()


__all__ = [
    "SECTION_WRITER_SYSTEM_PROMPT",
    "WRITE_DIRECTOR_SYSTEM_PROMPT",
    "WRITE_REVIEWER_SYSTEM_PROMPT",
    "build_section_writer_context",
    "build_write_director_context",
    "build_write_reviewer_context",
]
