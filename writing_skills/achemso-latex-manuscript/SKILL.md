---
name: achemso-latex-manuscript
description: Write ACS-style manuscript sections against the local achemso LaTeX template, using deterministic template edits and preserving BibTeX structure.
compatibility: local
metadata:
  catmaster-roles: "write_director section_writer write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing latex achemso acs"
  catmaster-suggested-tools: "apply_aider_edits read_research_pack review_research_context bash"
---

# achemso-latex-manuscript

## Overview
Use the bundled `achemso` assets as the fixed manuscript shell when the writing lane is producing an ACS-style draft. Prefer writing section/body fragments that will be assembled into the template over free-form generation of a full standalone document.

## Quick Start
Use when the section should be written against an ACS-style LaTeX template or when the output must stay close to `achemso` structure.

Read these assets first through the read-only skills mount:

- `@writing_skills/achemso-latex-manuscript/assets/achemso-demo.tex`
- `@writing_skills/achemso-latex-manuscript/assets/achemso-demo.bib`

## Suggested tools
- `apply_aider_edits`
- `read_research_pack`
- `review_research_context`
- `bash`

## Workflow
1. Read the `SKILL.md` and the two asset files before drafting the section.
2. Treat the mounted asset as the canonical outer shell and bibliography reference; do not try to edit the read-only mounted asset directly.
3. Write or update section-level `.tex` fragments under the writable run workspace instead of regenerating the entire manuscript wrapper.
4. Use `apply_aider_edits` for deterministic SEARCH/REPLACE edits to the working TeX file.
5. Keep bibliography keys and citation commands consistent with the template's BibTeX usage.
6. Treat `section_tex` and workspace `.tex` artifacts as the primary output; use `section_md` only as a concise reviewer shadow when useful.

### Template discipline

- Preserve the documentclass, package imports, and global preamble unless the task explicitly requires a change.
- Prefer editing section bodies, captions, and bibliography entries over restructuring the full template.
- Do not emit a second independent paper template when the system is already assembling an `achemso` outer shell.
- Do not invent BibTeX entries. If a citation key is unknown, leave a clear placeholder or gap note.
- Keep LaTeX macros simple and local; avoid adding fragile custom commands unless necessary.
- The mounted asset is reference-only; any edited TeX should live under the writable run workspace.

### Editing rules

- Use comments to mark inserted section blocks only when that helps later assembly or review.
- Keep figures and tables aligned with actual artifact paths and generated files.
- If you create a new `.tex` helper file, record the path in `latex_artifact_refs`.
- If the template and the evidence disagree, preserve the evidence and adapt the prose, not the other way around.

## Output Contract
Return section-level `section_tex` as the primary output. If you create or edit a workspace `.tex` fragment directly, record it in `latex_artifact_refs`. The outer `achemso` manuscript wrapper is assembled by the system.

## References
- Read [`../_references/style-and-revision-checks.md`](../_references/style-and-revision-checks.md) for sentence-level revision and consistency checks.
- Read [`../_references/submission-and-editorial-readiness.md`](../_references/submission-and-editorial-readiness.md) before treating the LaTeX section as journal-facing.
