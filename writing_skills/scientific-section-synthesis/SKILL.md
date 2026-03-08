---
name: scientific-section-synthesis
description: Draft a scientific manuscript section from evidence packs, memory, and retrieved context without inventing claims.
compatibility: local
metadata:
  catmaster-roles: "section_writer write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing section synthesis"
  catmaster-suggested-tools: "review_research_context read_research_pack bash_exec run_literature_research"
---

# scientific-section-synthesis

## Overview
Draft one evidence-grounded manuscript section at a time while preserving readable academic prose and a clear local argument.

## Quick Start
Use for Results, Discussion, Methods, and Introduction drafting when the writing agent needs to gather supporting context before writing.

## Suggested tools
- `review_research_context`
- `read_research_pack`
- `bash_exec`
- `run_literature_research`

## Workflow
1. Read the section brief and reduce it to a small evidence plan: required claims, required figures, required citations, and known gaps.
2. Pull only the most relevant packs, memory, historical context, and background literature for that section.
3. Draft around the evidence chain first, then shape it into natural manuscript prose.
4. Keep major claims supportable, but do not overload every sentence with traceability language.
5. Record unresolved gaps explicitly instead of burying them, but only where they matter to interpretation.

### Paragraph recipe

Use this default paragraph shape unless the section type clearly needs another pattern:

1. topic sentence stating the paragraph's point
2. evidence sentence(s) naming the figure, table, artifact, or citation
3. interpretation sentence that stays narrower than the evidence
4. bridge sentence to the next paragraph or subsection

### Section-specific reminders

- Introduction: define the gap early and stop once the reader is ready for the paper's question.
- Results: observations first, interpretation second.
- Discussion: answer the question, compare with prior work, then state limits.
- Methods: describe what was actually done and where the reproducibility anchors live.

### Style discipline

- Prefer repeated technical terms over decorative synonym changes.
- Cut clutter before adding rhetorical flourishes.
- Use active voice when it clarifies agency; use passive voice sparingly in methods when the process matters more than the actor.
- If a sentence makes two claims, split it.
- Keep path-heavy traceability in notes, tables, citations, or evidence maps rather than in every line of body prose.

## Output Contract
Return a readable manuscript section plus the supporting citations, artifact refs, figure refs, and claim-to-evidence map.

## References
- Keep claims narrower than the available evidence when uncertainty remains.
- Read [`../_references/style-and-revision-checks.md`](../_references/style-and-revision-checks.md) for the prose-editing and final consistency passes.
- Read [`../_references/section-patterns-and-story.md`](../_references/section-patterns-and-story.md) if the section drifts away from its structural role.
