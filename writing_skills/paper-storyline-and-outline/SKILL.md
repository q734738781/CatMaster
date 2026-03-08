---
name: paper-storyline-and-outline
description: Build a scientific storyline, paper outline, and section ladder from research materials.
compatibility: local
metadata:
  catmaster-roles: "write_director write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing outline storyline"
  catmaster-suggested-tools: "read_research_pack review_research_context"
---

# paper-storyline-and-outline

## Overview
Turn research materials into a coherent paper/story outline with section purposes and figure slots, emphasizing narrative economy over exhaustive coverage.

## Quick Start
Use when planning manuscript structure, deciding section order, or converting a campaign dossier into a publishable storyline.

## Suggested tools
- `read_research_pack`
- `review_research_context`

## Workflow
1. Define the paper in one sentence: question -> best supported answer -> why it matters.
2. Build the evidence ladder from the strongest figures, tables, and experimental or literature anchors first.
3. Choose a section order that follows the story rather than the chronology of the project.
4. Give each section one job only: context, evidence, interpretation, reproducibility, or synthesis.
5. Reserve figure and table slots only where they carry an argument the prose should not have to explain alone.
6. Keep a visible backlog of evidence gaps, weak comparisons, and sections that will need additional background retrieval, but do not let the plan read like a risk register.

### Planning heuristics

- One paper should usually carry one central advance. Sub-claims should support that advance rather than compete with it.
- Do not let the outline become a dump of everything the campaign touched.
- Title, abstract, results order, and conclusion should all describe the same contribution.
- If a proposed section cannot point to its required evidence, mark it as speculative and shrink it.
- Prefer a clean, human-readable section ladder over an over-instrumented planning document.

### Section brief checklist

For each section spec, record:

- the question the section answers
- the 2-4 claims the section must support
- the figure/table slots it depends on
- the evidence type required: computation, literature, artifact, or mixed
- what uncertainty or limitation must remain visible

## Output Contract
Return a structured writing plan with section purposes, required evidence, and figure/citation needs.

## References
- Prefer the source campaign dossier and structured evidence packs over raw transcripts.
- Read [`../_references/section-patterns-and-story.md`](../_references/section-patterns-and-story.md) when choosing section order or abstract/title logic.
- Read [`../_references/submission-and-editorial-readiness.md`](../_references/submission-and-editorial-readiness.md) before freezing the plan for a journal-facing draft.
