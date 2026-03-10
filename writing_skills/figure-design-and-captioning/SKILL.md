---
name: figure-design-and-captioning
description: Design evidence-carrying figures and captions from research outputs, with reproducible script and data references.
compatibility: local
metadata:
  catmaster-roles: "section_writer write_director write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing figures captions"
  catmaster-suggested-tools: "review_research_context read_research_pack generate_schematic_figure bash_exec render_structure_views analyze_images"
---

# figure-design-and-captioning

## Overview
Plan and generate manuscript figures that support specific claims, with explicit script/data provenance and concise captions.

## Quick Start
Use when a section needs structure renders, conceptual schematics, comparison plots, summary tables, or figure captions tied to computational outputs.

## Suggested tools
- `review_research_context`
- `read_research_pack`
- `generate_schematic_figure`
- `bash_exec`
- `render_structure_views`
- `analyze_images`

## Workflow
1. Start from the claim the figure must support; if the claim is unclear, the figure is not ready.
2. Identify the minimum artifact and data inputs needed to make that claim visible.
3. If the figure is a conceptual or mechanistic schematic, use `generate_schematic_figure` with a precise prompt and save the generated image under the writing workspace.
4. If the figure comes directly from coordinates or computed data, use reproducible scripts or rendering tools, and keep the script path.
5. Inspect the figure for readability, labeling, and whether the intended comparison is obvious.
6. Write a caption that states what is shown, under what conditions, and why it matters.

### Figure selection rules

- One figure should usually do one argumentative job.
- Prefer comparisons over isolated displays when the paper's claim depends on difference, ranking, or trend.
- Show uncertainty, spread, or sensitivity when the conclusion depends on robustness.
- If a table communicates the evidence more clearly than a plot, use a table.
- Do not use matplotlib or plotting libraries to typeset paragraphs, long bullet lists, or caption-like explanation inside the figure.
- Keep figure text sparse: short axis labels, short legends, short panel labels, and brief callouts only.
- Any explanation longer than a short phrase belongs in the caption or body text, not inside the graphic.

### Caption template

Captions should usually answer:

1. what is shown
2. what system, dataset, or condition is shown
3. what the reader should notice
4. what processing, normalization, or color meaning is essential

### Graph and image checks

- axes, units, legends, and abbreviations are defined
- color or symbol mapping is consistent across related figures
- image panels have meaningful labels and scale/orientation when relevant
- text does not simply duplicate the caption
- labels and annotations are short enough to stay stable at manuscript export size
- the script and source refs are sufficient to regenerate the display

## Output Contract
Return figure intent, output path, script path, source refs, and caption draft.

## References
- Do not invent figures when the underlying data or artifact path is missing.
- Read [`../_references/section-patterns-and-story.md`](../_references/section-patterns-and-story.md) for figure logic and evidence-first ordering.
- Read [`../_references/submission-and-editorial-readiness.md`](../_references/submission-and-editorial-readiness.md) before finalizing legends for a journal-facing draft.
