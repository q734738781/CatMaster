---
name: scientific-visualization
description: Prepare or revise data figures needing panel layout, statistical annotation or venue-specific formatting.
metadata:
    skill-author: K-Dense Inc.
---

# Scientific visualization

Use actual data and preserve units, comparisons, transformations and uncertainty.
A conceptual diagram is a different task; do not represent an illustration as
measured evidence. Slide text, tables and page layout remain native objects when
editable presentations are requested.

Choose the display for the scientific relationship readers need to see. Keep
interpretation in the caption and body; figure labels identify data rather than
typeset paragraphs. Inspect the rendered result at its intended size for missing
content, overlap, legibility and misleading scales.

Use the supplied project style and venue constraints. CatMaster's default direct
plotting style is Origin-like with the NPG categorical palette; the actual values
and rendering guidance are in
[publication-data-plotting](../../plot_worker/publication-data-plotting/SKILL.md).
Generic library presets below do not supersede that preference. Choose sequential
or diverging maps when numerical meaning requires them, and add non-color cues
when groups would otherwise be indistinguishable.

Save the requested final format, defaulting to a high-resolution PNG when no
format is specified. Avoid redundant format bundles. Keep source data and plotting
code when needed for subsequent work; disposable preview renders are scratch.

## Resources by need

- For multi-panel layout or difficult labeling, consult the relevant parts of
  [publication guidelines](references/publication_guidelines.md).
- For magnitude maps or accessibility choices, use
  [color palettes](references/color_palettes.md).
- For a named venue's output dimensions or file requirements, use
  [journal requirements](references/journal_requirements.md) as a starting point
  and check the actual venue instructions before submission.
- For a plotting technique or library example, use
  [matplotlib examples](references/matplotlib_examples.md). Examples illustrate
  code, not experimental data, statistical decisions or mandatory steps.

Optional Python helpers live in the mounted skill. Add
`$CATMASTER_SKILLS_ROOT/writing_specialist/scientific-visualization/scripts`
to the import path to use `figure_export.save_publication_figure` or
`style_presets`; palette assets are under `assets/`. The export helper defaults
to one PNG. Inspect signatures when selecting optional controls, and preserve
explicit project styling rather than blindly applying a generic preset.
