---
name: publication-data-plotting
description: Use this skill when the plot worker receives quantitative data or an existing data-native figure and must directly create, restyle, or repair a publication-ready plot with an Origin-like scientific aesthetic, deliberate palette selection, and rendered-image checks for text, annotation, and signal overlap.
license: project-local
allowed-tools: "ls glob grep read_file write_file edit_file execute"
---

# publication-data-plotting

## Overview

Turn supplied scientific data into a reproducible, publication-ready figure whose visual hierarchy makes the assigned scientific conclusion immediately legible.

## Quick Start

1. Read the figure brief and inspect the exact source data before choosing a chart.
2. Write or revise a reproducible plotting script and render vector output plus a raster preview.
3. Apply an Origin-like scientific style and a palette matched to the data semantics.
4. Open the preview at final size, repair every collision or weak visual signal, and export the final files.

## Allowed tools

- `ls`, `glob`, and `grep` to locate supplied data and existing figure code.
- `read_file` to inspect text/data inputs and the final raster preview.
- `write_file` and `edit_file` to create or revise the plotting script.
- `execute` to run plotting code and render outputs.

Do not use web search, image generation, or another agent to replace direct plotting from the supplied data.

## Workflow

### 1. Fix the claim and data semantics

Write the figure's one-sentence takeaway before plotting. Confirm the exact source paths, variables, units, category order, comparison baseline, uncertainty definition, replicate or sample-count meaning, and any transformation already authorized by the analysis. Choose the smallest chart or panel set that makes this evidence visible.

Do not recompute scientific results merely to decorate the figure. When a required semantic field is missing and cannot be inferred from the supplied evidence, preserve the data and flag that one blocking ambiguity rather than inventing it.

### 2. Build a reproducible plot

Use mature local libraries, normally Python with matplotlib, seaborn, pandas, NumPy, and SciPy when needed. Reuse an existing project plotting script when it is authoritative; otherwise create a reusable script under `scripts/` with the required CatMaster script header. Respect a user-specified language or established project backend.

Save to the requested path. Unless the user or venue specifies otherwise, provide one editable or vector format such as PDF or SVG and one 300 dpi or better PNG preview. Keep data loading, transformations, plotting, and export explicit in the script.

### 3. Apply an Origin-like scientific aesthetic

Start from a white canvas, clean axes, inward or otherwise consistent ticks, restrained grid use, readable final-size sans-serif typography, explicit units, controlled line and marker weights, compact legends, and aligned panels. Avoid decorative backgrounds, gradients, shadows, 3D effects, oversized titles, default rainbow colors, and unexplained visual encodings.

Choose color by meaning:

- use a restrained colorblind-safe qualitative palette for categories;
- use a perceptually ordered sequential map for magnitude;
- use a centered diverging map only when a scientifically meaningful midpoint exists;
- reserve the strongest accent for the evidence that carries the main claim;
- add marker, line-style, shape, or fill redundancy when grayscale or color-vision differences could merge groups.

The figure should resemble a carefully finished Origin publication graph, not a software-default screenshot. Exact font sizes, line widths, and dimensions follow the final journal size and panel density rather than a universal preset.

### 4. Inspect the rendered visual signal

Open the raster preview with `read_file` at the intended final dimensions. Check the rendered image, not only the plotting code, for:

- clipped axis labels, units, panel letters, legends, or annotations;
- legend, text, arrows, or significance marks covering points, curves, bars, error bars, or confidence regions;
- overlapping tick labels and unreadable scientific notation;
- weak contrast, indistinguishable series, or color carrying the only distinction;
- inconsistent axes, margins, baselines, panel alignment, or whitespace;
- dense labels or decorative elements competing with the core signal.

Move, shorten, rotate, or externalize labels; adjust margins, limits, panel proportions, legend placement, or encoding; then re-render and inspect again after any material layout change. Long interpretation belongs in the caption, not on the canvas.

### 5. Preserve scientific integrity

Show the supplied data faithfully. Do not drop inconvenient points, crop data to exaggerate separation, smooth or interpolate without authorization, hide uncertainty, use an unlabeled broken axis, or choose limits that create a misleading comparison. For bar charts, use a scientifically defensible baseline. State transformations and uncertainty semantics in the caption or handoff when they affect interpretation.

## Method-critical defaults

- The assigned scientific takeaway controls the visual hierarchy, but never changes the underlying values or statistical meaning.
- Palette choice must remain interpretable for color-vision differences and, when relevant, grayscale reproduction.
- Typography and spacing are judged at final display or print size, not at an enlarged development preview.
- Preserve exact units, category order, uncertainty definitions, sample-count meaning, and comparison baselines from the authoritative data.
- Hardware, accelerator, launcher, scheduler, package-build, and rendering-performance details are not figure QC or handoff content unless a known compatibility problem changes the rendered scientific result.

## Output Contract

Return the one-sentence figure takeaway, authoritative data path or paths used, plotting-script path, final vector or editable figure path, raster preview path, and any single condition that materially affects scientific interpretation. Do not create a separate manifest or acceptance checklist for an ordinary figure job.

## References

Use existing project style assets or journal specifications when the brief supplies them. The worker's rendered preview remains the final authority for clipping, overlap, visual hierarchy, and legibility.
