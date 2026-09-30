---
name: publication-data-plotting
description: Use this skill to create, restyle, or repair a scientific plot directly from supplied quantitative data or an existing data-native figure, with an Origin-like aesthetic, deliberate palette selection, and rendered-image checks for text, annotation, and signal overlap.
license: project-local
allowed-tools: "ls glob grep read_file write_file edit_file execute"
---

# publication-data-plotting

The optional `scripts/palette.py` provides the NPG constants used below, without
layout templates. It does not replace the style, semantic-color or rendered
inspection requirements in this skill.

## Overview

Turn supplied scientific data into a reproducible, publication-ready figure whose visual hierarchy makes the assigned scientific conclusion immediately legible.

## Quick Start

1. Read the figure brief and inspect the exact source data before choosing a chart.
2. Write or revise a reproducible plotting script and render one final figure in the selected format.
3. Apply an Origin-like scientific style and a palette matched to the data semantics.
4. Open the final PNG, or a disposable raster QA preview when the final format is not directly inspectable, repair every collision or weak visual signal, and keep only the selected final figure.

## Allowed tools

- `ls`, `glob`, and `grep` to locate supplied data and existing figure code.
- `read_file` to inspect text/data inputs and either the final PNG or a disposable raster QA preview.
- `write_file` and `edit_file` to create or revise the plotting script.
- `execute` to run plotting code and render outputs.

Do not use web search, image generation, or another agent to replace direct plotting from the supplied data.

## Workflow

### 1. Fix the claim and data semantics

Write the figure's one-sentence takeaway before plotting. Confirm the exact source paths, variables, units, category order, comparison baseline, uncertainty definition, replicate or sample-count meaning, and any transformation already authorized by the analysis. Choose the smallest chart or panel set that makes this evidence visible.

Do not recompute scientific results merely to decorate the figure. When a required semantic field is missing and cannot be inferred from the supplied evidence, preserve the data and flag that one blocking ambiguity rather than inventing it.

### 2. Build a reproducible plot

Use Python with matplotlib for ordinary quantitative plots, with pandas, NumPy, SciPy, or seaborn only where they help the analysis or layout. Reuse an existing project plotting script when it is authoritative; otherwise create a reusable script under `scripts/` with the required CatMaster script header. Respect an explicit user requirement for another backend.

Never ship matplotlib's default rcParams, default style, or default color cycle. Set the canvas, typography, axes, ticks, line and marker weights, legend, palette, and export settings deliberately so the result is an Origin-style scientific graph rather than a minimally edited library default.

Save exactly one final figure for each logical visualization. Use the format explicitly requested by the user or genuinely required by the downstream venue/interface; when neither specifies one, save one 300 dpi or better PNG. Do not retain equivalent PDF, SVG, TIFF, PNG, or editable copies for preview, convenience, or possible future use. If a non-raster final needs visual QA, render the preview under `/tmp/` and do not promote or report it as a deliverable. Keep data loading, transformations, plotting, and the selected export explicit in the script.

### 3. Apply an Origin-like scientific aesthetic

Start from a white canvas, clean axes, inward or otherwise consistent ticks, restrained grid use, readable final-size sans-serif typography, explicit units, controlled line and marker weights, compact legends, and aligned panels. Avoid decorative backgrounds, gradients, shadows, 3D effects, oversized titles, default rainbow colors, and unexplained visual encodings.

Use this Nature/NPG palette as the default categorical color source. Select only the colors the figure needs, assign them by scientific meaning, and keep each assignment stable across panels.

| Role | HEX | RGB |
|---|---|---|
| Main red | `#E64B35` | 230, 75, 53 |
| Cyan blue | `#4DBBD5` | 77, 187, 213 |
| Teal | `#00A087` | 0, 160, 135 |
| Deep blue | `#3C5488` | 60, 84, 136 |
| Coral red | `#F39B7F` | 243, 155, 127 |
| Muted light blue | `#8491B4` | 132, 145, 180 |
| Mint green | `#91D1C2` | 145, 209, 194 |
| Deep red | `#DC0000` | 220, 0, 0 |
| Brown | `#7E6148` | 126, 97, 72 |
| Sand | `#B09C85` | 176, 156, 133 |

Do not substitute a generic Morandi, pastel, or low-saturation palette, and do not fall back to matplotlib's default Tableau cycle. Choose color by meaning:

- use the Nature/NPG categorical colors for categories and check that adjacent series remain distinguishable;
- use a perceptually ordered sequential map for magnitude;
- use a centered diverging map only when a scientifically meaningful midpoint exists;
- reserve the strongest accent for the evidence that carries the main claim;
- add marker, line-style, shape, or fill redundancy when grayscale or color-vision differences could merge groups.

The figure should resemble a carefully finished Origin publication graph, not a software-default screenshot. Exact font sizes, line widths, and dimensions follow the final journal size and panel density rather than a universal preset.

### 4. Inspect the rendered visual signal

Open the final PNG with `read_file` at the intended final dimensions. If the selected final format is not directly inspectable, open a disposable raster rendering under `/tmp/` instead and remove or leave it as ignored scratch after QA. Check the rendered image, not only the plotting code, for:

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
- Nature/NPG palette assignments must remain interpretable for color-vision differences and, when relevant, grayscale reproduction; add non-color redundancy rather than replacing the palette with arbitrary muted tones.
- Typography and spacing are judged at final display or print size, not at an enlarged development preview.
- Preserve exact units, category order, uncertainty definitions, sample-count meaning, and comparison baselines from the authoritative data.
- Hardware, accelerator, launcher, scheduler, package-build, and rendering-performance details are not figure QC or handoff content unless a known compatibility problem changes the rendered scientific result.

## Output Contract

Return the one-sentence figure takeaway, authoritative data path or paths used, plotting-script path, the single final figure path, and any single condition that materially affects scientific interpretation. Do not return a temporary QA preview or create a separate manifest or acceptance checklist for an ordinary figure job.

## References

Use existing project style assets or journal specifications when the brief supplies them. The worker's rendered preview remains the final authority for clipping, overlap, visual hierarchy, and legibility.
