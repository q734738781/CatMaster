---
name: structure-visual-inspection
description: Use this skill when a task needs visual inspection of atomic structures, adsorption geometries, slab-site context, or image-based sanity checks before or alongside numerical analysis.
license: project-local
compatibility: local
allowed-tools: "render_structure_views analyze_images read_text_file"
metadata:
  catmaster-suggested-tools: "render_structure_views analyze_images read_text_file"
---

# structure-visual-inspection

## Overview
Use rendered structure views as auxiliary evidence for geometry sanity checks, visual comparison, and reportable artifacts.

## Quick Start
1. Render the structure with four standard views.
2. Inspect the panel directly first; only call image analysis when the visual question is still ambiguous.
3. Treat visual findings as supporting evidence.
4. Confirm critical geometric claims with numerical tools when available.

## Suggested tools
- render_structure_views
- analyze_images
- read_text_file

## Workflow
1. Render a default four-view panel first. Keep the first pass simple: default fit, default legend, no arbitrary camera tuning.
2. Increase `supercell` only when slab-site context is visually too local. Typical slab retries are `(2,2,1)` or `(3,3,1)`, not larger by default.
3. Use `analyze_images` only with a narrow question: orientation sanity check, obvious clash, site-family ambiguity, or pre/post-relax comparison.
4. When comparing candidates, keep the rendering preset and visual context aligned across all images. Do not mix one candidate with a larger supercell or looser fit unless the report says so.
5. If a visual conclusion would affect ranking, site assignment, or a scientific claim, hand back to numerical geometry tools or file-based metadata before finalizing.

## Method-critical defaults
- Visual evidence is never a substitute for exact bond lengths, coordination metrics, adsorption energies, or thermodynamic quantities.
- For slab-site interpretation, supercell context can materially change what looks like bridge/hollow/top. Make the chosen supercell explicit when it matters.
- If perspective or isometric views suggest a geometry issue, verify it with file-based structures or numeric geometry analysis before changing the workflow.
- Keep rendered legend and element identity explicit; do not infer species from color alone in the final answer.

## Output Contract
- Return the main panel image path.
- If the analysis used multiple images, return all image paths referenced in the conclusion.
- State whether the visual conclusion is obvious, tentative, or requires numerical confirmation.
- When image analysis was used, summarize the concrete findings and the main uncertainty.

## References
- Use the rendered panel for interpretable reporting artifacts, but keep the final scientific conclusion grounded in structure files and numerical results.
