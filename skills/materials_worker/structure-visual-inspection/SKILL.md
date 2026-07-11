---
name: structure-visual-inspection
description: Use this skill when a task needs visual inspection of atomic structures, adsorption geometries, slab-site context, or image-based sanity checks before or alongside numerical analysis.
license: project-local
compatibility: local
allowed-tools: "render_vesta_views read_file execute"
---

# structure-visual-inspection

## Overview
Use rendered structure views as auxiliary evidence for geometry sanity checks, visual comparison, and reportable artifacts.

## Quick Start
1. Call `render_vesta_views` on the structure and keep the default top, side, and isometric views.
2. Use `supercell=[2,2,1]` when a slab needs more in-plane adsorption-site context.
3. Inspect the returned `inspection_image` with `read_file`.
4. Confirm critical visual claims with numerical geometry analysis.

## Allowed tools
- render_vesta_views
- read_file
- execute

## Workflow
1. Use `render_vesta_views` for the report-facing and model-facing image artifact. The tool exports standardized top, side, and isometric VESTA views plus a labeled panel and metadata JSON.
2. Use the original cell first. For slab adsorption-site interpretation, repeat only in plane with `supercell=[2,2,1]`; do not repeat the vacuum direction merely to fill the image.
3. Use `read_file` with a narrow question for orientation sanity checks, obvious clashes, site-family ambiguity, termination context, or pre/post-relax comparison.
4. When comparing candidates, keep `views`, `supercell`, `image_scale`, and `display_width_angstrom` aligned unless the report records why one structure needed different framing.
5. If VESTA is unavailable, report the missing installation/configuration instead of silently presenting the Matplotlib helper as VESTA output. The local `code/render_structure_panel.py` remains a lightweight diagnostic fallback when the user explicitly accepts it.

## Method-critical defaults
- Visual evidence is never a substitute for exact bond lengths, coordination metrics, adsorption energies, or thermodynamic quantities.
- For slab-site interpretation, supercell context can materially change what looks like bridge/hollow/top. Make the chosen supercell explicit when it matters.
- Prefer VESTA output for report figures and multimodal structure inspection because it preserves familiar atom, bond, cell, lighting, and depth cues.
- If perspective or isometric views suggest a geometry issue, verify it with file-based structures or numeric geometry analysis before changing the workflow.
- Keep rendered legend and element identity explicit; do not infer species from color alone in the final answer.
- VESTA's license requires explicit acknowledgement for drawings used in publications. Preserve the citation recorded in the render metadata.

## Output Contract
- Return the main panel image path.
- Return the metadata JSON and any non-default supercell or display-width setting used.
- If the analysis used multiple images, return all image paths referenced in the conclusion.
- State whether the visual conclusion is obvious, tentative, or requires numerical confirmation.
- When image analysis was used, summarize the concrete findings and the main uncertainty.

## References
- Use the rendered panel for interpretable reporting artifacts, but keep the final scientific conclusion grounded in structure files and numerical results.
