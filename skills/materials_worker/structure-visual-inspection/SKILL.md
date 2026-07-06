---
name: structure-visual-inspection
description: Use this skill when a task needs visual inspection of atomic structures, adsorption geometries, slab-site context, or image-based sanity checks before or alongside numerical analysis.
---

# structure-visual-inspection

## Overview
Use rendered structure views as auxiliary evidence for geometry sanity checks, visual comparison, and reportable artifacts.

## Quick Start
1. Copy or execute `code/render_structure_panel.py` to render a first structure panel.
2. Tune camera vectors, fit, atom scale, supercell, tile size, and panel layout until the relevant geometry is visible.
3. Inspect the rendered panel with `read_file` when a multimodal read is actually needed.
4. Treat visual findings as supporting evidence.
5. Confirm critical geometric claims with numerical tools when available.

## Allowed tools
- execute
- read_file

## Workflow
1. Start with `python skills/materials_worker/structure-visual-inspection/code/render_structure_panel.py STRUCTURE -o OUTPUT.png --show-cell` from the repo root or copy the script into the writable workspace if the runtime needs a local artifact.
2. Tune deliberately. Common controls are `--supercell 2,2,1`, `--fit-scale 1.15`, `--padding 0.5`, `--atom-scale 0.45`, `--tile-size 1200,900`, and `--columns 2`.
3. For nonstandard orientations, write a small JSON list of view specs and pass `--views-json views.json`; each view has `name`, `title`, `camera_dir`, and `camera_up`.
4. Use `read_file` with a narrow question when you need multimodal help for orientation sanity checks, obvious clashes, site-family ambiguity, or pre/post-relax comparison.
5. When comparing candidates, keep the camera vectors, supercell, fit, atom scale, and visual context aligned across all images unless the report says why a candidate needed different settings.
6. If a visual conclusion would affect ranking, site assignment, or a scientific claim, hand back to numerical geometry tools or file-based metadata before finalizing.

## Method-critical defaults
- Visual evidence is never a substitute for exact bond lengths, coordination metrics, adsorption energies, or thermodynamic quantities.
- For slab-site interpretation, supercell context can materially change what looks like bridge/hollow/top. Make the chosen supercell explicit when it matters.
- Prefer the skill script for experiment-side inspection because camera and fit choices are explicit and reproducible. Do not add a generic OVITO backend to this helper; when OVITO is needed, write a task-specific workspace script with explicit renderer, camera, lighting, and export settings.
- If perspective or isometric views suggest a geometry issue, verify it with file-based structures or numeric geometry analysis before changing the workflow.
- Keep rendered legend and element identity explicit; do not infer species from color alone in the final answer.

## Output Contract
- Return the main panel image path.
- Return the script path and any non-default camera/rendering parameters used.
- If the analysis used multiple images, return all image paths referenced in the conclusion.
- State whether the visual conclusion is obvious, tentative, or requires numerical confirmation.
- When image analysis was used, summarize the concrete findings and the main uncertainty.

## References
- Use the rendered panel for interpretable reporting artifacts, but keep the final scientific conclusion grounded in structure files and numerical results.
