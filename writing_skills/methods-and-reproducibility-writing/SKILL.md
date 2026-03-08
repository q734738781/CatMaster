---
name: methods-and-reproducibility-writing
description: Write reproducible computational methods sections with explicit workflow, parameters, and artifact traceability.
compatibility: local
metadata:
  catmaster-roles: "section_writer write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing methods reproducibility"
  catmaster-suggested-tools: "review_research_context read_research_pack bash_exec"
---

# methods-and-reproducibility-writing

## Overview
Write methods text that is reproducible, bounded, and specific about computational workflow, software, and artifact locations.

## Quick Start
Use when drafting methods for calculation setup, screening strategy, convergence flow, figure generation, or data assembly.

## Suggested tools
- `review_research_context`
- `read_research_pack`
- `bash_exec`

## Workflow
1. Gather workflow-critical settings, software versions, artifact refs, and script paths from structured packs and workspace outputs.
2. State what was actually done, not what is usually done in the field.
3. Write enough detail for rerun and review without drowning the section in raw logs.
4. Point to files, scripts, generated figures, or result bundles when they matter for reproducibility.
5. Keep interpretation out of Methods unless the journal format explicitly combines sections.

### Minimum reproducibility checklist

Capture these when available:

- software, code version, and execution environment
- model or system definition
- key parameters and thresholds
- search, screening, convergence, or filtering criteria
- post-processing or analysis steps
- script paths and output artifact paths
- any unavailable detail that blocks exact reproduction

### Writing pattern

Organize the section as:

1. workflow overview
2. input systems or datasets
3. computational or analytical procedure
4. post-processing and figure generation
5. availability of scripts, source data, or artifacts

### Failure modes

- Do not write generic boilerplate detached from the actual campaign.
- Do not hide important thresholds or exclusions in vague language.
- Do not omit script or artifact references when the paper depends on generated visuals or derived data.

## Output Contract
Return methods prose plus the artifact/script refs needed to support reproducibility review.

## References
- Prefer verified workflow details from the source campaign over generic boilerplate.
- Read [`../_references/section-patterns-and-story.md`](../_references/section-patterns-and-story.md) for the default Methods structure.
- Read [`../_references/submission-and-editorial-readiness.md`](../_references/submission-and-editorial-readiness.md) before marking the section as journal-ready.
