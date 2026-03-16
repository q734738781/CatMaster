---
name: catalysis-prior-art-and-benchmarking
description: Use this skill when a catalysis workflow needs literature-grounded benchmark papers, prior-art mapping, representative method conventions, or evidence-backed comparisons.
license: project-local
compatibility: local
allowed-tools: "run_literature_research"
metadata:
  catmaster-suggested-tools: "run_literature_research"
---

# catalysis-prior-art-and-benchmarking

## Overview
Use this skill to ground catalysis planning in representative papers, benchmark conventions, and method choices reported in the literature.

## Quick Start
- For broad catalysis background or public benchmark context, online/web search may be enough.
- For representative papers on a catalyst/system, call `run_literature_research` with `depth=quick`.
- For adsorption/reference/dispersion conventions, use `depth=standard`.
- For narrow method disputes or evidence checks, use `depth=focused`.
- For full survey-style background reports, use `depth=deep_report` only when explicitly requested.

## Suggested tools
- run_literature_research

## Workflow
### 1. Ask a literature question that matches the scientific decision
Frame the query around the actual scientific need: representative papers, benchmark systems, reference-state conventions, dispersion policy, model chemistry, or open questions.

### 2. Keep literature aligned with the active workflow stage
Use literature to support planning decisions, not to replace direct tool outputs. A literature pack should refine scope, defaults, benchmarks, and evidence standards for the current catalysis workflow.
Prefer plain web/online exploration for broad orientation; use `run_literature_research` when you need paper-level citations, benchmark conventions, or a reusable evidence pack.

### 3. Extract benchmark conventions, not just paper titles
A good literature result for catalysis should identify what systems are commonly used, what comparison conventions recur, and where the literature disagrees.

## Method-critical defaults
- If literature is used to justify adsorption-energy conventions, reference states, dispersion treatment, or benchmark settings, those choices must be made explicit in proposal text or task packets.
- Do not mix literature-derived benchmark expectations with current-run numerical results.
- Use literature to bound method choices and evidence standards; final execution settings still need explicit task-level decisions.

## Output Contract
Return a literature pack that includes representative catalysis papers, a concise benchmark/method-convention summary, compact citations, and any open questions that still need project-specific resolution.

## References
- Pair with `literature-grounding` for general depth selection and trigger rules.
- Pair with `computational-heterogeneous-catalysis` when literature grounding is feeding into an execution plan.
