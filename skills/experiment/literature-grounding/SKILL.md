---
name: literature-grounding
description: Use this skill when the user asks for papers, prior work, benchmark conventions, prior-art mapping, or other explicit evidence grounding from published work, including heterogeneous-catalysis method benchmarks.
license: project-local
compatibility: local
allowed-tools: "run_literature_research"
---

# literature-grounding

## Overview
Use this skill to run the smallest literature-grounding workflow that can answer a paper, benchmark, or prior-art question with reusable evidence.

## Quick Start
- For broad public-background exploration or quick web context, ordinary online/web search is often enough.
- If the user asks for papers, prior work, representative literature, supporting evidence, benchmark conventions, prior-art mapping, or a reusable citation pack, call `run_literature_research`.
- When calling `run_literature_research`, keep `query` as the literature question and provide a short domain phrase in `topic` when possible; scholarly databases should not receive the full reporting instruction text.
- Within literature research itself, default to web-first orientation. Only escalate to OpenAlex / Semantic Scholar when paper metadata, citations, DOI/year/venue details, or explicit literature grounding are actually needed.
- Default to `depth=quick` for representative-paper requests.
- Use `depth=standard` for method conventions, benchmark framing, catalyst-system prior art, or comparative evidence.
- Use `depth=focused` for narrow method disputes, conflicting conventions, or targeted evidence checks.
- Use `depth=deep_report` only for explicit deep-review or survey-style requests.

## Allowed tools
- run_literature_research

## Workflow
### 1. Decide whether literature is actually needed
Only trigger literature work when the user explicitly asks for papers, prior work, supporting evidence, benchmark context, prior-art mapping, or literature-grounded method conventions.
Use normal web/online search first when the need is broad background, public-page summaries, or lightweight orientation rather than paper-level grounding.

### 2. Choose the smallest useful depth
Use `quick` for representative papers and fast grounding. Use `standard` for conventions, benchmark framing, catalyst-system prior art, or related-work summaries. Use `focused` when a narrow scientific question needs targeted evidence or when the literature disagrees on a method choice. Reserve `deep_report` for explicit deep-review requests.

### 3. Frame the literature question around the scientific decision
Ask for the literature object you actually need: representative papers, benchmark systems, reference-state conventions, dispersion policy, model chemistry, or open methodological disagreement. A good pack should extract conventions and decision-relevant comparisons, not just titles.

### 4. Route sources conservatively
Do not automatically hit both scholarly metadata APIs and public web in the same first pass. Start with public web when the request is broad, contextual, or public-summary-friendly. Use OpenAlex / Semantic Scholar when the user clearly needs paper-level grounding or when the web pass is too weak.

### 5. Keep the pack clean
Use the returned literature context pack as the planning/evidence object. Do not surface the raw retrieval process, intermediate search noise, or browser-style exploration steps to the main agent context.

## Method-critical defaults
- Literature grounding is not a default-on behavior. It should be explicitly requested by the user or justified by the planning need.
- Treat `run_literature_research` as a precise grounding path, not the default replacement for ordinary web search.
- Quick paper-finding requests should not silently escalate into deep literature reviews.
- When literature is used to justify quantitative settings, benchmark conventions, adsorption/reference-state policy, or dispersion treatment, carry those conclusions forward explicitly into proposal text or task packets.
- Do not mix literature-derived benchmark expectations with current-run numerical results; keep literature priors and project outputs separated.

## Output Contract
Return a compact literature pack with a short summary, a small key-paper set, citations/links, a concise convention-or-benchmark summary when relevant, confidence, and follow-up questions when uncertainty remains.

## References
- Pair with the active execution-stage skill after grounding the question; literature should refine planning defaults and evidence standards, not replace downstream structure or execution tools.
