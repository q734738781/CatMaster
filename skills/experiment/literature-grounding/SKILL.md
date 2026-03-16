---
name: literature-grounding
description: Use this skill when the user asks for papers, prior work, literature support, benchmark conventions, or other explicit evidence grounding from published work.
license: project-local
compatibility: local
allowed-tools: "run_literature_research"
metadata:
  catmaster-suggested-tools: "run_literature_research"
---

# literature-grounding

## Overview
Use this skill to decide when a request needs literature grounding and what depth of literature research is appropriate.

## Quick Start
- For broad public-background exploration or quick web context, ordinary online/web search is often enough.
- If the user asks for papers, prior work, representative literature, supporting evidence, benchmark conventions, or a reusable citation pack, call `run_literature_research`.
- When calling `run_literature_research`, keep `query` as the literature question and provide a short domain phrase in `topic` when possible; scholarly databases should not receive the full reporting instruction text.
- Within literature research itself, default to web-first orientation. Only escalate to OpenAlex / Semantic Scholar when paper metadata, citations, DOI/year/venue details, or explicit literature grounding are actually needed.
- Default to `depth=quick` for representative-paper requests.
- Use `depth=standard` for method conventions, benchmark framing, or comparative evidence.
- Use `depth=deep_report` only for explicit deep-review or survey-style requests.

## Suggested tools
- run_literature_research

## Workflow
### 1. Decide whether literature is actually needed
Only trigger literature work when the user explicitly asks for papers, prior work, supporting evidence, benchmark context, or literature-grounded method conventions.
Use normal web/online search first when the need is broad background, public-page summaries, or lightweight orientation rather than paper-level grounding.

### 2. Choose the smallest useful depth
Use `quick` for representative papers and fast grounding. Use `standard` for conventions, benchmark framing, or related-work summaries. Use `focused` when a narrow scientific question needs targeted evidence. Reserve `deep_report` for explicit deep-review requests.

### 3. Route sources conservatively
Do not automatically hit both scholarly metadata APIs and public web in the same first pass. Start with public web when the request is broad, contextual, or public-summary-friendly. Use OpenAlex / Semantic Scholar when the user clearly needs paper-level grounding or when the web pass is too weak.

### 4. Keep the pack clean
Use the returned literature context pack as the planning/evidence object. Do not surface the raw retrieval process, intermediate search noise, or browser-style exploration steps to the main agent context.

## Method-critical defaults
- Literature grounding is not a default-on behavior. It should be explicitly requested by the user or justified by the planning need.
- Treat `run_literature_research` as a precise grounding path, not the default replacement for ordinary web search.
- Quick paper-finding requests should not silently escalate into deep literature reviews.
- When literature is used to justify quantitative settings or benchmark conventions, carry those conclusions forward explicitly into proposal text or task packets.

## Output Contract
Return a compact literature pack with a short summary, a small key-paper set, citations/links, confidence, and follow-up questions when uncertainty remains.

## References
- Use `catalysis-prior-art-and-benchmarking` when the literature task is specifically about heterogeneous-catalysis benchmarks or method conventions.
