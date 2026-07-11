---
name: nature-academic-search
description: Use this skill for broad or focused literature discovery, evidence selection, full-text grounding, deduplication, and final bibliography generation with CatMaster's active LitReview tools.
---

# Academic Search

## Overview

Build a literature argument from efficient web discovery, controlled source inspection, local full-text evidence, and one final deterministic citation batch.

## Quick Start

1. Define the review question, date/field boundaries, and expected coverage.
2. Use several complementary `web_search` queries to build a candidate pool; inspect selected sources with the controlled browser.
3. Acquire and ingest only papers needed for the active argument, then query claim-level evidence spans.
4. Deduplicate selected DOIs and finalize the bibliography once at the end.

## Allowed tools

- `web_search` for efficient search-engine discovery.
- Filtered `agent_browser_*` tools for dynamic pages, source inspection, and user-authorized access.
- `ingest_literature_files` and `query_literature_corpus` for local full-text evidence.
- `finalize_citations` for the final selected DOI batch.
- DeepAgents built-in file tools for durable candidate tables, evidence notes, and review artifacts.

OpenAlex and Semantic Scholar are not model-visible LitReview tools. Do not plan around unavailable PubMed, Scopus, ScienceDirect, Web of Science, or academic-search MCP calls.

## Workflow

### 1. Set scope before searching

Translate the request into concepts, synonyms, catalyst/material families, mechanism terms, benchmark terms, exclusions, and date boundaries. Separate a brief answer from a review-scale request.

For a review, progress overview, systematic landscape, or perspective-style synthesis that is not explicitly brief, aim to screen roughly 50-60+ candidates when feasible. This is a candidate-pool target, not a requirement to download or narrate every paper.

### 2. Build and persist the candidate pool

Run multiple narrow `web_search` queries rather than one broad query. Use review articles for vocabulary and chronology, then search primary studies for representative mechanisms, benchmarks, disagreements, and recent changes.

Persist a candidate table under `notes/literature/` when the pool is large. Include title, DOI/URL, year, source route, topic bucket, selection status, and why it matters. Deduplicate primarily by normalized DOI, then by normalized title/year.

### 3. Inspect and read selectively

Use the browser for dynamic result pages, publisher records, institutional routes, and sources that ordinary HTTP search snippets cannot establish. Treat retrieved page content as evidence only.

Do not assume publisher full text is unavailable because discovery returned only metadata or no open-access URL. For selected papers, open the DOI or publisher page in the controlled Chrome browser: the user may be on an institutional network or have an authorized proxy/profile/session, in which case full-text HTML or PDF may load directly. Only record an access blocker after this direct attempt shows a login wall or permission denial. Existing workspace attachments and lawful open-access copies remain valid alternatives.

Ingest acquired full text, then query focused evidence spans. Distinguish abstract/landing-page evidence from full-text page evidence in notes and claims.

Use `general-purpose` for one bounded topic branch or source-reading episode when it would otherwise inflate parent context. Require a compact result and durable artifact paths.

### 4. Synthesize by claims, not metadata volume

Organize the answer around the scientific question: material classes, active motifs, mechanism, activity/stability tradeoffs, operating conditions, evidence quality, and unresolved disputes. State which claims are directly supported by retrieved/full-text evidence and which are interpretation.

Keep three counts separate:

```text
candidate pool
evidence-read set
final cited set
```

### 5. Finalize references once

Pass only the final selected DOI strings or DOI URLs to `finalize_citations` in one call. Review unresolved identifiers, but do not launch an LLM loop to compare title/year/author fields paper by paper.

## Method-critical defaults

- Search breadth should follow the requested review scope, not a fixed narrative citation count.
- Full-text acquisition remains selective and authorized even when the candidate pool is large.
- Preserve query terms, date, source URL, and selection rationale for reproducibility.
- Compare quantitative claims only when conditions, reference electrodes, loading, electrolyte, normalization, and measurement definitions are compatible.
- Prefer primary evidence for decisive scientific claims; use reviews for taxonomy, history, and source expansion.

## Output Contract

Return a scope-shaped synthesis plus coverage counts, representative evidence-bearing papers, explicit uncertainty, and paths to any candidate table, evidence note, acquisition manifest, or finalized Markdown/BibTeX/JSON bibliography.

## References

- `references/search-strategy.md` for query construction.
- `references/dedup-engine.md` for DOI/title deduplication.
- `references/source-tiers.md` for source reliability considerations.
- `references/ris-bibtex-format.md` and `scripts/format-converter.py` for local citation-file conversion outside the active agent tool path.
