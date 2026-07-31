---
name: nature-academic-search
description: Use this skill for broad or focused literature discovery, evidence selection from search summaries, abstracts, and selectively read sources, deduplication, and final bibliography generation with CatMaster's active LitReview tools.
---

# Academic Search

## Overview

Build a literature argument from efficient web discovery and the best evidence actually available, escalating to source inspection or full text only when the scientific claim requires it.

## Quick Start

1. Define the review question, date/field boundaries, and expected coverage.
2. Use several complementary `web_search` queries to build a candidate pool and read available substantive summaries or abstracts.
3. Inspect or acquire a source only when a decision-critical detail cannot be resolved from those summaries.
4. Deduplicate selected DOIs and finalize the bibliography once at the end.

## Allowed tools

- `web_search` for efficient search-engine discovery.
- Filtered `agent_browser_*` tools as a fallback for dynamic pages, decision-critical source inspection, and user-authorized access.
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

Search summaries and abstracts are usable evidence for the claims they explicitly support. A title and bibliographic record alone establish discovery, not scientific detail. State a material limitation or lower confidence in ordinary language when a conclusion rests only on partial evidence; do not require a numeric score or a formal evidence tier for every paper.

Use the browser only when a dynamic page or user-authorized route is needed to resolve a decision-relevant detail, or when the user explicitly asks for full-paper reading. Make at most one reasonable access attempt for a selected source in an ordinary review. If it fails, state that the full text was not checked and continue with other sources rather than trying alternate pages, mirrors, or downloads repeatedly. Existing workspace attachments and lawful open-access copies remain valid alternatives.

Ingest acquired full text, then query focused evidence spans. Distinguish abstract/landing-page evidence from full-text page evidence in notes and claims.

Use `general-purpose` for one bounded topic branch or source-reading episode when it would otherwise inflate parent context. Require a compact result and durable artifact paths.

### 4. Synthesize by claims, not metadata volume

Organize the answer around the scientific question: material classes, active motifs, mechanism, activity/stability tradeoffs, operating conditions, evidence quality, and unresolved disputes. State which claims are supported by available summaries or directly read sources and which are interpretation.

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
- Browser use and the number of downloaded papers are never review-completion targets.
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
