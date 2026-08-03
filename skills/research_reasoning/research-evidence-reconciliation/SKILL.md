---
name: research-evidence-reconciliation
description: Reconcile one Result with only the Hypotheses it genuinely tests, using canonical graph relations and decisive source content.
license: project-local
allowed-tools: "query_research_graph_sql query_literature_corpus acquire_literature_source read_document ls glob grep read_file web_search"
---

# research-evidence-reconciliation

## Overview

Judge one Result against its actual prediction and decision rule without grading papers, ranking branches, or inventing untested relationships.

## Quick Start

1. Query the Result, producing Experiment, tested Hypotheses, existing judgments, and direct refs.
2. Open the decisive source content and compare applicable older Results under matched conditions.
3. Separate observation, derived analysis, and causal interpretation.
4. Return only supported, opposed, or inconclusive relationships; an empty set is valid.

## Allowed tools

- `query_research_graph_sql`
- `query_literature_corpus`
- `acquire_literature_source`
- `read_document`
- `ls`
- `glob`
- `grep`
- `read_file`
- `web_search`

## Workflow

### 1. Establish what was tested

Query the Result, its producing Experiment, the Experiment's `tests` edges, decision rule, existing Result-to-Hypothesis relations, and direct refs. Compare conflicting older Results only when their relevant conditions are compatible.

### 2. Follow the real source owner

- For `artifact`, `thread`, and `message`, query the bound owner view, then follow a workspace path in the payload with `read_file`, `grep`, or `read_document`.
- For `note`, locate and read the workspace note with the generic filesystem tools.
- Treat corpus snippets as partial locators; continue result pages or open the identified source document.
- Retain `doi` and `url` as locators. Use selected-source acquisition only when full text is necessary and authorized; metadata alone is not article text.
- Treat `run` as provenance, not as scientific content.

### 3. Assign only direct relationships

Record `supports` only when a discriminating prediction is met, `opposes` only when it is contradicted, and `inconclusive` only when the Result directly tests the distinction but does not resolve it. Leave a Hypothesis unjudged when the Result does not directly test it under comparable conditions.

## Method-critical defaults

- Keep measured observations separate from derived analysis and interpretation.
- Report modality, conditions, independence or shared provenance, and limitations as scientific attributes.
- Do not calculate evidence strength, confidence, paper quality, venue prestige, or a composite score.
- Do not propose a new branch or select the next Experiment while acting as evidence judge.

## Output Contract

Return the Result ID and a flat set of directly justified Hypothesis relationships with concise reasons. Return an empty judgment set when no relationship is warranted; do not manufacture coverage.

## References

Use `research-graph-query` for the canonical logical schema and pagination rules before interpreting incomplete focus context.
