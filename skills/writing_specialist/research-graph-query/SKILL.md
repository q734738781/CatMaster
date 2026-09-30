---
name: research-graph-query
description: Use this skill when Writing needs to locate graph-linked Results, conflicting evidence, and source references before drafting or revising from a bound workspace Research Graph.
license: project-local
allowed-tools: "query_research_graph_sql"
---

# research-graph-query

## Overview

Use the bound Research Graph as a compact scientific index, then inspect the original sources needed for the writing claim.

## Quick Start

1. Use the current focus when present, or the node IDs and writing target supplied with the task.
2. Query relevant Hypotheses, all directly related Results, and their typed relations.
3. Follow Result refs to the necessary note, artifact, run, thread message, DOI, or URL.
4. Search the wider workspace only when the graph does not cover the claim.

## Allowed tools

- `query_research_graph_sql`

## Workflow

### 1. Locate the claim neighborhood

Query `research_nodes` and `research_edges` from the focus, handoff, or known node IDs. Include `supports`, `opposes`, and `inconclusive` Results; do not select only evidence that supports the intended narrative. Use `body_json` to locate the relevant observation, analysis, interpretation, conditions, and provenance fields.

### 2. Recover experiments and sources

For selected Results, query their producing Experiment and `research_refs`. The query tool declares the available logical tables and columns. `workspace_artifacts` and `thread_messages` contain only sources reachable through refs on the selected graph.

Use only the exact columns declared on the query tool's `sql` argument. Node-specific scientific fields are inside `research_nodes.body_json`; artifact locators such as `path`, `mime_type`, and `title` are inside `workspace_artifacts.payload_json`. Use SQLite JSON1 expressions such as `json_extract(n.body_json, '$.summary')` and `json_extract(a.payload_json, '$.path')` instead of guessing flattened columns or querying `sqlite_master`.

### 3. Read decisive owner material

Treat Result summaries and owner payloads as locators. Before using claim-critical numbers, conditions, mechanisms, or limitations, open the referenced owner through the normal workspace or source-reading capability. A DOI or URL identifies a source; it is not the source text.

### 4. Build the bounded writing handoff

Summarize the relevant findings and their meaning for the writing question, with source paths or Result IDs for selective recovery. Keep exact values, contrary evidence and conditions that change interpretation. If the graph is incomplete, perform a local workspace search and state the uncovered area rather than guessing another graph.

## Method-critical defaults

- Queries use the binding supplied for the current turn and return its `graph_id`;
  they do not select or rebind a graph. Never infer another target from older turns,
  titles, paths or completion flags. The recorded graph question is evidence context,
  not a replacement for the current writing brief.
- Preserve edge direction and typed relations. Creation order, importance, or SQL row order is not evidence strength.
- Use standard `LIMIT` plus `OFFSET` or keyset pagination when needed; do not assume an incomplete page is the complete evidence set.
- The graph accelerates navigation but does not replace the original data, literature, scripts, or records.

## Output Contract

Return a concise answer to the evidence question with the relevant findings, scientific meaning and inspected source handles. Mention a graph coverage gap when it affects the answer. This is a natural-language handoff, not a fixed field packet or a new persistent evidence schema.

## References

Use the ordinary writing skills for argument design and drafting after the evidence neighborhood has been resolved.
