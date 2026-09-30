---
name: research-graph-query
description: Query the complete current thread-bound Research Graph with standard read-only SQL when a partial focus snippet is insufficient.
license: project-local
allowed-tools: "query_research_graph_sql"
---

# research-graph-query

## Overview

Use read-only SQL to recover relevant graph state beyond the supplied focus snippet.

The focus snippet identifies the current working node. Queries do not change focus; return exact node IDs for findings about another branch.

## Quick Start

1. Query the focus node and its direct incoming and outgoing relations.
2. Follow only the hypotheses, results, refs, dependencies, launches, or planning rows needed for the task.
3. Use deterministic SQL pagination if the relevant row set does not fit one query.
4. Treat SQL order as display order, never as a scientific recommendation.

## Allowed tools

- `query_research_graph_sql`

## Workflow

### 1. Use the bound logical schema

The tool accepts one required `sql` string and queries the binding supplied for
the current turn, returning that target as `graph_id`. Older bindings, titles and
completion flags do not change it. A query does not select or rebind a graph.
Recorded questions and completion criteria describe the saved stage; use them as
context for the current assignment. Never qualify a table with `main` or query
SQLite schema tables.

Use the logical tables and columns declared in the query tool's `sql` schema.
Historical launch/planning records explain earlier coordination; they are not a
current work queue. Query them only when the task needs that history.

Do not guess flattened convenience columns. For example, `research_nodes` has no direct `claim`, `summary`, or `status` column (`state` is the canonical state column), and `workspace_artifacts` has no direct `path`, `mime_type`, `title`, or `description` column. Extract those payload fields explicitly:

```sql
SELECT n.node_id, n.state,
       json_extract(n.body_json, '$.claim') AS claim
FROM research_nodes AS n
```

```sql
SELECT a.artifact_id,
       json_extract(a.payload_json, '$.path') AS path,
       json_extract(a.payload_json, '$.mime_type') AS mime_type
FROM workspace_artifacts AS a
```

### 2. Recover the scientific neighborhood

Filter `research_nodes.node_id` by the supplied focus ID, then inspect incoming
and outgoing `research_edges` through `source_node_id` and `target_node_id`.
Join both endpoints to `research_nodes` when their scientific content is needed;
retain relation, scope, rationale and action. Do not infer these relationships
from similar titles.

For runnable eligibility, inspect every ready Experiment and require each `depends_on` target to have state `has_results`. Query refs separately, then open only decisive or conflicting sources through their normal owner. DOI and URL refs are locators, not source text.

### 3. Check before staging or judging

Cover the task-relevant focus, relations, prior Results and counterevidence. Inspect duplicate hypotheses when adding one, and true dependencies and the runnable frontier when choosing execution. Use ordinary `LIMIT` with `OFFSET` or a keyset for pagination and continue until the task-relevant rows are covered.

## Method-critical defaults

- Preserve typed relation direction; do not infer an edge from title similarity.
- Determine frontier eligibility only from Experiment state and satisfied dependencies.
- Do not use importance, cost, creation order, or SQL row order as route value.
- Do not treat `body_json`, refs, or owner payloads as complete until the required fields or sources have been opened.
- Treat platform availability, access/license state, hardware/software-build readiness, scheduler/receipt state, and performance telemetry as operational constraints, not scientific Hypotheses, decision rules, Results, or proposal branches.

## Output Contract

Return the exact node and relation IDs used for the conclusion, identify any remaining paginated rows that were not examined, and explain the supported interpretation or next check. Persist scientific changes through the available graph mutation tools.

## References

Use `research-evidence-reconciliation` for the Result-to-next-Hypothesis/Experiment reasoning and stopping reconsideration.

## Revisions and open research decisions

`revises` points from the new H or R to the older same-kind record. Inspect `action` (`replace`, `qualify`, `withdraw`), `scope` and `rationale` before using either record. Opposite observations under different conditions can coexist. Judgment edges retain their own conditions; a node color is only a relation summary.

`research_decisions.body_json` records the unresolved problem, user scope, stopping reason, basis node IDs, disposition, validation Result IDs and resumption conditions. `review_json` records its independent assessment and any recommended validation Experiment. Read existing decisions before repeating a review on the same evidence. These are key scientific decisions, not a task queue.

Start with the relevant neighborhood, then search other graph records and follow revision/counterevidence links as needed. A focus radius or the latest records are not an access boundary. Use recursive CTEs and standard pagination to follow longer chains.
