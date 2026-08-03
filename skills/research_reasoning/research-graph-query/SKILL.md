---
name: research-graph-query
description: Query the complete current thread-bound Research Graph with standard read-only SQL when a partial focus snippet is insufficient.
license: project-local
allowed-tools: "query_research_graph_sql"
---

# research-graph-query

## Overview

Use the bound read-only SQLite projection to recover canonical graph state that is absent from the injected partial focus snippet.

## Quick Start

1. Query the focus node and its direct incoming and outgoing relations.
2. Follow only the hypotheses, results, refs, dependencies, launches, or planning rows needed for the task.
3. Use deterministic SQL pagination if the relevant row set does not fit one query.
4. Treat SQL order as display order, never as a scientific recommendation.

## Allowed tools

- `query_research_graph_sql`

## Workflow

### 1. Use the bound logical schema

The tool accepts one required `sql` string. Graph identity and revision come from the trusted thread. Never qualify a table with `main` or query SQLite schema tables.

- `research_graphs(graph_id, title, question, completion_criterion, completed, orchestration_mode, archived, revision, created_at, updated_at)`
- `research_nodes(graph_id, node_id, kind, title, state, body_json, revision, created_at, updated_at)`
- `research_edges(graph_id, source_node_id, target_node_id, relation)`
- `research_refs(graph_id, node_id, ref_kind, ref_id)`
- `research_launches(launch_id, graph_id, experiment_node_id, idempotency_key, status, thread_id, run_id, lease_owner, lease_until, created_at, updated_at)`
- `research_planning(planning_id, graph_id, start_revision, status, thread_id, preview_json, lease_until, created_at, updated_at)`
- `workspace_artifacts(artifact_id, thread_id, payload_json, created_at, updated_at)`
- `thread_messages(row_id, thread_id, message_id, created_at, updated_at, payload_json, message_role, message_run_id)`

`body_json` and owner payloads are raw JSON. SQLite JSON1, joins, aggregates, window functions, and recursive CTEs are available. The two owner tables contain only rows reachable through this graph's `artifact`, `thread`, or `message` refs.

### 2. Recover the scientific neighborhood

Start from the focus rather than scanning titles:

```sql
SELECT n.node_id, n.kind, n.title, n.state, n.body_json
FROM research_nodes AS n
WHERE n.node_id = 'the_focus_id'
```

Then inspect typed relations and both endpoint bodies:

```sql
SELECT e.relation, source.node_id AS source_id, source.body_json AS source_body,
       target.node_id AS target_id, target.body_json AS target_body
FROM research_edges AS e
JOIN research_nodes AS source ON source.node_id = e.source_node_id
JOIN research_nodes AS target ON target.node_id = e.target_node_id
WHERE e.source_node_id = 'the_focus_id' OR e.target_node_id = 'the_focus_id'
ORDER BY e.relation, source.node_id, target.node_id
```

For runnable eligibility, inspect every ready Experiment and require each `depends_on` target to have state `has_results`. Query refs separately, then open only decisive or conflicting sources through their normal owner. DOI and URL refs are locators, not source text.

### 3. Check before staging or judging

Cover the focus and direct relations, relevant older Results and opposite judgments, duplicate Hypotheses, true execution dependencies, and the complete runnable frontier. Use ordinary `LIMIT` with `OFFSET` or a keyset for pagination and continue until the task-relevant rows are covered.

## Method-critical defaults

- Preserve typed relation direction; do not infer an edge from title similarity.
- Determine frontier eligibility only from Experiment state and satisfied dependencies.
- Do not use importance, cost, creation order, or SQL row order as route value.
- Do not treat `body_json`, refs, or owner payloads as complete until the required fields or sources have been opened.

## Output Contract

Return the exact node and relation IDs used for the conclusion, identify any remaining paginated rows that were not examined, and either stage one supported plan or report a genuine no-change outcome.

## References

Use `research-evidence-reconciliation` when the task is to judge a completed Result rather than propose a route.
