---
name: research-graph-writeback
description: Use this skill in Experiment or Literature Review work to decide whether newly verified scientific information should be written to the current turn-bound Research Graph, and to return source-grounded evidence to a parent when bound writeback tools are unavailable.
license: project-local
allowed-tools: "query_research_graph_sql record_bound_research_result mark_bound_research_experiment_failed"
---

# research-graph-writeback

## Overview

Use the turn-bound Research Graph as a concise scientific index while keeping the underlying run, artifact, note, or literature source as the evidence owner.

## Quick Start

1. Finish the scientific reading, execution, or analysis first.
2. If Graph tools are available, query only what is needed to check the target and avoid duplicate Results.
3. Write a Result only for new, source-supported scientific information; otherwise do not write.
4. Attach real owner refs and keep the Result concise enough to navigate back to them.

## Allowed tools

- `query_research_graph_sql`: inspect the Graph bound by the host to this turn.
- `record_bound_research_result`: record one distinct linked or standalone Result in that Graph.
- `mark_bound_research_experiment_failed`: record a concrete blocker only when the current focus is an Experiment.

If these tools are absent, return the evidence and owner paths to the parent agent. Do not infer a Graph target or perform another kind of Graph mutation.

## Workflow

### 1. Establish whether anything new was learned

A prepared input, downloaded paper, status check, explanation of an existing Result, ordinary wait, or completed chat turn is not itself a Result. A Result is a distinct observation, measurement, or derived scientific conclusion supported by evidence inspected in this turn.

Access or license state, hardware/platform readiness, scheduler or receipt state, software build, and performance telemetry are operational records, not Results. Keep them outside the Graph unless a known compatibility issue materially changed a scientific observation.

### 2. Check the relevant Graph context

Use `query_research_graph_sql` when the existing target, Results, or judgments are not already clear. Read enough owner evidence to confirm claim-critical conditions and numbers. Do not rely on a filename, final answer, or Graph summary alone.

Use the exact logical columns declared on the query tool's `sql` argument; do not guess convenience columns or inspect `sqlite_master`. Scientific fields such as `claim`, `objective`, and `summary` live inside `research_nodes.body_json`, while artifact fields such as `path`, `mime_type`, and `title` live inside `workspace_artifacts.payload_json`. Extract them with SQLite JSON1, for example `json_extract(n.body_json, '$.claim')` or `json_extract(a.payload_json, '$.path')`.

### 3. Record the smallest scientific unit

Use `record_bound_research_result` once per scientifically distinct outcome. The host links it to the focused Experiment when appropriate; without an Experiment focus, it remains standalone. Add only support, opposition, or inconclusive judgments warranted by the evidence.

### 4. Handle blockers narrowly

Use `mark_bound_research_experiment_failed` only for a specific condition that prevents the focused Experiment from obtaining a scientific result. Waiting, user questions, partial preparation, interruption, and an ordinary tool error are not scientific blockers.

## Method-critical defaults

- Preserve the actual method, conditions, comparison basis, and uncertainty needed to interpret the Result.
- Keep values and units faithful to the owner evidence; do not silently normalize or extrapolate them.
- Prefer zero writes over a duplicate or weakly supported Result.
- Do not invent an Experiment to explain a standalone literature or analysis Result.

## Output Contract

A Graph writeback contains a concise title and summary, the scientific details needed for interpretation, justified judgments when any, and real source refs. Without writeback tools, return the same evidence packet plus owner paths so the parent can decide whether to write it.

## References

The Research Graph is navigation, not the canonical raw record. Later Writing or Research work should follow Result refs to the original source before using claim-critical details.
