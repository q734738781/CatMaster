---
name: research-graph-writeback
description: Record or amend an evidence-backed scientific finding through bound Experiment or Literature Review graph tools, including during an ongoing study. Use for scientific writeback, not preparation or status updates.
license: project-local
---

# Research Graph writeback

Record a distinct observation, measurement or derived conclusion supported by
inspected evidence. Preparation, downloads, status updates and explanations of
existing Results do not themselves create a new Result. Operational records stay
with their owners unless a concrete compatibility issue changes the science.
Record a ready finding during a longer study; campaign completion or an existing
Graph Result is not a prerequisite for first writeback.

Use bound writeback tools when exposed. Without a selected graph or writeback
tools, return the finding and source paths without guessing another target. Query only the
context needed to identify the producing Experiment or avoid duplicates; use the
query tool's declared SQL columns, including JSON1 for fields inside `body_json`.
A new graph query is unnecessary when the relevant context is already available.

## Record the scientific unit

Populate `methods` with the actual data/evidence, representation, baseline, comparison,
analysis and validation needed to know what was tested. Put observations in `summary`
and the supported interpretation, limits and next discriminating question in `conclusion`.
For literature work, describe the actual search scope and comparison of evidence.
Do not infer that an entire hypothesis or method family failed from one narrow test.
When amending a Result, omitted method/conclusion fields preserve their previous values.
The placeholder "missing due to old record" means those details were never recorded.

An Experiment-produced execution Result requires an explicit Experiment focus.
For delegated work, reuse the existing Experiment with the same objective and
decision rule, including one registered at dispatch. A Graph binding alone does
not select its producing Experiment. Use `set_research_graph_focus` with that
node's actual ID, then `record_bound_research_result`. For example, an existing
`exp_123` is selected with `set_research_graph_focus(node_id="exp_123")`;
starting a separate execution thread does not create a new scientific branch.
Use `create_bound_research_experiment` only for authorized work with no matching
Experiment. A Literature Review finding does not need a retrospective Experiment:
when no Experiment focus exists, record it as a sourced standalone Result.

Use `record_bound_research_result` once per scientifically distinct outcome. If the same run later corrects or completes that observation, use `update_bound_research_result` with the existing result_node_id instead of creating a duplicate. Add only support, opposition, or inconclusive judgments warranted by completed scientific evidence, with their actual scope and scientific rationale. A separate evidence-summary agent is not a prerequisite. Reuse an existing Result when interpreting the same observation; independent reasoning can revise its judgment without duplicating the Result.

## Handle blockers narrowly

Use `mark_bound_research_experiment_failed` only after a real scientific execution route is exhausted by a specific condition that prevents the focused Experiment from obtaining a scientific result. Preparation, authorization, recovery, scheduler handling, interruption, an ordinary tool error, and a successful operational task may finish without a Result and are not scientific blockers.

A duplicate plan is not a failed experiment. Preserve and identify existing
duplicates in the handoff rather than marking successful work failed to tidy the graph.

After resolving a recorded blocker, use `resume_bound_research_experiment` rather than creating a replacement Experiment. Use `retract_bound_research_result` only for a mistaken Result created in the current run, before any judgment, when it is the focused Experiment's sole produced Result. For a later scientific withdrawal, preserve the old Result and report the new evidence and affected scope; add a `revises` relation only when that mutation is available. Do not delete established scientific history.

## Handoff

Keep actual methods, conditions, comparisons, values, units and uncertainty faithful
to the evidence. Use real source refs so later research or writing can inspect it.
Prefer no write to a duplicate or unsupported Result. Return the finding and owner
paths; no separate writeback report or mandatory operational inventory is needed.

## Tool scope and examples

Creation tools return graph_id, node_ids and the current revision. Reuse the returned node ID for focus/update operations; the graph target stays bound automatically. Each successful Experiment creation adds a new node; query current state before retrying an uncertain creation. `resume_bound_research_experiment` clears the focused Experiment's recorded blocker; it does not launch or resume computation.
