---
name: research-graph-control
description: Use this skill when a falsifiable research problem must continue across experiments or threads, or when a result should change the next hypothesis or experiment.
license: project-local
compatibility: CatMaster Research Specialist
allowed-tools: "list_research_graphs create_research_graph inspect_research_graph add_research_hypothesis add_research_experiment record_research_result mark_research_experiment_failed task"
---

# research-graph-control

## Overview

Maintain the shared scientific situation as one explicitly selected workspace Research Graph while leaving detailed notes, files, calculations, and receipts in their owning stores.

## Quick Start

Use the graph ID injected for a bound turn. If no graph is bound, list graphs and ask the user to choose when several are relevant; create one only when the work is genuinely multi-step or evidence-driven. A newly created graph is attached to the current thread by trusted runtime context, so do not ask for or invent a thread ID. Inspect before mutating, pass the inspected revision to one mutation, and inspect again after a revision conflict.

## Allowed tools

- `list_research_graphs`
- `create_research_graph`
- `inspect_research_graph`
- `add_research_hypothesis`
- `add_research_experiment`
- `record_research_result`
- `mark_research_experiment_failed`
- `task`

## Workflow

### 1. Select the graph explicitly

Continue the graph named in the bound context. Never choose among multiple graphs by semantic similarity. A one-off answer does not need a graph; use one when the problem has falsifiable hypotheses, several evidence-producing steps, cross-thread work, or a result that changes the next decision.

For a new graph, preserve the user's research question and add only hypotheses they supplied or hypotheses developed through a bounded scientific planning step. Do not create paper, person, method, debug, or generic concept nodes.

### 2. Add a complete scientific next step

A hypothesis needs a claim, rationale, observable predictions, and a coarse relative importance within the selected graph. An experiment proposal needs an objective, a concise executable plan, a decision rule, the owning execution lane, expected decision value, and coarse estimated compute cost. Use `draft` while a plan is incomplete and `ready` only when another researcher could prepare and run it without inventing missing scientific choices.

Keep scientifically distinct competing branches when evidence does not yet distinguish them. Use only `low`/`medium`/`high` for importance and expected value, and `none`/`low`/`medium`/`high` for compute cost. These fields order ready work; they do not state confidence, truth, or success probability. Connect hypotheses to experiments through their typed inputs. Add experiment dependencies only for true execution prerequisites; they must remain acyclic. Do not add confidence, novelty, composite scoring, precise invented resource estimates, layout, prompt, token, or generic metadata.

### 3. Execute through the normal owner

User or scheduler launches are handled by the graph launch service and ordinary permission boundaries. When coordinating a launched child, preserve its graph and experiment IDs. Put detailed inputs, calculation output, logs, managed-execution receipts, figures, and reports in normal workspace artifacts or runs rather than copying them into the graph.

### 4. Record evidence and continue

Record one concise Result for each scientifically meaningful attempt. Link it to its producing experiment and attach exact run, artifact, note, DOI, URL, thread, or message refs. Judge its effect independently for every relevant hypothesis as `supports`, `opposes`, or `inconclusive`; support and opposition may coexist and never make a hypothesis terminal.

Use `result --suggests--> hypothesis` when evidence motivates a new hypothesis. A Result can lead to another hypothesis or follow-up experiment, so the cycle may continue. If execution cannot proceed, record the concrete blocker with `mark_research_experiment_failed`; do not turn debugging branches into scientific hypotheses.

## Method-critical defaults

- Pass an explicit graph ID and the latest inspected graph revision to every mutation.
- On a stale-revision conflict, inspect and reconcile; never overwrite another thread's edit.
- Keep hypothesis claims falsifiable and experiment decision rules outcome-specific.
- Keep all competing evidence visible; do not collapse it to one confidence score.
- Preserve parallel hypothesis and experiment branches; automatic orchestration choosing one next launch does not remove the others.
- Use refs rather than copying long source content into graph nodes.
- Do not launch follow-up work merely because a frontier exists; respect the user's requested stopping point and normal approval rules.

## Output Contract

At handoff, state the graph title, the focused hypothesis, experiment, or result,
the typed relation changed, the concise scientific consequence, durable source
refs, and the next runnable or blocked decision. Keep the numeric revision for
tool concurrency rather than presenting it as scientific content. Mention
omitted nodes when working from a bounded graph view.

## References

- `devdocs/0728_major_update/04_workspace_research_graph.md`
- `catmaster/research/knowledge_graph/`
