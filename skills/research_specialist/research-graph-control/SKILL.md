---
name: research-graph-control
description: Use this skill when a falsifiable research problem must continue across experiments or threads, or when a result should change the next hypothesis or experiment.
license: project-local
allowed-tools: "list_research_graphs create_research_graph query_research_graph_sql add_research_hypothesis add_research_experiment stage_research_plan record_research_result set_research_result_judgment mark_research_experiment_failed task"
---

# research-graph-control

## Overview

Maintain the shared scientific situation as one explicitly selected workspace Research Graph while leaving detailed notes, files, calculations, and receipts in their owning stores.

## Quick Start

Use the graph ID injected for a bound turn. If no graph is bound, list graphs and ask the user to choose when several are relevant; create one only when the work is genuinely multi-step or evidence-driven. A newly created graph is attached to the current thread by trusted runtime context, so do not ask for or invent a thread ID. Query canonical state before mutating, use that revision for the next mutation, and query again after a revision conflict.

## Allowed tools

- `list_research_graphs`
- `create_research_graph`
- `query_research_graph_sql`
- `add_research_hypothesis`
- `add_research_experiment`
- `stage_research_plan`
- `record_research_result`
- `set_research_result_judgment`
- `mark_research_experiment_failed`
- `task`

## Workflow

### 1. Select the graph explicitly

Continue the graph named in the bound context. Never choose among multiple graphs by semantic similarity. A one-off answer does not need a graph; use one when the problem has falsifiable hypotheses, several evidence-producing steps, cross-thread work, or a result that changes the next decision.

For a new graph, preserve the user's research question and add only hypotheses they supplied or hypotheses developed through a bounded scientific planning step. Do not create paper, person, method, debug, or generic concept nodes.

In an internal graph-planning turn, delegate model-generated branch formation to `hypothesis_proposer`. The proposer queries the bound graph, searches evidence when useful, publishes meaningful provisional branches through its bound `stage_research_plan` tool, and returns a concise scientific memo in ordinary language. After staging, delegate current-revision Experiment comparison to `experiment_evaluator`; it publishes innovation and conservative recommendations without changing durable nodes. Do not ask either delegate for JSON, copy a tool payload into another call, or make it repeat graph/revision identifiers. Outside an internal planning turn, preserve user-supplied hypotheses and observations directly.

### 2. Add the smallest useful scientific next step

A user may contribute only a brief scientific statement. Preserve that input without inventing missing detail: a Hypothesis may start with only its claim, a draft Experiment may start with only its objective, and a Result may start with only its observation. Add rationale, observable predictions, sources, and coarse priority fields when they are known or can be developed from evidence. A ready Experiment is different: it needs a concise executable plan, a decision rule, the owning execution lane, and any known coarse estimated compute cost so another researcher can run it without inventing missing scientific choices. Keep an incomplete proposal as `draft`.

Keep as many scientifically distinct competing branches as the current evidence warrants, and stop when another branch would only repeat an existing one. Do not aim for a fixed Hypothesis/Experiment count or ratio. Temporary planning Experiments may remain drafts with only an objective; only a route with a usable plan and decision rule is ready for automatic execution. Hypothesis importance is an optional user priority and compute cost is an optional execution constraint; neither is scientific evidence or a host selection rule. Connect hypotheses to experiments through their typed inputs. Add experiment dependencies only for true execution prerequisites; they must remain acyclic. Do not add confidence, novelty, composite evidence scoring, precise invented resource estimates, layout, prompt, token, or generic metadata.

Choose node granularity by scientific decision, not by procedure. Preparation, source acquisition, structure generation, format conversion, parameter or convergence checks, smoke tests, individual conditions or replicates, and analysis stay inside one Experiment when they serve the same Hypothesis and decision rule. Split a step into another Experiment only when it can produce a standalone scientific Result that would change the next decision even if later steps never run. Operational fallback steps and project phases are not separate Hypotheses unless they assert genuinely distinct falsifiable claims. Likewise, keep observations from one execution in one Result when they jointly answer the same decision rule; split only when their scientific effects differ.

### 3. Execute through the normal owner

User or scheduler launches are handled by the graph launch service and ordinary permission boundaries. When coordinating a launched child, preserve its graph and experiment IDs. Put detailed inputs, calculation output, logs, managed-execution receipts, figures, and reports in normal workspace artifacts or runs rather than copying them into the graph.

### 4. Record evidence and continue

Record one concise Result for each scientifically meaningful observation or derived outcome. In ordinary scientific language, separate the observation or measurement from analysis and causal interpretation; include modality, applicable conditions, and provenance only when they change what the Result means. Access/license state, hardware/platform readiness, scheduler/receipt state, software build, and performance telemetry stay in their operational owner records rather than Hypotheses, decision rules, or Results unless a known compatibility issue materially changed the scientific observation. Do not assign the Result a global evidence grade. Link it to its producing graph experiment when it came from one. For a literature finding, collaborator result, historical observation, or other evidence obtained outside the graph, leave the producing experiment empty instead of inventing a retrospective Experiment node. Attach the exact run, artifact, note, DOI, URL, thread, or message ref when one exists. Judge only the hypothesis effects that the evidence actually addresses as `supports`, `opposes`, or `inconclusive`; these edges describe the Result-to-Hypothesis relation, not evidence strength. A new observation may remain unjudged until the next planning step. Use `set_research_result_judgment` to add, replace, or clear that one Result-to-Hypothesis judgment after inspecting the current revision. Support and opposition from different Results may coexist without making a hypothesis terminal.

Use `result --suggests--> hypothesis` when evidence motivates a new hypothesis. A Result can lead to another hypothesis or follow-up experiment, so the cycle may continue. If execution cannot proceed, record the concrete blocker with `mark_research_experiment_failed`; do not turn debugging branches into scientific hypotheses.

After a new Result, reconsider the meaningful runnable Experiment frontier rather than only the producing branch. A result can make another experiment more discriminating, redundant, premature, or newly necessary. Planning evaluates each current candidate under separate innovation and conservative policies. These temporary numbers compare only the same revision; they are not evidence, probability, or durable Experiment state. Automatic mode uses only an explicit current conservative recommendation and otherwise waits; manual mode displays both recommendations.

## Method-critical defaults

- Pass an explicit graph ID and the latest inspected graph revision to ordinary mutations. The bound `stage_research_plan` tool derives both from its trusted planning thread.
- On a stale-revision conflict, inspect and reconcile; never overwrite another thread's edit.
- Keep hypothesis claims falsifiable and experiment decision rules outcome-specific.
- Keep direct observations in Results and causal explanations in Hypotheses.
- Preserve claim-relevant evidence attributes in scientific language; do not add a strength score or fixed evidence-level field.
- Keep all competing evidence visible; do not collapse it to one confidence score.
- Preserve parallel hypothesis and experiment branches; automatic orchestration choosing one next launch does not remove the others.
- Use refs rather than copying long source content into graph nodes.
- Do not launch follow-up work merely because a frontier exists; respect the user's requested stopping point and normal approval rules.

## Output Contract

At handoff, state the graph title, the focused hypothesis, experiment, or result,
the typed relation changed, the concise scientific consequence, durable source
refs, and the next runnable or blocked decision. Keep the numeric revision for
tool concurrency rather than presenting it as scientific content. State when
reasoning began from the partial focus snippet and name additional canonical
SQL or source reads that materially affected the decision.

## References

- `devdocs/0728_major_update/04_workspace_research_graph.md`
- `catmaster/research/knowledge_graph/`
