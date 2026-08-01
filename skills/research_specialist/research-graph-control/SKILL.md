---
name: research-graph-control
description: Use this skill when a falsifiable research problem must continue across experiments or threads, or when a result should change the next hypothesis or experiment.
license: project-local
compatibility: CatMaster Research Specialist
allowed-tools: "list_research_graphs create_research_graph inspect_research_graph add_research_hypothesis add_research_experiment stage_research_plan record_research_result set_research_result_judgment mark_research_experiment_failed task"
---

# research-graph-control

## Overview

Maintain the shared scientific situation as one explicitly selected workspace Research Graph while leaving detailed notes, files, calculations, and receipts in their owning stores.

## Quick Start

Use the graph ID injected for a bound turn. If no graph is bound, list graphs and ask the user to choose when several are relevant; create one only when the work is genuinely multi-step or evidence-driven. A newly created graph is attached to the current thread by trusted runtime context, so do not ask for or invent a thread ID. Inspect before mutating, use that revision for the next mutation, and inspect again after a revision conflict.

## Allowed tools

- `list_research_graphs`
- `create_research_graph`
- `inspect_research_graph`
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

In an internal graph-planning turn, delegate model-generated branch formation to `hypothesis_proposer`. The proposer inspects the bound graph, searches evidence when useful, publishes meaningful provisional branches through its bound `stage_research_plan` tool, and returns a concise scientific memo in ordinary language. Do not ask it for JSON, copy its tool payload into another call, or make it repeat graph/revision identifiers. Outside an internal planning turn, preserve user-supplied hypotheses and observations directly.

### 2. Add the smallest useful scientific next step

A user may contribute only a brief scientific statement. Preserve that input without inventing missing detail: a Hypothesis may start with only its claim, a draft Experiment may start with only its objective, and a Result may start with only its observation. Add rationale, observable predictions, sources, and coarse priority fields when they are known or can be developed from evidence. A ready Experiment is different: it needs a concise executable plan, a decision rule, the owning execution lane, expected decision value, and coarse estimated compute cost so another researcher can run it without inventing missing scientific choices. Keep an incomplete proposal as `draft`.

Keep as many scientifically distinct competing branches as the current evidence warrants, and stop when another branch would only repeat an existing one. Do not aim for a fixed Hypothesis/Experiment count or ratio. Temporary planning Experiments may remain drafts with only an objective; only a route with a usable plan and decision rule is ready for automatic execution. Importance, expected value, and compute cost are optional coarse bands; leave them empty when they are not genuinely known. They do not state confidence, truth, or success probability. Connect hypotheses to experiments through their typed inputs. Add experiment dependencies only for true execution prerequisites; they must remain acyclic. Do not add confidence, novelty, composite scoring, precise invented resource estimates, layout, prompt, token, or generic metadata.

### 3. Execute through the normal owner

User or scheduler launches are handled by the graph launch service and ordinary permission boundaries. When coordinating a launched child, preserve its graph and experiment IDs. Put detailed inputs, calculation output, logs, managed-execution receipts, figures, and reports in normal workspace artifacts or runs rather than copying them into the graph.

### 4. Record evidence and continue

Record one concise Result for each scientifically meaningful observation. Link it to its producing graph experiment when it came from one. For a literature finding, collaborator result, historical observation, or other evidence obtained outside the graph, leave the producing experiment empty instead of inventing a retrospective Experiment node. Attach the exact run, artifact, note, DOI, URL, thread, or message ref when one exists. Judge only the hypothesis effects that the evidence actually addresses as `supports`, `opposes`, or `inconclusive`; a new observation may remain unjudged until the next planning step. Use `set_research_result_judgment` to add, replace, or clear that one Result-to-Hypothesis judgment after inspecting the current revision. Support and opposition from different Results may coexist without making a hypothesis terminal.

Use `result --suggests--> hypothesis` when evidence motivates a new hypothesis. A Result can lead to another hypothesis or follow-up experiment, so the cycle may continue. If execution cannot proceed, record the concrete blocker with `mark_research_experiment_failed`; do not turn debugging branches into scientific hypotheses.

After a new Result, reconsider the meaningful runnable Experiment frontier rather than only the producing branch. A result can make another experiment more discriminating, redundant, premature, or newly necessary. Recommend a next route with a short scientific reason when the evidence supports one; do not turn that judgment into a numeric utility. Automatic mode may launch the recommended runnable Experiment; manual mode leaves the choice to the user.

## Method-critical defaults

- Pass an explicit graph ID and the latest inspected graph revision to ordinary mutations. The bound `stage_research_plan` tool derives both from its trusted planning thread.
- On a stale-revision conflict, inspect and reconcile; never overwrite another thread's edit.
- Keep hypothesis claims falsifiable and experiment decision rules outcome-specific.
- Keep direct observations in Results and causal explanations in Hypotheses.
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
