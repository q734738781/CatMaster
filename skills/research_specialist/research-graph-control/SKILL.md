---
name: research-graph-control
description: Maintain hypotheses, experiments, results and consequential direction or stopping decisions in a bound Research Graph. Use for scientific progression or graph changes, not routine file inspection, status replies or an already answered one-off question.
license: project-local
---

# Research Graph control

Use the one graph supplied for the current turn. Its scientific query and write
operations use that binding automatically; do not supply or infer another graph ID.
The user interface handles graph selection and creation. A missing binding does
not prevent ordinary workspace analysis; report it if graph writeback is needed.
Use the latest inspected revision for mutations and reconcile conflicts before
retrying. Change thread focus explicitly when changing the branch being worked on.
The graph indexes scientific reasoning; detailed evidence stays in its source files.

Recorded questions and completion criteria describe the saved stage; apply current
user corrections while preserving constraints the user has not changed. Node
creation calls return new node IDs; reuse those IDs for subsequent work. If a
creation outcome is uncertain, query before retrying: another successful creation
makes another node.

## Preserve the scientific question

Research owns its assigned hypothesis–method–experiment–result cycle. Independent
branches retain that full responsibility; the root integrates their findings and
owns overall scope and completion. A completed method comparison does not finish
an unanswered research question. Ordinary Research stops at its requested task;
Persistent Research continues within the authorized stage.

- A Hypothesis may begin with the user's claim, an Experiment with its objective,
  and a Result with its observation. Add details from evidence, not invented fields.
- A ready Experiment needs a usable plan, outcome-specific decision rule, execution
  lane and known cost constraints. Keep incomplete proposals as drafts.
- Keep scientifically distinct competing explanations and meaningful alternatives;
  no fixed branch count or confidence/novelty score is required. Priority and cost
  are user/execution constraints, not scientific evidence or ranking rules.
- Choose node granularity by scientific decision. Preparation, acquisition,
  convergence checks, replicates and analysis stay inside one Experiment when
  they serve the same question. Split only for an independently meaningful Result.
- Dependencies describe real prerequisites and must remain acyclic. An external
  Experiment is an actionable laboratory handoff, never an automatic CatMaster run.
  A recommendation alone does not authorize calculations or laboratory work.

## Reconsider a consequential direction

Use `start_async_task(agent="research_challenger", task_cost="low")` when an
independent assessment could change premature narrowing, a missed method/source,
an assumed missing input, or a proposed stopping decision. Do not wait to label
it a stall. A routine Result or fulfilled bounded deliverable needs no consultation.

Supply the original objective and constraints, decisive request/source/Result refs,
the open decision and any stopping decision ID. Make your interpretation explicit
and challengeable; do not prescribe the answer, model or literature shortlist.
The shared `research-direction-recovery` skill supports this scientific assessment.
Act on returned evidence or explain why the current decision holds. Reuse existing
assessments on unchanged evidence and continue the same child for follow-up.

## Coordinate only the work needed

Delegate independent scientific questions as complete investigations with clear
objectives, source refs, authorization, deliverables, stopping conditions and
separate output paths. Use exposed task descriptions for cost and invocation fields.
When an Experiment already owns the delegated work, include its actual node ID
and objective in the brief so the executor can attach the Result to that node.
`start_async_task` shares the Graph but does not select an Experiment focus for
the child; task creation alone is not a second Experiment registration.
Before a substantial new method or source search in Persistent Research, check
`list_async_tasks(scope="research_graph")` for relevant in-flight work; inspect
matching briefs/results instead of repeating them.

When only asynchronous results remain, end the interim turn; completion resumes
the conversation. Do not hold it open with sleeps or status polling. Report
meaningful scientific transitions rather than routine reads or heartbeats.
Before first using multi-branch discussion, correcting an ongoing child, resolving
a peer question or changing this task's cost, read
[collaboration](references/collaboration.md) unless already in context. Discussion
messages do not start work; follow-ups continue an owned child. Peer visibility
does not grant control, and interrupting an agent does not cancel remote jobs.

## Preserve evidence and revise claims

The producer records a meaningful finding when ready, even during a longer study.
The root integrates existing Results without duplicating them. A shared script's
single-writer ownership does not make the scientific graph single-writer.

Each Result separates:

- `methods`: actual evidence/data, representation, conditions, baselines,
  comparisons and validation; literature work describes actual search and comparison.
- `summary`: observations and measurements with their units and relevant uncertainty.
- `conclusion`: supported interpretation, limits, alternatives and next question.

A low R² from a small descriptor set cannot rule out predictability. Examine
representation coverage, leakage, splits, baselines and sample size before
generalizing. “Missing due to old record” means knowledge was not recorded, not
negative evidence. Follow claim-critical source refs rather than relying on a label.

Attach real run/artifact/note/DOI/URL/thread/message refs. Link a producing Experiment
when one exists; do not invent a retrospective Experiment for literature or historical
evidence. Keep access, hardware, scheduler and build details out of scientific nodes
unless a concrete compatibility issue changes the observation.

Judgments are Result-to-Hypothesis `supports`, `opposes` or `inconclusive` relations
with scope and rationale, not global evidence grades. Opposite findings may coexist
under different conditions; preserve competing evidence. A Result may suggest a new
Hypothesis. For substantial correction, add the new H or R and use
`revise_research_claim` to link it to the older same-kind claim with `replace`,
`qualify` or `withdraw`; retain original observations and valid partial scope.
Reconcile revision conflicts instead of overwriting another researcher's judgment.

## Close at the user's stage

Mark completion when the actual deliverable is satisfied; a ready frontier alone
is no reason to expand the task. Only the root changes graph-wide scope or completion
through `update_research_graph_scope` or `set_research_graph_completion`.

Before ending unfinished Persistent Research, use `record_research_disposition`
to distinguish a scientific stall from user/budget/authorization boundaries or
real external waiting. A declared stall receives one independent challenger review.
Perform its concrete authorized validation and link the Result to the same decision,
unless an actual authorization/input/resource conflict, a moot or equivalent completed
check, or an achieved goal makes it unnecessary. An inconclusive outcome is not itself
a reason to park. If no useful authorized step remains, preserve open premises and
`resume_when`; do not repeat the review under another name.

During persistent synthesis and closeout, check relevant shared discussion and
answer consequential unresolved questions using actual methods and evidence.
Continue the owning branch when a material gap warrants authorized work; assignment
is not scientific resolution. Ordinary Research does not require this discussion pass.
A completed stage stays complete unless the user authorizes continuation.

Return the scientific consequence, decisive refs and next action or supported
stopping reason. Keep transaction revisions and scheduling details in tool state.
