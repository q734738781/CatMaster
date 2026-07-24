---
name: hypothesis-campaign-control
description: Use this skill when Research must compare competing scientific explanations through more than one evidence-producing check and keep hypothesis formation separate from evidence judgment.
license: project-local
compatibility: CatMaster Research Specialist
allowed-tools: "initialize_hypothesis_campaign extend_hypothesis_campaign inspect_hypothesis_campaign advance_hypothesis_campaign record_hypothesis_result task"
---

# hypothesis-campaign-control

## Overview

Move scientific content through a fixed role sequence: hypothesis proposer, execution owner, independent evidence judge, then the deterministic campaign controller.

## Quick Start

Delegate the question to `hypothesis_proposer`, then initialize the campaign from its structured output without silently rewriting it. Execute one returned packet through its named owner. Send every successful result to `evidence_judge` and record that judgment before advancing. Ask `hypothesis_proposer` for a revision only when the controller reports unresolved hypotheses without a useful remaining check.

## Allowed tools

- `initialize_hypothesis_campaign`
- `extend_hypothesis_campaign`
- `inspect_hypothesis_campaign`
- `advance_hypothesis_campaign`
- `record_hypothesis_result`
- `task`

## Workflow

### 1. Form the scientific plan

For a new branching campaign, call `task` once with `subagent_type=hypothesis_proposer`. Supply the research question and only the decision-relevant scientific context. The proposer owns:

- falsifiable competing hypotheses;
- rationale and observable predictions for each hypothesis;
- the smallest checks that distinguish them;
- the scientific decision rule for each check;
- coarse information value, cost, and real scientific dependencies.

Call `initialize_hypothesis_campaign` with that structured plan. Initialization saves
the plan but does not reserve a verification. Research coordinates the handoff but
does not author replacement hypotheses, scoring fields, or decision rules.

### 2. Execute one verification

Only `advance_hypothesis_campaign` reserves a verification and returns an
`EXECUTION PACKET`. Always pass one explicit action id.

- Delegate `literature` packets to `litreview_agent`.
- Delegate `experiment` packets to `experiment_specialist`.
- Perform a bounded `workspace` packet in Research.
- Ask the user for a `human` packet and wait.

Pass the complete hypothesis claims, predictions, task, and decision rule. Execute exactly one packet at a time. A packet never bypasses managed execution or protected-tool approval.

### 3. Judge the returned evidence independently

When execution succeeds, call `task` once with `subagent_type=evidence_judge`. Supply:

- the action id;
- all target hypotheses with their predictions;
- the decision rule;
- the scientific result;
- one exact DOI, URL, artifact path, run id, or explicit user source.

The judge must return exactly one `supports`, `opposes`, or `inconclusive` effect for every target hypothesis. Copy its structured output into `record_hypothesis_result`. Do not add hypotheses or verification actions in the result call. For execution failure, record only the concise failure reason.

### 4. Continue, revise, or wait

After recording one result, inspect again before selecting another action. A normal
user Research turn may continue sequentially when that is required by the original
request. A Map-launched Research turn handles only its reserved action.

When the controller reports `needs_hypothesis_revision`, delegate the current question, hypotheses, evidence judgments, completed checks, and failed checks to `hypothesis_proposer`. Apply only its new hypotheses and checks through `extend_hypothesis_campaign`.

Research Map does not steer the source chat thread. A manual click reserves the
selected action and creates a new ordinary Research thread. An asynchronous WebUI
worker can do the same serially for ranked non-human actions. Each execution thread
has its own DeepAgent checkpoint and Research Kernel, while its system context gives
the source campaign id for controller calls. Never substitute the execution thread id
for that campaign id.

The automatic worker is scheduling, not scientific authorship and not approval. It
does not call experiment or literature tools itself. Every launched child still uses
the normal Research delegation and permission mode. A protected tool can therefore
interrupt an automatic child for review. Stopping automatic Research prevents the
next launch and leaves the current child intact.

## Method-critical defaults

- Use one proposer call per initial plan or revision and one judge call per successful verification.
- Keep proposer and judge contexts separate; neither role may substitute for the other.
- Keep campaigns serial because the shared workspace has no isolated merge contract.
- Treat `information_value` and `cost` as coarse routing labels, not probabilities.
- Cost affects ranking only. It is not a permission gate.
- Automatic scheduling skips `human` actions; launch those manually and wait in the
  created Research thread for the user's evidence.
- An inconclusive result keeps the affected hypothesis unresolved.
- A failed check is not scientific evidence and is not silently retried.
- Keep full scheduler receipts, tool logs, and artifact inventories in their existing CatMaster stores, not in campaign state.

## Output Contract

At each handoff, expose:

- the campaign question and revision;
- complete hypothesis claims and predictions;
- the active action, owner, task, and decision rule;
- the evidence judge's summary, source, and per-hypothesis effects;
- the current hypothesis and verification statuses;
- the exact blocker or revision need when no packet is active.

## References

- `devdocs/ReasoningEnging/verification_cost_aware_hypothesis_engine/02_adr_minimal_hypothesis_network.md`
- `catmaster/research/hypothesis_engine/`
