# Changelog

This file records notable behavior changes from this point forward. It is not a
reconstruction of earlier development history. The manuals and technical
references describe the current system.

## Unreleased

### Agent Prompting

- Writing system prompts no longer prescribe claim counts or fixed review,
  polishing, and compilation pass counts. Conditional planning guidance now
  lives in writing skills, while runtime prompts retain qualitative completion
  conditions and hard safety or transaction limits.
- Shared specialist and named-worker prompts now reject model-invented hashes
  and ad hoc frozen contracts, schemas, manifests, baselines, lockfiles, or
  acceptance frameworks for ordinary one-off work while preserving artifacts
  required by real APIs, tools, reproducibility needs, and downstream consumers.
- CatMaster now explicitly replaces DeepAgents' auto-added `general-purpose`
  child for every specialist and named worker. The shared child remains a
  non-delegating context-isolation worker, inherits the caller's direct
  capability surface and staged skills, and adds bounded document access plus
  nonfatal tool-error handling without copying the full parent prompt. Its task
  brief, rather than a lane or blanket concurrency policy, defines scope and
  stopping conditions.

### Literature Review

- Direct Literature Review threads no longer receive Research Graph result or
  blocker writeback tools. Those actions are added only when the thread is
  actually bound to an Experiment node.
- All agent factories now use the same provider-aware search resolver. Codex
  OAuth and OpenAI roles, including self-evolution proposer/reviewer, receive
  hosted `web_search`; other providers receive CatMaster's search function.
  Tavily quota, authentication, rate-limit, and network failures are classified
  without exposing credentials, trip a run-scoped circuit, and can fall back to
  bounded scholarly-index discovery through the existing configuration flag.
- Literature Review no longer carries internal paper-count, acquisition-attempt,
  delegation-count, or batch-count targets in its active prompt and skills.
  Explicit user limits control discovery breadth, while broad reviews expand by
  coverage gaps and stop at saturation. Candidate discovery remains shallow
  until papers are selected for deeper evidence extraction.

### Research Graph

- Research planning now lets the proposer choose the number of scientifically
  distinct temporary branches instead of exposing fixed 12-Hypothesis and
  24-Experiment quotas. Temporary experiments may remain drafts until their
  execution plan and decision rule are known.
- The model-visible planning action now accepts scientific claims, objectives,
  source references, and semantic relationships. The host assigns temporary
  transaction IDs and resolves those relationships against the bound graph, so
  planning agents no longer exchange proposal IDs or graph revision fields.
- Research Graph layout now waits for React Flow to measure the current nodes
  before fitting the viewport, and node selection no longer retriggers layout
  or refits the graph.

### WebUI

- Steering queued during a run now yields after a completed tool/checkpoint
  boundary and continues from the same DeepAgent thread. Active scientific or
  remote tools are not cancelled to apply the newer instruction; turns without
  another tool boundary finish before steering starts.
- Completed turns now reconcile the live conversation with the persisted
  canonical message, including final Markdown and completed reasoning state.
  Long message-part pages retain their continuation reference, and the Todo
  inspector reads a full current-turn projection instead of the visible page.
- The ordinary Files projection now hides DeepAgents large-result offloads and
  known transient literature extracts while preserving them in workspace
  storage for diagnostics.

### Documentation

- Removed migration progress, completed implementation checklists, and
  future-removal notes from the DeepAgents reference documents.
- Established the repository rule that manuals describe current capabilities,
  configuration, limits, and verification. Notable behavior changes belong in
  this file.
