# Changelog

This file records notable behavior changes from this point forward. It is not a
reconstruction of earlier development history. The manuals and technical
references describe the current system.

## Unreleased

### WebUI

- Long assistant activity traces now consolidate the latest Todo state at the
  top and group reasoning, progress, and tool calls by subagent invocation.
  Activity appears between the Plan and final prose. Groups with many events or
  one substantial reasoning block collapse to their current or latest activity
  while preserving the complete trace on expansion; same-named parallel
  invocations remain separate.

### Runtime Reliability

- Codex OAuth model calls now retry a prematurely closed chunked SSE response
  twice with short bounded backoff. This recovery replays only the interrupted
  model call, not the complete specialist episode or previously completed local
  tool work; the existing longer overload retry policy remains separate.

### Agent Prompting

- Literature Review now has a named, non-delegating
  `litreview_worker_agent` for bounded discovery, source reading, extraction,
  and evidence-audit branches; `litreview_agent` retains coverage decisions,
  conflict resolution, and final synthesis. The Codex OAuth profile routes this
  worker and `writing_worker_agent` to GPT-5.6 Luna with xhigh reasoning while
  keeping their coordinators on GPT-5.6 Sol.
- Shared tool guidance now asks delegators to assess possible write overlap
  before launching concurrent subagents. Read-only branches remain freely
  parallel; potentially overlapping writers use separate output paths, one
  designated writer, or sequential execution without imposing mandatory
  per-task workspaces.
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

- Literature Review now exposes one selected-source acquisition tool instead of
  raw browser navigation, page-state, screenshot, and download primitives. The
  tool uses pinned ScanSci 1.9.0 direct OA adapters first, keeps pinned
  CloakBrowser 0.5.3 behind one internal ScanSci DOI-page fallback, validates
  PDF structure, page count, and paper identity, caches accepted files locally,
  and finally falls back to one local static-page snapshot. The separate downloader skill
  has been removed and its source-acquisition SOP merged into academic search.
- The Literature Review system prompt now retains scientific role and evidence
  boundaries only. Concrete acquisition, caching, corpus, delegation, and
  citation-finalization workflow guidance lives in the tool schemas and skills.
- Top-level Literature Review turns receive Research Graph query and Result
  writeback only when that turn is bound to a graph; the blocker action appears
  only for an Experiment focus. Unbound turns and Research-internal delegates
  have no bound mutation surface.
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
- Literature Review now describes evidence through claim-relative attributes:
  scientific modality, epistemic stage, access depth, claim relationship,
  condition fit, and independence/provenance. The active skill no longer ranks
  retrieval APIs or whole papers with source or evidence strength tiers.
- Literature pipeline triage now records `selected`, `deferred`, or `excluded`
  with reasons instead of assigning a six-component paper score. Citation
  support uses separate claim-relationship and access-depth attributes,
  reference checking reports `verification_status`, and reader source maps use
  `extraction_confidence` only for OCR and layout extraction quality.

### Research Graph

- Research Graph execution binding is now turn-scoped in long-lived WebUI
  threads. Experiment and Literature Review can explicitly record a sourced
  linked or standalone Result only against the graph/focus snapshot captured at
  turn start; internal delegates return evidence to their parent, and Writing
  remains read-only. A formal launch may continue across multiple idle turns,
  and Result/blocker writeback completes only the exact launch associated with
  that turn rather than a historical launch found by thread ID.
- Writing threads attached to a Research Graph now receive its partial focus
  context and can query the complete bound graph through the existing read-only
  SQL surface. The Writing coordinator uses Result relationships and refs as a
  navigation index before opening original evidence; writing workers receive a
  bounded author packet and no graph mutation tools. Research-delegated Writing
  inherits the parent thread's trusted graph binding.
- Research planning now starts from a partial focus snippet and exposes the
  complete bound graph through one read-only SQLite query tool whose visible
  input is only `sql`. The logical views preserve graph JSON and typed
  relationships while restricting artifact and message rows to references from
  the bound graph. Standard SQL pagination, recursive CTEs, window functions,
  and JSON1 remain available without host-side row truncation.
- Planning staging is now a pure preview. A separate evaluator assigns temporary
  innovation and conservative scores to every candidate Experiment for the
  current graph revision. Manual mode shows both recommendations; automatic
  mode uses the conservative recommendation and waits on missing, invalid,
  stale, or empty evaluation instead of selecting the first runnable node. The
  preview carries one proposer target and one normalized evaluation row set
  rather than duplicating route and score fields across provisional nodes.
- Automatic Experiment and Literature Review Result writeback now uses the
  shared evidence judge before the atomic graph mutation. Only relationships
  the Result actually addresses are recorded, and an empty judgment set is
  valid. A Result-focused planning turn may reuse existing Hypotheses or create
  a distinct new Hypothesis and its discriminating Experiment.
- Research Graph scientific text and semantic collections no longer have
  capacity-based schema limits. Corpus locators and DOCX/XLSX document reading
  now provide continuation cursors, and graph mutations return the exact changed
  entity and revision instead of an automatically truncated context projection.
  Corpus pages expose only query context, source locators, total count, and the
  next cursor; planning drafts leave blocking reasons to the dedicated failure
  transition rather than carrying them as candidate-science fields.
- Existing workspaces remove the retired Experiment `expected_value` field
  during schema migration and invalidate disposable planning previews so that
  the new revision is evaluated again rather than translating an old route
  recommendation.

- Research Graph Results and `evidence_judge` now preserve observation,
  derived analysis, interpretation, modality, conditions, and provenance as
  scientific attributes without adding evidence-level fields, confidence
  scores, or composite grades. Result-to-Hypothesis edges remain relational
  judgments rather than strength rankings.
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
- The completed-message push now replaces the live Todo inspector with its
  terminal canonical projection. Unfinished child-agent scratch plans no
  longer remain active after the turn has ended, even when the agent did not
  make a final `write_todos` call.
- Completed specialist-task cards now present the returned scientific Markdown
  as a content title, central conclusion, and short section outline. The
  LangGraph `Command` wrapper is retained only in the detailed record.
- The ordinary Files projection now hides DeepAgents large-result offloads and
  known transient literature extracts while preserving them in workspace
  storage for diagnostics.

### Documentation

- Removed migration progress, completed implementation checklists, and
  future-removal notes from the DeepAgents reference documents.
- Established the repository rule that manuals describe current capabilities,
  configuration, limits, and verification. Notable behavior changes belong in
  this file.
