# Changelog

This file records notable behavior changes from this point forward. It is not a
reconstruction of earlier development history. The manuals and technical
references describe the current system.

## Unreleased

- Published documentation contains maintained manuals and engineering changes;
  private investigations and replay artifacts stay outside the published tree.
  Cache probes write their default diagnostic output to the ignored `tmp/` directory.

- Scientific entry turns retain their selection or reuse the newest unarchived
  Graph, including completed stages; only an empty graph catalog initializes a
  first Graph. Research agents and delegates no longer receive workspace graph
  listing or creation tools. Scientific mutation schemas use the host-bound graph
  automatically, preserving node choices and revision checks.

- Graph-aware turns and background delegates receive an explicit current graph
  binding. The catalog reports the same accepted binding used by scoped SQL,
  independently of filtering or later thread selections. Graph context distinguishes
  saved questions/completion criteria from the current request; default graph and
  MiMo guidance no longer leave known bindings for the model to infer.

- Shared runtime guidance now applies to all DeepAgents role prompts, including
  Astra and self-evolution. MiMo adds a smaller model-specific emphasis and
  registers independently in specialist and self-evolution entrypoints, so startup
  order does not affect coverage. Role permissions and text/decision completion
  remain intact. Experiment/ML repetition is consolidated; discussion principles
  live in the recipient's system prompt while event notices retain query context.

- MiMo specialists and workers automatically receive a composed user-priority
  and system-usage prompt through native DeepAgents harness profiles, selected
  from the actual model in the existing LLM YAML. Explicit and inherited children
  use their own model guidance. No extra configuration is required; Astra retains
  its existing prompts, and tool bindings and reasoning replay remain unchanged.

- Nested worker activity keeps one message when a model stream changes its
  message ID. Reconnects and checkpoint updates retain the same tool results
  and text, and completed runs no longer mark remaining activity as failed.

- Self-evolution reflection, proposal, and review now use `auto` tool selection
  and accept ordinary text as a completed response without a corrective model
  turn. Explicit decision tools remain available for candidate processing and
  approval. Text, unsubmitted draft edits, and open findings are retained without
  inferring an action; the WebUI displays the textual conclusion.

- Self-evolution reflection can now open selected skill bodies, supporting files
  and workspace guidance through ordinary file tools. Its paginated catalog
  returns paths for the same staged versions. Proposal, review and evidence
  investigators share this readable guidance layout; disabled skills remain
  inspectable without activation.

- Self-evolution trace queries now return ready-to-read event handles; original
  records read correctly with the default field. Aggregate queries work over all
  public trace views, errors have native error status and concise recovery text,
  and large SQL results remain available through ordinary file tools. Reflection
  guidance emphasizes tool friction, scientific accuracy and reusable procedures.

- The Codex OAuth template now uses GPT-6 Luna for literature workers and thread
  titles, retaining xhigh and low reasoning respectively.

- Tool and companion skill guidance now explicitly states the PBE_54 default,
  selected remote stages, source-file selection, trajectory analysis choices,
  optional dataset splits and graph mutation semantics. Agent-facing wording
  describes callable behavior and recovery without unrelated backend machinery.
  Tool changes must review and update affected skills together.

- Ordinary scientific tools now preserve distinct batch outputs, accept explicit
  result files and NEB endpoints, expose native VASP inputs and method controls,
  and report partial failures and scientific convergence separately. Phonon
  atom mapping, paired k-path structures, conformer energy filtering, trajectory
  sampling, and dataset labels have focused correctness fixes.
- Tool interfaces add selectable remote stages, Materials Project pagination
  and cell choices, graph mutation handles, configurable active-learning
  selection, independent reviewer outcomes, and flexible local document builds.
  Literature retrieval and exports preserve complete results and explicit source
  constraints. Relevant skills describe the callable scope and parameter
  relationships.

- Self-evolution explicitly submits decisions through its declared terminal
  tools. A prose-only ending receives one correction within the same conversation,
  retaining evidence, candidate file edits and observations. Candidate changes
  continue through normal filesystem tools, with no patch payload.
- Tool-error handling preserves native LangGraph control signals so research
  capacity changes wait and resume instead of appearing as ordinary errors.
  Tool and skill guidance clarifies exact evolution run handles, reuse of an
  existing Experiment, and ORCA functional-token spelling.

- Shared narrative prompts and writing guidance now address Chinese coined
  shorthand, unexplained identifiers and opaque status labels in prose, headings
  and tables. Authoring and editorial review require ordinary explanations,
  classification chosen for the reader's question and introduced reference identifiers; the phrasing
  reference includes contextual Chinese examples.

- LLM output preservation now covers partial responses on streaming errors,
  Unicode reasoning, self-evolution citation and nonstandard content blocks,
  response details, and OpenRouter assistant replay. Partial responses retain
  failure status and do not count as completed calls. Compatibility specialist
  reports keep the authored answer and append new compilation results.

- Self-evolution now retains full model and tool debug observations, including
  native investigators and summaries, through the ordinary observability store.
  Thread-title generation and optional legacy monitor summaries use the same
  standalone invocation scope, with source links and per-call usage. Failed or
  cancelled invocations close pending calls without losing earlier usage.

- Codex OAuth writing profiles now use Astra medium reasoning, and the default
  Luna literature worker uses xhigh. The local configuration and shipped
  templates use these settings; thread-title generation keeps its existing level.

- Literature coordinator and worker guidance now organize reviews around
  scientific questions, findings and interpretation. Source checks address
  explicit verification requests or concrete inconsistencies; evidence attributes
  are an optional diagnostic reference. Literature reporting is separate from
  execution reporting, and research notes leave prose and figure-caption choices
  to authoring. User-guide examples follow the same scientific focus.

- Writing briefs leave unspecified editorial choices to the author and avoid
  invented writing restrictions. Narrative roles avoid competing explanations
  and qualifications and keep production compliance out of ordinary prose.
  Integration and coordinator acceptance assess reader relevance as well
  as factual accuracy, including when source notes supply the initial wording.
  Shared writing guidance explicitly develops authorial judgment: select what
  matters, explain the evidence, allocate detail by explanatory value and trust
  the reader's shared knowledge while retaining material uncertainty.

- Writing handoffs use free text with recoverable findings and sources, preserving
  section-based context isolation and whole-document editorial judgment. Fixed
  manuscript packet fields and always-on publication/Humanizer audit blocks are
  removed. Scientific-communication now owns prose guidance with optional phrasing
  and presentation references; the standalone Humanizer skill is retired with
  attribution retained. Manuscript guidance is selected for the actual task.
  Literature synthesis can read returned evidence while stopping new discovery.
  Legacy ACS section-return fields and materials literature depth modes are
  removed; PDF conversion and citation exports follow the requested deliverable.
  Visualization guidance routes to existing references instead of a long startup
  recipe. Model profiles and delegation/tool bindings are unchanged.

- Local shell execution can run bundled skill utilities directly through the
  run-bound `CATMASTER_SKILLS_ROOT`, including isolated stable/canary snapshots.
  Atomic validation and assembly guidance no longer requires copying or reading
  source first. Their console diagnostics expose actionable violations while
  retaining full pair/contact tables in JSON. Citation command examples use the
  mounted paths; per-reference conversion/validation progress is opt-in through
  `--verbose`. Atom-index freezing returns counts and the native constraint-file
  path instead of echoing the full input index list.
  DOI, metadata, PubMed and Scholar helpers also make per-record progress opt-in.
  Atomic helpers can explicitly print complete reports with `--verbose`; citation
  validation and panel rendering avoid duplicating diagnostics already saved to
  a requested report file, with verbose output retaining the full console view.

- Shared agent prompts now reuse skill instructions already in context and avoid
  compulsory startup reading. Operational guidance is shorter; research and remote
  execution skills load conditional details on demand. Citation/template workflows
  no longer request unrelated illustrations. Self-evolution checks proposed guidance
  for broad triggers and duplicated instructions while preserving full source access.
  Remote layouts now link directly to 16 task-specific references, preserving
  internal input contracts and using live task specs for parameter schemas.
  Task catalogs expose these references for native and MLFF tasks and resolve
  legacy bundled section links without changing private deployment configuration.
  Runtime and self-evolution guidance preserve non-obvious internal contracts
  before unfamiliar operations and evaluate savings including discovery and retry
  cost. Scientific writeback explicitly supports findings ready during an ongoing study.

- Research can launch `research_challenger` as an independent asynchronous task
  through the existing DBOS completion bridge. Persistent guidance and a shared
  direction-recovery skill address premature narrowing, missed methods and
  disconnected model-guided decisions while preserving branch H/E/R ownership.
  Existing `hypothesis_proposer` model bindings remain compatible.

- Codex now defaults `thread-id` to its stable session identity while respecting
  explicit header overrides. Provider-aware search augmentation resolves
  same-named function/native conflicts and preserves existing native search
  settings without adding duplicate tools.

- Codex diagnostics also retain an explicitly sent `thread-id`. A bounded manual
  DeepAgent probe exercises growing tool histories, optional native web search,
  thread identity and compaction, with per-request prefix and cache analysis.

- Self-evolution now persists per-call usage for reflection, proposal, review,
  native investigators and internal summaries through ordinary run observability.
  Cached input and reasoning details survive later failures; separate invocation
  records preserve retries and expose pending or missing usage without adding
  learning costs twice to the source research run.

- Codex requests now carry stable conversation-scoped cache keys and session
  headers across steps, compaction and model reconstruction. Explicit overrides
  remain supported, and independent delegated conversations stay separate.
- Token-based context retention uses the trigger's calibrated token scale,
  fixing repeated compaction that rewrote only the preceding summary.
- Selected Codex diagnostics capture terminal provider SSE metadata, including
  cache fields lost by LangChain projection and responses without Content-Type.
  Writing benchmarks enable bounded request/response capture before execution
  and report the capture counts.

- LLM profiles use high/medium reasoning in place of xhigh/high. The default
  writing coordinator, prose, presentation, plotting, review and compile-fixing
  roles share a separate Codex GPT-6 Astra low profile. Alternate provider
  profiles retain their model choices with reduced reasoning settings.
- OpenRouter request timeouts now convert CatMaster seconds to the adapter's
  milliseconds, avoiding premature timeouts and retries. A manual writing
  benchmark entrypoint runs the production writing lane in prepared workspaces.

- Context compaction uses actual same-model usage for consumed native media,
  preventing repeated compression caused by Office/PDF base64 estimates while
  honoring high image usage across Codex provider aliases. Compaction progress
  follows the model task across streaming message-ID changes and closes when
  the summary finishes. Native media and checkpoint history remain intact.
- Explicit dictionary-based subagents now receive the configured context
  compaction threshold and the same usage/restoration handling as root and
  compiled agents. Actual same-model text usage also triggers compaction across
  Codex provider aliases when the approximate count is too low.

- Persistent Research starts from its entrypoint and objective, with session
  pause/resume replacing Graph Auto/Manual and Update routes controls. Legacy
  Manual pauses survive until explicit continuation. Independent Research
  branches own method selection, repeated H/E/R cycles and timely Result writes;
  legacy planner records cannot narrow their tools. Ordinary replies no longer
  require a campaign disposition, and declared stopping reviews are delegated
  by the model rather than a host-generated tool call. Research activity includes
  nested tasks and the actual pause state; milestones show methods/conclusions
  and update in place when an existing Result is amended.

- Codex OAuth can sample final SDK-serialized requests for selected runs and
  roles into existing observability records, joined to usage by callback ID.
  Capture is off by default and preserves streaming and retry behavior.
- Presentation and editorial guidance delegates large visual checks into
  coherent page groups, returning located findings and reachable sources while
  retaining whole-document acceptance and a single deck editor.

- Scientific writing handoffs distinguish explanatory reports and talks, concise
  internal technical records, and venue-facing manuscripts. Unspecified report
  audiences default to collaborators unfamiliar with the project's computational
  background. Compact delivery messages no longer imply sparse documents; Writing
  assesses the overall explanation as well as individual pages and preserves
  required scientific coverage.

- Writing accepts actual documents and rendered presentations against the audience
  and user feedback, rather than relying on worker completion messages. Material
  content and visual defects return to the responsible authoring worker for scoped
  correction, with existing scientific checks reused and no extra review agent.

- Report and presentation handoffs preserve the scope of revision feedback instead
  of broadening it into content or design prohibitions. Scientific communication
  guidance connects method discrepancies to their effect on interpretation and
  scopes shared qualifications across reports and slides.

- Manuscript guidance distinguishes incidental drafting-session narration from
  scientific descriptions of agents, prompts, tools, workflows and data, preserving
  required disclosures. Research retains scientific plausibility and evidence-fit
  assessment without requiring an appended self-check section or checklist.

- Scientific reports and presentations share audience- and evidence-led content
  guidance, with a concise `scientific-communication` skill. Research routes
  substantial narrative reports from completed evidence to Writing; report
  handoffs no longer impose execution-audit fields. Presentation workers receive
  Humanizer and the existing Origin/NPG plotting guidance, and keep core evidence
  on the slides. Manuscript launch framing is scoped separately from progress
  reporting and its meaningful negative findings.

- Writing delegates editable PPTX work to `presentation_worker`, with full native
  file and shell capabilities, the EasySlides skill, and `generate_figure`.
  EasySlides Python dependencies are pinned in the control-plane environment;
  deployment scripts preinstall and carry its scripts, templates and references.
  Deck authoring defaults to native editable objects and rendered-slide inspection.

- `generate_figure` generates and edits image assets through OpenRouter with
  per-call model selection, reference images, and image options. Writing and
  its writing worker bind the general tool name; the previous Nano Banana name
  remains a compatibility alias. Profile templates default to GPT Image 2.5
  Sunburst. Editable presentations retain native text, tables and page layout.

- Literature Review and its workers can directly search OpenAlex and Semantic
  Scholar, look up records, and request seed-paper recommendations alongside
  provider-routed web search and dedicated source acquisition. Citation
  finalization and retained export scripts default to one BibTeX file; another
  format requires an explicit choice and is not emitted as a companion copy.

- Retired the imported Nature workflow skills and slide-layout templates, plus
  the standalone presentation plotting preset. Customer-validated Origin/NPG,
  Humanizer, academic-launch writing and rendered-artifact rules remain intact.
  Local literature guidance now lives in `literature-evidence-use`; retained
  citation utilities preserve complete metadata and unassessed candidate
  exports. Reporting guidance lives with `publication-launch-writing`.

- MLFF environments pin Sella 2.6.0 for its PRFO eigensolver optimization.
  GPU resource cards limit BLAS to one thread by default, with explicit resource
  overrides preserved; CPU resources retain their threading configuration.
- Shell GPU templates use DPDispatcher's native device assignment and GPU waves.
  A shared system `flock` coordinates batches across provider/general GPU cards
  on the same host/account. Deployment documentation distinguishes these settings
  from Slurm allocations and describes the private configuration updates required.
- Skill Evolution accepts local DBOS completion statuses and native run IDs,
  reads the selected turn's request and answer from workspace messages, and
  delivers learning results through the existing chat notifications. Checkpoint
  continuation retains the input episode; explicit Learn and historical evidence
  remain anchored to the selected run without rebuilding checkpoint snapshots.
- Async follow-up guidance prioritizes interruption for corrections to a child's
  current premise or instructions; additional evidence and later questions can
  queue behind valid work. Cards label supplementary instructions separately and
  show the corresponding run's processing state without claiming model receipt.
- LLM templates use Astra xhigh for specialist coordination and hypothesis
  proposals, while Astra workers retain high. Luna literature workers retain
  xhigh and thread titles retain low.
- Persistent Research's main researcher checks shared discussion during its
  existing turns and decides whether to answer, defer or continue an original
  child task. Branch-to-branch discussion remains available. Posting no longer
  creates automatic root-review runs, and unresolved messages do not block Graph
  completion. Shared discussion, peer discovery and closing checks are scoped to
  the Persistent Research owner and descendants; independent ordinary Research
  keeps child-task delegation without these additions. The UI shows recorded
  decisions and follow-ups, without an automatic root-wakeup control.
- Research can delegate independent ResearchSpecialist investigations with their
  own hypothesis–method–result cycle, isolated context and shared workspace Graph.
  Branches can start further independent investigations in the same shared pool.
  Native experiment/literature/writing workers inside a branch retain synchronous
  delegation. The main research session owns overall scope and completion.
- Background agent tasks have configurable shared and low/medium/high concurrency
  limits. Whole-task cost changes pause at native checkpoints and automatically
  resume the same task when admitted. Computational waits retain their slot;
  remote submission tools and Slurm job scheduling are unchanged. Activity shows
  cost and capacity waits, with separate capacity for foreground interaction.
- Agent follow-ups default to queueing new evidence behind running work. Explicit
  interrupt and WebUI Steer remain available for immediate corrections; a steer
  preserves the same task's cost reservation during the handoff.
- Result records, updates and Graph detail forms preserve Methods, Results and
  Conclusion. Missing method/conclusion fields in old records display an explicit
  old-record placeholder. Concurrent scientific writes retain revision checks.
  Background message IDs containing colons can be cited directly as Graph sources.

- XYZ and other coordinate previews open directly in MatterViz's 3D view without
  chemical valence checks or bond-order/charge assignment. Metal complexes and
  reaction geometries remain viewable, including trajectory frames; explicit
  chemical editing retains its validation.
- Background Activity and task details use the same current execution status.
  Historical duplicate cards and retained checkpoint task records no longer
  revive stopped tasks in the active rail. Refresh preserves real queued/running
  tasks across foreground turns without scanning old conversation payloads.
- Runtime deployment and remote packaging use the WebUI's in-process DBOS
  execution host without the retired Agent Server URL, launcher, or manifest.
  Full-repo sync preserves local execution state, login data, runtime directories,
  and secret files; deployment archives exclude them.
- WebUI startup and login no longer create `admin` or `default` workspaces.
  Without a selected workspace, users see a workspace creation/selection screen;
  stale links do not recreate deleted workspaces or launch thread requests.
- Native DeepAgents delta message persistence now reaches raw subagents and
  general-purpose workers, avoiding a full image-history copy at every checkpoint.
  Existing full-message SQLite checkpoints resume through LangGraph's native
  reader; historical saves remain intact.
- Native local agent execution persists model usage and publishes WebUI usage
  updates after each completed model call, retaining prior counts on recovery.
- Fresh background specialists receive their explicit task brief without an
  automatically prepended parent Research Graph request. Workspace/Graph bindings
  and normal evidence queries remain available.

- Local WebUI execution uses DBOS with a disk SQLite queue and native LangGraph
  workspace SQLite checkpoints. Background specialists run concurrently while
  users continue the foreground conversation. Task completion explicitly chooses
  parent continuation or notification only; stop/pause takes precedence.
- Removed the intermediate Agent Server launcher, SDK control plane, pickle
  persistence patches and online import path. Ordinary old SQLite conversations
  reuse their existing checkpoint binding. Offline imports remain outside the
  application. Research Graph scientific records remain in workspace.sqlite.
- Native checkpoint recovery preserves the accepted input across process death,
  including graph completion before workflow acknowledgement. Steer and rollback
  wait for graph teardown; failed-step continuation and native approvals retain
  their distinct controls. Background briefs, activity and completion policies
  remain visible in the WebUI.

- Local Agent Server state/history readers share the native checkpoint stores
  without reloading and retaining three additional pickle dictionaries per
  reader. Closing a reader leaves the owner's periodic flush and stores intact.
  Image content, checkpoint history and native continuation state are preserved;
  the in-memory backend still loads the retained history once per process.

- Legacy compressed conversations restore the nested summary message after
  Agent Server JSON transport. Subsequent DeepAgents compaction and continuation
  from already imported failed checkpoints retain the summary, cutoff and
  history references without importing the conversation again.
  Summary input uses the upstream factory default, avoiding a constructor-only
  45K trim that could replace a long tool trace with an empty-summary placeholder.

- Activity cards publish the latest update from each native state snapshot;
  transcript replay no longer cycles through historical tool names. Returning
  to a browser tab or reconnecting refreshes current thread state and messages
  before resuming live updates. Full process history remains readable.

- Async task lists and cards show excerpts of existing delegation instructions.
  Task details and `check_async_task` expose the original brief and current-run
  follow-up in full, including while running. Instructions survive compaction
  through native metadata; retained older run inputs are recovered on demand.
  No additional agent arguments or task scheduler are introduced.

- The chat's Running now spinner stays visible while a turn is active or runs
  are queued, including gaps between tool calls.
- Legacy async task acknowledgements also suppress automatic activity
  subscriptions, preventing queries of historical child runs after migration.
  Completed native tasks remain available for explicit activity inspection.
- Ordinary Research resumes after async specialist completion even when its
  attached Research Graph is already complete or archived. Persistent Research
  pause rules and native completion deduplication remain in effect.
- Async specialist cards open a live process viewer with nested-worker reasoning,
  explicit updates, tool inputs/results, and targeted Steer/Stop. Active children
  survive parent-turn changes; completed cards retain their history entry.
- Native async run creation opts into resumable subgraph/message streams through
  Agent Server HTTP middleware, without changing DeepAgents task scheduling.
- Composer settings, stop controls, and send strategies have aligned layouts and
  visible icons. Ctrl+Enter follows the selected running-task strategy.
- Native Steer/Stop interruptions retain partial output and display as
  interrupted activity instead of a synthetic task-failure alert.

- `general_execute` runs stage-local Python/Bash scripts through the existing
  single/batch DPDispatcher submission tools. Environment discovery reads
  descriptions and script permissions from resource cards; dedicated tasks
  remain preferred for supported operations.
- DeepAgents filesystem tools can read existing DPDispatcher receipts beneath
  the instruction snapshot mount through a more specific workspace route.
- Builtin source export collects static CatMaster imports recursively in one
  call, preserving separate original modules and package initializers. Repeated
  exports reuse identical files and protect workspace edits by default.
- Default writing coordination, prose workers and plot workers use GPT-6 Astra
  with high reasoning in the Codex OAuth and general templates. The Codex writing
  worker uses the existing OAuth worker profile; literature workers retain Luna
  xhigh. General templates retain their OpenRouter connection.
- Explicit YAML `temperature: null` now omits the parameter instead of being
  replaced by an environment/default value, including Astra OpenRouter requests.

- Native Codex OAuth authentication accepts `CATMASTER_CHATGPT_AUTH_PATH` for deployments with a writable private token store. LangChain still owns locking and refresh.

### Persistent Research and scientific memory

- Research and Persistent Research now progress through one ResearchSpecialist;
  Graph recovery no longer launches a second planner or candidate tournament.
- Independent scientific interpretation and next-check reasoning share the existing
  proposer. A declared persistent scientific stall triggers one native independent
  reconsideration and normally one authorized validation before parking.
- Judgment edges retain their conditions and reasons. Scientific revisions link
  preserved H/R records, and key stopping decisions retain open premises and
  validation outcomes for later navigation.
- Completed research stages stay complete after graph edits or late results.
  Internal continuations retain model/workspace bindings, and native async thread
  references are recognized within their bound workspace.

### WebUI conversation reliability

- Legacy imports restore private middleware state with native `Command.update`,
  preserving prior summaries and their cutoffs in the effective model context.
- The 258,000-token compaction setting is available again, independently of
  task-completion budgets, and configures upstream summarization directly even
  when the provider lacks a model context-window profile.
- Internal context summaries appear as a compact “正在压缩上下文…” activity
  instead of streaming their text into the assistant reply. The same boundary
  applies when rejoining a native run and when selecting its final answer.
- Legacy disk-backed conversations can continue when their checkpoint contains
  an empty `files` field absent from the current DeepAgents graph. Empty file
  state no longer causes HTTP 409; nonempty unsupported state stays protected.
- Local Agent Server startup includes a narrow workaround for upstream checkpoint
  flush registration defects. Real process tests cover multi-turn state, graceful
  restart, periodic persistence after forced termination, and native resume.
- Legacy conversations, including stopped or failed runs, retain their SQLite
  messages and state on the first native turn. Durable pending results and delta
  channel history are restored with upstream reducers; DeepAgents repairs
  unanswered tool calls. Old unfinished background records become interrupted
  history and do not restart completion watchers. Missing bound native threads
  and missing legacy sources with existing display history produce explicit errors.
- A persistent import marker prevents old history from returning after native
  state is cleared. Concurrent first imports and uncertain submission acceptance
  cannot trigger blind reimport. Ordinary submissions query thread metadata
  without fetching full state to decide whether to import. Run status queries
  select summary fields instead of retransmitting historical run inputs.

- Snapshot reload and SSE replay now share a cursor; positioned, durably written
  text deltas preserve numbers and prevent overlap from duplicating text.
- Streamed function-call aliases are consumed once. Tool input/output inspection
  reads the native projection's payload, and plans retain their actual state when
  a root reply finishes while background work continues.
- Background completion turns appear as collapsible runtime notices, not user
  speech. Declared report links and localized file sections populate Outputs;
  native child activity survives root completion and reload.

### Scientific Execution

- `mlff_vib` and `mlff_ts` now use verified native full Cartesian Hessians from
  MACE 0.3.16 and FairChem UMA 2.22. MACE supports its documented
  `(3N, N, 3)` tensor layout. UMA-S uses the correct general, uncompiled
  full-vmap path only when a fast `N*E` neighbor-topology memory estimate plus
  5% margin fits the currently usable CUDA memory; rejected preflights and
  actual CUDA OOMs fall back to finite differences under `auto`. UMA-M uses
  finite differences without attempting a native Hessian. UMA runtime pins
  move to `fairchem-core==2.22.0` and `torch==2.13.0`. MatterSim and ORB-v3
  retain serial finite differences.

### LLM Runtime

- The Codex OAuth profile now uses GPT-6 Astra high for coordinator and
  technical-worker roles. Luna literature workers remain xhigh and title calls
  remain low. Astra profiles explicitly omit temperature through native provider
  kwargs, avoiding the generic config's temperature fallback.
- The validated LangChain pins are now `langchain==1.4.0`,
  `langchain-core==1.6.2`, and `langchain-openai==1.6.0`, including upstream
  reasoning-item boundary preservation. The OpenAI SDK and agent execution
  stack retain their existing pins.
- WebUI stream reattachment now inherits the native run's stream modes. This
  keeps live messages and tool updates flowing after a WebUI restart with the
  pinned SDK/Agent Server query-parameter behavior. Reconciliation uses the
  native run's message identity so a Research Graph notice referencing that
  run cannot hide its unfinished chat answer.
- Research Graph execution now passes the turn's graph, focus, and launch to
  Agent Server as run configuration. Bound Result and blocker writeback can
  finish the correct launch instead of leaving it running after a result was
  recorded. Unbound turns explicitly clear the launch binding.
- Agent Server instruction memory is now scoped to the canonical workspace
  path instead of the display name. Same-named workspaces no longer share
  memory in the server store. Direct local runners retain their existing
  workspace-local namespace; old shared namespaces are not copied because
  their workspace ownership can be ambiguous.
- Writable specialists and workers now keep one canonical persisted format per
  logical scientific artifact. Reference libraries default to BibTeX, ordinary
  figures/renders to one high-resolution PNG, and structures/trajectories to the
  single native format needed downstream when no format is specified. Plotting
  QA conversions stay temporary, and equivalent multi-format outputs require an
  explicit user request or a real interface/venue contract.
- WebUI refresh now restores the active thread within the current browser tab,
  lands loaded conversations on their latest message, and reuses the workspace
  resolved by the thread list instead of independently scanning every project
  space for the initial messages, artifacts, and event stream.
- WebUI reasoning traces now reconcile LangChain's completed reasoning snapshot
  with the reasoning blocks already streamed for that model call. Aggregation-only
  spacing differences no longer create a duplicate trace block, adjacent streamed
  summaries remain visually separated, and model-end reasoning still fills the
  trace when no live reasoning was available.
- Literature source acquisition now uses exact-pinned ScanSci PDF 1.14.0 with
  Patchright 1.62.2 as its preferred browser backend and CloakBrowser 0.5.10 as
  the compatible fallback. CatMaster retains its direct legal-OA-first routing,
  adds the configured official Elsevier API before the bounded DOI-page
  fallback, and exposes optional Supplementary Information retrieval through
  the existing high-level acquisition tool while keeping main-PDF validation.
- Literature Review now exposes `batch_acquire_literature_sources` for a
  selected identifier list supplied directly or through a workspace text,
  CSV, or TSV file. It reuses the single-source legal and verified acquisition
  path, normalizes duplicates, rejects invalid lists before downloading, and
  enforces a non-bypassable 50-row limit; prompts and the literature skill
  guide routine batches toward coherent 10-30-source sets.
- Agent Server is now the sole WebUI authority for conversation threads, runs,
  double-text handling, checkpoints, interrupts, and stopping. The six
  CatMaster entries are registered as LangGraph graph factories, and the WebUI
  is a thin BFF/projection over `langgraph-sdk` rather than a second run-control
  plane. The custom AgentTask store, dependency scheduler, completion inbox,
  streaming runner, and local run-control flags have been removed. DeepAgents
  remains exact-pinned at `0.7.11`; Agent Server support adds exact
  `langgraph-sdk==0.4.4`, `langgraph-cli[inmem]==0.4.31`,
  `langgraph-api==0.13.3`, and `langgraph-runtime-inmem==0.33.3` pins.
- Agent Server projections now stream root model text while a run is active,
  preserve the complete native HITL interrupt envelope, and resume
  `HumanInTheLoopMiddleware` with its documented `decisions` payload. Files
  explicitly named in a successful final `## Files` section are registered as
  directly openable artifact cards when they exist in the workspace. Startup
  reconciliation also rebuilds a missing local display row when Agent Server
  accepted a run immediately before the WebUI process stopped.
- Agent Server tool-call projection now coalesces LangChain's streamed argument
  chunks by call ID and index, including the normal later chunks whose ID is
  null, and treats provider item IDs plus complete ToolCall records as one
  invocation. This prevents thousands of anonymous tool cards and replay events
  from one real turn. Ordered projection writes run outside the FastAPI event
  loop, and early UI events no longer claim a native run directory before its
  owning RunContext has created it.
- Persistent Research now launches background domain work through DeepAgents
  native async subagents in independent Agent Server threads/runs. Task Context
  reads native async-task state and sends child Steer/Stop actions directly to
  Agent Server. A recoverable completion subscriber converts a terminal child
  into one semantic root input for result verification and synthesis; restart
  reconciliation redelivers only notifications not already recorded, without
  owning the child lifecycle. Completion wakeups carry a native run-metadata
  idempotency marker so restart recovery does not replay one completed child.
  LangGraph v3 lifecycle envelopes remain protocol metadata rather than
  assistant answer text.
- Running-thread input now exposes Agent Server's `interrupt`, `enqueue`,
  `rollback`, and `reject` strategies as Steer, Queue, Replace, and Reject if
  busy. Stop targets a concrete native run and never claims that an already
  submitted remote scheduler job was canceled. The bundled Agent Server
  remains a development-only service, but local source use now has one process
  entry: `start_webui.sh` starts, checks, and stops both the loopback Agent
  Server and WebUI. An explicitly configured non-loopback Agent Server remains
  externally managed; automated deployments require `CATMASTER_AGENT_SERVER_URL`
  for a persistent service.
- Removed CatMaster's prescribed numeric execution budgets from the active LLM
  profile and every shipped configuration template. Async specialists and
  Research Graph exploration have no CatMaster-imposed model-call, token,
  elapsed-time, or source-count cutoff by default; they converge against their
  explicit scientific completion or failure criteria. A deployment may still
  impose a visible resource-concurrency boundary.
- Removed CatMaster's fixed-three consumed-image replay rewrite. Native
  `read_file` image results now remain unchanged in active history and durable
  checkpoints until DeepAgents performs its normal conversation compaction,
  avoiding repeated mutation of older prompt prefixes.
- Per-run LLM usage fallback no longer counts usage-bearing AI messages replayed
  inside LangGraph checkpoint `values` or `updates`; only finalized model-end
  payloads (or legacy message projections) enter the current run totals.
- Literature and Writing handoffs reuse canonical source handles and batch
  related corrections instead of repeating source extraction or opening one
  audit episode per issue. They do not impose fixed turn or correction-pass
  limits. Bundled worker reasoning remains high or xhigh, and general-purpose
  children continue to inherit their owning agent's model.
- Fresh subagent re-delegation keeps contexts isolated while carrying forward
  the prior final handoff's exact validated action parameters, authoritative
  paths, completed conclusions, and remaining work. Unchanged objectives reuse
  that compact handoff instead of repeating the same preflight or forwarding raw
  tool history.
- `md_trajectory_summary` now discovers and inventories native ASE `.traj`
  files, reports their frame and atom counts, and exports the last frame as a
  one-frame `.traj` while retaining its inventory-only role.
- Ordinary Research threads can remain attached to the same workspace Research
  Graph without inheriting Persistent Research orchestration. New threads no
  longer copy a currently selected Persistent Research entry, and switching a
  top-level Persistent thread to an ordinary entry retains graph/focus context
  while demoting it to an independent primary thread. Session steering,
  planning/comparison ownership, and scheduler child routing now require an
  explicit Persistent Research root; stale root records are migrated and graph
  orchestration ownership is restored to the sole valid root when possible.
- Experiment, Materials, Dynamics, and ORCA/xTB now share three atomistic skills
  for constraint-guided rigid-fragment assembly, geometry-first validation and
  recovery, and optional reconstruction from clear literature figures.
  ASE/SciPy reference scripts provide multi-start rigid pose
  optimization and PBC-aware absolute plus covalent-radius-normalized contact
  checks without adding another registered tool. The literature-figure skill
  can use one bounded general-purpose multimodal branch to produce a
  source-separated morphology brief for the construction script; this visual
  description is modeling guidance, not a validation gate. The execution
  prompts treat immediate optimization, SCF, energy, or force failures after
  assembly as possible construction failures: rebuild from the last chemically
  valid components before changing numerical settings or resubmitting, then use
  rendered inspection and physical pre-relaxation only after the numerical
  geometry gate. `mlff_sp` now accepts `task_config.document_extxyz=true` to
  write a same-basename extxyz calculation record with atom-resolved forces;
  its maximum-force summary uses raw forces so fixed atoms cannot hide a local
  geometry problem.
- Skill Evolution now defaults unconfigured workspaces to `auto`, so an
  eligible reviewer-approved Follow-auto revision is selected for the next
  run. Explicit workspace choices and deployment overrides for `off`,
  `observe`, or `auto` remain authoritative.
- Persistent Research now keeps one stable Research Session folder in the
  WebUI. Experiment execution threads appear inside it, while automatic
  planning and pair-comparison threads stay out of the ordinary conversation
  list and remain available to developer diagnostics. The root conversation
  projects current background activity, durable status, progress, and decision
  counts even when its own turn is idle. It also provides separate controls for
  steering the active execution, pausing later automation, resuming it, opening
  the active child, and inspecting Research Graph decisions. Explicit persisted
  parent and role fields preserve this grouping through refresh and service
  restart; ambiguous older execution threads remain in a visible related-work
  group instead of receiving guessed ownership.
- Persistent Research now writes every recorded Result back to its stable root
  conversation as a milestone, including a directly openable report artifact
  when the Result cites one. The Research Session panel keeps the latest Result
  and report visible, and scientific wait, no-change, blocker, and completion
  states produce explicit root-thread closeouts; a wait is labeled unfinished
  rather than looking like a completed or silently stopped turn.
- Every specialist, worker, reasoning delegate, and proposal checkpoint now
  receives the resolved absolute project `files/` directory as its current
  workspace boundary and shell-location reference. Filesystem function tools are
  explicitly distinguished as a virtual namespace rooted at `/` and must not
  receive that physical absolute path. The `execute` contract identifies the
  physical directory as its starting cwd, prefers workspace-relative paths, and
  forbids host-file exploration outside it; explicitly
  mounted virtual skill and memory roots remain reachable only through their
  virtual paths.
- Every task-facing specialist, worker, reasoning delegate, and proposal
  checkpoint now receives one shared artifact rule: file persistence is opt-in,
  empty manifest/audit ceremony is suppressed, and disposable OCR, conversions,
  one-off scripts, logs, and intermediate tables go under virtual `/tmp/`
  (`files/tmp/`, or `tmp/` from shell) and stay out of normal handoffs.
- Writing now uses one `writing_worker_agent` for drafting, revision, explicit
  prose polishing, and final integration. The redundant `academic_polisher`
  role, `writing_polisher_agent`, and direct prose-overwrite tool have been
  removed; review defaults use `write_reviewer`, while TeX repair falls back to
  `task_runner`. The bundled Codex OAuth profile routes `section_writer`
  explicitly to `google/gemini-3.7-flash` with `high` reasoning through
  OpenRouter and does not perform a hidden key-dependent model fallback.
- The WebUI now exposes `Persistent Research`, which uses the existing
  ResearchSpecialist and workspace Research Graph while creating or adopting an
  `auto` graph from the first instruction. Its lightweight scheduler advances
  one real Experiment at a time; it does not use the retired Hypothesis Engine,
  MCTS state, global scores, or a dual-score evaluator. A successful branch-
  proposer turn now atomically admits its complete staged H/E set, including
  drafts and unselected alternatives, before a selection-only pass compares the
  full ready frontier. Each comparison is a fresh isolated DeepAgent call over
  host-selected canonical H/E/R evidence, excludes proposer/reviewer/search
  assessments and earlier choices, reverses the final A/B order, and requires
  the winner to beat `wait` before auto launch. Manual and automatic graphs use
  the same revision-bound pair outcomes. The Research Graph panel records
  explicit decision preferences and shows pair reasons, decisive sources, and
  unresolved tradeoffs without score bars or per-route admission approvals.
  A proposer no-change response still enters clean selection when the current
  graph already contains ready Experiments. Comparator guidance treats method,
  representation, parameters, sampling, and implementation as downstream
  execution choices rather than reasons to stop; `wait` is reserved for a
  concrete failed route or an external, user, safety, cost, or authorization
  boundary that prevents every bounded candidate from advancing the goal.
  An implementation-ready `external` Experiment now preserves a laboratory or
  collaborator synthesis/measurement handoff without entering pair selection or
  any CatMaster launch. The graph and catalog show handoffs awaiting Results;
  the inspector offers direct external Result entry, which reconnects the
  returned observation to the same Experiment and resumes normal planning.
  Failed or incomplete planning remains stale and retryable instead of being
  misclassified as a successful plan or no-change outcome.
- LangChain OpenAI is pinned to `1.5.1` with `langchain-core==1.5.4`.
  This published release includes upstream fix `#39635`, so streamed Responses
  calls retain the completed encrypted reasoning item for later `store=false`
  replay without a CatMaster adapter workaround.

### Evidence Reachability and Self-Evolution

- Skill Evolution now treats the reflector target as an anchor and the
  proposer-selected owner as the single candidate-chain identity. Each pass
  starts from the target's current open delta while consolidated observations,
  exact-version skill-use contrasts, and their runs remain lazily queryable.
  The original anchor and resolved owner are stored separately, so rerouted
  deferred evidence is recovered when a later real episode directly targets
  its final owner.
  `defer` keeps evidence open, `ignore` consumes a non-durable finding, bounded
  `needs_revision` repair continues the exact draft, and ordinary new evidence
  starts from the selected effective version instead of inheriting a rejected
  draft. Independent approval continues through `auto` without a second human
  confirmation.
- Stored PDF, DOCX, XLSX, and PPTX files now use one DeepAgents `read_file`
  interface. Compact documents use provider-native file blocks; files that
  would overfill one request are returned as bounded local text with explicit
  integer offsets. Later document pages reuse one incremental process-local
  parse, while a normal source edit starts a fresh parse. The parallel
  `read_document` tool, opaque document cursors, and hash-bound continuation
  have been removed. Durable checkpoints omit inline file bytes and tell the
  agent to reopen the stable workspace path.
- A public URL opened through `open_public_page` or `find_in_page` is fetched
  once into a readable `literature/public_pages/` workspace source. Character
  and match continuation use the returned `source_path` and explicit offsets
  locally, without another URL request or an opaque snapshot/hash identifier.
  Literature-corpus result offsets remain explicit. OpenAlex cursor paging and
  Semantic Scholar offset paging remain exposed through their native provider
  contracts, and canonical scholarly records retain complete returned author
  lists and abstracts.
- Shared `nature-shared` directories are plain readable asset packages rather
  than independently discoverable skills. They no longer carry their own
  `SKILL.md` or manifest, and skill tests no longer freeze optional asset
  imports or exact `always_load` lists. Repository guidance also forbids
  treating ordinary trusted workspace files as hostile network inputs or
  adding proactive SHA-256, tamper, fingerprint, or secure-manifest layers to
  general tools without a real external protocol or demonstrated failure.
- Usage and machine-time summaries derive from complete event sets and publish
  their covered event count and ID range. Self-Evolution skill-use projection
  pages the complete event set internally but exposes semantic read/use/outcome
  state and exact pagination errors rather than a mechanical event inventory.
  Large model-visible tool fields keep exact readable offload references in the
  final tool message, and an offload write failure becomes a visible failure
  instead of a clipped success.
- Self-Evolution jobs now anchor one complete user episode to its final physical
  run while preserving an explicitly selected older run. Exact-target history
  is complete and pageable without attaching several run databases at once;
  mixed reflection batches retain successful items and allow only a selected
  failed item to be retried. Every retry is keyed to its immutable predecessor,
  so a failed retry can create a distinct successor without collapsing onto an
  earlier attempt. Candidate revisions remain immutable under concurrent
  proposal and revision work.
- Self-Evolution trace and history tools now publish their exact read-only table
  columns. Rejected queries and invalid tool arguments return model-visible
  correction results, so the reflector can retry in the same ReAct loop while
  direct programmatic queries remain strict. Oversized SQL results are rejected
  before entering model context and remain fully reachable through explicit SQL
  pagination or the stable event-field continuation reader. Event views now
  include callback and parent-callback IDs for causal branch reconstruction.
- Self-Evolution reflection, proposal, and review now run on the DeepAgents
  harness. Large tool results are offloaded to an invocation-local readable
  backend, long episodes receive evidence-aware automatic summarization and
  context-overflow recovery, and a constrained read-only evidence investigator
  may isolate bounded trace branches. Temporary context never becomes part of
  a candidate revision; proposal and review filesystem authority remains
  unchanged.
- Durable findings from one reflection batch are persisted before candidate
  materialization. Findings with the same exact target now share one evidence
  bundle and one proposer/reviewer pass instead of producing redundant
  revisions in the same episode; later episodes still advance the immutable
  revision normally.
- Evolution history no longer reports a reviewer's active, uncommitted
  candidate transaction as a corrupt revision. Committed revisions with missing
  or malformed descriptors still return their exact read error.
- Self-Evolution validation is now agent-owned: the host keeps only real path,
  authorization, immutable-revision, and release-pointer boundaries. Candidate
  loadability is checked through the same DeepAgents middleware as the active
  runtime, and exact loader diagnostics plus the current bundle are returned to
  the proposer for bounded repair. Fixed section order, placeholder, prose,
  reference, file-count, tool-registry, and similar formal quality vetoes have
  been removed.
- `allowed-tools` skill metadata is now completely inert in CatMaster skill
  discovery and loading. It is neither parsed nor audited and cannot filter a
  skill, candidate, or runtime capability. Explicit tool authorization remains
  in the existing tool-policy boundary.
- Self-Evolution workers now inspect the complete mounted skill tree and
  authorized workspace history, may correct the initial owner anchor, and can
  open full raw semantic events without a host-generated mechanical inventory.
  Multiple findings may point to the same initial target, and jobs without a
  durable SOP update retain an explicit `no_change` result and the reflector's
  stated reason.
- Self-Evolution now treats reviewer `approve` as activation eligibility rather
  than the start of a routine human release queue. `needs_revision` performs up
  to two automatic immutable repairs, and persistent workspace `off`, `observe`,
  and `auto` modes control whether approved `auto_head` revisions remain dormant
  or become the selected next-run version. Disabled and pinned targets are never
  overridden.
- Effective Skills state is shared by self-evolution and fresh specialist-run
  staging. Top-level chat agents can inspect or apply exact enable, pin, mode,
  rollback, and genuine held-authorization changes through a typed capability
  bound to the current user-message ID. Automatic activation, `no_change`, and
  real clarification requests return concise updates to the originating chat;
  automatic, chat, and GUI transitions retain one ordered before/after history
  record per state change without duplicate activation-result events.
- Canary use and outcome telemetry is observational. CatMaster no longer
  converts success counters into semantic approval, automatically stops a
  canary after failure, silently demotes an active pointer, or rewrites a
  candidate after target drift. Exact reviewed-revision selection is governed
  by workspace mode and per-target policy; human input is reserved for a
  genuinely held authorization, safety, or unresolved-preference boundary.
- Materials Project search keeps 50 as a visible, overridable default and labels
  default versus agent-selected subsets explicitly. Adsorption generation now
  persists a deterministic all-sites record and continues by stable site offset
  using the declared slab path, slab ID, mode, and adsorption distance without
  exposing or requiring a slab digest.

### WebUI

- Redesigned the workspace with a unified warm-white and forest-green theme,
  sidebar navigation, a compact conversation header, a full-width composer,
  and editable new-conversation starting points. Agent and permission controls
  now sit in the composer; native run controls retain their existing behavior.
  Task Context can collapse on desktop and surfaces conversation artifacts and
  the latest research report through the shared preview. Files, Research Graph,
  message cards, and narrow-screen drawers share the same visual language.
- Workspace startup now selects the newest user-facing root conversation rather
  than a more recently updated background execution thread. Native async
  specialist branches remain openable from Task Context but do not appear as
  ordinary thread-navigation rows; generated and legacy projected titles are
  bounded so a long objective cannot take over the Chat header.
- New threads now keep an empty persisted title while displaying `New thread`
  as a placeholder. The first ordinary user message supplies an immediate local
  title, and an explicitly configured low-effort Luna call can refine it in the
  background without delaying the research turn. Compare-and-set persistence
  preserves manual renames and thread ordering; legacy placeholder titles are
  backfilled locally, and `thread.updated` now carries the public thread
  projection needed for live rail updates.
- Long-running turns now keep a compact **Running now** panel above the
  composer. It projects active persisted tool parts from the complete current
  turn, keeps the oldest operation first, and shows the tool's actual start
  time plus a browser-local elapsed clock without heartbeats or scheduler
  polling. Major research phases can emit sparse semantic progress updates;
  the latest update remains visible above the Plan card while earlier updates
  stay in a compact history, and reasoning blocks have visible source
  boundaries. A restart reconciles
  same-instance orphan turns to Interrupted while leaving another WebUI
  instance's active threads untouched.
- Agent filesystem operations now accept the UI-visible `files/` prefix as an
  alias for the existing workspace-files root, and `sandbox:/files/...` links
  in assistant Markdown open the corresponding file in the shared large
  preview instead of being rendered with an empty destination.
- Assistant Markdown can now render `sandbox:/files/...` workspace images as
  responsive figure cards inside the conversation. The browser resolves them
  through an authenticated thread-scoped image route, keeps image bytes out of
  persisted messages, opens the source file in the large preview when clicked, and
  replaces missing or non-image sources with a compact failure notice.
- Chat's right rail now belongs to task context and currently shows the Plan
  projection across its full height. The former file-preview tabs and Todo
  height splitter have been removed. Images, files, artifacts, reports, and
  complete activity details instead reuse one near-full-viewport modal with a
  dimmed backdrop, top-right close control, Escape/backdrop dismissal, focus
  containment and restoration, and mobile full-screen layout.
- Skill Evolution is now available in trusted no-login deployments under the
  fixed local `admin` actor. Its post-run worker and complete management surface
  use the same workspace `off | observe | auto` lifecycle as login deployments,
  while the no-login worker stays outside authenticated `users/*` roots.
- Login cookies are now namespaced by a stable WebUI instance ID. CLI launches
  default that ID to the listening port, so concurrent instances on one hostname
  no longer overwrite or delete each other's browser session; `--instance-id`
  and `CATMASTER_WEBUI_INSTANCE_ID` provide an explicit stable override.
- Todo updates remain complete in the persisted trace, but Chat now projects
  repeated `write_todos` snapshots from one user task, including checkpoint
  continuations, into one final Plan card per semantic agent role. DeepAgents'
  semantic agent name takes precedence over opaque `tools:<id>` namespaces and
  callback backfill, so coordinator and worker plans are no longer duplicated
  or mislabeled by a later observation.
- Threads can now be renamed in place from the workspace rail, with the new
  title persisted through the existing thread update API. The desktop workspace
  rail can be resized by pointer or keyboard and remembers its width. File trees
  in both the rail and Files workspace preserve overflowing names and expose
  horizontal scrolling instead of making the rest of a filename unreachable.
- Authenticated tool-call details page complete stored input and safely redacted
  output by cursor. Candidate revisions expose pageable whitelisted proposal,
  review, validation, and response-evidence files; Research Graph mutation
  history is pageable by event ID. Artifact and remote-receipt registration no
  longer drops declared records beyond the former inline discovery caps, and
  per-item registration failures remain visible.
- New public artifact records use UUID identifiers and reuse the stored
  thread/path record on repeated registration instead of deriving the public ID
  from a path hash. Run-state observability events no longer add an unused state
  hash.
- A failure raised while LangGraph is executing now exposes **Continue from
  checkpoint** on the latest failed turn. The action resumes the same durable
  DeepAgent thread with `None` input, so it retries the failed graph tail
  without adding a synthetic user message or replaying completed checkpointed
  steps. Stale and non-resumable failure cards cannot start a continuation.
- Monitor now provides an expandable per-model-label token table with
  uncached input, cached input, cache-write, output, total-token, and call
  counts while keeping raw provider usage metadata out of the public view.
- Skill Evolution processing history now labels every terminal job explicitly,
  including deferred evidence, ignored non-durable findings, no durable change,
  an execution lapse covered by existing guidance, candidate creation, partial
  completion, and legacy results. It displays each finding's reflected change,
  rationale, anchor and resolved owner, and evidence handles, and pages beyond
  the initial job set. The tab badge counts unresolved job failures together
  with only candidates awaiting a genuine in-chat clarification. Processing
  history keeps its natural height inside the page scroll, so long result cards
  remain reachable instead of being clipped below the candidate grid.
- The existing Skill Evolution page now opens a complete effective-Skills
  manager from **Manage skills**. It provides cursor-paged targets and immutable
  versions, workspace evolution mode with an exact auto-change preview, selected
  version versus `auto_head` and latest draft, enable/disable and Follow auto or
  Pinned controls, and automatic/chat/GUI history. Changes apply to subsequent
  runs; candidate and processing cards remain secondary diagnostics. Save
  feedback also distinguishes an enabled next-run selection from a disabled
  target that will not be staged.
- Long assistant activity traces now consolidate the latest Todo state at the
  top and group reasoning, progress, and tool calls by subagent invocation.
  Activity appears between the Plan and final prose. Groups with many events or
  one substantial reasoning block collapse to their current or latest activity
  while preserving the complete trace on expansion; same-named parallel
  invocations remain separate.

### Runtime Reliability

- Fresh `read_file` images now remain native through the model call that consumes
  them. Subsequent model requests and durable checkpoints retain the three most
  recently consumed images inline and replace older image payloads with stable
  workspace-path hints, so long visual-analysis threads can reopen source images
  without replaying every prior base64 payload.
- All lane coordinators, workers, reasoning subagents, and proposal-review modes
  now inherit one prominent audit-reuse stop rule. Before starting or delegating
  audit-like work, they must reuse a completed result for the same claim on
  unchanged evidence; different names, agents, decompositions, parent ownership,
  or final synthesis do not justify another pass. A follow-up must identify an
  explicit independent-assessment requirement, changed evidence, a failed or
  incomplete prior answer, or a concrete contradiction, and inspect only the
  affected scope.
- A bound Research Graph no longer causes a one-off calculation to visit the
  hypothesis proposer. Hypotheses now remain falsifiable physical, chemical,
  or materials claims rather than computational recipes. Research preserves
  explicit user and established project, reproduction, or comparison constraints;
  Experiment and the responsible worker own every unspecified model, method,
  parameter, numerical setting, and calculation stage.
- Experiment delegation now keeps simple one-off briefs close to the user's
  stated objective, inputs, constraints, and deliverable instead of expanding
  them into production-style validation plans. Scientific workers treat a
  complete dedicated-analyzer result as their default stopping point and open
  raw engine output only for missing, conflicting, failed, or explicitly
  requested verification. Experiment and Literature Review retain shared
  post-result Graph sampling guidance even outside Research, but defer it until
  a new reusable scientific Result exists. Direct Research, Experiment, and
  Literature Review turns now attach to the sole active workspace Graph or
  create one when none exists. Experiment writeback retains an explicit
  Experiment focus, while Literature Review can preserve a sourced standalone
  Result; ambiguous unbound or internal work returns an evidence packet.
- Shared specialist and worker policy now treats hashes/checksums,
  engine/framework/schema versions, runtime environments, machine/scheduler
  layout, and transfer-integrity metadata as non-blocking operational details by
  default. Agents no longer create secure manifests, digest inventories,
  compatibility matrices, repeated integrity passes, or similar validation
  layers merely because a task uses transfer, checkpoint, retry, archives, or
  managed execution; explicit user requests and concrete existing machine or
  result-changing compatibility requirements remain authoritative. The staged
  skill catalog no longer asks successful scientific work to report remote
  receipt IDs, collect model-file SHA-256 values, benchmark an acceleration
  choice across hardware/runtime stacks, or inventory package/OS metadata for a
  normal data handoff; receipt inspection remains available for returned remote
  failures.
- DeepAgents is pinned at 0.7.4. CatMaster now owns the common agent contract:
  every DeepAgent stack receives one explicit Todo middleware and one
  backend-bound filesystem allowlist; writable roles expose recursive `delete`
  behind workspace and memory-root guards, while reasoning and self-evolution
  review roles remain read-only. `write_file` keeps the native whole-file
  replacement behavior with a model-visible read-first warning, and the shared
  prompt restores only the missing completion, recovery, progress, and
  proportional-validation invariants.
- The structured `execute` exit code now reaches observability records, thread
  events, persisted tool-part metadata, and the public WebUI tool card. A
  nonzero process exit remains a successful tool transport rather than being
  rewritten as a framework exception. Native Codex OAuth `apply_patch` custom
  output is also retained in streamed tool results instead of appearing empty.
- The control-plane LangChain, LangGraph, and provider integrations remain
  exactly pinned. MCP remains on the latest validated 1.x line until its 2.0
  boundary is validated.
- Codex OAuth `apply_patch` calls that LangChain v3 streaming exposes only as a
  `non_standard` provider block now recover their missing scheduler metadata at
  the model-result boundary. The original custom-call block remains intact for
  Responses API replay, and already-normalized calls are not duplicated.
- Codex OAuth Responses replay now removes only malformed historical
  `web_search_call` items whose required action discriminator was lost by the
  current LangChain adapter. Valid hosted-search calls, assistant text, and the
  durable checkpoint remain unchanged, so a following turn no longer fails
  before model execution.
- Codex OAuth model calls now retry a prematurely closed chunked SSE response
  twice with short bounded backoff. This recovery replays only the interrupted
  model call, not the complete specialist episode or previously completed local
  tool work; the existing longer overload retry policy remains separate.

### Scientific Tooling

- Registered ORCA and CP2K boots now run eligible single-node Slurm jobs from
  compute-node-local scratch and copy all scientific outputs back to the shared
  DPDispatcher stage. Multi-node or unknown allocations retain in-place shared
  execution; failed scientific runs still return partial files, while a failed
  copy-back is reported as task failure instead of false success.
- ORCA keeps its high-volume working files in node-local scratch but now writes
  `job.out` directly to the stable shared DPDispatcher stage, so committed
  output becomes visible while the calculation is running and is not dependent
  on final node-local stage-out. Reader visibility remains subject to the
  site's shared-filesystem cache interval.
- ORCA method-selection guidance now separates the requested operation from
  electronic-method choice. For unprescribed routine main-group work it presents
  `r2SCAN-3c` as the geometry and requested-frequency starting candidate and
  `WB97M-V/def2-TZVPP` as a final-DFT candidate for relative energies, barriers,
  conformers, and noncovalent energies. B3LYP remains available for explicit
  continuity or task-specific evidence but is no longer the familiar-default
  fallback; gaps, spectroscopy, transition metals, multireference cases, and
  higher-level calculations follow property-specific selection instead.
- CP2K and LAMMPS preparation now stage complete native input files plus explicitly mapped dependencies without generating a host-selected method, force field, boundary, timestep, ensemble, restart, or convergence recipe. Their skills provide editable native examples, and the old recipe-only preparation helpers are no longer active.
- xTB and CREST now share a minimal native-stage surface containing exact ordered argv tokens and explicit file mappings. Registered execution invokes those tokens unchanged, transfers selected nested and hidden inputs, and returns each attempt into a new result directory without merging engine outputs over the authored stage; `crest_execute` replaces the former flat `crest_run` overrides.
- ORCA preparation now keeps one generic native keyword/block surface plus a two-endpoint NEB stager. Charge and multiplicity are required, periodic structures are rejected, endpoint atom mapping is checked, and execution reconciles PAL only in a runtime copy while preserving the canonical input and other PAL controls.
- CP2K, ORCA, xTB/CREST, and LAMMPS analysis now separates process completion, task convergence, and property availability. Missing frequencies remain unknown, final energies/optimization metrics are selected, ORCA shielding is not labeled as chemical shift, and unrelated runs no longer receive automatic relative energies.
- Quantitative trajectory analysis now requires the physical interval between stored frames and an explicit MSD fit window. LAMMPS dump coordinates, image flags, triclinic cells, boundary conditions, IDs, and normalized RDF are interpreted from native columns; periodic generic XYZ/ASE trajectories require an explicit wrapped/unwrapped declaration rather than an inferred coordinate convention.

### Agent Prompting

- The writing worker now has three explicit system-level prose constraints:
  avoid defensive self-weakening, avoid fragmented phrases/staccato sentences
  and bullet-point chains in place of prose, and maintain paragraph-to-paragraph
  claim-evidence-interpretation continuity. The default Codex OAuth profile now
  routes `section_writer` to GPT-5.6 Sol with high reasoning. Final manuscript
  drafts and prose-heavy reports must clear all three constraints before being
  marked final. The finalization pass now reads the complete `humanizer` pattern
  and word/phrase watch lists, applies its false-positive guidance, and explicitly
  checks negative parallelisms and tailing negations such as `not only ... but
  ...`, `not just ...`, `not merely ...`, generic `not X but Y` framing, and
  clipped negative endings instead of relying on one remembered keyword.
- Quantitative plotting now defaults to deliberately styled matplotlib output:
  Origin-like axes, ticks, typography, and line work are required, matplotlib's
  default style and color cycle are forbidden, and the Nature/NPG categorical
  palette replaces the former low-saturation pastel guidance. The four core
  prompt colors are `#E64B35`, `#4DBBD5`, `#F39B7F`, and `#8491B4`; the plotting
  skills carry the full ten-color palette and rendered visual-QA contract.
- Experiment briefs and scientific workers now treat reliable, reproducible,
  final, production, publication, conservative, robust, and QC as scope labels,
  not implicit requests for tighter numerical settings or extra calculation
  stages. Workers begin from a documented method-appropriate normal setting and
  tighten only for an explicit user requirement, a method/source-specific need,
  or observed convergence, noise, or sensitivity evidence. CP2K guidance now
  separates ordinary GPW/SCF/geometry starting candidates from system- and
  observable-specific convergence. LAMMPS guidance selects unit style and force
  norm before the tolerance, distinguishes analytic, ML, and molecular force
  fields, and separates tolerances from recovery ceilings and machine-precision
  toy checks. ORCA retains its manual-backed `TightSCF` recommendation only for
  an actually requested frequency calculation rather than ordinary SP/Opt work,
  and now prefers `WB97M-V` over `WB97X-V` as the unprescribed general-purpose
  main-group final-energy candidate while keeping both method- and
  property-dependent. The `-V` examples do not add a second dispersion model or
  automatically tighten ORCA's normal SCF and grid settings. ORCA task examples
  now expose only operation-specific keywords and native blocks; electronic
  methods, basis sets, dispersion, and solvation remain separate scientific
  choices so an unrelated scan, IRC, or property example cannot become an
  accidental method default. Shared runtime policy now treats every skill
  example as a potentially stale, non-authoritative fragment rather than a
  method prescription. LAMMPS, CP2K, xTB, and CREST references likewise expose
  task, algorithm, or argv fragments without bundling an unrelated force field,
  electronic method, molecular state, solvent, system, or numerical setting.
- A dedicated `vasp-implicit-solvation` SOP now owns VASPsol/LSOL preparation
  beyond `vasp_prepare`. It strongly prefers a converged, compatible, same-state
  vacuum `WAVECAR` with `ISTART=1` and `ICHARG=0`, requires agent-side restart
  staging and runtime read confirmation, forbids cross-structure or cross-state
  reuse, and keeps `IDIPOL`, `LDIPOL`, and `DIPOL` absent by default. The general
  VASP input skill now routes LSOL work to that SOP. VASP preparation, band,
  NEB, and dimer tools now define `enable_dipole=true` consistently as the full
  slab correction (`IDIPOL=3`, `LDIPOL=True`, and center-of-mass `DIPOL`), while
  `false` writes none of those tags. They continue to warn when a final INCAR
  combines `LSOL` with active dipole correction while preserving explicit
  overrides.
- Writing now exposes `nature-writing` as its single general scientific-
  manuscript skill. The former `scientific-writing` entry was removed after
  its useful generic IMRAD, study-design reporting, and professional-report
  support were folded into the routed `nature-writing` resources; duplicate
  citation, plotting, and sentence-style guidance now stays with the existing
  specialist skills that own those capabilities. The stale Research-lane copy
  of `nature-writing` was also removed because Research already delegates
  author-facing publication work to Writing.
- Nature's shared references are staged as the upstream `nature-shared` asset
  package without making that directory an independently callable skill.
  Calling skills can resolve the Terminology Ledger and other shared files
  through stable `/.deepagents/skills/.../nature-shared/...` paths.
- Writing now applies one shared academic-launch system contract in the Writing
  coordinator, drafting worker, and polisher. Paper planning, revision,
  compression, experiment organization, and display selection center the
  strongest evidence-supported publishable advantage; project chronology,
  unsolicited defensive self-assessment, irrelevant comparison dimensions,
  and non-scientific hardware/platform detail are excluded from journal-facing
  narrative while necessary claim-changing qualifications remain precise.
- Writing now includes a dedicated `plot_worker`. The Codex OAuth profile routes
  it to GPT-5.6 Sol with high reasoning. It turns supplied quantitative data
  directly into reproducible Origin-like publication plots, selects palettes by
  scientific semantics, and inspects rendered previews for clipping, collisions,
  legibility, and overlap between text and visual signals before handoff.
- Literature Review now has a named, non-delegating
  `litreview_worker_agent` for bounded discovery, source reading, extraction,
  and evidence-audit branches; `litreview_agent` retains coverage decisions,
  conflict resolution, and final synthesis. The Codex OAuth profile routes this
  worker to GPT-5.6 Luna with xhigh reasoning, routes `writing_worker_agent` to
  GPT-5.6 Sol with high reasoning, and keeps their coordinators on GPT-5.6 Sol
  with xhigh reasoning.
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
- Scientific provenance, hash identity, and execution contracts are now three
  separate prompt policies. Ordinary scientific QC preserves scientific inputs,
  methods, conditions, evidence, and results while leaving hardware, launcher,
  scheduler, build, license, receipt, and performance fields in runtime records
  unless a concrete failure, result-changing incompatibility, or explicit
  user request makes them relevant. A user may explicitly request inspection,
  comparison, recording, or reporting of any operational field, and may request
  an otherwise unnecessary contract artifact; those requests override the
  default reporting boundary.
- Managed scientific execution now treats CPU, GPU, accelerator, launcher, and
  build choice as operational routing rather than a default scientific gate.
  Workers select one compatible registered path and do not require cross-device
  equivalence or alternate-backend smoke runs unless the user asks or a concrete
  result-changing compatibility issue has been observed. Submission retry bounds
  remain operational safeguards and no longer imply a scientific `NO_GO` or a
  global experiment-wide recovery quota.
- Research-to-Experiment-to-worker computation briefs now preserve scientific
  objectives, invariants, comparison criteria, authority, cost, and stopping
  conditions without prescribing an execution playbook. Workers may make and
  report scientifically equivalent implementation corrections within that
  boundary. A failed specialist-selected worker, task keyword, backend, or step
  sequence now triggers an internal, scientifically equivalent revised
  delegation before any human blocker is reported; Research reroutes work that
  falls outside Experiment's worker ownership. Human input is reserved for cases
  with no authorized equivalent or a required change to user-controlled science,
  cost, time, safety, or authorization. Routine preparation, smoke, submission,
  and recovery are no longer forced into separate delegation episodes.
- CatMaster now explicitly replaces DeepAgents' auto-added `general-purpose`
  child for every specialist and named worker. The shared child remains a
  non-delegating context-isolation worker, inherits the caller's direct
  capability surface and staged skills, and adds bounded document access plus
  nonfatal tool-error handling without copying the full parent prompt. Its task
  brief, rather than a lane or blanket concurrency policy, defines scope and
  stopping conditions.

### Literature Review

- Literature Review now derives its working shape from the actual brief and
  reassesses convergence after each useful evidence batch. It returns a usable
  synthesis once more discovery is unlikely to change the answer, boundary, or
  next action, and a user stop prevents new branches. Selected-source
  acquisition now shares canonical DOI, arXiv, PMID, and public-URL identities
  plus in-flight work across one parent/worker research run; only transient
  failures or a materially changed authorized route are retried.
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
  writeback only when that turn is bound to a graph; Result, blocker, update,
  resume, and retraction calls still require an explicit Experiment focus.
  Unbound turns and Research-internal delegates have no bound mutation surface.
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

- Thread focus is now an explicit persisted action rather than a side effect of
  selecting a node. Top-level Research, Experiment, Literature Review, and
  Writing can set or clear focus; Result writeback from Experiment or Literature
  Review requires an Experiment focus, and Experiment can atomically create and
  focus a new graph Experiment when it explicitly adopts work.
- A bound Result can be corrected in place with graph/node revision checks while
  preserving its relationships, or retracted in the same run only while it is
  unjudged and solely produced by the focused Experiment. Human Result deletion
  requires a reason. Retractions recompute producer state and remain visible in
  recent graph mutation history in the WebUI. Blocked Experiments can be
  resumed to `ready` or `draft` after their blocker is removed.
- Preparation, authorization, acquisition, recovery, build, scheduler, and
  platform-diagnostic episodes may close without manufacturing a scientific
  Result. Result judging and graph writeback remain limited to completed
  scientific observations, while Research alone owns model-visible graph-scope
  corrections.
- The bound SQL tool now exposes its exact logical columns at the tool surface
  and identifies node, artifact, and message fields stored inside JSON payloads.
  Experiment writeback and query skills use the same JSON1 examples, avoiding
  guessed columns such as `workspace_artifacts.path` without adding duplicate
  database fields or allowing SQLite schema introspection.
- Research Graph planning now chooses node granularity by scientific decision
  rather than procedural step. Setup, acquisition, conversion, convergence or
  smoke checks, individual conditions or replicates, and analysis remain inside
  one Experiment when they serve the same Hypothesis and decision rule; a new
  node is reserved for a standalone Result that can change the next decision.
- An unbound thread's first Research turn now binds the workspace's sole open
  graph, or creates a manual graph from that request when none exists, before
  the turn context is frozen. Multiple open graphs still require an explicit
  choice, and direct Experiment, Literature Review, and Writing turns remain
  opt-in.
- Bound-graph SQL and scientific-node mutation tools are no longer shown on
  unbound turns; unbound Research retains graph listing and creation. Temporary
  plan staging and the Experiment evaluator are exposed only for a trusted
  active internal planning turn at the current graph revision.
- Research Graph identity and launch ownership remain host-bound in long-lived
  WebUI threads. Experiment and Literature Review can write only through their
  current explicit Experiment focus; internal delegates return evidence to
  their parent, and Writing remains scientifically read-only. A formal launch
  may continue across multiple idle turns, and Result/blocker writeback
  completes only the exact matching launch rather than a historical launch
  found by thread ID.
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
- Evidence judging is restricted to already completed scientific Results and no
  longer audits proposals, plans, platform feasibility, operational readiness,
  or preflight. Research Graph nodes likewise exclude access/license,
  hardware/platform, scheduler/receipt, build, and performance state unless a
  known compatibility issue materially changed a scientific observation.
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

### Self-Evolution

- Self-evolution reflection, proposal, and review now navigate the canonical
  observability databases through host-bound read-only `trajectory_runs` and
  `trajectory_events` views. A lightweight index contains the run/task, outcome,
  exact query schema, diagnostics, and stable evidence handles rather than a
  mechanical event/tool/token inventory. Normalized and authorized raw event
  bodies remain queryable with SQL/JSON1 and exact continuation handles instead
  of being copied into prompts or candidate files. Oversized query results are
  rejected without truncation.
- Model, tool, and task events now have one normalized semantic projection that
  keeps LangChain content blocks, parsed tool calls, model-visible tool results,
  and boundaries while excluding provider ciphertext, envelopes, deltas, and
  duplicate callback records. One reflection may emit several independently
  evidenced targets; each is proposed and reviewed independently, execution
  lapses create no durable observation, and the per-item outcome is persisted.
- Failed or recovery-review evolution jobs are never retried automatically. A
  logged-in user may explicitly retry one selected job; the retry points to its
  predecessor and reuses the same immutable run evidence without altering the
  original job.
- Every successor candidate revision, including a requested revision, selected
  failed-item retry, or same-target update with new evidence, now starts from
  the complete predecessor bundle or memory and exposes its proposal, review,
  and validation artifacts read-only. Supported deltas are refined rather than
  discarded by restarting from the stable owner.

### WebUI

- Steering queued during a run now yields after a completed tool/checkpoint
  boundary and continues from the same DeepAgent thread. Active scientific or
  remote tools are not cancelled to apply the newer instruction; turns without
  another tool boundary finish before steering starts.
- Completed turns now reconcile the live conversation with the persisted
  canonical message, including final Markdown and completed reasoning state.
  Long message-part pages retain their continuation reference, and the Plan
  rail reads a full current-turn projection instead of the visible page.
- The completed-message push now replaces the live Plan projection with its
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
