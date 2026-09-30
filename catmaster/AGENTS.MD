# CatMaster Runtime Guide

## External API Verification

Before changing active runtime code that depends on external framework APIs,
especially LangChain, LangGraph, DeepAgents, OpenRouter/provider bridges, or
streaming event contracts, verify the current API shape from official
documentation and the locally installed package source/signatures first. Do not
infer event payloads, message schemas, stream fields, or provider-specific
metadata by guesswork. When local behavior and public docs differ, document the
compatibility reason in the code or test and prefer stable documented surfaces
such as LangChain `content_blocks`, LangGraph v3 protocol events, typed
projections, and declared tool schemas.

This repository now uses a DeepAgent-native specialist runtime as the active path. Treat the current implementation under `catmaster/specialists/` as the source of truth.

Maintainer notes for CatMaster's non-standard DeepAgents/LangChain/OpenRouter
interaction layers are in [deepagents_interactions.md](../docs/deepagents_interactions.md).
Current multimodal file behavior is documented in
[deepagents_multimodal.md](../docs/deepagents_multimodal.md).

## 1) Active architecture

### Public entrypoints

The active runtime exposes these lanes:

- `research`
- `persistent_research`
- `experiment`
- `writing`
- `peer_review`
- `literature_review`

They are built in [runtime.py](specialists/runtime.py).

### Specialist roles

- `ResearchSpecialist`
  - The only orchestration-capable specialist.
  - Can delegate domain work to `experiment_specialist`, `writing_specialist`, `peer_review_specialist`, and `litreview_agent`.
  - Uses async `research_challenger` for consequential direction checks, independent interpretation, scoped judgments, hypothesis revision and useful next checks. It can mutate scientific graph records but cannot execute experiments. A bound Graph alone does not require delegation for a one-off calculation.
  - For reported physical/chemical properties, benchmark values, trends, mechanism-relevant quantities, spectra, adsorption/formation/reaction energies, barriers, band gaps, stability metrics, and thermodynamic quantities, default to literature or existing workspace evidence before proposing new DFT.
  - If literature/workspace evidence is not found but the property is calculable, state the evidence gap and tell the user they can explicitly request a calculation; do not launch DFT by default from a property lookup.
  - Ordinary Research fulfills the requested task. Persistent Research and its independent Research branches own complete H/E/R cycles within their respective objectives, including scientific method selection, execution delegation, interpretation and further authorized investigation. A completed experiment does not complete an unfinished research question. Root completion concerns the user's overall objective; branch completion concerns the delegated question.
  - `research` and `persistent_research` share the same builder, research_lead role and specialist hierarchy. Persistent mode advances within the authorized stage through the same native root. A declared scientific stall enters one independent challenger reconsideration and normally one concrete authorized validation. Completion, explicit pause and actual authorization/resource boundaries take priority. Literature/recommendations never authorize new calculations or laboratory execution.
  - An Experiment with `execution_lane=external` is an implementation-ready laboratory or collaborator handoff. It may be `ready`, but it never enters the automatic runnable frontier and cannot create a CatMaster child launch. Preserve its concise actionable plan, decision rule, hypotheses, and sources; the laboratory result is recorded later on the same Experiment and resumes ordinary evidence-driven planning. External ownership is neither `draft` nor `blocked`.
  - Before scientific closeout, assess plausibility and evidence-claim fit using available results, method/QC findings and literature context. Integrate material conditions and unresolved questions into the requested answer; do not require a separate self-assessment paragraph, section or checklist. Explicit verification or diagnostic requests retain their requested discussion.
- `ExperimentSpecialist`
  - Bounded execution specialist.
  - May delegate only to `materials_worker`, `dynamics_worker`, `ml_worker`, and `orca_xtb_worker`.
  - Uses direct public-source checking only for narrow best-practice, literature, or official documentation checks. Keep this limited to focused implementation grounding, not broad review.
  - Do not answer that CatMaster cannot calculate solely because the specialist thread lacks direct execution tools. If a task fits a worker-owned domain or managed execution path, delegate the bounded scientific calculation before declaring a capability blocker. If a specialist-selected worker or route fails, prefer a scientifically equivalent revised delegation over asking the user. Do not substitute a standalone probe unless the user requested capability inspection or one concrete unknown prevents a usable delegation.
  - `materials_worker` owns structure preparation, screening, relaxation, single-point ranking, and path optimization. `dynamics_worker` owns all MLFF MD, restart, trajectory-QC, and LAMMPS workflows, including MD that continues a materials workflow. MLFF dataset/model lifecycle work, including MACE training, fine-tuning, and benchmark evaluation, remains with `ml_worker`.
- `WritingSpecialist`
  - Writing coordinator, not a free-form research agent.
  - Owns substantial scientific reports, synthesis of existing literature evidence, progress presentations and manuscripts. Research routes narrative deliverables here even when their evidence comes from calculations; execution records and QC remain Experiment work. Writing does not start new experiments to complete a report.
  - Should delegate substantive drafting, explicit prose polishing, and final integration to `writing_worker_agent`.
  - Uses shared `scientific-communication` for evidence selection and audience-appropriate narrative, and `publication-launch-writing` for manuscript construction with on-demand reporting references and venue templates. Progress reports retain scientifically meaningful negative findings and changes of direction; manuscript framing does not erase them.
  - Delegates editable presentations to `presentation_worker`, which reads the presentation, writing-quality and plotting skill roots. It uses scientific-communication for authored slide text and scripts and checks that core scientific evidence appears on the pages themselves.
  - Delegates quantitative and data-native figure production to `plot_worker`, which owns direct matplotlib plotting, deliberate Origin-like styling instead of library defaults, the Nature/NPG categorical palette, and rendered overlap/legibility inspection.
  - The coordinator and writing worker share concise scientific-communication guidance. Manuscript-specific argument, template and submission guidance lives in skills selected for the actual task. Narrative reports retain their own purpose; language examples are optional editing references.
  - The coordinator owns editorial acceptance of the actual document or rendered presentation against the audience, purpose and user feedback. Worker completion does not substitute for this assessment. Reuse completed scientific checks; batch concrete corrections to the responsible authoring worker and inspect affected content after revision. The coordinator does not rewrite the final artifact itself, and optional aesthetic alternatives do not require more cycles.
  - Uses direct public-source checking only for tightly bounded background supplementation needed by the current manuscript or note.
- `PeerReviewSpecialist`
  - Editor-style review coordinator for one canonical manuscript PDF.
  - Delegates bounded review episodes to `peer_review_worker_agent`, keeps those episodes sequential because they share the workspace, and returns the editor decision, full reviewer comments, and durable review memo.
  - Does not run experiments, rewrite the manuscript, or take over research planning.
- `litreview_agent`
  - Literature-review coordinator for both direct entry and Research delegation.
  - Delegates scientific question groups to `litreview_worker_agent` for discovery, selected-source reading and extraction, then integrates the findings into the requested answer. Focused source checks address explicit verification requests or concrete inconsistencies. The worker uses CatMaster's high-level acquisition contract; a second download-only agent is not part of the runtime.

All lanes inherit one shared audit-reuse rule from the shared execution
policy. Before starting or delegating audit-like work, reuse a completed result
that addresses the same claim on unchanged evidence. Different names, agents,
decompositions, stages, parent ownership, or final synthesis do not make the work
new. A new pass is allowed only when independent assessment is explicitly
required, relevant evidence changed, the earlier result failed or omitted a
required answer, or a concrete contradiction was found. Any follow-up must name
that trigger and stay limited to the affected claims and their direct
dependencies.

Layered capability visibility is part of the runtime contract. Research passes
execution needs through `experiment_specialist` instead of inspecting remote
capability itself. Experiment may use the remote task catalog only to choose the
responsible domain and avoid impossible constraints; concrete resource checks,
worker skill/SOP interpretation, and managed submissions belong to the worker
agent that owns the domain.

Cross-layer computation briefs preserve scientific intent without freezing an
implementation playbook:

- A simple one-off calculation or analysis brief stays close to the user's
  wording: objective, supplied input or canonical path, explicit constraints,
  and requested deliverable. Add only missing scientific information required
  to execute it. Do not inflate it into a production-grade validation contract
  with invented comparisons, independent QC, analytic checks, directory
  schemes, report fields, precision demands, or convergence demands.
- Labels such as reliable, reproducible, final, production, publication,
  conservative, robust, or QC do not themselves request a tighter numerical
  standard or another calculation stage. Reproducibility normally means
  preserving the scientifically material settings actually used. Do not use
  these labels to add tighter tolerances, higher cutoffs or iteration ceilings,
  greater precision or sampling, optimization, frequencies, replicas, or
  comparisons that the user did not request.
- Research-to-Experiment briefs carry the objective, authoritative inputs and
  evidence, constraints explicitly supplied by the user or already established
  by the project's protocol, reproduction target, or comparison definition,
  authorized scope or cost, and the requested
  deliverable or stopping point. A coordinator or hypothesis-proposer suggestion
  does not become binding merely because it appeared upstream. Research owns
  scientifically meaningful model, representation, method and validation choices
  and explains which ones define the comparison. Experiment and its responsible
  worker choose implementation and unspecified numerical settings; they can
  select an appropriate method for a one-off calculation when none is specified.
  Research briefs do not
  prescribe workers, tool-call order, probes, backends, launchers, file-edit
  sequences, retries, or operational gates.
- Experiment may add the responsible domain worker and canonical artifact paths.
  Keep preparation, scientifically equivalent corrections, execution, recovery,
  and domain QC in one worker episode when they serve the same bounded scientific
  objective; split work at scientific decision boundaries rather than mechanical
  phases.
- Workers own implementation choices inside the stated invariants and authority.
  They may correct syntax, layout, compatible execution routing, runtime details,
  and restart or recovery mechanics without an approval round trip, then report
  material corrections in their handoff.
- If a named worker, task keyword, backend, or prescribed sequence is unsuitable,
  that is an internal delegation mismatch rather than a human blocker. The worker
  returns the failed assumption and needed capability to its immediate delegator;
  the coordinator preserves the scientific model and binding conditions, rewrites
  the brief, and selects a concretely different compatible worker or route.
- When the equivalent route lies outside Experiment's worker ownership, a
  Research-delegated Experiment returns the mismatch to Research for rerouting.
  Neither level asks the user merely to repair a specialist-authored brief.
- Human input is blocking only when no authorized scientifically equivalent route
  remains, or proceeding would change an explicit user requirement, a
  user-controlled scientific choice, approved cost/time, safety or destructive
  authority, or another boundary agents cannot change.

Treat an explicit user requirement or clearly labeled scientific invariant as
binding. Other lower-level suggestions in a parent brief are advisory and may be
revised by the receiving agent. Standalone capability probes are exceptional:
use one only when the user requested a capability check or one concrete unknown
prevents a scientifically usable delegation.

Subagent contexts share the workspace even when their task calls execute
concurrently. Before launching multiple subagents, consider whether their work
may modify overlapping files or directories. Read-only branches may run in
parallel without special isolation. If write overlap is plausible, give the
tasks separate output paths, designate one writer, or run them sequentially;
preserve concurrent changes rather than overwriting or undoing them. A lane may
apply a stricter rule when its workflow requires one canonical writer; the
peer-review lane currently runs its delegated review episodes sequentially.

Execution guidance for the active experiment path:

- `ExperimentSpecialist` may inspect the remote task catalog to form clean worker briefs, but managed submission belongs to the worker agents.
- Scientific engine execution with a registered managed path should stay managed remote execution. This includes VASP, CP2K, ORCA, xTB/CREST, LAMMPS, and MACE execution/training/MD paths.
- Local command/Python capability is for general preparation, file inspection, lightweight glue scripts, dependency setup for bounded local steps, and post-processing. Do not use it as a fallback to run scientific engine binaries, boot scripts, `sbatch`, or `mpirun` when a managed path fits.
- If a managed path is missing or unavailable, return the precise missing task/config/layout context to the coordinator so it can try a scientifically equivalent registered delegation. Report a human blocker only when no authorized equivalent remains; do not silently switch to local engine execution unless the user explicitly requested local-only execution or a dry run.
- A listed registered task whose task spec reports a configured execution binding has sufficient deployment provenance for normal submission. Do not turn intentionally hidden administrator fields such as resource revisions, queue/account/module/executable/license identifiers, or prior success receipts into end-user prerequisites. Treat infrastructure as blocked only after a concrete catalog, spec, or submission error.
- Treat hardware, accelerator, launcher, rank layout, and build choice as operational routing. Select one registered path that fits the prepared scientific input; do not require CPU/GPU, accelerator/reference, or cross-build equivalence as a default preflight, scientific gate, or production prerequisite. Compare execution backends only when the user explicitly requests it or a concrete compatibility problem may have changed the scientific result.
- Bound automatic retry of a failed managed stage to avoid duplicate work, but do not turn that operational retry bound into a scientific `NO_GO` or a global experiment-wide recovery quota.
- If a dedicated tool does not cover a bounded non-engine execution task, prefer implementing the missing step with local command/Python capability and mature third-party libraries inside the workspace over stopping at the tool boundary.
- A successful dedicated analyzer return is the worker's default scientific
  source of record. Once it contains the requested result, completed state, and
  scientifically material flags, stop. Re-open raw engine output only for a
  missing requested field, unresolved error or result-changing warning,
  conflicting evidence, or an explicit user request for raw or independent
  verification; do not reparse it merely to confirm the analyzer.
- Local command calls run inside a workspace-local Python environment layered on top of the main environment. If a bounded local task needs a missing Python package, install it there instead of modifying the main environment.
- If method defaults, package behavior, or scientific best practice are uncertain, prefer a narrow literature or official documentation check before improvising. Do not turn that into a broad literature campaign.
- Start numerical work from a documented, method-appropriate normal setting,
  made explicit where the native input or comparison requires it. Tighten only
  for an explicit user requirement, a method/source-specific requirement, or
  concrete current-run evidence of relevant nonconvergence, numerical noise,
  or result sensitivity. Change only the affected independent control. An
  iteration/evaluation ceiling is recovery capacity rather than an accuracy
  target, and an analytic QC comparison is not a request for machine-precision
  agreement.
- If the missing implementation becomes heavy, high-throughput, multi-step, or clearly worth rerunning, materialize it as a reusable workspace script under `scripts/` instead of hiding the whole workflow in one long ephemeral shell command.
- Give reusable workspace scripts a concise purpose and explain non-obvious inputs or method choices when useful. Do not require operational author/date fields for ordinary scripts.

Agent-visible tool schema rules:

- Treat tool schemas as an LLM-facing interface, not only as a Pydantic validation artifact. Legal JSON Schema can still be confusing after LangChain, provider, MCP, or registry conversion.
- Separate "may be omitted" from "may be null". Avoid `T | None = None` / `Optional[T] = None` for active-path tool inputs unless `null` is a real domain value the model should choose.
- For optional object/list/string controls, prefer non-nullable defaults: `Field(default_factory=dict)`, `Field(default_factory=list)`, or `""`. If legacy callers may send `null`, coerce it in a `model_validator(mode="before")` rather than exposing nullable schema.
- Avoid exposing `anyOf`/`oneOf` nullable unions for method-critical control fields, especially `dict`, `list`, `str`, enum, and override/config fields. Nullable unions can be stripped, normalized to the wrong branch, or read by the LLM as if `null` were the only safe value.
- Describe the intended empty value directly, such as "leave empty when using task_name" or "omit or pass `{}` when no overrides are needed"; do not tell the model to pass `null` unless that is the intended method value.
- For method-critical overrides, expose a named non-nullable object such as `template_overrides`, document accepted keys in the tool/catalog surface, and test that the override renders into the actual command or payload.
- Test the final schema that the agent sees, not just `BaseModel.model_json_schema()`. Check `StructuredTool.args_schema`, `ToolRegistry.as_openai_tools(...)`, and any sanitizer/provider adapter that may alter `anyOf`, defaults, required fields, or nullability.

### Operational metadata and validation restraint

Runtime validation must remain proportional to the requested scientific or
technical result. Hashes/checksums, signatures, file sizes and inventories;
schema/manifest/contract, engine/framework/package/build versions; runtime
environments and environment variables; executable/module/queue/task bindings;
receipt/context IDs; CPU/GPU identity, MPI/OpenMP and scheduler layout; transfer,
archive, cache, deployment, and performance metadata are operational details,
not default scientific QC, evidence, prerequisites, or blockers.

Do not proactively collect, compare, freeze, expose, or ask agents to reason
about those fields in ordinary tool use. Do not block a configured managed task
on version equality, environment or hardware equivalence, checksum agreement,
schema requalification, or a complete operational inventory. Routine
agent-visible returns should contain only the status, recovery handle or
attempt/output paths, and error information needed for the next action. Keep
internal task configuration, loader/transfer fields, runtime environment,
resource binding, schema/build versions, and detailed receipts out of prompts,
skills, task projections, scientific Results, and Research Graph records unless
the user requests them or they explain a concrete result-changing failure.

Do not build new defensive infrastructure around trusted scientific stages.
Forbidden default additions include secure manifests, per-file digest or
fingerprint tables, compatibility matrices, version/capability parsers, frozen
schema baselines, repeated local/transfer/remote integrity passes,
environment-equivalence gates, custom archive-security validators, and new
provenance/receipt schemas. A transfer, checkpoint, retry, manifest, archive, or
remote job is not by itself evidence that hashing or another validation layer is
needed. Prefer the existing transport and its ordinary failure semantics.

Do not reinterpret ordinary workspace tools as network-defense mechanisms.
User-owned documents, structures, slabs, models, stages, caches, and generated
files are trusted workspace state by default; their normal use does not justify
proactive SHA-256 calculation, tamper checks, fingerprint fields, secure
manifests, or hash-derived tool interfaces. An agent editing such a file is an
ordinary visible workspace action, not an attack. Use the smallest integrity
mechanism only when an external machine protocol or actual trust boundary
requires it, the user explicitly asks for it, or concrete corruption evidence
cannot be diagnosed through ordinary I/O and task failures.

Exceptions must be narrow: an explicit user request; an existing external API,
file-format, authentication, authorization, transaction, or downstream machine
contract; a concrete execution failure or known compatibility issue that can
change the result; or a version/environment detail that is scientifically
causal to reproducibility. Existing format parsers, authentication, cache,
deduplication, artifact identity, and remote-recovery mechanisms may retain the
minimal internal metadata their real protocols require. Do not project those
implementation details into agent QC or create parallel checks. Maintainers
changing external framework integrations must still follow the API-verification
rule at the top of this file; that is a development obligation, not a routine
scientific-run gate.

## 2) Current literature architecture

CatMaster has one active literature Agent construction path: the DeepAgents
`litreview_agent`, used directly by `literature_review` and compiled once under
`research`. Do not add a second LangGraph/LangChain literature Agent behind a
tool or compatibility route.

`litreview_agent` receives:

- one provider-routed `web_search` capability: hosted OpenAI web search for
  `codex_oauth`/OpenAI Responses models, otherwise the CatMaster Tavily tool;
- native OpenAlex and Semantic Scholar metadata search and record lookup,
  plus Semantic Scholar seed-paper recommendations, with provider pagination;
- high-level single-source and bounded batch acquisition tools that try direct
  legal open-access routes first, use the configured official Elsevier API for
  matching ScienceDirect DOIs, optionally use one internal ScanSci browser
  DOI-page fallback, validate PDF structure and identity, can request
  Supplementary Information for selected DOIs, and otherwise save one readable
  static page per source. A batch accepts a direct identifier list or a
  workspace list file, deduplicates normalized identifiers, and rejects more
  than 50 input rows without truncating them;
- local full-text corpus ingest/query tools;
- one deterministic batch citation finalizer.

It declares one leaf `litreview_worker_agent` for scoped scientific questions,
selected-source reading, extraction and focused source checks. The coordinator
owns coverage and integration into the scientific answer. Literature returns
findings, their meaning and recoverable sources; its reporting guidance is
separate from computational execution reporting. Research notes leave final
wording, chapter order and caption choices to the writing task. CatMaster explicitly
replaces DeepAgents' auto-added `general-purpose` child with one shared
context-worker specification. That child normally inherits the current layer's
model and capabilities; Literature Review binds its fallback to the same model
and tools as `litreview_worker_agent`. CatMaster passes staged skill roots
explicitly and adds bounded document access plus nonfatal tool-error handling.
Both literature workers are non-coordinating leaves; the named worker is the
normal delegation target, while `general-purpose` remains a compatibility
fallback for context isolation.

`open_public_page` and `find_in_page` are not part of the LitReview model-visible
surface. Literature Review and its leaf workers can directly call
`search_openalex`, `search_semantic_scholar`, `get_openalex_record`,
`get_semantic_scholar_record`, and `recommend_semantic_scholar`. OpenAlex also
serves as an internal fallback during final citation resolution. Other
specialists and workers receive provider-routed web search for narrow checks.
Do not bind hosted and Tavily tools under the same `web_search` name in one agent.

Search summaries, abstracts, and verified metadata remain valid Literature
Review evidence for the claims they actually support. Selected-source
acquisition is used when decision-critical details require more depth, not as a
per-paper completion criterion. The agent reads returned local artifacts rather
than controlling browser state or repeatedly fetching the same page. Do not add
numeric confidence scores, mandatory per-paper evidence labels, or entitlement
checks to enforce this policy.

Physical/chemical property lookups are a literature-first case: answer from public literature, citation-grade metadata, or existing workspace evidence before proposing new calculations. Missing property evidence is not implicit approval to run DFT; ask for or invite an explicit calculation instruction when the property is computable.

Final metadata normalization happens only after paper selection in one bounded
deterministic call. Do not reintroduce a per-record LLM metadata loop.
Bibliography export defaults to one BibTeX `.bib` file. Another format is an
explicit choice for a user request or downstream requirement, not an automatic
companion export. Unresolved identifiers remain visible in the tool response.

## 3) Runtime state and persistence

### Run state

For WebUI conversations, DBOS owns durable execution and queue state. Native
LangGraph SQLite owns checkpoints and interrupt/resume state. CatMaster binds
workspaces and projects UI events; do not add a second scheduler or checkpoint
format. Each deployment has one host and one control SQLite, with queue partition
concurrency one per workspace/thread.

Reuse each WebUI thread's stable `deepagent_thread_id`. Direct entrypoints build
the matching specialist through `SpecialistRunner` with its native SQLite saver,
Store, prompt, tools, skills and backend. New input contains only the current
HumanMessage and attachments; checkpoint state supplies history. Semantic runtime
notifications are HumanMessage turns, never dynamic system instructions.

Research delegates detached specialists through the DBOS background tool adapter.
New task IDs give isolated contexts; follow-up reuses an explicit task. Preserve
native worker nesting and general-purpose delegation. Completion policy is an
explicit `resume_parent` or `notify`; stop/pause overrides automatic wakeups.

Use DBOS queue/cancel operations and native LangGraph checkpoint/Command APIs for
enqueue, reject, steer, rollback and approval resume. Do not infer intent from
message roles, or emulate recovery by replaying UI transcript text. Stable run IDs
in native checkpoint metadata prevent duplicate inputs after process failure.

Direct CLI/domain `run_state.json` is an artifact/domain snapshot, not the WebUI
execution authority. Research Graph continues to own scientific nodes, results,
relationships and accepted launch bindings in workspace.sqlite.

Do not add new active-path features that depend on:

- `task_state.json`
- `FINAL_REPORT.md`
- `MEMORY/**`
- `update_memory`
- memory patch flows

### Threading and memory

All local specialist execution uses workspace SQLite persistence:

- `metadata/deepagent_threads.sqlite`: native conversational checkpoints.
- `metadata/deepagent_memory.sqlite`: native long-term memory Store.

The stable DeepAgent thread ID supplies short-term continuity. Keep the existing
workspace project ID memory namespace. Runtime startup must not scan or import
intermediate pickle stores; exceptional historical conversion is an offline task.

Long-term memory uses:

- `/.deepagents/AGENTS.md`
- `/memories/AGENTS.md`

Persistent memory is for durable project conventions, validated reusable knowledge, and stable workflow guidance only. Do not store transient task requests, scratch execution logs, speculative findings, or one-off run details.

### Workspace layout reminder

For active-path generated artifacts inside the workspace `files/` tree:

- keep user-facing topic outputs grouped by topic
- keep disposable OCR text, conversions, exploratory snippets, one-off scripts,
  logs, and intermediate tables under `files/tmp/`; agents see it as virtual
  `/tmp/` (`tmp/` in shell), and normal discovery and handoffs ignore it
- keep writing outputs under `writing/`
- keep reusable generated helper code and heavy deterministic execution logic under `scripts/`
- especially prefer `scripts/` for high-throughput screening helpers, large batch post-processing, and other multi-run reproducible pipelines
- for agent-created or substantially revised scripts, require an opening comment header with code writing date, responsible/related agent, implementation principle, and purpose

Every writable specialist and worker also receives the shared single-format
artifact rule. Persist one canonical file format per logical scientific
artifact, using the explicit or real downstream-required format and otherwise
defaulting to `.bib` for reference libraries, high-resolution `.png` for
ordinary figures/renders, and the single native structure/trajectory format
needed next. Source data, a script, and its resulting figure are different
artifacts; equivalent-format copies are not. Keep disposable conversion or QA
renders under `/tmp/`, preserve pre-existing user files, and allow duplicate
formats only on an explicit user request or a real external contract.

### Multimodal tool-content rules

Active specialist threads use one DeepAgents `read_file` interface with two document dispositions. Compact PDF and modern Office files may use provider-native file blocks; documents that do not fit one safe request use a bounded local text view with ordinary integer offsets. Raw document bytes do not remain in checkpoint history.

Maintenance rules for active-path code:

- User-submitted images, documents, audio, and video should be stored as workspace artifacts first. Images and supported media may be sent to the first model call as standard content blocks when the configured model capability allows it. Send PDF, DOCX, XLSX, and PPTX natively only after the same bounded preflight used by stored-file reads; otherwise send the stable path and let `read_file` page through text.
- Persisted WebUI thread messages must keep artifact references, MIME, size, and sent/degraded status. Do not persist raw base64 media payloads in `messages.jsonl` or thread metadata.
- Fresh `read_file` image results must remain multimodal through the next model call. Keep image-bearing message history unchanged between ordinary model calls; do not add a fixed-count, fixed-age, or rolling replacement layer for consumed images. Let DeepAgents handle old media at its normal summarization or context-overflow compaction boundary.
- Keep `read_file` as the only general file-reading tool. Its compact-document branch may deliver PDF, DOCX, XLSX, and PPTX as native file blocks. Its large-document branch must return bounded normalized text plus `next_offset`; do not expose a parallel `read_document`, opaque cursor, content hash, or host-selected cache path.
- DeepAgents 0.7.4 ignores `offset` and `limit` for binary reads and excludes `read_file` from generic large-result eviction. Preserve the CatMaster document middleware and backend guard that prevent an oversized binary block from reaching the next model request. OOXML container checks are internal parser safety, not agent-visible validation metadata.
- Never implement document overflow handling by writing a `read_file` result to disk and telling the agent to call `read_file` again. That creates a recursive read/offload loop.
- Replay compatibility handling belongs at the checkpoint or model/provider boundary. Persist a stable workspace-path hint instead of inline PDF/Office bytes without mutating a safe compact current-turn delivery or image results.
- When a later turn needs image, audio, video, PDF, or Office-document understanding, re-open the stored artifact with `read_file(file_path=...)`. PDF pages may also be rendered selectively. Do not rely on old inline media surviving indefinitely inside replayed history.
- Prefer attachments, file IDs, file paths, or generated artifact URLs over embedding large base64 blobs in multi-turn history.

This is a replay-safety rule for the active DeepAgent stack, not a ban on multimodal prompting. The goal is to keep `thread_id`-based continuity while preserving current-turn and fresh image/tool-result multimodal context.

## 4) Workspace Research Graph

Scientific state belongs to the workspace Graph in `metadata/workspace.sqlite`.
Threads bind a graph and a focus; the active execution owner is the graph's
`orchestration_thread_id`. The graph remains readable across threads and does
not become a conversation transcript or another task engine.

Research, Persistent Research, Experiment and Literature Review retain a valid
existing binding. Unbound turns attach the newest unarchived graph by creation
time, including completed stages; initialize a first graph only when none is
available. Graph listing, creation and switching belong to the host and user
interface. Agents receive only bound scientific query/write tools, with no graph
selector in scientific mutation inputs. Persistent Research begins from the submitted objective,
without a separate Auto or Update routes step. Session pause/resume governs
automatic continuation; an explicit continuation also adopts a legacy Manual graph.
An existing binding remains on the same graph, including after delivery;
status messages and runtime notifications do not silently reactivate paused or
completed research. A one-off task does not require artificial H/E/R records.

ResearchSpecialist owns scientific progression. It delegates through native
DeepAgents specialists and receives completion turns on the same native LangGraph
thread. Graph recovery reconciles already accepted work and delivers relevant
external evidence edits to that root; it never selects or launches scientific
work. Historical planning/comparison records remain inspectable.

The asynchronous independent `research_challenger` connects Results, earlier evidence and
counterevidence to scoped judgments, revised hypotheses and next Experiments.
It may recommend a next check and incrementally mutate graph records. It cannot
execute the check, write general files or issue shell commands; it retains
`ls`, `glob`, `grep`, native `read_file`, graph SQL, search and selected-source
acquisition. Its own `research_reasoning` skill root is staged explicitly.
Its own thread completes through the existing DBOS bridge and resumes the root.
Independent challenges may also address premature narrowing, missed methods and
assumed missing inputs before a stall is declared. The root chooses delegation;
Research branches retain full H/E/R responsibility.
Routine Results do not require a separate independent summary. Direct
Experiment and Literature Review retain their own scientific checks and Result
writeback; formal Peer Review retains its specialist hierarchy.

Keep Hypothesis, Experiment and Result as the scientific node kinds. Optional
importance/cost bands express supplied priorities and coarse execution demand,
not confidence or scientific value. A ready Experiment needs a usable plan,
decision rule and execution lane. External Experiments are actionable laboratory
or collaborator handoffs and cannot launch CatMaster work.

Result-to-Hypothesis judgments carry `scope` and `rationale`. `revises` links a
new H or R to an older same-kind claim, with action `replace`, `qualify` or
`withdraw` and an explicit scope and reason. Retain original observations and
valid partial findings. Substantial reinterpretation appends a new claim and
revision link; ordinary typo corrections may use local updates. Same-run
unreferenced mistaken Result cleanup retains its existing transactional limits.
No host rule automatically refutes or reruns downstream science.

Consequential stopping decisions live in `research_decisions`, with the issue,
authorization, basis node IDs, independent review, validation Experiment/Result
links and open premises. This is scientific memory, not a queue or all-action
ledger. Reuse the same decision for unchanged evidence. Persistent closeout
uses native middleware: an explicit stall requires independent reconsideration once,
then requires a validation outcome or concrete permitted exception before
parking. The model chooses the independent delegation; middleware never inserts
an AI tool call on its behalf. Ordinary replies do not require a global disposition.
Dynamic instructions are HumanMessage inputs; stable policy remains in system
prompts. User cancellation and checkpoint continuation remain native controls.

The current user stage defines completion. A real negative answer can satisfy a
judgment task; silently narrowing an unfulfilled user goal cannot. Mark completed
when the requested deliverable is met and stop. Scientific edits and late results
do not reopen delivery. Manual Graph edits never enable automation.

Graph context is explicitly partial. Standard read-only SQL exposes all bound
scientific records, relation attributes and decisions; source-owner records are
available when reachable through refs. Preserve SQLite JSON1, joins, recursive
CTEs, aggregation, window functions and ordinary pagination. A fixed snippet or
recent-results list is never a semantic access limit. Do not add preview/search/
detail wrappers, capacity restrictions, truth scores, content hashes or an
independent graph-summary authority. Follow revisions and counterevidence beyond
the initial neighborhood when the question requires it.

Graph SQL uses connection-local bound views plus SQLite read-only and authorizer
controls. Reject writes, schema access, ATTACH and other workspace tables. Normal
mutations use current revision CAS and return the changed entity and new revision;
a conflict calls for a targeted refresh and does not change role identity. Keep
original papers, files and detailed analyses in their existing owners, with real
DOI/URL/note/artifact/thread refs. A sourced Result need not invent a retrospective
Experiment merely to fit the graph shape.

Keep the `research_execution` skill root available to Experiment and Literature
Review agents, including direct entry outside Research, so a completed valuable
scientific Result is not lost merely because Research did not launch the turn.
This is post-result guidance, not preflight: do not read or apply it at task
start, during ordinary preparation/execution, or for status-only work. Direct
scientific entry turns use the host binding policy above. A bound Experiment writes through an explicit
Experiment focus, while Literature Review may write a sourced standalone Result.
An unbound turn or internal delegate returns a concise evidence packet
and owner paths without guessing a Graph target.

## 5) Final output contract

The active specialist output contract is intentionally soft and compact.

Final specialist responses should answer the user's request first and use the
shape the user asked for. Use compact `Summary` / `Facts` / `Files` sections
only for handoff-style closeouts where they make the result easier to inspect;
do not force those headings on ordinary conversational answers or literature
reviews.

For review-style or literature answers, use task-appropriate headings and state
the coverage boundary. Distinguish screened or retrieved candidates from the
smaller set of papers highlighted as key examples. Do not let a small familiar
set become the default depth for a genuinely broad review. Keep discovery-breadth
guidance in the literature skill, where it can be conditioned on the user's
scope; do not put paper-count targets in the specialist system prompt or replace
them with an internal numeric quota. Explicit paper ranges and brief, focused, limited-scope, or
non-systematic requests control discovery breadth. Broader reviews should expand
across the important coverage buckets until additional searches are mostly
duplicative, while the narrative may still highlight a smaller key-paper set.

Do not reintroduce heavy structured payloads like:

- `constraints`
- `open_questions`
- `decisions`
- `next_steps`
- duplicated artifact inventories

Proposal review is the exception: proposal checkpoints still use a structured `ProposalCheckpoint` schema because the human-interrupt flow depends on it.

## 6) Skills

### Skill roots

The skill layout below includes the `research_reasoning` root required by the
Research Graph implementation described above. Add and stage that root in the
same functional change as its query and subagent bindings:

- `skills/materials_worker/*`
- `skills/dynamics_worker/*`
- `skills/ml_worker/*`
- `skills/orca_xtb_worker/*`
- `skills/atomistic/*` for shared constraint-guided structure assembly,
  numerical/visual geometry validation, and geometry-first failure recovery.
- `skills/research_specialist/*`
- `skills/research_reasoning/*` for graph query and evidence reconciliation.
- `skills/litreview_agent/*`
- `skills/writing_specialist/*`
- `skills/presentation_worker/*` for editable deck authoring and rendering.
- `skills/writing_quality/*` for shared scientific communication and optional phrasing references.
- `skills/plot_worker/*` for direct publication-data plotting and visual QA.
- `skills/execution/*` for shared managed-execution support.

At runtime they are staged into:

- `files/.deepagents/skills/materials_worker`
- `files/.deepagents/skills/dynamics_worker`
- `files/.deepagents/skills/ml_worker`
- `files/.deepagents/skills/orca_xtb_worker`
- `files/.deepagents/skills/atomistic`
- `files/.deepagents/skills/research_specialist`
- `files/.deepagents/skills/research_reasoning`
- `files/.deepagents/skills/litreview_agent`
- `files/.deepagents/skills/research_execution`
- `files/.deepagents/skills/writing_specialist`
- `files/.deepagents/skills/presentation_worker`
- `files/.deepagents/skills/writing_quality`
- `files/.deepagents/skills/plot_worker`
- `files/.deepagents/skills/execution`

Worker agents and entrypoint specialists bind skills via DeepAgent `skills=[...]`, not through the old custom skill runtime.

### Skill authoring

Skills must follow the DeepAgent-style layout:

- one directory per skill
- a `SKILL.md` file
- YAML frontmatter at the top

Loader-required frontmatter fields:

- `name`
- `description`

Optional descriptive frontmatter fields:

- `license`
- `compatibility`

Existing `allowed-tools` declarations are inert upstream-compatible metadata.
CatMaster does not parse or audit them, and authors should not add them to imply
access control. Tool-choice guidance belongs in ordinary SOP prose when useful.
The runtime-visible tool surface is determined by the specialist/subagent
`tools=[...]` binding and its real authorization boundary.

Never implement a loader, middleware, registry, staging, discovery, validation,
or filtering routine that rejects, skips, hides, fails to stage, or otherwise
makes a skill unreadable because `allowed-tools` is absent, empty, stale,
contains an unknown tool name, or differs from the current tool binding. The
skill must remain discoverable and loadable when a recommended tool is
unavailable; enforce tool availability only at the actual tool binding or call
boundary. Tests must not make `allowed-tools` agreement a skill-loading
requirement.

When a tool surface changes, update the relevant skills in the same change.
Apply the repository-wide Agent-Facing Explanations and Tool-Skill Maintenance
rules: check every affected domain/shared/reference entry and the final visible
schema. Explain callable behavior and recovery, without backend machinery or
inaccessible agent roles. Record the per-tool guidance review for broad changes.
Do not hard-code concrete tool-name routing into generic specialist prompts. Managed execution prompts should stay abstract and worker-scoped; put concrete remote/managed tool choices in skills, tool schemas/descriptions, allowlists, and registry wiring. Coordinator specialists should know that worker-owned managed execution exists and should delegate to the worker, but they should not receive detailed remote tool-selection contracts for tools they cannot call.

Prompt visibility follows the actual invocation surface. A prompt may name a
delegate or tool only when that recipient can call it in the current graph.
Leaf task prompts must not mention invisible parents, coordinators, siblings,
downstream agents, UI placement, continuation scheduling, or host routing.
Give a leaf only its objective, authoritative references, explicit user
constraints, deliverable, and stopping condition; the runtime routes its final
result.

Skill loading uses progressive disclosure: expose role-appropriate metadata, let
the agent select the needed procedure, and load conditional references only when
relevant. Reuse already-read instructions while they remain in context. Do not
turn matching a skill domain into compulsory startup reading. Keep scientific
method, communication guidance and non-obvious internal contracts intact when
shortening prose. Read the relevant contract before an unfamiliar operation;
parameter types alone may not convey its layout, ownership or recovery rules.
Apply the repository's total-cost standard to runtime and self-evolution prompts,
using `../skills/AGENTS.MD` for detailed authoring rules.

Treat skill examples as non-authoritative, possibly stale fragments. Prefer examples that isolate one task keyword, native block, or argv fragment; do not bundle unrelated methods, structures, potentials, basis sets, solvents, and numerical settings into a complete scientific input that an agent may mistake for a default. The runtime must still preserve the complete native input surface so the agent can compose the actual task without rebuilding the execution harness.

Keep task-shaping numerical heuristics such as candidate counts, claim counts,
or review/compile pass targets in the relevant skill, where they can be
conditioned on task scope and stopping evidence. Specialist and worker system
prompts should express the qualitative objective and completion condition.
Exact counts may remain there only when they are genuine safety, concurrency,
transaction, or output-shape invariants rather than planning advice.

Do not encourage model-generated governance ceremony or files that merely prove
work occurred. Manifests, inventories, audit/QC tables, checklists, READMEs,
reports, frozen contracts, schemas, baselines, and lockfiles require a user
request, an existing interface, or decision-relevant downstream content. The
mere presence of a transfer, checkpoint, retry, archive, schema, or remote task
is not such a requirement.
Do not use these objects to justify secure manifests, digest inventories,
version matrices, environment gates, or repeated integrity checks. Preserve
real repository and API contracts; the target is invented process work, not
necessary interfaces.

## 7) Tool conventions

### Registration and returns

All agent-visible tools should be registered in `catmaster/tools/registry.py` and exposed as LangChain tools.

Tool functions should:

- validate inputs with a Pydantic model
- return LangChain-native `(content, artifact)`
- raise `CatMasterToolExecutionError` for business failures with a model-visible `public_message`
- construct `content` explicitly inside each tool as the model-visible operational result; include status, counts, recovery handles, and the small set of newly generated files needed for the next step
- match `content` to the tool's actual semantics: query, browser, catalog, and numeric-result tools must not synthesize file paths merely to satisfy a common shape
- keep full structured payloads and large file inventories in `artifact`; for a large file list, expose only its count plus an explicit manifest or output-root path in `content`
- never infer model-visible content by scanning arbitrary `artifact` or MCP `structuredContent` field names; heterogeneous tools own their own return semantics, and input paths already present in the tool call should not be echoed automatically
- never assume a LangChain/MCP `artifact` or `structuredContent` field is visible to the model; test each tool's final `ToolMessage.content` when an output path or recovery handle is operationally required
- keep both the registered input-model docstring and the tool function docstring tagged on the first line with a short `[scope/stage]` label
- use short, stable labels such as `[structure/discovery]`, `[bulk/modeling]`, `[defect/modeling]`, `[electronic/prepare]`, `[phonon/modeling]`, `[vasp/prepare]`, `[vasp/analysis]`, `[md/analysis]`, `[mace/execute]`, `[ml/prepare]`, `[ml/analysis]`, `[structure/viz]`, `[thermo/gas]`, or `[literature/metadata]`

Because the active tool registry uses the input model docstring as the tool description, keep the first-line tag and summary synchronized between the input model docstring and the tool function docstring.

### Error behavior in the active path

In the active specialist runtime, tools are wrapped as non-fatal. Tool exceptions are converted into model-visible error messages instead of directly killing the lane.

This applies to both:

- top-level specialists
- subagents

Do not design active-path workflows that depend on tool exceptions terminating the run.

### Managed execution

Remote or managed execution tools are part of the platform capability surface, even when the corresponding executables are not visible in the local interactive shell.

When developing or expanding DPDispatcher-backed or other managed execution tools:

- update the relevant skills, tool schemas/descriptions, allowlists, and registry wiring in the same change
- keep any runtime prompt language abstract: state capability boundaries and ownership, not concrete tool-selection playbooks
- for low-level remote submission tools, keep task-specific stage directory layouts in skills/catalog metadata and require workers to prepare and verify the declared layout before submission
- managed execution should prioritize producing the requested result when safe: bounded automatic recovery is acceptable for likely transient remote failures, but receipts/context must be preserved so live or orphaned remote work is not ignored
- for heterogeneous ML tasks, prefer managed tools when they fit, but do not let missing tool coverage block local script-based execution

### Paths

User-visible artifacts belong under the workspace files tree, not metadata.

Prefer:

- project files under `files/`
- internal runtime state under `metadata/`

Use path helpers from `catmaster.tools.base` rather than manual string joins.

## 8) Writing lane expectations

`WritingSpecialist` is a coordinator.

Current expectations:

- write from existing workspace evidence only
- use audience, document purpose and concrete scientific findings to select the narrative; keep decisive structures, observations and performance comparisons in the report body or on the slides
- write as a knowledgeable colleague addressing an attentive reader: develop the important evidence into explanation and supported judgment, allocate detail by explanatory value, and rely on shared knowledge for routine implications
- use compact natural-language briefs for writing tasks, carrying the user objective, useful findings, evidence paths and explicit user requirements; leave unstated editorial choices to the author rather than adding writing prohibitions, hypothetical misreading checklists or universal caption fields
- authors select and integrate evidence notes using reader relevance; include conditions and qualifications when they affect interpretation or decisions, and keep production instructions in the work rather than narrating compliance
- preserve context isolation through coherent section or question groups and final integration; pass enough shared purpose and neighboring context to preserve the whole argument
- establish the delivery purpose and reader's assumed knowledge in the handoff: explanatory scientific report or talk, concise technical record for informed readers, or venue-facing manuscript; unspecified report/talk audiences default to advisors or experimental collaborators unfamiliar with the project's computational theory and history, while explicit internal technical requests retain their concise form
- keep completion-message brevity separate from document depth; preserve requested scientific coverage and the context/source paths needed to recover fuller findings, and permit source reading for explanation without repeating established scientific validation
- do not reopen broad research loops from writing
- delegate substantive drafting, revision, explicit prose polishing, and final integration to `writing_worker_agent`; there is no separate polisher role or agent
- delegate quantitative and data-native figure jobs to `plot_worker`; its default deliverable is a deliberately configured Origin-style matplotlib figure using the Nature/NPG categorical palette, never matplotlib's default style or color cycle; keep conceptual, mechanistic, and workflow image generation separate
- apply shared scientific-communication guidance to narrative work; select manuscript, template and submission skills only for the corresponding task
- authorial manuscript voice omits incidental drafting-session narration; agents, prompts, tools, workflows, files and runs remain valid scientific content when they are study objects, relevant methods or data, or required disclosures
- use the scientific-communication phrasing reference when a passage needs language revision; preserve scientific meaning and necessary negation rather than performing a mandatory keyword audit
- judge the actual argument, coverage, scientific meaning and reader relevance as well as local facts; repair explanations crowded out by qualifications or production commentary at their scope, without a fixed review cycle or separate self-assessment artifact
- if producing a TeX bundle, the worker should run compile directly and use diagnostics to repair bundle issues

Do not assume there is still a separate active `compile_agent` in the writing lane. Match the current implementation in `catmaster/specialists/runtime.py`.

## 9) Usage, logging, and UI

### Usage

Usage accounting is based on LangChain `UsageMetadataCallbackHandler`, not the old `LLM_USAGE` trace event.

Each completed LLM call must update `usage_summary.json` immediately and emit
the thread-native `usage.updated` event. The final turn flush is a fallback,
not the primary accounting point. LangGraph v3 model-end messages provide a
deduplicated fallback when callback propagation does not expose the finalized
`AIMessage.usage_metadata`.

Run usage includes per-model and per-role call counts, input/output/total
tokens, cache reads/writes, and reasoning tokens when the provider returns
them. Do not count the same call twice when both a LangChain callback and a
stream event expose its usage.

### Logging

The active runtime uses LangChain step logging and UI event callbacks. Avoid reintroducing verbose prompt/context dumps.

### WebUI

The WebUI is run-state and SSE driven. Active-path UI work should align with the current streaming model.

## 10) Code policy

When making changes:

- prefer `catmaster/specialists/` for active-path behavior
- prefer DeepAgent-native mechanisms over provider-specific shims
- do not add new provider-specific shims unless the change explicitly needs them

## 11) Testing guidance

When active-path behavior changes, prefer updating tests around:

- `tests/test_specialist_runtime.py`
- WebUI live-state tests
- tool-specific tests
- usage summary tests

If a tool or prompt change affects skill-guided behavior, update the corresponding skill files too.
