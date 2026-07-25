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

## 1) Active architecture

### Public entrypoints

The active runtime exposes these lanes:

- `research`
- `experiment`
- `writing`
- `peer_review`
- `literature_review`

They are built in [runtime.py](specialists/runtime.py).

### Specialist roles

- `ResearchSpecialist`
  - The only orchestration-capable specialist.
  - Can delegate to `experiment_specialist`, `writing_specialist`, and `litreview_agent`.
  - For reported physical/chemical properties, benchmark values, trends, mechanism-relevant quantities, spectra, adsorption/formation/reaction energies, barriers, band gaps, stability metrics, and thermodynamic quantities, default to literature or existing workspace evidence before proposing new DFT.
  - If literature/workspace evidence is not found but the property is calculable, state the evidence gap and tell the user they can explicitly request a calculation; do not launch DFT by default from a property lookup.
  - Defaults to on-demand, stage-bounded research. After a delegated result, check scientific reasonableness and evidence-claim fit, then close out once the user's requested or explicitly approved stage is complete. Record weak evidence and unresolved questions as limitations or future frontier items rather than automatically dispatching another round.
  - Its research-lane closeout should include a `Scientific Reasonableness Check` section covering plausibility, evidence-claim fit, method/QC/literature context, and unresolved gaps.
- `ExperimentSpecialist`
  - Bounded execution specialist.
  - May delegate only to `materials_worker`, `dynamics_worker`, `ml_worker`, and `orca_xtb_worker`.
  - Uses direct public-source checking only for narrow best-practice, literature, or official documentation checks. Keep this limited to focused implementation grounding, not broad review.
  - Do not answer that CatMaster cannot calculate solely because the specialist thread lacks direct execution tools. If a task fits a worker-owned domain or managed execution path, delegate a bounded calculation/probe before declaring a capability blocker.
  - `materials_worker` may keep managed MLFF inference workflows together, including screening, relaxation, single-point ranking, path optimization, and connected MLFF dynamics. `dynamics_worker` may also run MLFF MD when dynamics or trajectory analysis is the primary task, and owns LAMMPS workflows. MLFF dataset/model lifecycle work, including MACE training, fine-tuning, and benchmark evaluation, remains with `ml_worker`.
- `WritingSpecialist`
  - Writing coordinator, not a free-form research agent.
  - Should delegate substantive drafting to `writing_worker_agent`.
  - Uses direct public-source checking only for tightly bounded background supplementation needed by the current manuscript or note.

Layered capability visibility is part of the runtime contract. Research
delegates remote-capability probes to `experiment_specialist`; experiment may
use the remote task catalog only to make worker briefs accurate; concrete
resource checks, worker skill/SOP interpretation, and managed submissions belong
to the worker agent that owns the domain.

In the current shared-workspace runtime, all specialist and worker subagent
episodes are sequential by default. A parent must issue at most one subagent
delegation in a model response, wait for it to
finish, inspect its result and workspace effects, and only then decide whether
another delegation is required. Do not launch multiple subagents in the same
turn: they share the workspace and can race on scripts, outputs, and cleanup.
A future parallel-hypothesis runtime may opt into concurrency only when it
provides explicit workspace/write-scope isolation and a deterministic merge
contract.

Execution guidance for the active experiment path:

- `ExperimentSpecialist` may inspect the remote task catalog to form clean worker briefs, but managed submission belongs to the worker agents.
- Scientific engine execution with a registered managed path should stay managed remote execution. This includes VASP, CP2K, ORCA, xTB/CREST, LAMMPS, and MACE execution/training/MD paths.
- Local command/Python capability is for general preparation, file inspection, lightweight glue scripts, dependency setup for bounded local steps, and post-processing. Do not use it as a fallback to run scientific engine binaries, boot scripts, `sbatch`, or `mpirun` when a managed path fits.
- If a managed path is missing or unavailable, return a precise blocker with the missing task/config/layout context instead of silently switching to local execution, unless the user explicitly requested local-only execution or a dry run.
- A listed registered task whose task spec reports a configured execution binding has sufficient deployment provenance for normal submission. Do not turn intentionally hidden administrator fields such as resource revisions, queue/account/module/executable/license identifiers, or prior success receipts into end-user prerequisites. Treat infrastructure as blocked only after a concrete catalog, spec, or submission error.
- If a dedicated tool does not cover a bounded non-engine execution task, prefer implementing the missing step with local command/Python capability and mature third-party libraries inside the workspace over stopping at the tool boundary.
- Local command calls run inside a workspace-local Python environment layered on top of the main environment. If a bounded local task needs a missing Python package, install it there instead of modifying the main environment.
- If method defaults, package behavior, or scientific best practice are uncertain, prefer a narrow literature or official documentation check before improvising. Do not turn that into a broad literature campaign.
- If the missing implementation becomes heavy, high-throughput, multi-step, or clearly worth rerunning, materialize it as a reusable workspace script under `scripts/` instead of hiding the whole workflow in one long ephemeral shell command.
- Every agent-created or substantially revised reusable script under `scripts/` must start with a concise comment header containing code writing date, responsible/related agent, implementation principle, and purpose. Include key inputs/outputs when helpful.

Agent-visible tool schema rules:

- Treat tool schemas as an LLM-facing interface, not only as a Pydantic validation artifact. Legal JSON Schema can still be confusing after LangChain, provider, MCP, or registry conversion.
- Separate "may be omitted" from "may be null". Avoid `T | None = None` / `Optional[T] = None` for active-path tool inputs unless `null` is a real domain value the model should choose.
- For optional object/list/string controls, prefer non-nullable defaults: `Field(default_factory=dict)`, `Field(default_factory=list)`, or `""`. If legacy callers may send `null`, coerce it in a `model_validator(mode="before")` rather than exposing nullable schema.
- Avoid exposing `anyOf`/`oneOf` nullable unions for method-critical control fields, especially `dict`, `list`, `str`, enum, and override/config fields. Nullable unions can be stripped, normalized to the wrong branch, or read by the LLM as if `null` were the only safe value.
- Describe the intended empty value directly, such as "leave empty when using task_name" or "omit or pass `{}` when no overrides are needed"; do not tell the model to pass `null` unless that is the intended method value.
- For method-critical overrides, expose a named non-nullable object such as `template_overrides`, document accepted keys in the tool/catalog surface, and test that the override renders into the actual command or payload.
- Test the final schema that the agent sees, not just `BaseModel.model_json_schema()`. Check `StructuredTool.args_schema`, `ToolRegistry.as_openai_tools(...)`, and any sanitizer/provider adapter that may alter `anyOf`, defaults, required fields, or nullability.

## 2) Current literature architecture

CatMaster has one active literature Agent construction path: the DeepAgents
`litreview_agent`, used directly by `literature_review` and compiled once under
`research`. Do not add a second LangGraph/LangChain literature Agent behind a
tool or compatibility route.

`litreview_agent` receives:

- one provider-routed `web_search` capability: hosted OpenAI web search for
  `codex_oauth`/OpenAI Responses models, otherwise the CatMaster Tavily tool;
- a filtered, stateful `agent-browser` MCP surface for dynamic pages and
  user-authorized browser sessions;
- local full-text corpus ingest/query tools;
- one deterministic batch citation finalizer.

It declares no custom literature or metadata worker. DeepAgents' native
`general-purpose` subagent is the only child and inherits the same tools and
browser session for bounded context-isolated work.

`open_public_page`, `find_in_page`, OpenAlex, and Semantic Scholar tools are not
part of the LitReview model-visible surface. OpenAlex may remain an internal
deterministic fallback during final citation resolution. Other specialists and
workers receive the same provider-routed search surface for narrow checks. Do
not bind hosted and Tavily tools under the same `web_search` name in one agent.

Physical/chemical property lookups are a literature-first case: answer from public literature, citation-grade metadata, or existing workspace evidence before proposing new calculations. Missing property evidence is not implicit approval to run DFT; ask for or invite an explicit calculation instruction when the property is computable.

Final metadata normalization happens only after paper selection in one bounded
deterministic call. Do not reintroduce a per-record LLM metadata loop.

## 3) Runtime state and persistence

### Run state

The active runtime writes `run_state.json` in the run directory. This is the active run snapshot used by the WebUI and resume flow.

Active multi-turn continuity for specialists comes from the DeepAgent `thread_id` bound to the chat session. Do not manually inject chat-history summaries or reconstructed transcript text into active specialist prompts.

WebUI thread entrypoints must preserve DeepAgent-native session semantics:

- Reuse the WebUI thread's stable `deepagent_thread_id` as the LangGraph
  `configurable.thread_id`, even when the user switches between supported
  entrypoints in the same thread. Do not create entrypoint-specific checkpoint
  namespaces unless checkpoint fork/isolation support is intentionally added.
- Direct entry into `research`, `experiment`, `writing`, `peer_review`, or
  `literature_review` must still build that entrypoint's DeepAgent through the
  normal `create_deep_agent` path with the matching specialist
  `system_prompt`, tools, subagents, checkpointer, store, and backend. Do not
  bypass specialist system prompts by routing direct entrypoint requests through
  a generic agent wrapper.
- A new user turn should pass only the current user message/attachments to the
  agent runtime and rely on the DeepAgent/LangGraph checkpoint for conversation
  continuity. Do not reconstruct prior WebUI `messages.jsonl` rows into prompt
  text as an active-path context strategy.

Do not add new active-path features that depend on:

- `task_state.json`
- `FINAL_REPORT.md`
- `MEMORY/**`
- `update_memory`
- memory patch flows

### Threading and memory

The active runtime uses DeepAgent/LangGraph persistence:

- workspace-level SQLite thread checkpoint store:
  - `deepagent_threads.sqlite`
- workspace-level SQLite long-term memory store:
  - `deepagent_memory.sqlite`

Short-term conversation continuity uses a stable DeepAgent `thread_id`.

Long-term memory uses:

- `/.deepagents/AGENTS.md`
- `/memories/AGENTS.md`

Persistent memory is for durable project conventions, validated reusable knowledge, and stable workflow guidance only. Do not store transient task requests, scratch execution logs, speculative findings, or one-off run details.

### Workspace layout reminder

For active-path generated artifacts inside the workspace `files/` tree:

- keep user-facing topic outputs grouped by topic
- keep writing outputs under `writing/`
- keep reusable generated helper code and heavy deterministic execution logic under `scripts/`
- especially prefer `scripts/` for high-throughput screening helpers, large batch post-processing, and other multi-run reproducible pipelines
- for agent-created or substantially revised scripts, require an opening comment header with code writing date, responsible/related agent, implementation principle, and purpose

### Multimodal tool-content rules

Active specialist threads use bounded document ingestion plus DeepAgents-native multimodal input when the model and provider support it. Images remain multimodal content blocks; PDF and modern Office documents are parsed through bounded document paths so raw document bytes do not enter checkpoint history.

Maintenance rules for active-path code:

- User-submitted images, documents, audio, and video should be stored as workspace artifacts first. Images and supported media may be sent to the first model call as standard content blocks when the configured model capability allows it. PDF, DOCX, XLSX, and PPTX should be sent as bounded parsed text; render only selected PDF pages when visual inspection is needed.
- Persisted WebUI thread messages must keep artifact references, MIME, size, and sent/degraded status. Do not persist raw base64 media payloads in `messages.jsonl` or thread metadata.
- Fresh `read_file` image results must remain multimodal through the next model call. Do not blanket-textualize image `ToolMessage` content immediately after tool execution.
- Do not use `read_file` for PDF, DOCX, XLSX, or PPTX. `read_document` returns bounded parsed text directly and must stay below the large-tool-result eviction threshold. Use `pages` for PDF pages and PPTX slides only. Visual PDF analysis should render a bounded page selection to images, then inspect those images.
- Office content extraction belongs to `python-docx`, `openpyxl`, and `python-pptx`; do not manually unzip OOXML and parse its XML. A ZIP metadata preflight may reject encrypted or oversized containers before those libraries open them.
- Never implement document overflow handling by writing a `read_file` result to disk and telling the agent to call `read_file` again. That creates a recursive read/offload loop.
- Replay compatibility handling belongs immediately before the model/provider call. Degrade old inline PDF/Office payloads to explicit path/MIME text references without mutating image results or requiring another file-tool call.
- When a later turn needs image, audio, or video understanding, re-open the stored artifact with `read_file(file_path=...)`. For PDF/Office documents, use `read_document(file_path=..., pages=...)`; PDF pages may also be rendered selectively. Do not rely on old inline media surviving indefinitely inside replayed history.
- Prefer attachments, file IDs, file paths, or generated artifact URLs over embedding large base64 blobs in multi-turn history.

This is a replay-safety rule for the active DeepAgent stack, not a ban on multimodal prompting. The goal is to keep `thread_id`-based continuity while preserving current-turn and fresh image/tool-result multimodal context.

## 4) Research Kernel

`ResearchSpecialist` maintains a lightweight research board file at:

- `files/research_kernels/<thread_id>/kernel.json`

The schema is intentionally small:

- `question`
- `hypotheses`
- `run_cards`
- `frontier`
- `conclusion_draft`

`run_cards` are compact and decision-oriented. Each card keeps only:

- `source`
- `summary`
- `facts`
- `artifacts`

Keep `hypotheses` to the currently active 3-5 items only.

`frontier` is a non-executing record of unresolved questions and recommended
future work. It does not authorize another experiment or literature round. A
requested stage may close with frontier items still present when the limitations
are stated clearly.

## 5) Final output contract

The active specialist output contract is intentionally soft and compact.

Final specialist responses should answer the user's request first and use the
shape the user asked for. Use compact `Summary` / `Facts` / `Files` sections
only for handoff-style closeouts where they make the result easier to inspect;
do not force those headings on ordinary conversational answers or literature
reviews.

For review-style or literature answers, use task-appropriate headings and state
the coverage boundary. Distinguish screened or retrieved candidates from the
smaller set of papers highlighted as key examples. Do not let 15-20 papers
become the default review depth. When the user asks for a review, research
progress overview, systematic landscape, or perspective-style synthesis and has
not asked for a quick/brief answer, plan perspective-level coverage: aim to
screen roughly 50-60+ candidate papers and include or save a bibliography or
candidate table at that scale when feasible. A concise narrative may still
highlight a smaller key-paper set.

Do not reintroduce heavy structured payloads like:

- `constraints`
- `open_questions`
- `decisions`
- `next_steps`
- duplicated artifact inventories

Proposal review is the exception: proposal checkpoints still use a structured `ProposalCheckpoint` schema because the human-interrupt flow depends on it.

## 6) Skills

### Skill roots

The active skill layout is:

- `skills/materials_worker/*`
- `skills/dynamics_worker/*`
- `skills/ml_worker/*`
- `skills/orca_xtb_worker/*`
- `skills/research_specialist/*`
- `skills/litreview_agent/*`
- `skills/writing_specialist/*`
- `skills/execution/*` for shared managed-execution support.

At runtime they are staged into:

- `files/.deepagents/skills/materials_worker`
- `files/.deepagents/skills/dynamics_worker`
- `files/.deepagents/skills/ml_worker`
- `files/.deepagents/skills/orca_xtb_worker`
- `files/.deepagents/skills/research_specialist`
- `files/.deepagents/skills/litreview_agent`
- `files/.deepagents/skills/writing_specialist`
- `files/.deepagents/skills/execution`

Worker agents and entrypoint specialists bind skills via DeepAgent `skills=[...]`, not through the old custom skill runtime.

### Skill authoring

Skills must follow the DeepAgent-style layout:

- one directory per skill
- a `SKILL.md` file
- YAML frontmatter at the top

Recommended frontmatter fields:

- `name`
- `description`
- `license`
- `compatibility`
- `allowed-tools`

`allowed-tools` is a skill-level recommendation list. It does not define the runtime-visible tool surface. Visible tools are determined by the specialist/subagent `tools=[...]` binding.

When a tool surface changes, update the relevant skills in the same change.
Do not hard-code concrete tool-name routing into generic specialist prompts. Managed execution prompts should stay abstract and worker-scoped; put concrete remote/managed tool choices in skills, tool schemas/descriptions, allowlists, and registry wiring. Coordinator specialists should know that worker-owned managed execution exists and should delegate to the worker, but they should not receive detailed remote tool-selection contracts for tools they cannot call.

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
- do not reopen broad research loops from writing
- delegate substantive drafting and revision to `writing_worker_agent`
- if producing a TeX bundle, the worker should run compile directly and use diagnostics to repair bundle issues

Do not assume there is still a separate active `compile_agent` in the writing lane. Match the current implementation in `catmaster/specialists/runtime.py`.

## 9) Usage, logging, and UI

### Usage

Usage accounting is based on LangChain `UsageMetadataCallbackHandler`, not the old `LLM_USAGE` trace event.

Run usage is written to `usage_summary.json`, including per-role breakdowns when available.

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
