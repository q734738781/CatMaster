# CatMaster Runtime Guide

This repository now uses a DeepAgent-native specialist runtime as the active path. Treat the current implementation under `catmaster/specialists/` as the source of truth.

Legacy note:

- Older orchestrator / director-style code under `catmaster/agents/` is deprecated but still present in the repo.
- Unless you are explicitly maintaining that legacy stack, do not use it as the source of truth for current prompts, routing, tool policy, or runtime behavior.
- When active-path behavior and legacy code disagree, follow `catmaster/specialists/runtime.py` and the active specialist tests.

## 1) Active architecture

### Public entrypoints

The active runtime exposes exactly three lanes:

- `research`
- `experiment`
- `writing`

They are built in [runtime.py](/home/chenhh/python_projects/CatMaster/catmaster/specialists/runtime.py).

### Specialist roles

- `ResearchSpecialist`
  - The only orchestration-capable specialist.
  - Can delegate to `experiment_specialist`, `writing_specialist`, and `litreview_agent`.
- `ExperimentSpecialist`
  - Bounded execution specialist.
  - May delegate only to `materials_worker`, `ml_worker`, and a lightweight `literature_agent`.
- `WritingSpecialist`
  - Writing coordinator, not a free-form research agent.
  - Should delegate substantive drafting to `writing_worker_agent`.
  - May use the lightweight `literature_agent` only for tightly bounded background supplementation needed by the current manuscript or note.

Do not design new features around the old `director -> task runner -> memory patch` stack unless you are explicitly maintaining legacy code. The active path is the DeepAgent specialist runtime.

## 2) Current literature architecture

Literature support is split into two forms:

- `litreview_agent`
  - Used by `research`
  - A compiled DeepAgent subagent
  - Internally delegates to:
    - `literature_agent` for broad Tavily-backed public-web review
    - `metadata_agent` for exact scholarly metadata
- lightweight `literature_agent`
  - Used by `experiment`
  - Also used by `writing` for narrow background/context supplementation only
  - Exposes only a lightweight Tavily-backed `internet_search` tool for quick external grounding

The metadata side is intentionally separate from broad review:

- `search_openalex`
- `search_semantic_scholar`
- `get_openalex_record`
- `get_semantic_scholar_record`
- `recommend_semantic_scholar`

These are for exact paper metadata, DOI/year/venue/authors/citation resolution, and seed-paper expansion. They are not the broad-review layer.

## 3) Runtime state and persistence

### Run state

The active runtime writes `run_state.json` in the run directory. This is the active run snapshot used by the WebUI and resume flow.

Active multi-turn continuity for specialists comes from the DeepAgent `thread_id` bound to the chat session. Do not manually inject chat-history summaries or reconstructed transcript text into active specialist prompts unless you are explicitly maintaining legacy code or a separate non-specialist flow.

Do not add new active-path features that depend on:

- `task_state.json`
- `FINAL_REPORT.md`
- `MEMORY/**`
- `update_memory`
- memory patch flows

Those belong to the legacy orchestration stack and are no longer the primary runtime contract.

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

### Multimodal tool-content rules

Active specialist threads may use multimodal model input in the current turn when the model and provider support it, but do not assume raw multimodal tool outputs are safe to replay indefinitely through every provider bridge or message role.

Maintenance rules for active-path code:

- Keep long-lived `tool` message history text-first. Prefer concise textual tool outputs plus workspace artifact paths over inline base64 image/file payloads.
- If a tool naturally returns binary or multimodal content, persist the artifact as a file and keep only a short textual summary, file path, and MIME hint in model-visible history when possible.
- Do not depend on LangChain-standard multimodal blocks like `{"type": "image", ...}` being replay-safe for provider-backed `tool` history. Provider wire formats may require provider-native blocks such as OpenAI-style `image_url`, and some bridges reject non-text `tool.content`.
- When a later turn needs image or file understanding, re-open the artifact in that turn or use a dedicated analysis tool to turn it into textual findings. Do not rely on old inline image tool payloads surviving indefinitely inside the thread store.
- Prefer attachments, file IDs, file paths, or generated artifact URLs over embedding large base64 blobs in multi-turn history.
- If runtime compatibility is uncertain, sanitize non-text `tool` messages before model invocation instead of dropping the whole thread. Preserve only the minimum textual placeholder needed for continuity.

This is a compatibility rule for the active DeepAgent stack, not a ban on multimodal prompting. The goal is to keep `thread_id`-based continuity while avoiding provider/schema failures from replaying raw multimodal `tool` history across turns.

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

## 5) Final output contract

The active specialist output contract is intentionally soft and compact.

Final specialist responses should contain only these sections:

- `Summary`
- `Facts`
- `Files`

Do not reintroduce heavy structured task-era payloads like:

- `constraints`
- `open_questions`
- `decisions`
- `next_steps`
- duplicated artifact inventories

Proposal review is the exception: proposal checkpoints still use a structured `ProposalCheckpoint` schema because the human-interrupt flow depends on it.

## 6) Skills

### Skill roots

The active skill layout is:

- `skills/experiment/*`
- `skills/machine_learning/*`
- `skills/writing/*`

At runtime they are staged into:

- `files/.deepagents/skills/experiment`
- `files/.deepagents/skills/machine_learning`
- `files/.deepagents/skills/writing`

Specialists bind skills via DeepAgent `skills=[...]`, not through the old custom skill runtime.

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

## 7) Tool conventions

### Registration and returns

All agent-visible tools should be registered in `catmaster/tools/registry.py` and exposed as LangChain tools.

Tool functions should:

- validate inputs with a Pydantic model
- return LangChain-native `(content, artifact)`
- raise `CatMasterToolExecutionError` for business failures with a model-visible `public_message`
- keep both the registered input-model docstring and the tool function docstring tagged on the first line with a short `[scope/stage]` label
- use short, stable labels such as `[structure/discovery]`, `[bulk/modeling]`, `[defect/modeling]`, `[electronic/prepare]`, `[phonon/modeling]`, `[vasp/prepare]`, `[vasp/analysis]`, `[md/analysis]`, `[mace/execute]`, `[ml/prepare]`, `[ml/analysis]`, `[structure/viz]`, `[thermo/gas]`, or `[literature/metadata]`

Because the active tool registry uses the input model docstring as the tool description, keep the first-line tag and summary synchronized between the input model docstring and the tool function docstring.

### Error behavior in the active path

In the active specialist runtime, tools are wrapped as non-fatal. Tool exceptions are converted into model-visible error messages instead of directly killing the lane.

This applies to both:

- top-level specialists
- subagents

Do not design active-path workflows that depend on tool exceptions terminating the run.

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

The WebUI is run-state and SSE driven. Active-path UI work should align with the current streaming model rather than polling-only or legacy task-state assumptions.

## 10) Legacy code policy

This repository still contains legacy orchestration code under areas like `catmaster/agents/`, but it is not the primary architecture.

When making changes:

- prefer `catmaster/specialists/` for active-path behavior
- prefer DeepAgent-native mechanisms over custom compatibility layers
- do not add new task-era compatibility shims unless a change explicitly targets legacy support

## 11) Testing guidance

When active-path behavior changes, prefer updating tests around:

- `tests/test_specialist_runtime.py`
- WebUI live-state tests
- tool-specific tests
- usage summary tests

If a tool or prompt change affects skill-guided behavior, update the corresponding skill files too.
