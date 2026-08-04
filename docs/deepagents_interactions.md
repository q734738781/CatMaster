# DeepAgents integration reference

> This is a maintainer reference for the active runtime. For user-facing
> capabilities and workflows, start with the
> [CatMaster user manual](user-guide/README.en.md).

CatMaster uses DeepAgents and LangGraph for specialist execution while retaining
ownership of workspace files, artifacts, run state, model configuration, and
WebUI events. This document defines the current integration boundaries.

## Runtime ownership

`catmaster/specialists/runtime.py` builds lane specialists and workers with
`create_deep_agent(...)`.

Each active agent receives:

- a model selected through `catmaster/llm/factory.py`;
- tools and skills for its current role;
- a workspace filesystem backend;
- workspace SQLite checkpoint and store objects;
- a stable `thread_id` derived from the WebUI chat session;
- CatMaster middleware for bounded tool-error recovery.

DeepAgents checkpoints live in:

- `metadata/deepagent_threads.sqlite`
- `metadata/deepagent_memory.sqlite`

The WebUI reads the current run snapshot from
`metadata/runs/<run_id>/run_state.json`. Checkpoints provide conversation
continuity and are not the user-facing run-status format.

When WebUI steering arrives during a run, CatMaster uses LangGraph's runtime
static breakpoint after the top-level `tools` node. The active tool is allowed
to finish, its result is checkpointed, and the queued user message then starts
from that same thread checkpoint. Runs with no further tool boundary finish
normally before the queued message starts. This is a local OSS runtime
integration; it does not depend on LangSmith Deployment double-texting APIs.

## Model calls and delegation

Model request settings come from the selected CatMaster profile. The factory
passes `timeout_s` and an explicitly configured `max_retries` to the provider
integration. Codex OAuth templates leave `max_retries` unset and use the pinned
OpenAI SDK default for transport, rate-limit, and HTTP server errors. A narrow
model middleware handles transient overloads that arrive inside an already
accepted HTTP 200 stream, including the structured `server_is_overloaded` code
and the provider's canonical retry-later or request-ID messages. It is attached
through the DeepAgents `openai-codex` provider profile, so the same behavior
applies to specialists, named workers, declarative subagents, and CatMaster's
explicit `general-purpose` child. The specialist runner only retries a completed
episode when its final report cannot be parsed.

CatMaster supplies one explicit subagent named `general-purpose`, which replaces
the auto-added DeepAgents child for every specialist and named worker. It is a
bounded context worker, not a coordinator: the child completes one self-contained
task brief and cannot delegate further. DeepAgents supplies the current model,
processed direct tools, permissions, interrupt policy, and its standard child
middleware. CatMaster explicitly passes the caller's staged skill roots and adds
bounded document access plus nonfatal tool-error handling. The child does not
receive the caller's full specialist prompt or persistent memory.

## CatMaster middleware

`SpecialistRunner._build_default_middleware()` installs
`catmaster_nonfatal_tool_errors`.

When a bound tool raises an exception, this middleware returns a typed error
`ToolMessage` with a compact artifact. The current agent can inspect the error
and decide whether to correct its inputs, choose another method, or report the
blocker. Model request retry remains separate and provider-owned.

The supporting functions are:

- `_nonfatal_tool_error_result` in `catmaster/specialists/runtime.py`
- `tool_error_to_message` in `catmaster/runtime/tool_output_adapter.py`

## Multimodal content

Current-turn attachments use LangChain content blocks when the selected model
supports the media type. CatMaster stores uploaded binaries as workspace
artifacts and keeps raw base64 out of durable WebUI thread messages. DeepAgents
`read_file` image results remain multimodal for the next model call.

PDF and modern Office documents use bounded document readers. Provider-specific
conversion happens at the model boundary, not inside scientific tools.

See [Multimodal files](deepagents_multimodal.md) for the supported user flow,
persistence rules, and provider behavior.

## Codex OAuth `apply_patch`

Codex OAuth roles receive a LangChain custom tool named `apply_patch`. It accepts
the freeform V4A patch envelope and can add, update, move, or delete several
files in one call.

Execution is restricted to the current project `files/` root. The implementation
uses workspace locking, atomic replacement for individual files, path traversal
checks, symlink checks, and model-visible conflict errors.

The model emits a `custom_tool_call`. DeepAgents executes it through the normal
tool scheduler and returns `custom_tool_call_output` on the next model call.
For LangChain v3 event streaming, CatMaster restores missing scheduler metadata
from the completed provider block at the Codex OAuth model-result boundary. The
original block is retained unchanged for Responses API replay, and recovery is
skipped when LangChain already supplied the tool call.
`/memories` remains a routed DeepAgents store, so persistent memory edits use
`edit_file` rather than this workspace patch tool.

The live acceptance script is:

```bash
PYTHONPATH=. \
  /home/chenhh/miniconda3/envs/catmaster-dev/bin/python \
  tests/manual/codex_oauth_apply_patch_live.py --workers 3
```

## OpenRouter content conversion

`CatMasterChatOpenRouter._create_message_dicts(...)` converts standard
LangChain media blocks into the shapes accepted by OpenRouter:

- image base64 blocks become `image_url` data URLs;
- file base64 blocks become OpenRouter `file` blocks with `file_data`;
- existing provider-native `image_url` and `file` blocks pass through.

Unsupported replayed blocks become explicit text references before SDK
validation. The conversion is scoped to CatMaster's OpenRouter subclass and
does not modify tool results or patch installed packages.

The relevant functions are in `catmaster/llm/factory.py`:

- `_sanitize_openrouter_message_dicts`
- `_sanitize_openrouter_content`
- `_openrouter_data_url`
- `_attach_openrouter_cache_control`

## Dependency and verification contract

`requirements/pc-conda.yml` is the source of truth for DeepAgents, LangChain,
OpenRouter, and OpenAI package versions.

After changing one of these dependencies, run:

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python -m pytest \
  tests/test_native_apply_patch.py \
  tests/test_openrouter_message_sanitizer.py \
  tests/test_specialist_runtime.py \
  tests/test_llm_factory_extra_body.py -q
```

Also verify:

- image and PDF handling in a fresh thread;
- replay through the same `thread_id`;
- OpenRouter serialization after a media-bearing message;
- context compaction after media-bearing tool results;
- one Codex OAuth patch call when that provider is enabled.

Notable changes to these contracts belong in the repository
[Changelog](../CHANGELOG.md).
