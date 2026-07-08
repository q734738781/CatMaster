# DeepAgents Interaction Notes

This note documents CatMaster's non-standard interaction points with DeepAgents,
LangChain, and OpenRouter. The design rule is to keep these interactions small:
prefer DeepAgent-native behavior, modify only CatMaster-owned runtime
boundaries, and remove local code once upstream behavior makes it redundant.

## Current Active Path

The active path is `catmaster/specialists/runtime.py`.

CatMaster creates DeepAgents with:

- `create_deep_agent(...)` for the lane specialist and worker agents.
- LangGraph checkpoint/store objects backed by workspace SQLite files.
- A stable `thread_id` derived from the WebUI chat session.
- CatMaster middleware inserted into every specialist and worker agent.

The SQLite files are:

- `metadata/deepagent_threads.sqlite`
- `metadata/deepagent_memory.sqlite`

The WebUI-facing run snapshot remains `metadata/runs/<run_id>/run_state.json`.
Do not use DeepAgent checkpoints as the user-visible run status source.

## Middleware Added By CatMaster

`SpecialistRunner._build_default_middleware()` currently injects two custom
middleware objects.

### `catmaster_retry_semantic_model_failures`

Purpose:

- Retry model responses that are syntactically accepted but unusable.
- Retry provider/schema exceptions that commonly come from OpenRouter transport
  or SDK validation instability.

Related functions:

- `_validate_model_response_for_retry`
- `_validate_ai_message_for_retry`
- `_is_retryable_model_exception`

Keep this layer while using OpenRouter and long-lived DeepAgent threads. It is
the final defense for transient provider/schema failures and unusable assistant
outputs. It must not rewrite multimodal `ToolMessage` content; provider-specific
compatibility handling belongs at the model adapter boundary.

Possible future removal:

- The retry logic can be narrowed if provider errors become rare and covered by
  the model client's own retry policy.

### Removed: `catmaster_textualize_multimodal_tool_results`

CatMaster previously inserted a tool-result middleware named
`catmaster_textualize_multimodal_tool_results`. That layer has been removed.
It erased the DeepAgents-native multimodal path by converting fresh
`read_file` image/PDF/file content blocks into text before the next model call
could consume them.

DeepAgents' built-in `read_file` can return messages like:

```json
{"type": "image", "base64": "...", "mime_type": "image/png"}
```

Those blocks are the desired active-path representation. They should remain
available to the next model call whenever the selected provider/model supports
the media type. Do not reintroduce a blanket tool-output textualizer.

Durable thread history must still avoid raw user-upload base64. The WebUI stores
uploaded binaries as artifacts and only passes inline content blocks in the
current turn. DeepAgents `read_file` tool outputs are owned by the DeepAgents
runtime; if provider replay compatibility is needed, handle it at the provider
adapter boundary or through an explicit legacy-thread migration, not by mutating
fresh tool results globally.

### `catmaster_nonfatal_tool_errors`

Purpose:

- Convert tool exceptions into model-visible error `ToolMessage`s instead of
  killing the whole agent run.
- Preserve a compact artifact for WebUI/tool trace inspection.

Related functions:

- `_nonfatal_tool_error_result`
- `tool_error_to_message` in `catmaster/runtime/tool_output_adapter.py`

Keep this layer. The specialist runtime depends on bounded recovery from tool
failures.

Possible future removal:

- Only remove if every tool path is guaranteed to return explicit error
  messages and DeepAgents provides equivalent non-fatal behavior for tool
  exceptions.

## OpenRouter Boundary Sanitizer

OpenRouter-specific message normalization lives in `catmaster/llm/factory.py`.

Related functions:

- `_sanitize_openrouter_message_dicts`
- `_sanitize_openrouter_content`
- `_openrouter_data_url`
- `_attach_openrouter_cache_control`

This is not a provider-wide shim. CatMaster now depends on
`langchain-openrouter>=0.2.1` and `openrouter>=0.9.1`; that stack natively
wraps file content messages with the current SDK `Chat*Message` classes and
preserves `role`. CatMaster does not monkey-patch
`langchain_openrouter.chat_models._wrap_messages_for_sdk`.

Older checkpoints may contain LangChain-style blocks such as:

```json
{"type": "image", "id": "...", "mime_type": "image/jpeg"}
```

The OpenRouter SDK expects provider-native chat blocks such as `image_url`, and
PDF/file inputs use OpenRouter's `file` block shape. The factory sanitizer
therefore converts CatMaster/LangChain standard content blocks into native
OpenRouter blocks at request time:

- `{"type": "image", "base64": "...", "mime_type": "..."}`
  becomes an `image_url` data URL block.
- `{"type": "file", "base64": "...", "mime_type": "...", "filename": "..."}`
  becomes an OpenRouter `file` block with `file_data`.
- Existing provider-native `image_url` and `file` blocks pass through unchanged.

Only genuinely unsupported replayed blocks are degraded to a textual placeholder
before SDK validation.

Keep this sanitizer while old checkpoints exist and OpenRouter is used. It is
intentionally scoped to CatMaster's `ChatOpenRouter` subclass at
`CatMasterChatOpenRouter._create_message_dicts(...)`.

Possible future removal:

- Remove after old checkpoints are migrated or discarded and OpenRouter/LangChain
  accept the standard content block shapes CatMaster emits without local
  conversion.
- Do not re-add a global `langchain_openrouter` monkey patch unless a focused
  test proves the upstream wrapper regressed.

## Dependency Expectations

Current versions are pinned in `requirements/pc-conda.yml`:

- `langchain-openrouter==0.2.5`
- `openrouter==0.11.1`
- `langchain-deepseek==1.1.0`
- `deepagents==0.6.12`

DeepAgents issue
`langchain-ai/deepagents#2873` tracks a related multimodal summarization problem:
image blocks can be mishandled when conversation history is summarized/offloaded.

Do not assume an upstream DeepAgents upgrade fixes CatMaster's problem without
testing these cases:

- `read_file` on an image/PDF page
- multi-turn replay through the same `thread_id`
- OpenRouter model call after the image-bearing tool message exists
- summary/compaction after image-bearing tool messages exist

## Interaction Budget

Interactions that are currently justified:

- Non-fatal tool error conversion.
- Model retry for empty/invalid assistant output.
- Provider/schema retry for OpenRouter validation/EOF failures.
- WebUI upload artifact storage plus current-turn content block injection.
- OpenRouter request-boundary conversion for standard image/file blocks and
  legacy checkpoint compatibility.

Interactions that should not be added:

- Manual chat transcript reconstruction into specialist prompts.
- New memory-patch flows for active specialists.
- Provider-specific message rewriting inside tools.
- Inline base64 artifacts in model-visible durable history.
- Extra summarization or compaction layers unless a measured context limit
  problem requires them.

Interactions that are candidates for removal later:

- OpenRouter replay sanitizer, after old checkpoints are migrated or discarded.
- Provider/schema exception retry breadth, if OpenRouter client retries and
  schema handling stabilize.
- Any local patch to installed DeepAgents packages, once equivalent constructor
  options or upstream defaults exist.

Interaction already removed:

- Global OpenRouter file-wrapper monkey patch. Current
  `langchain-openrouter>=0.2.1` / `openrouter>=0.9.1` handles file blocks with
  the new SDK message classes, so CatMaster no longer replaces
  `_wrap_messages_for_sdk`.
- Runtime `ToolMessage` multimodal textualization. Fresh DeepAgents `read_file`
  media blocks now remain multimodal through the next model call.

## Maintenance Checklist

When changing DeepAgents, LangChain, or OpenRouter versions:

1. Run the focused compatibility tests:

   ```bash
   pytest tests/test_openrouter_message_sanitizer.py tests/test_specialist_runtime.py -q
   ```

2. Run the OpenRouter factory tests:

   ```bash
   pytest tests/test_llm_factory_raw_http_logging.py tests/test_llm_config_prompt_cache_retention.py -q
   ```

3. Manually check one replay scenario with a workspace that has existing
   `deepagent_threads.sqlite` history containing image/PDF `read_file` calls.

4. If failures mention `Unmarshaller`, `tool.content`, `image_url`, or
   `content.str`, inspect the checkpoint history before changing tool logic.
   The failure is often replay-schema incompatibility, not the scientific tool
   result itself.
