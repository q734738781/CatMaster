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

`SpecialistRunner._build_default_middleware()` currently injects three custom
middleware objects.

### `catmaster_retry_semantic_model_failures`

Purpose:

- Retry model responses that are syntactically accepted but unusable.
- Retry provider/schema exceptions that commonly come from OpenRouter transport
  or SDK validation instability.
- Sanitize old replayed `ToolMessage` history before model invocation.

Related functions:

- `_validate_model_response_for_retry`
- `_validate_ai_message_for_retry`
- `_is_retryable_model_exception`
- `_sanitize_model_request_for_history`

Keep this layer while using OpenRouter and long-lived DeepAgent threads. It is
the final defense for existing checkpoints that may already contain bad message
blocks.

Possible future removal:

- The retry logic can be narrowed if provider errors become rare and covered by
  the model client's own retry policy.
- The request sanitizer can be removed only after old thread stores have been
  migrated or discarded and upstream no longer writes replay-unsafe multimodal
  tool content.

### `catmaster_textualize_multimodal_tool_results`

Purpose:

- Convert non-text `ToolMessage.content` into a compact textual placeholder
  before it enters long-lived agent history.
- Preserve useful continuity data such as tool name, file path, MIME type, and
  block id.
- Avoid persisting base64 images/PDF pages into `deepagent_threads.sqlite`.

Related functions:

- `_sanitize_tool_result_for_history`
- `_sanitize_tool_message_for_history`
- `_tool_content_needs_textualization`
- `_textualized_tool_content`
- `_multimodal_tool_block_reference`

This exists because DeepAgents' built-in `read_file` can return messages like:

```json
{"type": "image", "base64": "...", "mime_type": "image/png"}
```

Those blocks are useful for the current model call, but they are not safe as
long-term `tool` history across provider bridges. OpenRouter's SDK expects
provider-native blocks such as `image_url`, and some paths reject multimodal
`tool.content` entirely.

Keep this layer. It is the preferred durable fix because it prevents new
checkpoint pollution.

Possible future removal:

- Remove only if DeepAgents changes `read_file`/filesystem middleware to avoid
  inline binary/multimodal tool history, or if CatMaster stops using long-lived
  DeepAgent thread persistence.

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
- `_unsupported_openrouter_block_text`
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
some provider paths reject multimodal content on `tool` messages entirely. The
factory sanitizer converts unsupported replayed blocks into textual
placeholders before the SDK validates the request. For `role=tool`, it is more
strict: any non-text block is textualized, even if the block type is otherwise
valid for user chat input.

Keep this sanitizer while old checkpoints exist and OpenRouter is used. It is
intentionally scoped to CatMaster's `ChatOpenRouter` subclass at
`CatMasterChatOpenRouter._create_message_dicts(...)`.

Possible future removal:

- Remove after old checkpoints are migrated or discarded and the runtime-level
  tool textualizer has been deployed long enough that no new polluted
  checkpoints are produced.
- Do not re-add a global `langchain_openrouter` monkey patch unless a focused
  test proves the upstream wrapper regressed.

## Dependency Expectations

Current versions are pinned in `requirements/pc.txt`:

- `langchain-openrouter==0.2.5`
- `openrouter==0.11.1`
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
- Textualizing non-text tool history before persistence/model replay.
- OpenRouter request-boundary sanitation for legacy checkpoints.

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
