# DeepAgents Multimodal Migration Status

Last reviewed: 2026-07-07

This document records the migration from CatMaster's old path-only multimodal
handling to DeepAgents-native multimodal behavior. The goal is not to preserve
the old workaround layer. The active agent path should follow the current
DeepAgents/LangChain content-block model while keeping CatMaster's artifact
persistence and replay safety.

## Source Basis

Local package versions checked in the `catmaster` environment:

- `deepagents==0.6.12`
- `langchain==1.3.11`
- `langchain-core==1.4.8`
- `langchain-openai==1.3.3`
- `langchain-openrouter==0.2.5`
- `langchain-anthropic==1.4.8`
- `openai==2.44.0`
- WebUI `@assistant-ui/react==0.14.24`

Official/current behavior used as the target:

- DeepAgents supports multimodal user messages, multimodal `read_file`, and
  multimodal custom tool outputs when the selected model supports them.
- DeepAgents recommends storing large media in a backend and passing paths or
  URLs for long-running histories.
- LangChain messages support provider-native content blocks and standard
  `content_blocks`; initializing with `content_blocks` also populates `content`.
- OpenAI Responses uses `input_image` for image inputs and `input_file` for
  file/PDF inputs; PDF `detail` controls visual page processing.
- assistant-ui simple adapters cover images/text, and custom attachment
  adapters are required for broader file handling.

Reference URLs:

- https://docs.langchain.com/oss/python/deepagents/multimodal
- https://docs.langchain.com/oss/python/deepagents/backends
- https://docs.langchain.com/oss/python/langchain/messages
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/guides/file-inputs
- https://www.assistant-ui.com/docs/guides/attachments

## Implemented CatMaster State

Active-path implementation is now content-block-first for supported current-turn
attachments and DeepAgents `read_file` tool results.

- Frontend:
  - `catmaster/webui/frontend/src/v2/useCatMasterThreadRuntime.js` uses
    `CatMasterAttachmentAdapter`, not the previous image/text-only adapters.
  - `catmaster/webui/frontend/src/v2/messageAdapters.js` serializes filename,
    MIME type, size, data URL, and text metadata into
    `/api/threads/{thread_id}/submit`.
- WebUI backend:
  - `catmaster/runtime/multimodal_blocks.py` owns MIME/kind detection,
    capability checks, attachment summaries, and standard content-block
    construction.
  - `ThreadAgentLoopService.prepare_submit_attachments()` stores attachments
    under `files/attachments/<thread_id>/...`, registers artifacts, and builds
    current-turn blocks when the selected model capability allows it.
  - Persisted user messages contain artifact parts and sidecar metadata only;
    raw data URLs/base64 are not stored in thread messages or monitor events.
- DeepAgent invocation:
  - `StreamingSpecialistRunner.arun_turn()` accepts a `content` list and sends
    `{"role": "user", "content": content_blocks_or_text}` to DeepAgents.
  - The text-only prompt is still used for run state, UI preview, and research
    goal metadata.
- Provider boundary:
  - OpenRouter conversion remains scoped to
    `CatMasterChatOpenRouter._create_message_dicts(...)`.
  - Standard image base64 blocks convert to `image_url` data URLs.
  - Standard file/PDF base64 blocks convert to OpenRouter `file.file_data`.
- Sanitization:
  - `catmaster_textualize_multimodal_tool_results` is removed.
  - `_sanitize_model_request_for_history()` is no longer invoked.
  - Fresh multimodal `ToolMessage` content survives to the next model request.

Remaining deliberate limits:

- Audio/video current-turn attachments are supported by the block builder but
  disabled by default until provider/model serialization is verified per config.
- `analyze_images.py` remains a deprecated compatibility module; active guidance
  now points to DeepAgents `read_file`.
- Legacy checkpoints with provider-incompatible media should be handled through
  provider-boundary fallback or explicit migration, not by reintroducing blanket
  runtime textualization.

## Implementation Status

- [x] Current-turn image attachments reach DeepAgents as image content blocks.
- [x] Current-turn PDF attachments reach DeepAgents as file content blocks.
- [x] User-uploaded binaries are stored as artifacts before model invocation.
- [x] Persisted WebUI thread messages avoid raw media base64/data URLs.
- [x] `multimodal.prepared` monitor events expose path/MIME/sent status without
  base64.
- [x] DeepAgents `read_file` image/PDF results remain multimodal in runtime
  middleware.
- [x] OpenRouter adapter converts standard image/file blocks at the provider
  boundary instead of erasing fresh tool output.
- [x] Runtime prompt guidance no longer requires `general-purpose` delegation
  for multimodal analysis.
- [ ] Add explicit audio/video provider serialization tests before enabling
  those modalities by default.
- [ ] Decide whether to remove `analyze_images.py` completely or keep it as a
  declared fallback for non-multimodal providers.
- [ ] Add a legacy checkpoint migration/inspection utility if real old thread
  stores contain incompatible media blocks.

## Target Architecture

The target behavior is content-block-first for the current turn and
reference-first for persistent history.

1. The user submits text plus attachments.
2. CatMaster stores each binary attachment as a workspace artifact.
3. CatMaster builds current-turn content blocks from the stored artifact when
   the model/provider supports that modality.
4. The DeepAgent receives one user message whose `content` is a list of
   LangChain/DeepAgents standard content blocks.
5. Persistent CatMaster thread messages keep artifact parts, file paths, MIME,
   size, and preview metadata, not large inline base64.
6. DeepAgents built-in `read_file` may return multimodal content blocks. Those
   blocks must be visible to the next model call at least within the current
   active context.
7. Long-lived history safety is handled by a boundary-aware compaction/replay
   policy, not by immediate tool-result textualization.

## Non-Negotiable Design Rules

- Do not preserve the current blanket `catmaster_textualize_multimodal_tool_results`
  behavior.
- Do not solve multimodality by asking the model to infer from file paths only.
- Do not write base64 blobs into `messages.jsonl` as normal long-term thread
  state.
- Do not make PDF/image understanding depend on hidden specialized tools when
  DeepAgents-native `read_file` and current-turn content blocks can carry the
  modality.
- Do not add provider-specific payload construction before checking whether
  LangChain content blocks already serialize correctly for the provider.
- Keep old provider failure protection only as a fallback around replayed or
  stale messages, not around fresh current-turn or fresh `read_file` outputs.

## Detailed Audit Checklist

The implementation status above is authoritative for the active code path. The
detailed checklist below is kept as an audit matrix for follow-up hardening,
provider-specific expansion, and test coverage.

### Phase 1 - Contracts And Data Model

- [ ] Add a typed internal attachment model, for example
  `PreparedAttachment`, with:
  - [ ] `artifact_id`
  - [ ] `workspace_path`
  - [ ] `filename`
  - [ ] `mime_type`
  - [ ] `size_bytes`
  - [ ] `kind` (`image`, `pdf`, `text`, `document`, `audio`, `video`,
        `unsupported`)
  - [ ] `current_turn_block`
  - [ ] `history_part`
  - [ ] `warnings`
- [ ] Replace `prepare_submit_attachments()` return type with a structured
  result instead of `(parts, prompt_suffix)`.
- [ ] Add explicit size caps:
  - [ ] hard upload cap remains enforced before write
  - [ ] current-turn inline block cap for base64 media
  - [ ] provider-specific warning when an attachment is stored but not sent as a
        current-turn block
- [ ] Keep `ArtifactPart` for WebUI rendering, but add enough metadata to
  reconstruct current-turn blocks from the saved file.
- [ ] Add `ThreadSubmitRequest.attachments` validation/schema tests for images,
  PDFs, text, and unsupported files.

### Phase 2 - Frontend Attachment Surface

- [ ] Keep assistant-ui as the frontend attachment primitive.
- [ ] Replace the current adapter setup with a dedicated CatMaster composite:
  - [ ] image adapter
  - [ ] text adapter
  - [ ] PDF/document adapter
  - [ ] unsupported-file adapter that still submits metadata and lets backend
        store the file when possible
- [ ] Preserve client-side previews:
  - [ ] thumbnails for images
  - [ ] filename/MIME/size rows for PDF/documents
  - [ ] text preview for small text files
- [ ] Ensure `requestFromAssistantAppend()` emits stable fields:
  - [ ] `type`
  - [ ] `filename`
  - [ ] `mime_type`
  - [ ] `data` or `text`
  - [ ] `size_bytes` when available
  - [ ] no raw assistant-ui implementation object required by backend logic
- [ ] Add frontend tests for image, text, PDF, and unsupported attachment
  serialization.

### Phase 3 - Current-Turn DeepAgent Input

- [ ] Change `ThreadAgentLoopService.submit()` to pass structured multimodal
  turn input to `launch_turn()`.
- [ ] Change `launch_turn()` to pass `content_blocks` or `content` list through
  to `StreamingSpecialistRunner.arun_turn()`.
- [ ] Change `StreamingSpecialistRunner.arun_turn()` from:
  - [ ] old: `{"messages": [{"role": "user", "content": str(prompt)}]}`
  - [ ] new: `{"messages": [{"role": "user", "content": content_blocks_or_text}]}`
- [ ] Keep `user_prompt` as a text-only field for run state, research goal
  seeding, logs, and UI preview.
- [ ] When attachments are present, include a text block that lists the stored
  paths and states whether each file was also sent as a multimodal block.
- [ ] Add tests proving the current agent input contains:
  - [ ] text block
  - [ ] image block for an image attachment
  - [ ] file block for a PDF attachment
  - [ ] no raw data URL in persisted message JSON

### Phase 4 - Content Block Normalization

- [ ] Implement one normalization module, for example
  `catmaster/runtime/multimodal_blocks.py`.
- [ ] Convert stored attachments to LangChain/DeepAgents standard blocks first:
  - [ ] text: `{"type": "text", "text": "..."}`
  - [ ] image by URL/path when possible: `{"type": "image", "url": "..."}`
  - [ ] image by base64 when necessary:
        `{"type": "image", "base64": "...", "mime_type": "..."}`
  - [ ] file/PDF:
        `{"type": "file", "source_type": "base64", "mime_type": "...", "data": "...", "filename": "..."}`
- [ ] Keep provider-native conversion in the model adapter only when required:
  - [ ] OpenAI Responses should be allowed to map via LangChain to
        `input_image`/`input_file`.
  - [ ] OpenRouter should be allowed to map via LangChain/OpenRouter serializer.
  - [ ] Anthropic should use LangChain standard/provider conversion.
- [ ] Add local serialization tests using installed package internals:
  - [ ] `ChatOpenAI(..., use_responses_api=True)` maps image blocks to
        `input_image`.
  - [ ] `ChatOpenAI(..., use_responses_api=True)` maps PDF/file blocks to
        `input_file`.
  - [ ] `ChatOpenRouter` preserves or converts image/file blocks into accepted
        chat content.
- [ ] Add a model-capability check helper that can conservatively decide whether
  to send image/PDF/audio/video blocks for the configured model.

### Phase 5 - Remove The Wrong Sanitizer

- [x] Delete or disable `catmaster_textualize_multimodal_tool_results` from
  `SpecialistRunner._build_default_middleware()`.
- [x] Remove `_sanitize_tool_message_for_history()` from the active runtime path;
  provider-boundary compatibility now handles supported OpenRouter image/file
  conversion without mutating fresh DeepAgent tool results.
  - [x] fresh current-turn tool messages keep multimodal blocks
  - [ ] add a legacy checkpoint migration utility if real old stores require it
  - [x] read-file multimodal blocks retain path/MIME metadata
  - [x] stale or provider-incompatible unknown blocks become explicit text
        references at the provider boundary, not silent omissions
- [x] Ensure `_sanitize_model_request_for_history()` is not invoked in a way that
  removes the immediately preceding `read_file` multimodal result.
- [ ] Add tests that fail under the old sanitizer:
  - [x] `read_file` image ToolMessage reaches the next model request as an image
        block.
  - [x] `read_file` PDF ToolMessage reaches the next model request as a file
        block.
  - [ ] old replayed media beyond the safe window becomes a path/MIME text
        reference.
- [x] Update CatMaster AGENTS guidance to say replay sanitization is
  boundary-aware, not blanket textualization.

### Phase 6 - DeepAgents Built-In `read_file`

- [ ] Trust DeepAgents `read_file` for supported media formats instead of
  custom image-analysis routing.
- [ ] Verify CatMaster's backend/read permissions expose `files/attachments/**`
  to the active DeepAgent filesystem backend.
- [ ] Add integration tests with the local DeepAgents middleware source behavior:
  - [ ] image extension returns `ToolMessage.content_blocks`
  - [ ] PDF extension returns `ToolMessage.content_blocks`
  - [ ] text files still return line-numbered text
- [ ] Remove prompt wording that implies attached images must be inspected via a
  deprecated special tool.
- [ ] Provide concise prompt guidance: if a user references an attached file by
  path, use `read_file(file_path=...)` for actual inspection.

### Phase 7 - Tool Surface Cleanup

- [ ] Decide whether `analyze_images.py` stays:
  - [ ] preferred option: remove it from active documentation and keep only as a
        private compatibility module until no tests import it
  - [ ] fallback option: register it as an explicit fallback tool named
        `analyze_images` only for models/providers where `read_file`
        multimodal blocks are unavailable
- [ ] If keeping the fallback:
  - [ ] register it in `catmaster/tools/analysis/__init__.py`
  - [ ] register it in `ToolRegistry`
  - [ ] add it only to allowlists where it is useful
  - [ ] update its docstring so it no longer claims the built-in path already
        works until the main migration lands
- [ ] Keep `review_pdf_manuscript` and `peer_review_pdf_manuscript`, but
  refactor shared PDF block construction to the common multimodal block module.
- [ ] Keep `generate_nanobanana_figure` as image output/generation; do not mix it
  with image understanding.

### Phase 8 - Provider Compatibility

- [ ] Define a small `ModelMultimodalCapability` structure:
  - [ ] supports images
  - [ ] supports PDFs/files
  - [ ] supports audio
  - [ ] supports video
  - [ ] supports tool-result media
  - [ ] preferred block style
- [ ] Populate capability defaults from provider/model config, with a
  conservative unknown fallback.
- [ ] Add configuration override in `llm.yaml`, for example:
  - [ ] `models.<label>.multimodal.images: true`
  - [ ] `models.<label>.multimodal.pdfs: true`
  - [ ] `models.<label>.multimodal.tool_results: true`
- [ ] OpenAI provider:
  - [ ] use `use_responses_api=True`
  - [ ] prefer LangChain standard blocks and verify serialization
  - [ ] support PDF `detail` when needed
- [ ] OpenRouter provider:
  - [ ] preserve current OpenRouter schema sanitizer only for outbound replayed
        messages that are actually incompatible
  - [ ] verify image and PDF payloads against `langchain-openrouter`
        serialization
- [ ] Anthropic provider:
  - [ ] verify image/PDF block formats through `langchain-anthropic`
  - [ ] do not assume OpenAI `input_*` block names outside OpenAI.
- [ ] If capability is missing, degrade explicitly:
  - [ ] store artifact
  - [ ] tell the model/user that the attachment was not sent visually
  - [ ] route to text extraction or an available fallback tool when possible

### Phase 9 - Persistence And Replay Policy

- [ ] Persist user messages as:
  - [ ] text part
  - [ ] artifact part(s)
  - [ ] structured sidecar with MIME/path/size/capability info
- [ ] Do not persist current-turn base64 in `messages.jsonl`.
- [ ] Store enough information to rebuild current-turn blocks before the agent
  starts, but not after arbitrary history replay.
- [ ] Preserve media paths in run state and artifacts so the agent can call
  `read_file` in later turns.
- [ ] Add a short "media context expired" reference policy for older turns:
  - [ ] keep path/MIME/title
  - [ ] state that the agent should re-open the file with `read_file`
  - [ ] do not pretend the model still sees old image/PDF pixels.

### Phase 10 - UI And Monitor

- [ ] Show user attachments as artifact/file chips in the conversation.
- [ ] Show whether each attachment was:
  - [ ] stored only
  - [ ] sent to model as image/file block
  - [ ] sent as text excerpt
  - [ ] rejected/unsupported
- [ ] Add a Monitor event for multimodal turn preparation:
  - [ ] attachment count
  - [ ] block count
  - [ ] MIME summary
  - [ ] store paths
  - [ ] provider capability outcome
- [ ] Do not expose base64 in Events by default; keep expanded JSON redacted or
  path-only for media payloads.

### Phase 11 - Testing Matrix

- [ ] Unit tests:
  - [ ] attachment normalization
  - [ ] MIME detection
  - [ ] safe filename/path behavior
  - [ ] current-turn block construction
  - [ ] no base64 in persisted thread messages
  - [ ] boundary-aware sanitizer
  - [ ] provider serialization snapshots
- [ ] WebUI tests:
  - [ ] assistant-ui image attachment submission
  - [ ] text attachment submission
  - [ ] PDF attachment submission
  - [ ] unsupported file display/degrade behavior
- [ ] Runtime tests:
  - [ ] `StreamingSpecialistRunner.arun_turn()` receives a list `content`
        payload when attachments exist
  - [ ] DeepAgent `read_file` image/PDF result survives to next model call
  - [ ] older multimodal messages compact to textual references only after the
        defined boundary
- [ ] End-to-end smoke tests:
  - [ ] ask about an attached image
  - [ ] ask about an attached PDF figure
  - [ ] ask the agent to re-open a prior attachment path in a second turn
  - [ ] run with OpenAI Responses provider
  - [ ] run with OpenRouter provider when configured

### Phase 12 - Documentation And Developer Rules

- [ ] Update `catmaster/AGENTS.MD` multimodal section:
  - [ ] current-turn media uses content blocks
  - [ ] long-term history uses artifact references
  - [ ] sanitizer is replay-boundary-only
  - [ ] use `read_file` for later media inspection
- [ ] Update `catmaster/webui/AGENTS.MD`:
  - [ ] assistant-ui attachment adapters are the frontend source of truth
  - [ ] attachment state must expose stored/sent/degraded status
- [ ] Update tool authoring guidance:
  - [ ] text summary plus artifact path for large generated media
  - [ ] custom tools may return multimodal blocks only when the next model call
        is expected to consume them
  - [ ] large binary content should be persisted and referenced
- [ ] Remove or rewrite misleading comments in `analyze_images.py`.

## Suggested Implementation Order

1. Build the block normalization module and tests.
2. Change backend attachment preparation to return structured artifacts plus
   current-turn blocks.
3. Change `launch_turn()` and `arun_turn()` to pass content blocks.
4. Remove the blanket tool-result textualizer and add boundary-aware replay
   sanitization tests.
5. Extend frontend adapters to PDFs/documents and update UI status display.
6. Refactor PDF/image tools to use the shared block module or mark them fallback
   only.
7. Add provider capability config and serialization tests.
8. Update AGENTS/docs and run end-to-end smoke tests.

## Definition Of Done

- [ ] A user can attach an image and ask a question; the first DeepAgent model
  call receives an image block, not just a path.
- [ ] A user can attach a PDF and ask a question; the first DeepAgent model call
  receives a file/PDF block when supported.
- [ ] A later turn can reference the stored attachment path and the agent can
  inspect it with DeepAgents `read_file`.
- [ ] DeepAgents `read_file` image/PDF outputs are not immediately erased by
  CatMaster middleware.
- [ ] Persisted thread messages do not contain raw media base64.
- [ ] Provider-incompatible old media messages degrade to explicit path/MIME
  references.
- [ ] `analyze_images.py` is either restored as a declared fallback or removed
  from active guidance.
- [ ] Tests cover OpenAI Responses serialization, OpenRouter serialization, WebUI
  attachment submission, and sanitizer behavior.

## Open Decisions

- [ ] Whether to support audio/video in the first migration or only images/PDFs.
- [ ] Whether current-turn PDF blocks should default to OpenAI `detail=low` or
  expose `detail=high` through a UI/backend flag for visually dense papers.
- [ ] Whether OpenRouter multimodal should be enabled by default per model label
  or only through explicit `llm.yaml` capability flags.
- [ ] Whether media-heavy inspections should auto-delegate to subagents to keep
  the main research thread compact.
- [ ] Whether to create a dedicated `media_inspector` worker or rely entirely on
  DeepAgents `read_file` inside existing specialists.
