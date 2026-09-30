# Multimodal files

> This document describes the current attachment and file-reading behavior.
> WebUI users can start with
> [Working in the WebUI](user-guide/04-webui.en.md#what-happens-to-attachments).

CatMaster stores every accepted attachment as a workspace artifact before an
agent uses it. The selected model profile determines whether the current turn
also receives a native content block or only a stored file reference.

## What users can provide

The WebUI accepts images, PDFs, modern Office documents, text files, structures,
and other project files.

| File kind | Current-turn behavior | Later access |
|---|---|---|
| Image | Sent as an image content block when the model profile supports images | Open with `read_file` |
| PDF | A compact document may be sent as a native file block; otherwise only its stored path is sent | Open with `read_file`; large files return bounded text with `next_offset` |
| DOCX, XLSX, PPTX | A compact document may be sent as a native file block; otherwise only its stored path is sent | Open with `read_file`; large files return bounded text with `next_offset` |
| Text, Markdown, JSON, CSV, logs, and source files | Sent as a bounded text excerpt when appropriate | Open with `read_file` |
| Structure and scientific data files | Stored as project artifacts and handled by the relevant structure, trajectory, volume, or analysis tools | Open from Files or through the matching scientific tool |
| Audio, video, legacy Office, oversized, or unsupported media | Stored as an artifact; the current model may receive only the path and a warning | Use a supported converter or external workflow |

The current-turn summary tells the agent where each attachment was stored and
whether it was sent to the model or stored only.

## Storage and conversation history

Attachments are stored under:

```text
files/attachments/<thread_id>/
```

The persisted conversation keeps the artifact identity, workspace path,
filename, MIME type, size, representation status, and warnings. Raw media
base64 and data URLs are not written into ordinary thread history or monitor
events.

Native LangGraph checkpoints are separate from these WebUI projections: they
retain the complete message state, including inline images returned by
`read_file`. Specialists and workers use the native incremental message channel,
so intervening checkpoints use saved message writes instead of repeating the
whole image history. Periodic full snapshots and historical full checkpoints
still contain those images. DeepAgents 0.7.11 compaction records a summary and
cutoff for model input while retaining the raw message log. Its media offload does not remove
images from existing checkpoints. Consequently, a long image-heavy conversation
can have compact model input and large checkpoint storage at the same time. See
[local persistence](deepagents_interactions.md#execution-and-storage)
for checkpoint storage and conversation continuity.

## Images in agent replies

An agent can place a workspace image directly in its Markdown answer with a
reference such as:

```markdown
![CO2 conversion and C2+ selectivity](sandbox:/files/figures/co2_hydrogenation.png "Benchmark trend")
```

The stored message keeps this small workspace reference. When the thread is
displayed, the WebUI resolves it to an authenticated, thread-scoped image URL
and serves the current file from that thread's workspace. PNG, JPEG, GIF,
WebP, SVG, and other browser-supported image MIME types use the same route.
The UI does not copy image bytes into Markdown or thread history. Clicking a
workspace image opens the same file in the large workspace preview; a missing or
non-image path becomes a compact failure notice instead of a broken image.

Inline rendering is meant for figures that carry the answer, such as a
structure render, mechanism diagram, or key curve. The durable workspace file
remains the source of truth and is still available from Files or an artifact
card. Scratch images under `files/tmp/` should not be embedded in a final
answer.

This split supports both immediate inspection and long-running project
continuity. Native `read_file` image results remain unchanged in active message
history rather than being replaced by a CatMaster age- or count-based retention
layer. DeepAgents handles older media together with the rest of the conversation
when its normal summarization or context-overflow compaction runs. The original
workspace artifact remains available through `read_file` afterward.

Compaction counts native media using the same model's most recent reported usage
once those files or images have been consumed, then adds later messages and the
current system/tool overhead. This avoids treating Office/PDF base64 as ordinary
text and missing high-detail image usage through a provider-label mismatch. New
unmeasured media still uses upstream estimation; no images, file blocks or history
are removed by this counting adjustment. OpenAI's
[file-input processing](https://developers.openai.com/api/docs/guides/file-inputs)
distinguishes parsed document and spreadsheet input from its base64 transport.

Presentation guidance uses the existing `general-purpose` task delegate for
context-heavy visual review. Checkers read coherent groups of rendered pages in
isolated contexts and return located findings and source paths. The author keeps
editing ownership, assesses the full narrative and resolves particular issues
through direct reading as needed. Revision checks cover changed pages and affected
transitions while reusing accepted findings elsewhere. This is agent guidance;
it does not cap image access, remove message content or change compaction triggers.

## Model capability checks

`ModelMultimodalCapability` controls whether a profile accepts images, PDFs,
documents, audio, video, and multimodal tool results. OpenAI, Codex OAuth, OpenRouter,
Anthropic, Gemini, and generic LangChain profiles start with conservative
image and document support. Audio and video are disabled by default.

Deployments can override these fields in the model profile:

```yaml
provider_options:
  multimodal:
    images: true
    pdfs: true
    documents: true
    audio: false
    video: false
    tool_results: true
    current_turn_inline_limit_bytes: 33554432
```

If a profile does not support a file kind, CatMaster still preserves the file
and reports that it was stored only.

## DeepAgents and provider behavior

The active agent receives one user message containing a text summary plus any
supported current-turn content blocks.

DeepAgents `read_file` returns native image and file content blocks. In 0.7.11,
binary reads load the complete file, ignore `offset` and `limit`, and are not
protected by generic large-result eviction. CatMaster therefore keeps the
native path only for documents that pass a bounded preflight. A large or
text-heavy PDF/Office file is intercepted before the next model call and is
returned through the same `read_file` tool as normalized, line-paginated text.
Continuation uses the stable file path and an integer offset; there is no
parallel `read_document`, opaque cursor, hash identity, or automatic sidecar.

For images, CatMaster leaves the DeepAgents message history unchanged between
compaction events. It does not continually replace the oldest consumed image as
new images arrive. When DeepAgents summarizes an older part of the conversation,
its media-aware history offload stores inline media under
`/conversation_history/media/` and gives the summary a stable reference that can
be reopened with `read_file` if needed.

The compatibility backend also supplies compact Word and Excel bytes
through the native contract because the pinned release's extension table
includes PDF and PowerPoint but omits DOCX and XLSX. Codex OAuth is mapped to
the OpenAI file capability for the same reason.

Provider conversion stays inside the model adapter:

- OpenAI and Codex OAuth use LangChain and Responses API serialization.
- OpenRouter converts standard image and file blocks in
  `CatMasterChatOpenRouter._create_message_dicts(...)`.
- Other providers receive standard LangChain blocks supported by their
  integration.

Scientific tools return artifacts and concise model-visible results. They do
not build provider-specific chat payloads.

## WebUI status

The message shows attachments as artifacts. The Monitor
`multimodal.prepared` event reports:

- the number of attachments and model content blocks;
- MIME types and workspace paths;
- whether each file was sent, parsed, or stored only;
- any size or capability warning.

The event omits raw binary content.

## Implementation references

The main code paths are:

- `catmaster/runtime/multimodal_blocks.py`
- `catmaster/runtime/deepagents_backend.py`
- `catmaster/runtime/document_reads.py`
- `catmaster/webui/agent_loop.py`
- `catmaster/webui/frontend/src/v2/messageAdapters.js`
- `catmaster/llm/factory.py`

Relevant tests cover attachment preparation, compact native document reading,
large-document pagination before a model call, provider conversion,
persistence, and DeepAgents tool-result handling.

Notable changes to this behavior belong in the repository
[Changelog](../CHANGELOG.md).
