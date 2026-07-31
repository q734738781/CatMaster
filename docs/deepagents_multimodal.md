# Multimodal files

> This document describes the current attachment and file-reading behavior.
> WebUI users can start with
> [Working in the WebUI](user-guide/04-webui.en.md#what-happens-to-attachments).

CatMaster stores every accepted attachment as a workspace artifact before an
agent uses it. The selected model profile determines whether the current turn
also receives visual content, extracted text, or only a file reference.

## What users can provide

The WebUI accepts images, PDFs, modern Office documents, text files, structures,
and other project files.

| File kind | Current-turn behavior | Later access |
|---|---|---|
| Image | Sent as an image content block when the model profile supports images | Open with `read_file` |
| PDF | Parsed through the bounded document reader; selected pages can be rendered for visual inspection | Open with `read_document`, optionally with page selection |
| DOCX, XLSX, PPTX | Parsed through the bounded document reader | Open with `read_document` |
| Text, Markdown, JSON, CSV, logs, and source files | Sent as a bounded text excerpt when appropriate | Open with `read_file` or the matching document tool |
| Structure and scientific data files | Stored as project artifacts and handled by the relevant structure, trajectory, volume, or analysis tools | Open from Files or through the matching scientific tool |
| Audio, video, legacy Office, oversized, or unsupported media | Stored as an artifact; the current model may receive only the path and a warning | Use a supported converter or external workflow |

The current-turn summary tells the agent where each attachment was stored and
whether it was sent to the model, parsed as text, or stored only.

## Storage and conversation history

Attachments are stored under:

```text
files/attachments/<thread_id>/
```

The persisted conversation keeps the artifact identity, workspace path,
filename, MIME type, size, representation status, and warnings. Raw media
base64 and data URLs are not written into ordinary thread history or monitor
events.

This split supports both immediate inspection and long-running project
continuity. A later turn can reopen the stored path instead of replaying the
original binary payload through every model call.

## Model capability checks

`ModelMultimodalCapability` controls whether a profile accepts images, PDFs,
documents, audio, video, and multimodal tool results. OpenAI, OpenRouter,
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

DeepAgents `read_file` can return image content blocks. Those blocks remain
available to the next model call when the profile supports multimodal tool
results. PDF and modern Office documents use `read_document`, which limits
extracted content and supports page selection.

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
- `catmaster/runtime/document_access.py`
- `catmaster/webui/agent_loop.py`
- `catmaster/webui/frontend/src/v2/messageAdapters.js`
- `catmaster/llm/factory.py`

Relevant tests cover attachment preparation, bounded document reading,
provider conversion, persistence, and DeepAgents tool-result handling.

Notable changes to this behavior belong in the repository
[Changelog](../CHANGELOG.md).
