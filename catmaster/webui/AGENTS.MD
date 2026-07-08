# WebUI Development Guide

The active WebUI is a FastAPI + React thread workspace. Do not treat this
package as a Gradio application, and do not make the old run monitor the
primary UX.

## 1) Source of Truth

The conversation page should be assistant-ui first.

Use `@assistant-ui/react` primitives and runtime patterns for the central chat
experience whenever they fit:

- thread layout and message list
- message parts and assistant/user message rendering
- composer input and send/stop controls
- tool-call and rich message extension points
- runtime/store integration boundaries

CatMaster-specific code should adapt backend state into assistant-ui concepts
and provide domain renderers for artifacts, tool receipts, interrupts, files,
and scientific multimodal content. Do not rebuild generic chat primitives when
assistant-ui already provides the behavior.

If a feature cannot be expressed through assistant-ui cleanly, document the
reason near the implementation and keep the custom surface narrow.

## 2) Current Stack

- backend: FastAPI
- frontend: React/Vite with `@assistant-ui/react`
- live updates: semantic SSE from thread endpoints
- state model: thread records, message parts, artifact records, interrupt
  records, and replayable thread events

The current v2 frontend entrypoint is:

- [CatMasterWorkspace.jsx](frontend/src/v2/CatMasterWorkspace.jsx)

The legacy `App.jsx` and `/api/session/{ctx}/run/*` endpoints may remain for
debugging and compatibility, but new user-facing work should target the thread
workspace and modern thread API.

## 3) UI Architecture Rules

### Assistant-UI First

The center conversation pane should be built primarily from assistant-ui
components and extension hooks. Custom CatMaster components belong at the
domain boundary:

- artifact cards and previews
- chemistry/materials structure viewers
- file and report references
- tool receipt summaries
- interrupt approval/edit/reject controls
- ask-user response parts

Avoid parallel custom implementations of message lists, composers, thread
stores, or generic tool-call state unless they are temporary migration code.

### Thread Workspace

The default page is a single CatMaster Workspace:

- left rail: workspace switcher, thread list, project file tree
- center: assistant-ui conversation thread and composer
- right inspector: artifact/file/tool/interrupt details

Do not reintroduce separate Home, Monitor, and Files tabs as the primary
navigation. Files stay available through the left rail and right inspector.

### Multimodal Rendering

Messages may contain text, structured tool events, artifact references, file
references, molecular/structure previews, images, PDFs, remote receipts, and
interrupt cards.

Represent these as typed message parts or artifact references first. Use
assistant-ui extension points for rendering, and keep heavy content behind
artifact preview/content/download routes rather than embedding large payloads in
chat history.

assistant-ui attachment adapters are the composer source of truth. Attachment
state sent to the backend must include stable filename, MIME, size, and content
fields, and backend/UI metadata should distinguish stored-only, sent as
image/file/text block, and unsupported/degraded attachments without exposing
base64 in thread history or monitor events.

### Streaming First

Prefer:

- semantic SSE event handling
- incremental message part reconciliation
- explicit thread status
- artifact cards that open in the inspector

Do not introduce new active UI behavior that depends on:

- legacy `task_state.json`
- `FINAL_REPORT.md`
- Gradio component semantics
- the old large runtime snapshot stream as the main state source

## 4) Backend Contract Assumptions

The backend should expose a modern agent-loop contract while preserving
CatMaster's existing subagent delegation architecture, tool surface, and skills.

Assume thread-native endpoints for:

- create/list/read/update threads
- submit/stop/resume thread turns
- stream thread events by `last_seq`
- read thread messages and artifacts
- preview/content/download artifact endpoints

The frontend should not infer agent lifecycle from logs or filenames. If UI
behavior depends on additional event fields, fix the backend event contract
instead of adding fragile frontend-only workarounds.

## 5) Artifact, Tool, and Interrupt Semantics

Artifacts are first-class UI records. Messages should store artifact IDs and
preview metadata, not base64 payloads or arbitrary absolute paths.

Tool calls should be represented as structured events and rendered as compact
receipt cards in the conversation, with detailed inspection available in the
right panel.

Interrupts should render as message parts and inspector review cards. Resume
payloads should use explicit decision types such as `approve`, `edit`,
`reject`, and ask-user `respond`.

Default permission mode is `auto`. The UI must keep an explicit Review/HITL
toggle available for users who want protected tool calls to pause for approval.
Both modes should map to backend interrupt policy rather than bypassing backend
safety checks in the frontend.

Prompt steering while a thread is running should be queued as a steering request
and applied at a safe boundary. Do not hard-cancel remote/scientific work solely
to apply steering.

## 6) Frontend Implementation Conventions

- Prefer code under `frontend/src/v2/` for new thread-workspace work.
- Prefer assistant-ui primitives before adding CatMaster-specific chat UI.
- Keep CatMaster-only artifact/receipt/interrupt metadata available to custom
  renderers while assistant-ui receives normalized messages.
- Use internal scrolling for dense rails and inspectors.
- Keep text and controls compact enough for repeated operational use.
- Do not add branch/retry UI until backend checkpoint fork support exists.

## 7) Build and Verification

When changing the WebUI:

- rebuild the frontend bundle with `npm run build`
- keep backend event fields and frontend expectations aligned
- verify assistant-ui message rendering, composer behavior, streaming, tool
  cards, artifact cards, interrupt cards, stop, refresh/rejoin, and artifact
  preview routes

If a change modifies active WebUI semantics, update focused tests in `tests/`
without asserting long prompt text or exact optional wording.
