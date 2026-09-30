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
- lifecycle source: DBOS durable workflows/queue with native LangGraph SQLite
  checkpoints and interrupts
- WebUI state model: rebuildable thread/message/activity projections, artifact
  records, and replayable semantic events

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
- right context rail: native graph run/background-specialist activity, the
  current turn's Plan, and compact Research progress in one semantic
  scroll
- large preview overlay: artifact, file, structure, report, table, image, and
  detailed activity inspection opened from conversation entries

Do not reintroduce separate Home, Monitor, and Files tabs as the primary
navigation. Files stay available through the left rail and the full Files view.
Do not put document or artifact previews back into the narrow right context
rail. Previewable conversation entries open one shared modal surface with a
dimmed backdrop, a large central canvas, and explicit close behavior.

### Multimodal Rendering

Messages may contain text, structured tool events, artifact references, file
references, molecular/structure previews, images, PDFs, remote receipts, and
interrupt cards.

Represent these as typed message parts or artifact references first. Use
assistant-ui extension points for rendering, and keep heavy content behind
artifact preview/content/download routes rather than embedding large payloads in
chat history.

Assistant Markdown may embed a browser-displayable workspace image with a
`sandbox:/files/...` source. Resolve that source at display time through the
authenticated thread-scoped image endpoint; do not rewrite the persisted
message or add base64 to it. The inline image should open its source file in the
large preview overlay, and a missing or non-image source should render a compact failure
notice.

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
- artifact cards that open in the large preview overlay

Do not introduce new active UI behavior that depends on:

- legacy `task_state.json`
- `FINAL_REPORT.md`
- Gradio component semantics
- the old large runtime snapshot stream as the main state source

## 4) Backend Contract Assumptions

The backend should expose a thin adapter over DBOS execution and native LangGraph while preserving
CatMaster's specialist delegation architecture, tool surface, skills, artifact
registry, and scientific Research Graph.

Assume thread-native BFF endpoints for:

- create/list/read/update threads
- submit/stop/resume native thread runs with explicit run identity
- update/stop native async specialist runs
- stream thread events by `last_seq`
- read thread messages and artifacts
- preview/content/download artifact endpoints

DBOS owns the queue and run lifecycle; LangGraph owns checkpoints and interrupts. CatMaster stores a UI projection but must not add a local scheduler,
turn FIFO, control flags, task DAG, or completion inbox. The frontend should not
infer agent lifecycle from logs or filenames. If UI behavior depends on
additional event fields, fix the backend projection/event contract instead of
adding fragile frontend-only workarounds.

## 5) Artifact, Tool, and Interrupt Semantics

Artifacts are first-class UI records. Messages should store artifact IDs and
preview metadata, not base64 payloads or arbitrary absolute paths.

Tool calls should be represented as structured events and rendered as compact
receipt cards in the conversation, with detailed inspection available in the
shared preview overlay.

Interrupts should render as message parts and inline review cards. Resume
payloads should use explicit decision types such as `approve`, `edit`,
`reject`, and ask-user `respond`.

Default permission mode is `auto`. The UI must keep an explicit Review/HITL
toggle available for users who want protected tool calls to pause for approval.
Both modes should map to backend interrupt policy rather than bypassing backend
safety checks in the frontend.

While a thread is running, the composer must require one explicit running-thread
input strategy: `interrupt` (Steer), `enqueue` (Queue), `rollback`
(Replace), or `reject` (Reject if busy). Preserve those meanings instead of
implementing a hidden frontend or backend queue. Do not claim that stopping a
local agent run cancels already submitted remote/scientific work. Persistent
Research root input remains on the root thread; use the native async-specialist
Steer or Stop action to target one child branch directly.

## 6) Frontend Implementation Conventions

- Prefer code under `frontend/src/v2/` for new thread-workspace work.
- Prefer assistant-ui primitives before adding CatMaster-specific chat UI.
- Keep CatMaster-only artifact/receipt/interrupt metadata available to custom
  renderers while assistant-ui receives normalized messages.
- Use internal scrolling for dense rails and preview canvases.
- Keep text and controls compact enough for repeated operational use.
- Do not add branch/retry UI until backend checkpoint fork support exists.

## 7) Build and Verification

When changing the WebUI:

- rebuild the frontend bundle with `npm run build`
- keep backend event fields and frontend expectations aligned
- verify assistant-ui message rendering, composer behavior, streaming, tool
  cards, artifact cards, interrupt cards, stop, refresh/rejoin, and artifact
  preview routes and modal open/close behavior

If a change modifies active WebUI semantics, update focused tests in `tests/`
without asserting long prompt text or exact optional wording.

## Research continuity

Persistent Research uses one root ResearchSpecialist. Durable background completion
and relevant external scientific graph edits enqueue semantic HumanMessage turns
on that same root. Workspace/model bindings follow the accepted session. Graph
recovery does not start planner or candidate-tournament runs.

Selecting Persistent Research and submitting its objective starts the session;
there is no separate Graph Auto/Manual or Update routes control. Research pause
and resume reuse thread Stop and explicit submit. Preserve legacy Manual pauses
until user continuation, and use the same pause projection in execution and UI.
Aggregate nested background tasks as well as historical execution children.
Result amendments refresh their existing milestone without creating duplicate
Results or additional model wakeups.

The graph panel displays judgment scopes/reasons, scientific revision links,
and consequential stopping decisions with their independent reconsideration and
validation outcome. Running Plan remains conversation state; scientific memory
stays in the workspace Graph. Completed stages and paused automation do not
automatically restart after edits or late results.
