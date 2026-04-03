# WebUI Development Guide

The active WebUI is a FastAPI-backed, React-based interface. Do not treat this package as a Gradio application.

## 1) Current stack

- backend: FastAPI
- frontend: React
- live updates: SSE (`text/event-stream`)
- state model: `run_state.json` plus streamed runtime events

The primary frontend entrypoint is:

- [App.jsx](/home/chenhh/python_projects/CatMaster/catmaster/webui/frontend/src/App.jsx)

## 2) UI architecture rules

### Streaming-first

The active UI is driven by backend push, not by legacy polling assumptions.

Prefer:

- SSE event handling
- live snapshot reconciliation
- explicit run/activity state from the backend

Do not introduce new UI behavior that depends on:

- legacy `task_state.json`
- `FINAL_REPORT.md`
- Gradio component semantics

### Chat-first layout

The main page is a chat-first interface.

Current expectations:

- the central column is the conversation timeline
- live thinking stays in the chat stream
- right-side panels are auxiliary views such as Todo and Tool trace
- long-running sessions should scroll inside the chat container rather than extending the full page height

Do not regress the layout into a monitor-first or form-first interface.

## 3) Live state semantics

The frontend should distinguish between:

- persisted chat history
- transient live assistant state
- per-agent Todo and Tool trace views

Transient runtime information should not be accidentally written into durable chat history unless the backend explicitly persists it.

### Agent-aware views

The runtime may have multiple active agents or subagents.

Frontend code should preserve agent-aware grouping for:

- thinking/live LLM state
- todo lists
- tool trace rows

When showing aggregate views, prefer stable ordering and avoid excessive re-sorting on every streamed event.

## 4) Backend contract assumptions

Assume the backend provides:

- run snapshots
- streamed runtime events
- active run identity
- per-agent live state
- usage summaries

If UI behavior depends on additional event fields, fix the backend event contract instead of adding fragile frontend-only workarounds.

## 5) Frontend implementation conventions

- Prefer existing React patterns already used in `App.jsx`
- Keep derived UI state small and explicit
- Avoid introducing redundant local mirrors of server state when a streamed snapshot is already authoritative
- Use internal scrolling for dense panels such as Todo and Tool trace
- Preserve readable ordering in multi-agent views:
  - Todo grouped by status
  - Tool trace with running work first

## 6) Build and verification

When changing the WebUI:

- rebuild the frontend bundle
- keep backend event fields and frontend expectations aligned
- verify that live streaming still works for:
  - thinking
  - todo updates
  - tool trace updates
  - final answer insertion

If a change modifies active WebUI semantics, update the relevant tests in `tests/` in the same change.
