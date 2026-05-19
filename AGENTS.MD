# CatMaster Repository Guide

Use this file for repository-wide development rules. More specific guidance lives in:

- `catmaster/AGENTS.MD` for the active DeepAgent specialist runtime.
- `catmaster/webui/AGENTS.MD` for the FastAPI/React WebUI.
- `skills/AGENTS.MD` for skill authoring.

## Testing

Avoid brittle tests that assert long prompt text or exact optional skill names. Prompts and skills are intentionally editable, composable runtime guidance, not stable public APIs.

When tests need coverage for prompts or skills, prefer stable behavioral contracts such as generated runner structure, tool bindings, staged skill roots matching the current source tree, schema fields, state transitions, or one narrowly scoped sentinel for a critical invariant.
