# CatMaster Repository Guide

Use this file for repository-wide development rules. More specific guidance lives in:

- `catmaster/AGENTS.MD` for the active DeepAgent specialist runtime.
- `catmaster/webui/AGENTS.MD` for the FastAPI/React WebUI.
- `skills/AGENTS.MD` for skill authoring.

## Testing

Avoid brittle tests that assert long prompt text or exact optional skill names. Prompts and skills are intentionally editable, composable runtime guidance, not stable public APIs.

When tests need coverage for prompts or skills, prefer stable behavioral contracts such as generated runner structure, tool bindings, staged skill roots matching the current source tree, schema fields, state transitions, or one narrowly scoped sentinel for a critical invariant.

## Runtime Prompt Authoring

Do not hard-code concrete tool-selection playbooks into generic system prompts. Generic prompts may describe role boundaries, delegation ownership, capability classes, and abstract policies such as preferring a fitting managed execution path over ad hoc local reimplementation.

Concrete tool-name guidance belongs in tool schemas, tool descriptions, skills, allowlists, and registry wiring. When a tool surface changes, update the relevant skill and tool binding rather than adding another prompt sentence that names the tool.

Coordinator specialists should not receive remote managed-execution contracts for worker-owned tools. Keep managed execution and remote receipt-handling guidance scoped to the worker/subagent layer that actually owns and can call those tools.
