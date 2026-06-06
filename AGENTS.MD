# CatMaster Repository Guide

Use this file for repository-wide development rules. More specific guidance lives in:

- `catmaster/AGENTS.MD` for the active DeepAgent specialist runtime.
- `catmaster/webui/AGENTS.MD` for the FastAPI/React WebUI.
- `skills/AGENTS.MD` for skill authoring.

## Testing

Avoid brittle tests that assert long prompt text or exact optional skill names. Prompts and skills are intentionally editable, composable runtime guidance, not stable public APIs.

When tests need coverage for prompts or skills, prefer stable behavioral contracts such as generated runner structure, tool bindings, staged skill roots matching the current source tree, schema fields, state transitions, or one narrowly scoped sentinel for a critical invariant.

## Dependency Management

Keep deployment dependencies reproducible. Runtime requirement files should use exact versions for dependencies that CatMaster imports, configures, or intentionally exposes as part of agent workflows. Scientific Python, pymatgen, pandas, ASE, MACE/torch, DPDispatcher, Pydantic, LangChain, DeepAgents, OpenRouter, and WebUI-adjacent packages are especially likely to change APIs between releases.

When adding a Python dependency, add it to the smallest relevant requirements file with an explicit version after testing the relevant workflow. Do not use a local environment freeze as the sole source of truth: audit the imports, deployment role, and package metadata first. Do not add broad unpinned packages such as `package`, `package>=x`, or `package<next-major` unless there is a concrete compatibility reason documented next to the requirement.

Keep requirement-file ownership clear. `requirements/pc.txt` is the CatMaster control-plane/WebUI/agent environment. `requirements/gpu.txt` is only the MACE/GPU execution add-on for remote boot scripts and optional local MACE tools; do not add LLM, LangChain, OpenRouter, WebUI, vector-search, or DPDispatcher control-plane packages there unless GPU-side code directly imports them.

When upgrading a pinned dependency, verify the import paths and the focused workflows that use it before changing the pin. For packages that recently split or moved APIs, such as pymatgen and provider/agent SDKs, include a short note in the PR or commit describing the compatibility check.

Remove stale direct dependencies instead of carrying them indefinitely. If a package is only needed transitively, let the owning package declare it unless CatMaster directly imports it or needs to constrain a known fragile integration boundary.

## Runtime Prompt Authoring

Do not hard-code concrete tool-selection playbooks into generic system prompts. Generic prompts may describe role boundaries, delegation ownership, capability classes, and abstract policies such as preferring a fitting managed execution path over ad hoc local reimplementation.

Concrete tool-name guidance belongs in tool schemas, tool descriptions, skills, allowlists, and registry wiring. When a tool surface changes, update the relevant skill and tool binding rather than adding another prompt sentence that names the tool.

Coordinator specialists should not receive remote managed-execution contracts for worker-owned tools. Keep managed execution and remote receipt-handling guidance scoped to the worker/subagent layer that actually owns and can call those tools.
