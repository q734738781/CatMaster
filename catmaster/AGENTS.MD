# Agent Development Guide (Codex)

This repository implements **CatMaster**, a task-based orchestration system for computational catalysis workflows. The system is **tool-driven and auditable**:

- The LLM produces plans and step decisions.
- Tools perform concrete work (structure generation, batch job dispatch, file operations, parsing).
- The runtime records events, toolcalls, artifacts, and memory updates for traceability and resumability.

This document defines coding conventions intended for code-generation agents (e.g., Codex) to prevent style drift and to keep changes consistent with the existing architecture.

---

## 1) Repository layout and responsibilities

### Key packages

- `catmaster/agents/`
  - Orchestration, planning, task execution control, summarization, and prompt templates.
  - Only agents should directly call the LLM.
- `catmaster/runtime/`
  - Deterministic run state, tracing, artifact persistence, file-based memory operations, context pack building.
  - Anything involving `metadata/runs/...` lives here.
- `catmaster/tools/`
  - All executable “capabilities” exposed to the LLM.
  - Tools must validate inputs with Pydantic schemas and return standardized outputs.
- `catmaster/ui/`
  - Reporter abstraction and UI event bus (console UI removed; WebUI lives under `catmaster/webui/`).

### Data model (high level)

- `task_state.json` is the run’s single source of truth for:
  - `user_request`, `plan`, `tasks`, `observations`, and `status`.
- File-based memory (`files/MEMORY/**` + `metadata/memory/events.jsonl`) is the durable memory layer:
  - Director merges structured task results into memory index/topics and appends immutable event logs.
- Artifacts/toolcalls are persisted under `metadata/runs/<run_id>/...`.

---

## 2) Language, formatting, and style

### Python version and typing

- Use modern Python typing (PEP 585): `list[...]`, `dict[...]`, `Optional[...]`, etc.
- Prefer `from __future__ import annotations` at the top of non-trivial modules.
- Add type hints to public functions and non-trivial helpers.

### Formatting

- 4-space indentation.
- Keep functions small and single-purpose; extract helpers instead of deeply nesting.
- Prefer `pathlib.Path` over raw string path manipulation.

### Imports

- Standard library imports first, then third-party, then `catmaster.*`.
- Avoid unused imports; remove them rather than leaving them for later.


### Naming

- Modules: `snake_case.py`
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Prefer explicit names over abbreviations unless domain-standard (e.g., `neb`, `vasp`, `mace`).

---

## 3) Error handling and return conventions

### Tools must return standardized outputs

Tools should return the structure accepted by langchain's internal tool schema.
### Determinism

- Tools must be deterministic given identical inputs (unless randomness is explicitly required and logged).
- Avoid timestamps in output filenames unless required; if used, return the resolved path explicitly.

---

## 4) Path and workspace policy (critical)

CatMaster uses a project-space layout:

- **Project files root**: `project_space/files` (LLM/task-visible files)
- **Metadata root**: `project_space/metadata` (run metadata, traces, internal artifacts)

Rules:

1. Tools should operate in `project_space/files` unless they are explicitly metadata tools.
2. If a tool produces an artifact that future tasks must read, write it into project files and return a files-relative path via `workspace_relpath(...)`.
3. Use `resolve_workspace_path(...)` (or scoped path helpers) for path resolution and validation. Do not manually join paths with strings.
4. Do not use environment-variable-based workspace selection in new code. Project space should be passed through params/instances (for example CLI args, `WebSession`, `Orchestrator`) or explicit function arguments.

---

## 5) Tool development checklist (how to add/modify tools)

### 5.1 Input schema (Pydantic)

Every tool must have a `BaseModel` input schema with `Field(..., description="...")`:

- Keep fields stable; changing names/types is a breaking change.
- `ToolExecutor` rejects extra fields strictly. If you introduce a new field, update the planner prompts if they produce params for that tool.

Backward compatibility guidance:

- If you must rename a field, support both old and new names for at least one release (e.g., accept `mp_id` and `mp_ids`) and normalize internally.

### 5.2 Tool function signature and output

- Tool function signature: `def tool_name(payload: dict) -> tuple[Any, dict]:`
- First line should validate: `params = InputModel(**payload)`
- Required return is LangChain-native `(content, artifact)` tuple.
- For business failures, raise `CatMasterToolExecutionError` with a model-visible `public_message`.
- Do not introduce new output helper protocols for tool returns.
- Do not return arbitrary dict/list/string and rely on runtime guessing; non-native returns are treated as errors.

### 5.3 Registering tools

When adding a tool:

1. Implement tool module under the correct category:
   - `catmaster/tools/geometry_inputs/`
   - `catmaster/tools/execution/`
   - `catmaster/tools/retrieval/`
   - `catmaster/tools/misc/`
2. Re-export in the category `__init__.py`.
3. Register in `catmaster/tools/registry.py`:
   - import function + InputModel
   - call `register_tool("tool_name", tool_func, InputModel)`

Migration scope policy:
- Only tools registered/exposed to agent tool surface must follow the strict output contract immediately.
- For unregistered/internal tools, defer migration until they are prepared for registry exposure.

### 5.4 Large outputs: write files, return paths

For high-throughput workflows, avoid returning huge lists/dicts directly from tools.

Preferred pattern:

- Write a JSON summary file into a user-workspace output directory.
- Return:
  - `summary_json_rel`: workspace-relative path to JSON
  - optionally `counts`/small stats for quick display

### 5.5 Batch tools: required behaviors

For tools operating on directories:

- Accept `input_dir` and `output_dir/output_root`.
- **Never** place `output_dir` inside `input_dir` (must validate).
- Exclude `metadata/` and staging directories (`*_batch_*`) from input scanning.
- Ensure outputs are isolated and do not become inputs on subsequent scans.
- Produce a deterministic mapping from input paths to output paths. Prefer mirroring the input layout under
  `output_root`, and for file inputs expand into a folder (e.g., `input_root/a/c.vasp` -> `output_root/a/c/...`).

---

## 6) Agents and prompts

### Structured outputs (Pydantic)

- Use Pydantic models + structured-output APIs (e.g., LangChain `with_structured_output`) for any LLM response that is meant to be machine-read.
- Do not rely on JSON fences or ad-hoc JSON parsing in new code.

### Keep schema stable
If you change a schema, you must update:

- prompt templates
- Pydantic models and any validation/normalization logic
- downstream consumers (Orchestrator/TaskRunner/Summarizer)

### Events and UI

When adding new run phases or behaviors, emit events via the Reporter to preserve observability. Follow existing patterns in `catmaster/ui/reporters.py` and `catmaster/ui/events.py`.

---

## 7) Memory rules

- Do not write ad-hoc long-form memory blobs into prompts or metadata files.
- Use `files/MEMORY/MEMORY.md` as the short index (autoload excerpt), and put details in `files/MEMORY/topics/*.md`.
- Use `apply_aider_edits` for deterministic SEARCH/REPLACE updates when memory or other project files must be synchronized.
- `MemoryStore.merge_task_result(...)` is compatibility-only (event logging), not the canonical MEMORY updater.
- Keep immutable audit trails in `metadata/memory/events.jsonl`.
- Paths recorded in memory files must be project-files-relative paths (no `metadata/...`).

---

## 8) Logging, tracing, and reproducibility

- Prefer structured persistence over console logs:
  - tool inputs/outputs via `ArtifactStore`
  - run/task/tool events via `TraceStore`
- Use the existing run directory layout; do not create ad-hoc directories under `metadata/` without coordination.
- When writing JSON, use UTF-8 and `ensure_ascii=False`.

---

## 9) Repository hygiene

- Do not add or commit `__pycache__/` or `*.pyc`.
- Avoid committing large generated artifacts; tools should write them to the workspace at runtime, not to the repository.
- Keep changes minimal and localized; do not refactor unrelated modules in the same change.

### Test execution in conda (critical)

- When running tests in the project conda environment, use:
  - `conda run -n catmaster python -m pytest ...`
- Do **not** use `conda run -n catmaster pytest ...` directly. In this repository it can pick the wrong interpreter and cause false `importorskip` results (tests appear skipped even when dependencies exist in the env).

---

## 10) Quick reference: tool skeleton

Remote tool rule (DPDispatcher/GPU):
- If a tool needs to ship a script to the remote workdir, stage it under `task_script/` and run it via
  `python task_script/<script>.py ...`. Do not rely on syncing the full repo to the remote host.

```python
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class ExampleToolInput(BaseModel):
    path: str = Field(..., description="Input path (workspace relative).")
    output_dir: str = Field("outputs/example", description="Output path (workspace relative).")


def example_tool(payload: Dict[str, Any]):
    params = ExampleToolInput(**payload)
    in_path = resolve_workspace_path(params.path, must_exist=True)
    out_dir = resolve_workspace_path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "result.json"
    out_file.write_text("{}", encoding="utf-8")

    if not out_file.exists():
        raise CatMasterToolExecutionError(
            tool_name="example_tool",
            public_message="result file was not created",
            artifact={"data": {"output_dir_rel": workspace_relpath(out_dir)}},
        )

    return (
        "example_tool completed.\n"
        f"output_dir_rel={workspace_relpath(out_dir)}",
        {
            "tool_name": "example_tool",
            "data": {"output_dir_rel": workspace_relpath(out_dir)},
        },
    )
```

---

## 11) MCP adoption rules (for upcoming tool development)

Prioritize MCP-ization for cross-cutting capabilities with clear standard interfaces:

- filesystem
- browser / web search
- external services
- scheduler / remote execution
- database / retrieval

Do **not** prioritize MCP-ization yet for local domain tools:

- adsorption
- slab
- NEB
- thermo
- MACE

Rationale:

- These local domain tools are deeply coupled to the current Python/ASE/pymatgen/workspace contract.
- Forcing full MCP migration now would introduce high engineering cost.
- Standardization payoff is much higher first for filesystem / web / browser classes than for tightly coupled local scientific workflows.
