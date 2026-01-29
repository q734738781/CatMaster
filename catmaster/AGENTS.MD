# Agent Development Guide (Codex)

This repository implements **CatMaster**, a task-based orchestration system for computational catalysis workflows. The system is **tool-driven and auditable**:

- The LLM produces plans and step decisions.
- Tools perform concrete work (structure generation, batch job dispatch, file operations, parsing).
- The runtime records events, toolcalls, artifacts, and whiteboard patches for traceability and resumability.

This document defines coding conventions intended for code-generation agents (e.g., Codex) to prevent style drift and to keep changes consistent with the existing architecture.

---

## 1) Repository layout and responsibilities

### Key packages

- `catmaster/agents/`
  - Orchestration, planning, task execution control, summarization, and prompt templates.
  - Only agents should directly call the LLM.
- `catmaster/runtime/`
  - Deterministic run state, tracing, artifact persistence, whiteboard operations, context pack building.
  - Anything involving `.catmaster/runs/...` lives here.
- `catmaster/tools/`
  - All executable “capabilities” exposed to the LLM.
  - Tools must validate inputs with Pydantic schemas and return standardized outputs.
- `catmaster/ui/`
  - Reporters (console/live) and event bus.

### Data model (high level)

- `task_state.json` is the run’s single source of truth for:
  - `user_request`, `plan`, `tasks`, `observations`, and `status`.
- The whiteboard (`.catmaster/whiteboard.md`) is the durable, human-readable memory:
  - Updated via deterministic whiteboard ops (UPSERT/DEPRECATE).
- Artifacts/toolcalls are persisted under `.catmaster/runs/<run_id>/...`.

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

Tools should return the structure produced by `catmaster.tools.base.create_tool_output`:

- `status`: `"success"` or `"failed"`
- `tool_name`: tool identifier
- `data`: dict payload
- `warnings`: list of strings
- `error`: error message if failed
- `execution_time`: optional float (seconds)

Do not return raw objects, custom dict shapes, or non-dict values.

### Exceptions inside tools

- Prefer returning `create_tool_output(..., success=False, error=...)` for expected validation or runtime failures.
- If a tool raises an exception, the task runner will wrap it into a failed tool result, but this should be reserved for truly unexpected errors.

### Determinism

- Tools must be deterministic given identical inputs (unless randomness is explicitly required and logged).
- Avoid timestamps in output filenames unless required; if used, return the resolved path explicitly.

---

## 4) Path and workspace policy (critical)

CatMaster distinguishes two “views” of the filesystem:

- **User view**: the workspace root (excludes `.catmaster/`)
- **System view**: `.catmaster/` (run metadata, traces, internal artifacts)

Rules:

1. Tools should operate in **user view** unless there is a strong reason otherwise.
2. The `task_runner` role cannot access `view="system"` via file tools.
3. If a tool produces an artifact that future tasks must read, it must be written into the **user workspace**, and the tool must return a **workspace-relative** path using `workspace_relpath(...)`.
4. Use `resolve_workspace_path(...)` and `resolve_view_path(...)` for all path resolution and validation. Do not manually join paths with strings.

---

## 5) Tool development checklist (how to add/modify tools)

### 5.1 Input schema (Pydantic)

Every tool must have a `BaseModel` input schema with `Field(..., description="...")`:

- Keep fields stable; changing names/types is a breaking change.
- `ToolExecutor` rejects extra fields strictly. If you introduce a new field, update the planner prompts if they produce params for that tool.

Backward compatibility guidance:

- If you must rename a field, support both old and new names for at least one release (e.g., accept `mp_id` and `mp_ids`) and normalize internally.

### 5.2 Tool function signature and output

- Tool function signature: `def tool_name(payload: dict) -> dict:`
- First line should validate: `params = InputModel(**payload)`
- Return `create_tool_output(...)` in all normal cases.

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
- Exclude `.catmaster/` and staging directories (`*_batch_*`) from input scanning.
- Ensure outputs are isolated and do not become inputs on subsequent scans.
- Produce a deterministic mapping from input paths to output paths (e.g., flatten relative paths using `__`).

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

## 7) Whiteboard and memory rules

- Do not directly edit `.catmaster/whiteboard.md` with ad-hoc text operations in new code.
- Use whiteboard ops (UPSERT/DEPRECATE) and apply them with `whiteboard_ops_apply_atomic(...)`.
- Allowed sections are limited (see `catmaster/runtime/whiteboard_ops.py`); do not invent new sections without updating validators, templates, and context pack builder.

Whiteboard content rules:

- Paths recorded in Key Files must be user-workspace relative paths (no `.catmaster/...`).
- Use stable IDs (FACT/FILE/CONSTRAINT identifiers) and UPSERT existing IDs when updating.

---

## 8) Logging, tracing, and reproducibility

- Prefer structured persistence over console logs:
  - tool inputs/outputs via `ArtifactStore`
  - run/task/tool events via `TraceStore`
- Use the existing run directory layout; do not create ad-hoc directories under `.catmaster/` without coordination.
- When writing JSON, use UTF-8 and `ensure_ascii=False`.

---

## 9) Repository hygiene

- Do not add or commit `__pycache__/` or `*.pyc`.
- Avoid committing large generated artifacts; tools should write them to the workspace at runtime, not to the repository.
- Keep changes minimal and localized; do not refactor unrelated modules in the same change.

---

## 10) Quick reference: tool skeleton

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class ExampleToolInput(BaseModel):
    path: str = Field(..., description="Input path (workspace relative).")
    output_dir: str = Field("outputs/example", description="Output path (workspace relative).")


def example_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = ExampleToolInput(**payload)
    in_path = resolve_workspace_path(params.path, must_exist=True)
    out_dir = resolve_workspace_path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Comments for logics here
    out_file = out_dir / "result.json"
    out_file.write_text("{}", encoding="utf-8")

    return create_tool_output(
        tool_name="example_tool",
        success=True,
        data={"result_json_rel": workspace_relpath(out_file)},
    )
```
