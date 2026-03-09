# CatMaster
```
+----------------------------------------------------------+
|   _____      _    __  __              _                  |
|  / ____|    | |  |  \/  |            | |                 |
| | |     __ _| |_ | \  / |  __ _  ___ | |_  ___  _ __     |
| | |    / _` | __|| |\/| | / _` |/ __|| __|/ _ \| '__|    |
| | |___| (_| | |_ | |  | || (_| |\__ \| |_| ___/| |       |
|  \_____\__,_|\__||_|  |_| \__,_||___/ \__|\___||_|       |
|        \_____________________________________________\   |
|         \_____________________________________________\  |
|                                                          |
|   Agentic Catalysis Research and Scientific Writing      |
|                                                          |
+----------------------------------------------------------+
```

CatMaster is an open-source agent system for computational catalysis workflows. It has moved well beyond a single planner-plus-tools loop: the current codebase has separate execution, research, and writing lanes, shared file-based memory, run-history retrieval, MCP-backed filesystem tooling, role-scoped skills, and a manuscript production path that can assemble and compile a TeX paper bundle.

## What CatMaster Is Now

The current architecture is organized around four lane-level workflows:

- `standard`: proposal -> director -> task runner -> memory patch. This is the full execution lane for multi-step work.
- `fast`: a proposal-free execution lane driven by `fast_director` for quicker bounded tasks.
- `research`: a campaign-level planner that can dispatch literature review, bounded child experiments, ask human input, conclude, and hand off to writing.
- `writing`: a dedicated writing lane with `write_director -> section_writer -> write_reviewer -> assemble -> finalize`, centered on TeX manuscript generation.

That lane split is the biggest architectural change relative to the older README and `docs/abilities.md`. CatMaster is no longer just a task executor with memory. It is now a multi-lane system with:

- LangGraph-based orchestration for execution, research, and writing.
- Shared project memory in `files/MEMORY/**`, plus campaign-local research state.
- A run ledger with history retrieval and reranking.
- Two skill roots: `skills/` and `writing_skills/`.
- A prompt-first writing workflow that can read project evidence, generate figures, assemble an `achemso` manuscript, and run a compile-fix pass.

## Technical Highlights

- Graph-based runtime instead of a monolithic orchestrator.
  - `catmaster/agents/graph.py`
  - `catmaster/agents/research_graph.py`
  - `catmaster/agents/writing_graph.py`

- Lane-specific agent roles and tool surfaces.
  - Execution roles: `proposal`, `director`, `task_runner`, `memory_patch`
  - Research roles: `research_lead`
  - Writing roles: `write_director`, `section_writer`, `write_reviewer`, `academic_polisher`, `tex_compile_fixer`

- Shared project workspace semantics.
  - The real working root is always project `files/`.
  - `metadata/runs/<run_id>` stores audit data, task state, traces, reports, and UI events.
  - Research campaign file outputs live in `files/research/<campaign_id>/`.
  - The active manuscript bundle lives in `files/manuscript/`, with rollover to `files/manuscript_archive/`.

- Research lane as a campaign controller, not just another execution run.
  - `RunLiterature`
  - `RunExperiment`
  - `RunWriter`
  - `AskHuman`
  - `Conclude`

- Writing lane as a first-class system, not a report formatter.
  - Prompt-first request interface
  - Fixed `achemso` shell
  - Section-level TeX generation
  - Figure planning vs realized figure artifact separation
  - Final compile-fix pass via `pdflatex`

- Metadata-driven skills.
  - `skills/` for execution/research skills
  - `writing_skills/` for manuscript and figure-writing skills
  - Visibility filtered by role and lane metadata, not just hardcoded lists

- MCP filesystem integration and role-scoped tool access.
  - Proposal/director/task-runner roles get different filesystem surfaces
  - Writing director and section writer get their own writing-specific tool bundle

## Current Capability Surface

### Execution and planning

- Full execution lane with proposal review, HITL pause/resume, replanning, memory patching, and final reporting.
- Fast lane for proposal-free bounded execution.
- Child experiment runs launched from the research lane while still working against the shared project `files/` root.

### Research lane

- Campaign persistence with `ResearchStore`
- Literature review workflow with structured packs
- Experiment dispatch and pack persistence
- Ask-human pause/resume continuity
- Context brokerage using:
  - project memory
  - run history
  - campaign state
  - persisted research artifacts
- `RunWriter` handoff using a fixed manuscript-style house prompt

### Writing lane

- Direct prompt-based writing runs from the WebUI
- Optional retrieval context from a source research campaign
- `write_director` planning with director-grade tools
- `section_writer` with:
  - MCP filesystem access
  - `bash_exec`
  - literature/research read tools
  - schematic figure generation
  - `apply_aider_edits`
- Review loop focused on manuscript quality rather than audit-log prose
- Fixed `achemso`-style final assembly into:
  - `files/manuscript/MANUSCRIPT.tex`
  - `files/manuscript/sections/*.tex`
  - `files/manuscript/figures/*`
  - `files/manuscript/references.bib`
- Final compile-fix step that requires `pdflatex`

### Scientific tooling

The tool registry still covers the core catalysis workflow:

- molecule generation from SMILES
- slab construction and surface preparation
- selective dynamics helpers
- supercells
- adsorption-site enumeration
- adsorbate placement and batch adsorption generation
- NEB geometry/input generation
- VASP input preparation
- VASP and MACE execution through DPDispatcher
- Materials Project retrieval
- structure rendering and image analysis

### Memory and history

- Shared file-based memory under `files/MEMORY/**`
- Memory index synthesis for agent context
- Run ledger indexing and history retrieval
- Research context review over prior project runs, memory, and campaign artifacts

## Project Layout

- `catmaster/`: orchestration graphs, runtime services, WebUI, and tools
- `configs/`: LLM and DPDispatcher configuration
- `skills/`: execution and research skills
- `writing_skills/`: writing-specific skills and template assets
- `reference_scripts/`: helper scripts and reference templates
- `devdocs/`: internal development/finalization notes
- `demos/`: example prompts and workflow demos

## Project Space Layout

Within a project space, the important paths are:

- `files/`
  - the real workspace root for agent operations
- `files/MEMORY/`
  - shared project memory
- `files/research/<campaign_id>/`
  - research campaign file outputs
- `files/manuscript/`
  - active manuscript bundle produced by the writing lane
- `files/manuscript_archive/<run_id>/`
  - archived prior manuscript bundles
- `metadata/runs/<run_id>/`
  - per-run audit layer: task state, traces, reports, UI events, tool logs
- `metadata/research_campaigns/<campaign_id>/`
  - structured research campaign state

## Environment Setup

CatMaster targets real catalysis environments: local workstation + optional GPU machine + remote CPU/VASP cluster.

### Python

Typical local setup:

```bash
conda create -n catmaster python=3.11
pip install -r requirements/pc.txt
```

If the same machine is also your GPU-side execution host, install both:

```bash
pip install -r requirements/pc.txt
pip install -r requirements/gpu.txt
```

### OVITO

`render_structure_views` prefers OVITO. `ovito` is already included in `requirements/pc.txt`.

If Ubuntu is missing OpenGL runtime symbols:

```bash
sudo apt update
sudo apt install -y libopengl0
```

### Node.js

MCP filesystem tools run through `npx @modelcontextprotocol/server-filesystem`, so Node.js is required.

```bash
brew install node
```

Then verify:

```bash
node -v
npm -v
npx -v
```

### LaTeX

The writing lane now expects `pdflatex` for the final compile-fix pass. If `pdflatex` is not available, manuscript finalization fails instead of silently degrading.

### Materials Project

```bash
export MP_API_KEY=YOUR_API_KEY
```

### POTCAR setup

Pymatgen still requires local POTCAR availability. Download the VASP POTCAR files and configure Pymatgen accordingly:

https://pymatgen.org/installation.html

### DPDispatcher

DPDispatcher runtime config lives in `configs/dpdispatcher/`.

Setup:

```bash
cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml
```

Then fill in SSH / queue / environment details in `machines.yaml`.

Remote execution assumptions:

- passwordless SSH
- Slurm-style HPC on CPU/VASP side
- Python 3.10+ available on compute nodes
- remote VASP or MACE runtime already installed

## LLM Configuration

CatMaster uses `configs/llm.yaml`, with credentials coming from environment variables.

Typical setup:

```bash
export OPENROUTER_API_KEY="..."
```

Current notable config areas:

- `models`: provider/model registry
- `agents`: role -> model mapping
- `image_generation`: image model and static image config for schematic figure generation
- `writing.author_name`: fixed manuscript author name, default `CatMaster`

The current role map includes the writing and research stack, not just the original execution roles.

## Quick Start

Run the WebUI:

```bash
python -m catmaster.webui --project-space-root ./project_space
```

Equivalent:

```bash
python main.py --project-space-root ./project_space
```

Then open:

```text
http://127.0.0.1:7860
```

Supported top-level lanes in the current UI/runtime:

- `standard`
- `fast`
- `research`
- `writing`

## Writing Workflow Notes

CatMaster's current writing behavior is intentionally different from the old `dossier -> single writer` design.

- Research can hand off to writing through `RunWriter`.
- Direct writing runs are prompt-first.
- Writing is TeX-first.
- The active manuscript bundle is stable and inspectable at `files/manuscript/`.
- Figures are treated as explicit manuscript assets, not just planning placeholders.

## Development Notes

- `devdocs/` contains internal design/finalization notes and is not the primary user-facing documentation.
- The old `docs/abilities.md` summary has been retired; this README is now the canonical top-level capability overview.
