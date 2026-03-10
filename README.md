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

CatMaster is an open-source agent system for computational catalysis workflows. It is built for the real lab setup: files on your workstation, optional GPU-side screening, remote VASP jobs on a cluster, too many structures to inspect by hand, too many calculations to prepare manually, and too much evidence to turn into a clean paper draft at the end.

If you want an agent that can do more than chat about catalysis, CatMaster is meant to help with the actual work:

- inspect the current workspace before doing anything stupid
- plan bounded scientific tasks
- prepare and launch real calculations
- keep project memory across runs
- run a multi-step research campaign
- write a manuscript-grade TeX bundle from the resulting evidence

## What CatMaster Can Do

### Run scientific work, not just describe it

CatMaster has two execution lanes:

- `standard`
  - proposal -> director -> task runner -> memory patch
  - for multi-step work that benefits from planning and review
- `fast`
  - a quicker proposal-free lane for bounded execution

That means it can:

- inspect files, folders, and prior outputs
- build structures and prepare inputs
- call scientific tools and shell commands
- run bounded workflows with MCP filesystem access and domain tools
- update shared project memory with the durable results

### Drive a research campaign

The `research` lane is the campaign brain. It does not just execute one task and stop.

It can:

- review literature
- dispatch bounded child experiments
- pause for human feedback when a decision is actually blocked
- persist campaign state across resumes
- decide when the evidence is good enough to move into writing

Its action space is explicit:

- `RunLiterature`
- `RunExperiment`
- `RunWriter`
- `AskHuman`
- `Conclude`

### Write a paper, not a lab log

The `writing` lane is built for manuscript production.

It can:

- read workspace evidence, project memory, run history, and research artifacts
- plan paper structure with `write_director`
- draft section-level TeX with `section_writer`
- review and revise sections
- generate lightweight schematic figures
- assemble a fixed-paper bundle
- run a compile-fix pass with `pdflatex`

The final output lives in a stable manuscript root:

- `files/manuscript/MANUSCRIPT.tex`
- `files/manuscript/sections/*.tex`
- `files/manuscript/figures/*`
- `files/manuscript/references.bib`

### Keep everything traceable

CatMaster is meant for work you may need to resume, audit, or defend later.

It keeps:

- shared project memory in `files/MEMORY/**`
- research campaign state in `metadata/research_campaigns/<campaign_id>/`
- run audit data in `metadata/runs/<run_id>/`
- reports, traces, tool logs, and UI event streams

So you can stop a run, come back later, inspect what happened, and continue without losing the thread.

## Why It Works

CatMaster is not one giant agent with every tool dumped on it.

It works by splitting the system into lane-specific workflows and role-specific tool surfaces:

- `standard`
- `fast`
- `research`
- `writing`

Backed by:

- `catmaster/agents/graph.py`
- `catmaster/agents/research_graph.py`
- `catmaster/agents/writing_graph.py`

Main agent roles:

- execution: `proposal`, `director`, `task_runner`, `memory_patch`
- research: `research_lead`
- writing: `write_director`, `section_writer`, `write_reviewer`, `academic_polisher`, `tex_compile_fixer`

That split is what lets CatMaster do different kinds of work well:

- execution agents can focus on getting tasks done
- research agents can focus on campaign decisions
- writing agents can focus on manuscript structure, prose, figures, and TeX output

## Core Technical Highlights

- LangGraph-based orchestration for execution, research, and writing
- shared project memory in `files/MEMORY/**`
- run ledger indexing and history retrieval
- MCP filesystem integration with role-scoped access
- metadata-driven skills from both `skills/` and `writing_skills/`
- fixed manuscript assembly with figure handling and compile-fix
- research-to-writer handoff for turning campaign evidence into a draft

## Scientific Capability Surface

### Geometry and input preparation

- molecule generation from SMILES
- slab construction and surface preparation
- selective dynamics helpers
- supercells
- adsorption-site enumeration
- adsorbate placement and batch adsorption generation
- NEB geometry and INCAR preparation
- VASP input preparation

### Execution

- VASP execution through DPDispatcher
- MACE relaxation and batch screening
- remote task forwarding for CPU/GPU environments

### Retrieval and evidence gathering

- Materials Project retrieval
- run-history retrieval and review
- research pack access
- workspace inspection and file reading through MCP filesystem tools

### Writing and figures

- TeX-first section writing
- schematic figure generation
- figure planning vs realized figure tracking
- fixed manuscript assembly
- compile-fix workflow

## Project Layout

- `catmaster/`: orchestration graphs, runtime services, WebUI, and tools
- `configs/`: LLM and DPDispatcher configuration
- `skills/`: execution and research skills
- `writing_skills/`: writing skills and template assets
- `reference_scripts/`: helper scripts and reference templates
- `devdocs/`: internal development and finalization notes
- `demos/`: example prompts and workflow demos

## Project Space Layout

Inside a project space, the important paths are:

- `files/`
  - the real workspace root for agent operations
- `files/MEMORY/`
  - shared project memory
- `files/research/<campaign_id>/`
  - research campaign file outputs
- `files/manuscript/`
  - active manuscript bundle produced by the writing lane
- `files/manuscript_archive/<run_id>/`
  - archived manuscript bundles from prior writing runs
- `metadata/runs/<run_id>/`
  - per-run audit layer: task state, traces, reports, UI events, tool logs
- `metadata/research_campaigns/<campaign_id>/`
  - structured research campaign state

## Environment Setup

CatMaster targets the common catalysis setup: local workstation + optional GPU machine + remote CPU/VASP cluster.

### Python

Typical local setup:

```bash
conda create -n catmaster python=3.11
pip install -r requirements/pc.txt
```

If the same machine is also your GPU-side execution host:

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

The writing lane expects `pdflatex` for the final compile-fix pass.

### Materials Project

```bash
export MP_API_KEY=YOUR_API_KEY
```

## Credits

CatMaster's writing skill stack includes adapted ideas and skill content from the `claude-scientific-skills` project by K-Dense AI:

- https://github.com/K-Dense-AI/claude-scientific-skills

### POTCAR setup

Pymatgen requires local POTCAR availability. Download the VASP POTCAR files and configure Pymatgen accordingly:

https://pymatgen.org/installation.html

### DPDispatcher

DPDispatcher runtime config lives in `configs/dpdispatcher/`.

Typical setup:

```bash
cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml
```

Then fill in SSH, queue, and environment details in `machines.yaml`.

Remote execution assumptions:

- passwordless SSH
- Slurm-style HPC on the CPU/VASP side
- Python 3.10+ available on compute nodes
- remote VASP or MACE runtime already installed

## LLM Configuration

CatMaster uses `configs/llm.yaml`, with credentials coming from environment variables.

Typical setup:

```bash
export OPENROUTER_API_KEY="..."
```

Important config areas:

- `models`
  - provider/model registry
- `agents`
  - role -> model mapping
- `image_generation`
  - static image model config for schematic figure generation
- `writing.author_name`
  - fixed author name for manuscript assembly, default `CatMaster`

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

Supported top-level lanes:

- `standard`
- `fast`
- `research`
- `writing`

## Good First Uses

If you want to get a feel for the system quickly, try one of these:

- ask `fast` lane to inspect an existing catalysis workspace and summarize what is already there
- ask `standard` lane to prepare a bounded VASP or MACE workflow
- ask `research` lane to turn a vague catalyst question into a campaign with literature and experiments
- ask `writing` lane to turn existing workspace evidence into a compact ACS-style TeX manuscript

## Notes

- `devdocs/` is for internal development and finalization notes.
- This README is the main top-level capability overview for new users.
