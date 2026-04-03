# CatMaster
**THIS README WAS WRITTEN AGES AGO AND NOTHING EXCEPT ENVIRONMENT SETTINGS SHOULD BE TREATED AS REAL CAPABILITY OF THE SYSTEM. USE IT WITH YOUR CASES!**

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

### VASPKIT

`vaspkit` is recommended for thermochemistry correction tools such as `vaspkit_adsorbate_thermo_correction` and `vaspkit_gas_thermo_correction`.

If you already have a local VASPKIT build, make sure it is on `PATH`:

```bash
export PATH=~/vaspkit/bin:$PATH
vaspkit
```

To persist that for future shells:

```bash
echo 'export PATH=~/vaspkit/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

If `vaspkit` is unavailable, CatMaster will fall back to ASE thermochemistry:

- `502` fallback uses ASE `IdealGasThermo`
- `501` fallback uses ASE `HarmonicThermo` with a `50 cm^-1` low-frequency floor

The ASE fallback is intentionally compatibility-oriented, but VASPKIT remains the preferred backend when you need the closest match to VASPKIT output conventions.

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
./start_webui.sh
```

Equivalent:

```bash
python -m catmaster.webui --project-space-root ./project_space
```

Also works:

```bash
python main.py --project-space-root ./project_space
```

The helper script will:

- launch with `conda run -n catmaster`
- export the repository root into `PYTHONPATH`
- default `--project-space-root` to `./project_space`

Useful overrides:

```bash
CATMASTER_CONDA_ENV=catmaster ./start_webui.sh --port 7991
CATMASTER_PROJECT_SPACE_ROOT=/path/to/workspace ./start_webui.sh
```

### Execute Timeout Note

In the current specialist runtime, the default `execute` backend timeout is `12h`
(`43200s`). This default applies when the agent calls `execute(command=...)`
without an explicit `timeout=...`.

However, explicit per-call `execute(timeout=...)` values are still validated by
DeepAgent's installed `FilesystemMiddleware`, whose default
`max_execute_timeout` is typically `3600s`. If you want the LLM to be able to
request longer explicit timeouts such as `6h` or `24h`, the simplest local-dev
solution is to manually patch your installed `deepagents` package.

Typical place to edit:

```text
<your-python-env>/site-packages/deepagents/middleware/filesystem.py
```

What to change:

- raise the default `max_execute_timeout` from `3600` to your desired limit
  such as `21600` (`6h`) or `86400` (`24h`)
- or pass a larger `max_execute_timeout=...` anywhere `FilesystemMiddleware(...)`
  is constructed inside the installed `deepagents` package

This is a local environment patch, so it may need to be re-applied after
upgrading or reinstalling `deepagents`.

Deploy a runtime checkout:

```bash
scripts/deploy_runtime.sh --target ../CatMaster_Run
```

After deploy, the runtime can be started directly from its root:

```bash
cd ../CatMaster_Run
./start_webui.sh --port 7991
```

The deploy script now defaults the runtime project space to `./project_space` inside the target directory. You can still override it either at deploy time with `--project-space-root`, or at run time with `CATMASTER_PROJECT_SPACE_ROOT=/path/to/workspace`.

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
