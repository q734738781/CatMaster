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
|     An Automated Agent System for Catalysis Research     |
|                                                          |
+----------------------------------------------------------+
```
CatMaster is a **open-source** task-based LLM orchestration and tooling framework for computational materials workflows. It provides a structured planning/execution loop, tool registry for geometry/input preparation and job submission (VASP, MACE via DPDispatcher), and unified tracing/reporting (task state, tool calls, memory events, and final reports).

## Highlights

- Task-based orchestrator with plan review and file-based memory (`files/MEMORY/**` + `metadata/memory/events.jsonl`) carried along tasks.
- Tool registry for materials workflows (Materials Project retrieval, slab construction, adsorption site enumeration, VASP/MACE job submission), and long-tail tools with powerful LLM and python_exec
- HITL (human-in-the-loop) intervention for blocked runs with replanning.
- Unified run artifacts and traces (event/tool/patch traces, observations, final report).
- Demo scripts and documented application cases in `demos/`, with result summaries in `demos/examples/`. You can try them!

## Project layout

- `catmaster/`: core orchestration, runtime, tools, and UI.
- `demos/`: runnable scripts for end-to-end use cases and examples.
- `docs/`: capability and design notes (start with `docs/abilities.md`).

## Environment setup

CatMaster is designed for real-world research environments, aiming at production-level research—for example, you have supercomputer cluster access for DFT calculations and an optional GPU server for ML-related research, while processing files on your laptop. Because of this, the environment setup is a bit complex, but definitely worth it. Once you set it up, you can open multiple terminals, raise many questions, have your coffee break, and let the LLMs do trivial things for you. But before that, you should carefully read the following setup guidance:

First, you should download the source code of the project (Find it in release/source code.zip, or directly download the repository if you like dev version), place somewhere (recommned: WSL for convinient environemnt setting) and preparew the environment:

### Python dependencies

Pick the right requirements set for your environment:
- `requirements/pc.txt`: local machine dependencies.
- `requirements/gpu.txt`: GPU/MACE-side dependencies.

The CPU side cluster normally do not need python dependencies, If you wish to use local as GPU-server workload (not dedicated gpu-server), you should install the dependencies in pc.txt and gpu.txt on the same local machine. Virtual environment recommended

Typical installation:
```bash
conda create -n catmaster python=3.11
pip install -r requirements/pc.txt
```

### Node.js (for MCP filesystem tools)

CatMaster can integrate MCP filesystem tools (`@modelcontextprotocol/server-filesystem`) via `npx`, so Node.js is required.

Install with Homebrew:
```bash
brew install node
```

If Homebrew prints a PATH hint for Node, append Node's bin to `~/.bashrc` (example):
```bash
echo 'export PATH="$(brew --prefix node)/bin:$PATH"' >> ~/.bashrc
# if you installed a versioned formula, use that name instead, e.g.:
# echo 'export PATH="$(brew --prefix node@22)/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:
```bash
node -v
npm -v
npx -v
```

### WebUI

Run the web workbench (recommended):
```bash
python -m catmaster.webui --project-space-root /path/to/project_space
```

Or use the main entry point:
```bash
python main.py --project-space-root /path/to/project_space
```

Project space selection is parameter-driven (`--project-space-root`/constructor arguments), not environment-variable driven.

**For GPU SIDE**: Ensure the remote host has the Python/MACE runtime; the task scripts are forwarded via DPDispatcher, so syncing the full repo is not required.
**For CPU SIDE**: VASP execution now runs a forwarded Python boot script; ensure the CPU cluster provides Python 3.10+ in the job environment (module/conda).

### Materials Project

Register and set your Materials Project API key in ~/.bashrc if you do not want LLM to raise error and make a human-in-the-loop intervention request if they can not find relevant structure:
```bash
export MP_API_KEY=YOUR_API_KEY
```

### Pymatgen POTCAR Configs:

Pymatgen needs POTCAR files to generate VASP inputs. You should download the POTCAR files from VASP portal and place them in a local dir.
Then refer to https://pymatgen.org/installation.html to setup POTCARs for pymatgen.


### DPDispatcher config (configs folder only)

All DPDispatcher runtime configs are unified under `configs/dpdispatcher/`.

- `machines_template.yaml`: tracked template for deployment.
- `machines.yaml`: local machine config (gitignored).
- `resources.yaml`: tracked resource presets.
- `tasks.yaml`: tracked task templates and default `resources` bindings.

Setup:
1. Copy template: `cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml`
2. Fill SSH/queue/env details in `machines.yaml`.
3. Verify machine names referenced in `resources.yaml`.
4. Keep task command patterns in `tasks.yaml` unless you know your site-specific changes are required.

**FOR ANY EXECUTION ON REMOTE MACHINES, PASSWORDLESS SSH CONNECTIONS (PRIVATE KEY LOGIN) SHOULD BE CONFIGURED BEFORE LAUNCHING THE CODE.**

### CPU (VASP) requirements

- Slurm-based HPC environment.
- Python 3.10+ available on compute nodes (for `task_script/vasp_boot.py`).
- VASP available in PATH or sourced via an environment script. Make sure launch command in tasks.yaml matches your site setup.
- DPDispatcher uses SSH to submit jobs; the MPI bootstrap should be set for SSH.

### GPU (MACE) requirements

- GPU host with CUDA/cuDNN and a Python environment for MACE.
- **Important:** MACE jobs forward the MACE script via DPDispatcher; you only need the remote runtime environment.

### VASP remote environment script

To execute vasp, you need to ensure VASP is correctly installed in cpu server and slurm system configured if you use slurm. (Hint: Taobao may help you?). Use the reference shell in `reference_scripts/catmaster_env_vasp.sh` as a template and update paths for your site. The current file contains site-specific paths and should be adapted before use. We are planning for support more DFT backends, however, for catalysis research, we have to admit the VASP is the dominant tool and have the best corresponding environment (guidance, comparable results, even the LLMs are more familiar with it and have better chance for fill-in correct params etc.)

Reference (from `reference_scripts/catmaster_env_vasp.sh`):
```bash
export PATH=/public/software/vasp.6.4.1-vtst-sol/bin:$PATH
export PYTHONPATH=/public/home/abcdefg/catmaster_code:$PYTHONPATH  # optional: only if you run custom Python not forwarded
source /public/software/vasp.6.4.1-vtst-sol/env.sh
ulimit -s unlimited
export I_MPI_HYDRA_BOOTSTRAP=ssh
```

Notes:
- `I_MPI_HYDRA_BOOTSTRAP=ssh` (SSH_HYDRA) must be set for DPDispatcher on some MPI stacks.
- Ensure the VASP environment script is sourced and `vasp_std` is in PATH.

## LLM Configuration
CatMaster now supports multiple providers via `configs/llm.yaml` (with secrets from env). To use LLM, acquire your API key and export it (e.g. ~/.bashrc):

```bash
export OPENAI_API_KEY="sk-projxxxxxxxxxxx"
# Optionally, change endpoint:
export OPENAI_BASE_URL="https://your-proxy-or-custom-endpoint/v1"
```

### Switch provider via configs/llm.yaml

Driver-template example (OpenRouter chat-completions + OpenAI responses):
```yaml
tool_calling_profiles:
  openrouter_chat_completions:
    driver: openai_chat_completions
    supports_builtin_tools: false
    parallel_tool_calls: false
    request_options: {}
    extra_body: {}
  openai_responses:
    driver: openai_responses
    supports_builtin_tools: true
    parallel_tool_calls: true
    request_options: {}
    extra_body: {}

models:
  "openai/gpt-5.2:online":
    provider: openrouter
    model: openai/gpt-5.2:online
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    tool_calling:
      profile: openrouter_chat_completions
      request_options: {}
      extra_body:
        # OpenRouter/provider-specific fields go here.
        # prompt_cache_retention: 24h

  "gpt-5.2":
    provider: openai
    model: gpt-5.2
    api_key_env: OPENAI_API_KEY
    tool_calling:
      profile: openai_responses
      request_options: {}
      extra_body: {}

agents:
  proposal: "openai/gpt-5.2:online"
  director: "openai/gpt-5.2:online"
  task_runner: "openai/gpt-5.2:online"
  memory_patch: "openai/gpt-5.2:online"
  summary: "gpt-5.2"

agent_policies:
  proposal:
    browse_tools_enabled: true
```

## Quick start (WebUI)

After environment setup, start CatMaster from the WebUI entry:

```bash
python -m catmaster.webui --project-space-root ./project_space
```

Equivalent command:

```bash
python main.py --project-space-root ./project_space
```

Then open:

```text
http://127.0.0.1:7860
```

Notes:
- You still need your model provider key (e.g., `OPENAI_API_KEY`) in the environment.
- Run summaries are saved in each project under `reports/FINAL_REPORT.md`.

## Application cases

- Start with `docs/abilities.md` for current capabilities.
- See runnable examples under `demos/`.
- Browse `demos/examples` for summarized runs, key files and final reports from those demos.

## Common environment variables

- `MP_API_KEY`: Materials Project API key.
- `OPENAI_API_KEY`: OpenAI API key.
- `OPENROUTER_API_KEY`: OpenRouter API key (when using OpenRouter models).
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible endpoint.
- `OPENROUTER_BASE_URL`: optional OpenRouter endpoint override.

---

Finally, the project is currently in its prototype/conceptual validation stage and under active development, you can open issues if you meet problems when using this system.

## License
This project is licensed under the Apache-2.0 License.

## Citation
If you find CatMaster useful in your research, please consider citing:
```
@misc{chen2026catmasteragenticautonomouscomputational,
      title={CatMaster: An Agentic Autonomous System for Computational Heterogeneous Catalysis Research}, 
      author={Honghao Chen and Jiangjie Qiu and Yi Shen Tew and Xiaonan Wang},
      year={2026},
      eprint={2601.13508},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2601.13508}, 
}
```
