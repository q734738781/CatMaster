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
CatMaster is a **open-source** task-based LLM orchestration and tooling framework for computational materials workflows. It is build on the classical Planner-Executor-Summarizer architecture now, provides a structured planning/execution loop, tool registry for geometry/input preparation and job submission (VASP, MACE via DPDispatcher), and unified tracing/reporting (task state, tool calls, whiteboard diffs, and final reports).

## Highlights

- Task-based orchestrator with plan review, structured whiteboard memory carried along tasks, and per-task summaries to support complex question.
- Tool registry for materials workflows (Materials Project retrieval, slab construction, adsorption site enumeration, VASP/MACE job submission), and long-tail tools with powerful LLM and python_exec
- HITL (human-in-the-loop) intervention for blocked runs with replanning.
- Unified run artifacts and traces (event/tool/patch traces, observations, final report).
- Demo scripts and documented application cases in `demos/` and result summary in `docs/examples/`. You can try them!

## Project layout

- `catmaster/`: core orchestration, runtime, tools, and UI.
- `configs/dpdispatcher/`: DPDispatcher routing/machines/resources/task templates.
- `demos/`: runnable scripts for end-to-end use cases.
- `devdocs/demos/`: summarized demo outputs and final reports.
- `docs/`: capability and design notes (start with `docs/abilities.md`).
- `reference_scripts/`: environment setup helpers (e.g., VASP remote env).

## Environment setup

CatMaster is designed for real-world research environments, aiming at production-level research—for example, you have supercomputer cluster access for DFT calculations and an optional GPU server for ML-related research, while processing files on your laptop. Because of this, the environment setup is a bit complex, but definitely worth it. Once you set it up, you can open multiple terminals, raise many questions, have your coffee break, and let the LLMs do trivial things for you. But before that, you should carefully read the following setup guidance:

First, you should download the source code of the project, place somewhere (recommned: WSL for convinient environemnt setting) and preparew the environment:

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

**For GPU SIDE**: Copy and place the catmaster folder in a remote dir, referred to <REMOTE_CODE_ROOT>, and fill the DPDispatcher configuration files that will provided in sections later.

### Materials Project

Register and set your Materials Project API key in ~/.bashrc if you do not want LLM to raise error and make a human-in-the-loop intervention request if they can not find relevant structure:
```bash
export MP_API_KEY=YOUR_API_KEY
```

### Pymatgen POTCAR Configs:

Pymatgen needs POTCAR files to generate VASP inputs. You should download the POTCAR files from VASP portal and place them in a local dir.
Then refer to https://pymatgen.org/installation.html to setup POTCARs for pymatgen.


### DPDispatcher config (templates, with required fields)

DPDispatcher config can be provided via:
- `$CATMASTER_DP_CONFIG` / `$CATMASTER_DP_TASKS`
- `~/.catmaster/dpdispatcher.yaml` or `~/.catmaster/dpdispatcher.d/*`
- `configs/dpdispatcher/*` directly under the project dir (relative to main.py)

**FOR ANY EXECUTION ON REMOTE MACHINES, PASSWORDLESS SSH CONNECTIONS (PRIVATE KEY LOGIN) SHOULD BE CONFIGURED BEFORE LAUNCHING THE CODE.**

Below are typical **template** examples (local+CPU Slurm Cluster+Dedicated GPU Server(e.g. AutoDL)) based on the current config files. Replace placeholders, but keep the structure and critical commands intact.
`machines.yaml` (template; keep `env_setup` blocks):\n
```yaml
cpu_server:
  batch_type: Slurm
  context_type: SSHContext
  local_root: <LOCAL_WORK_ROOT>
  remote_root: <REMOTE_WORK_ROOT>
  remote_profile:
    hostname: <CPU_LOGIN_HOST>
    port: 22
    username: <USERNAME>
    key_filename: <PATH_TO_SSH_KEY>
  env_setup: |
    ulimit -s unlimited # Any commands you want to use for 

gpu_server:
  batch_type: Shell
  context_type: SSHContext
  local_root: <LOCAL_WORK_ROOT>
  remote_root: <REMOTE_WORK_ROOT>
  remote_profile:
    hostname: <GPU_HOST>
    port: 22
    username: <USERNAME>
    key_filename: <PATH_TO_SSH_KEY>
  env_setup: |
    ulimit -s unlimited
    export PATH=<CONDA_BIN>:$PATH
    eval "$(conda shell.bash hook)"
    conda activate <GPU_ENV_NAME>
    export PYTHONPATH=<REMOTE_CODE_ROOT>:$PYTHONPATH
```

`resources.yaml` (template; keep `source_list` and Slurm flags as needed):\n
```yaml
vasp_cpu:
  machine: cpu_server
  number_node: 1
  cpu_per_node: <CPU_PER_NODE>
  queue_name: <CPU_QUEUE>
  group_size: 1
  custom_flags:
    - "#SBATCH -t <D-HH:MM:SS>"
    - "#SBATCH --export=ALL"
  source_list:
    - "<REMOTE_VASP_ENV_SCRIPT>"

mace_gpu:
  machine: gpu_server
  number_node: 1
  cpu_per_node: <CPU_PER_NODE>
  gpu_per_node: 1
  queue_name: ""
  group_size: 1
  custom_flags: []
```

`tasks.yaml` (template; **do not change the command patterns unless you know what you are doing :)**
```yaml
vasp_execute:
  command: "mpirun -n $SLURM_NTASKS vasp_std > vasp_stdout.out 2>&1" # This is a typical comman patten, change it to fit your server.
  forward_files:
    - "*"
  backward_files:
    - "*"
  task_work_path: "."

mace_relax:
  command: "python -m catmaster.tools.execution.mace_jobs --structure {structure_file} --fmax {fmax} --steps {maxsteps} --model {model}"
  forward_files:
    - "{structure_file}"
  backward_files:
    - opt.*
    - summary.json
  task_work_path: "."

mace_relax_dir:
  command: "python -m catmaster.tools.execution.mace_jobs --input {input_path} --output_root {output_root} --fmax {fmax} --steps {maxsteps} --model {model}"
  forward_files:
    - "input"
  backward_files:
    - "output"
  task_work_path: "."
```

`router.yaml` (template): # This file will be considered to remove in future versions
```yaml
tasks:
  mace_relax:
    resources: mace_gpu
    defaults:
      model: medium-mpa-0
  vasp_execute:
    resources: vasp_cpu
```


Notes:
- Keep `env_setup` / `source_list` aligned with your site environment scripts and MPI setup.
- For GPU MACE jobs, the **code folder must exist on the GPU host**; set `PYTHONPATH` in `env_setup` accordingly.
- For Slurm/VASP, ensure your env script exports `vasp_std` and sets MPI bootstrap (e.g., `I_MPI_HYDRA_BOOTSTRAP=ssh` when required).
- Under normal scenarios, you should not modify the name of tasks.yaml in case of program will not find suitable task.
- 
### CPU (VASP) requirements

- Slurm-based HPC environment.
- VASP available in PATH or sourced via an environment script. Make sure launch command in tasks.yaml matches your site setup.
- DPDispatcher uses SSH to submit jobs; the MPI bootstrap should be set for SSH.

### GPU (MACE) requirements

- GPU host with CUDA/cuDNN and a Python environment for MACE.
- **Important:** MACE jobs need the code folder available on the GPU host. Copy/sync your repo there and set `PYTHONPATH` accordingly.

### VASP remote environment script

To execute vasp, you need to ensure VASP is correctly installed in cpu server and slurm system configured if you use slurm. (Hint: Taobao may help you?). Use the reference shell in `reference_scripts/catmaster_env_vasp.sh` as a template and update paths for your site. The current file contains site-specific paths and should be adapted before use. We are planning for support more DFT backends, however, for catalysis research, we have to admit the VASP is the dominant tool and have the best corresponding environment (guidance, comparable results, even the LLMs are more familiar with it and have better chance for fill-in correct params etc.)

Reference (from `reference_scripts/catmaster_env_vasp.sh`):
```bash
export PATH=/public/software/vasp.6.4.1-vtst-sol/bin:$PATH
export PYTHONPATH=/public/home/abcdefg/catmaster_code:$PYTHONPATH
source /public/software/vasp.6.4.1-vtst-sol/env.sh
ulimit -s unlimited
export I_MPI_HYDRA_BOOTSTRAP=ssh
```

Notes:
- `I_MPI_HYDRA_BOOTSTRAP=ssh` (SSH_HYDRA) must be set for DPDispatcher on some MPI stacks.
- Ensure the VASP environment script is sourced and `vasp_std` is in PATH.

## LLM Configuration
This project currently only support OpenAI's model, but theoretically, it can be seamlessly changed to other model provider's models, but this feature is in currently only in future plan. To use LLM, you should acquire your API key, and set in in your environment (e.g. ~/.bashrc):

```bash
export OPENAI_API_KEY="sk-projxxxxxxxxxxx"
# Optionally, you may want to change the endpoint:
export OPENAI_BASE_URL="https://your-proxy-or-custom-endpoint/v1"
```

You can modify the main.py for other models manually. Currently the system do not require any structured output and interact with LLMs with only plain text, so theoretically, you can directly switch to other langchain chatmodel by modifying the main.py. We plan to release a unified LLM interface in next minor release.

## Quick test (LLM entry)

After tough environment settings, finally you can enjoy the CatMaster system. Congratulations!

This repo includes a user-friendly entry point `main.py` for prompt-based runs.

Example (prompt string):
```bash
python main.py \
  --workspace workspace/my_run \
  --model gpt-5.2 \
  --reasoning-effort medium \
  --prompt "Compute O2 in a box and report energy per atom."
```

Example (prompt file):
```bash
python main.py \
  --workspace workspace/my_run \
  --model gpt-5.2 \
  --prompt-file prompts/o2.txt
```

Notes:
- `--clean` is optional and off by default (use it only when you want to delete the workspace).
- Set `CATMASTER_WORKSPACE` instead of `--workspace` if preferred.
- You still need your model provider key (e.g., `OPENAI_API_KEY`) in the environment.

## Application cases

- Start with `docs/abilities.md` for current capabilities.
- See runnable examples under `demos/`.
- Browse `demos/examples` for summarized runs, key files and final reports from those demos.

## Common environment variables

- `CATMASTER_WORKSPACE`: workspace root for artifacts and reports.
- `MP_API_KEY`: Materials Project API key.
- `OPENAI_API_KEY` (or your provider key).
- `CATMASTER_DP_CONFIG` / `CATMASTER_DP_TASKS`: DPDispatcher config overrides.

---

Finally, enjoy your catalysis research in the age of LLM! The project is currently in its prototype/conceptual validation stage and under active development, you can open issues if you meet problems when using this system.

## License
This project is licensed under the Apache-2.0 License.

[![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=q734738781)](https://github.com/anuraghazra/github-readme-stats)