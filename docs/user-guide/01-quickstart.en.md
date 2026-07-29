# 1. Quick installation and first conversation

English | [中文](01-quickstart.zh.md) | [Contents](README.en.md) | [Next](02-concepts.en.md)

If an administrator has already given you a CatMaster URL, skip to "Your first WebUI session." To run CatMaster locally, complete the minimal setup below. Full model routing, shared deployment, and external program configuration are in [Chapter 10](10-deployment-operations.en.md).

## Minimal local installation

The CatMaster control plane uses one conda environment. From the repository root:

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

Update an existing environment with:

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
```

The MACE, UMA, MatterSim, and ORB requirements files describe isolated remote MLFF environments. They do not replace the control-plane environment.

## Configure one working model

For a first installation, copy the standard template only if `configs/llm.yaml` does not exist. Keep and edit an existing profile instead of overwriting it:

```bash
cp -n configs/llm.template.yaml configs/llm.yaml
```

The template uses OpenRouter model labels. Export the key:

```bash
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

For persistent local variables, create a private file from the example:

```bash
cp -n .env.example .env.local
chmod 600 .env.local
```

CatMaster does not load `.env.local` automatically. Load it before starting:

```bash
set -a
source .env.local
set +a
```

Never commit real credentials. If you use OpenAI, Anthropic, DeepSeek, Gemini, an OpenAI-compatible endpoint, or Codex OAuth, change the provider, model, and matching fields as described under [LLM configuration](10-deployment-operations.en.md#configure-the-llm). Replacing only the key variable is not enough.

## Start the WebUI

Create a project-space root and bind the service explicitly to the local host:

```bash
mkdir -p "$HOME/catmaster_projects"

CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

Open:

```text
http://127.0.0.1:7991
```

MatterViz and the lazy Ketcher molecule editor are shipped in the built WebUI assets. The first start may also install pinned JSmol compatibility assets for OUTCAR vibration and fallback previews, so it can take longer than later starts. Check status and logs with:

```bash
./start_webui.sh --status
tail -f .runtime/webui.log
```

For direct error output, start in the foreground:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991
```

Stop the background service with:

```bash
./start_webui.sh --stop
```

## Your first WebUI session

The default start shows a login page. Register an account and CatMaster creates a personal project area with a `default` workspace. On a shared server, each account is restricted to its own user directory.

For the first exercise, create a workspace named `quickstart` and a new thread. Choose Experiment and set permission mode to Review. This exposes worker delegation and tool activity while keeping later generic file edits or remote submissions behind approval.

Upload any CIF or POSCAR. A source checkout includes `tests/assets/Fe.cif` as a convenient interface test. Send:

```text
Use Experiment to inspect the crystal structure I just attached. Identify its workspace path, elements,
cell, periodicity, atom count, and any suspicious short contacts. Then ask the Materials worker to create
a 2x2x2 supercell at quickstart/Fe_2x2x2.vasp.

Choose an appropriate structure tool and explain how the cell and atom count changed.
This is a structure-only task. Do not query or submit any remote task.
```

This request reveals the normal CatMaster workflow. Chat should show Progress, a `materials_worker` delegation, and a `supercell` tool card. `supercell` writes its declared output within one domain-tool call, so the current Review mode may not display an approval card for this step. State the output path before sending, inspect the tool arguments, and verify the artifact and file afterward. The generated structure should open in the MatterViz preview; **Open Structure Workbench** exposes its editable base atoms, cell, measurements, and Save As controls.

The Fe transformation is only a test. It verifies that the LLM can use the current tool schema, Experiment can delegate Materials, the worker can read an attachment and write a project file, and artifacts, Files, and Monitor all describe the same operation.

Approval cards appear later for real remote submission calls. Local file edits and domain tools run without Review approval. Chapter 4 gives the exact boundary.

If literature is your main use case, upload a paper, select Literature Review, and send:

```text
Read this paper closely. Confirm the attachment path and readable page range, then identify the research
question, main evidence chain, and limitations. Separate methods, direct observations, and author interpretation.
Retain page or source anchors. In this turn, give me the reading plan first. Do not download other papers
or write a broader review.
```

## What to inspect after the first run

Chat shows agent, worker, and tool activity. Files confirms that requested artifacts exist in the project. Monitor records model calls, tool status, errors, and run scale. Open all three after the first exercise to connect the response, the process, and the files.

If the agent replies but does not create the requested file, expand the tool card and check for a rejected Review action, a bad path, or a warning. If the model never calls a tool, inspect Monitor and the WebUI log, then use [Troubleshooting](11-reference-troubleshooting.en.md) to verify model capability and configuration.

## What you do not need yet

The first local conversation does not require VASP, CP2K, LAMMPS, ORCA, xTB, CREST, MACE, or a cluster account. It also does not require every literature API, browser profile, VESTA, VASPKIT, Pandoc, or LaTeX. Add those capabilities after the basic WebUI path works.

For a temporary trusted single-machine test, login can be disabled:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991 --no-login
```

No-login mode opens the shared `admin` workspace and disables Skill Evolution. Never bind it to a LAN or public address.

The next chapter explains the agent, worker, skill, tool, and artifact relationships you just observed.
