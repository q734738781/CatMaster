# 1. Quick start

English | [Contents](README.en.md) | [Next](02-concepts.en.md)

This chapter gives a reproducible local startup path. At the end, you should be
able to register or sign in, create a workspace, choose an entrypoint, and
receive one model response. Remote scientific software is not required yet.

## 1.1 Prerequisites

- A Linux workstation or server.
- A working conda installation.
- Network access and a key for the selected LLM provider, or existing Codex
  OAuth credentials for the current system user.
- Installation-time access to conda, pip, npm, and the JSmol download source.
  See [Deployment and operations](10-deployment-operations.en.md) for offline
  preparation.
- At least 20 GB of free space is a practical starting point for the control
  plane. Project data and returned calculation results require additional room.

The complete control-plane environment is owned by:

```text
requirements/pc-conda.yml
```

Do not replace it with `requirements/mace.txt`, `requirements/uma.txt`,
`requirements/mattersim.txt`, or `requirements/orb.txt`. Those files describe
isolated remote MLFF provider environments.

## 1.2 Create the environment

From the repository root:

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

To update an existing environment:

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
```

Check the interpreter and WebUI command:

```bash
python --version
python -m catmaster.webui --help
```

## 1.3 Install the Literature Review browser

`agent-browser` is needed only for the controlled browser route used by
Literature Review, but installing it during initial setup is convenient:

```bash
npm install -g agent-browser@0.31.1
agent-browser install
agent-browser doctor --offline --quick
agent-browser mcp --help
```

CatMaster starts the MCP subprocess itself. Do not copy a global Codex MCP
entry into CatMaster. Complete institutional sign-in, CAPTCHAs, QR codes, and
one-time passwords yourself in the browser. Never put passwords, cookies, or a
browser profile in a project space.

## 1.4 Configure the first model

Copy the standard template:

```bash
cp configs/llm.template.yaml configs/llm.yaml
```

The standard template uses OpenRouter model labels. Export its key:

```bash
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

For another provider, do more than swap the key. Follow [LLM and runtime
configuration](03-llm-configuration.en.md) and update the provider, model, role
bindings, and provider-specific fields.

Configuration and secrets can be stored separately. One practical approach is:

```bash
cp .env.example .env.local
chmod 600 .env.local
```

The template uses `KEY=value` lines, so export its entries when sourcing it:

```bash
set -a
source .env.local
set +a
```

CatMaster does not auto-load `.env.local`. Never commit a file containing real
keys.

## 1.5 Create the project root

The project root contains one or more user workspaces:

```bash
mkdir -p "$HOME/catmaster_projects"
```

With the default account mode, each user's data is placed under
`users/<username>/` inside this root. See [Concepts and project
spaces](02-concepts.en.md) for the full layout.

## 1.6 Start the WebUI safely

Set the project root, bind address, and port explicitly:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

Open:

```text
http://127.0.0.1:7991
```

`start_webui.sh` runs in the background by default. Its embedded defaults are
`0.0.0.0:7991`, while `python -m catmaster.webui` defaults to
`127.0.0.1:7860`. The manual always supplies explicit values to avoid accidental
network exposure and port ambiguity.

The first startup may download and install a fixed JSmol asset bundle for
structure previews. It can therefore take longer than later starts.

## 1.7 First sign-in and smoke check

Sign-in and self-registration are enabled by default. Usernames are normalized
to lowercase, contain letters, numbers, dots, underscores, or hyphens, and must
be 3 to 40 characters. Passwords must be 8 to 256 characters. Registration also
uses a small arithmetic challenge.

After signing in:

1. Keep the default workspace or create a test workspace.
2. Create a thread.
3. Select `Experiment` and set permission mode to `Review`.
4. Send: `List the current project files and explain files versus metadata. Do
   not create anything.`
5. Confirm that Chat receives incremental output and Monitor records a run.

This checks the LLM, thread storage, and streaming UI. It does not test a cluster
or scientific executable.

## 1.8 Routine commands

Inspect status and logs:

```bash
./start_webui.sh --status
tail -f .runtime/webui.log
```

Run in the foreground for diagnosis:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh --foreground
```

Stop the background server:

```bash
./start_webui.sh --stop
```

Use another conda environment name:

```bash
CATMASTER_CONDA_ENV=<ENV_NAME> \
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

## 1.9 Local no-login mode

No-login mode exposes an open `admin` space and disables Skill Evolution. Use it
only on a trusted local machine:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991 --no-login
```

Never bind `--no-login` mode to a LAN or public address.

## 1.10 Next steps

- Read [Concepts and project spaces](02-concepts.en.md) before adding real data.
- Read [LLM and runtime configuration](03-llm-configuration.en.md) to tune role
  models.
- For cluster calculations, start with [Remote machines and
  execution](08-remote-execution.en.md). Do not edit private active config and
  immediately submit a production job without a smoke test.
