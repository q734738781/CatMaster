# 10. Installation, model configuration, and deployment

[Previous](09-tools-skills-evolution.en.md) | [Contents](README.en.md) | [Next](11-reference-troubleshooting.en.md)

This chapter is for users who install CatMaster, configure models, or operate a server. Ordinary users do not need to learn every YAML field. They only need an accurate view of the agents, remote tasks, and external programs enabled by their deployment.

## Control-plane environment

The WebUI, agent runtime, materials tools, and most local analysis share `requirements/pc-conda.yml`:

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

Update an existing environment with:

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
```

The MACE, UMA, MatterSim, and ORB-v3 requirement files describe isolated remote environments. Installing them all into the control plane creates unnecessary torch, CUDA, and model conflicts and does not register remote tasks.

## Configure the LLM

CatMaster routes models by role. One model can serve every role, or a deployment can assign different models to coordination, workers, writing, review, vision, and low-frequency candidate proposal/review. Start from the standard template:

```bash
cp -n configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

A minimal single-model profile is:

```yaml
models:
  main:
    provider: openrouter
    model: <OPENROUTER_MODEL_ID>
    temperature: 1.0
    reasoning:
      effort: high
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1

agents:
  proposal: main
  director: main
  task_runner: main
  memory_patch: main
  summary: main
```

`main` is an internal CatMaster label. The `model` value is the provider model ID. Every value under `agents` must refer to a defined label.

### Mapping roles to the five agents

| Role | Main use | Typical fallback |
|---|---|---|
| `proposal` | Task proposal and initial decomposition | Required |
| `director` | Experiment coordination and general decisions | Required |
| `task_runner` | Materials, Dynamics, ML, and ORCA/xTB workers | Required |
| `memory_patch` | Project memory and skill candidates | Required |
| `summary` | Summary and general review fallback | Required |
| `research_lead` | Research agent | `director` |
| `research_state_updater` | Research state updates | `research_lead` |
| `hypothesis_proposer` | Falsifiable hypothesis and verification-plan formation | `research_lead` |
| `evidence_judge` | Independent evidence interpretation | `research_state_updater` |
| `write_director` | Writing coordinator | `research_lead` |
| `section_writer` | Writing worker | `task_runner` |
| `write_reviewer` | Writing checks and review | `summary` |
| `academic_polisher` | Conservative prose polishing | `summary` |
| `tex_compile_fixer` | TeX compilation repair | `academic_polisher` |
| `tool_selector` | General tool-selection support | `task_runner` |
| `image_analyzer` | Image understanding | `task_runner` |
| `literature_deep_research` | Literature Review | `director` |
| `self_evolution_proposer` | Improvement candidate generation | `memory_patch` |
| `self_evolution_reviewer` | Independent candidate review | `write_reviewer` |

A cost-conscious profile can use a faster model for `task_runner` and stronger models for Research, Writing, and review. Tool calling, image support, and long context must be verified against provider documentation and a real smoke call rather than inferred from the model name.

### Providers and credentials

Supported profile providers are `openai`, `openrouter`, `deepseek`, `gemini`, `oai_compatible`, `langchain`, `anthropic`, and `codex_oauth`. Common key variables include `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, and `ANTHROPIC_API_KEY`. Compatible endpoints specify their key variable through `api_key_env` and their endpoint explicitly.

Reasoning fields are provider-specific. OpenAI and OpenRouter use `reasoning.effort`; some compatible services use `reasoning_effort`; Anthropic thinking belongs in provider-specific kwargs. Copy the matching repository template instead of moving one provider's fields unchanged to another.

Keep real keys in environment variables or an external secret manager. An LLM YAML may contain a private endpoint, but it should not contain plaintext secrets. `.env.local` is not loaded automatically:

```bash
set -a
source .env.local
set +a
```

Codex OAuth uses the current operating-system user's credentials:

```bash
python -c \
'from langchain_openai.chatgpt_oauth import login_chatgpt_device; login_chatgpt_device()'

export CATMASTER_LLM_CONFIG=configs/llm_codex_oauth.template.yaml
```

The Codex OAuth template passes `timeout_s: 180` and leaves `max_retries`
unset, so transport errors, rate limits, and HTTP server errors use the pinned
OpenAI SDK default. The Codex backend can also accept an HTTP 200 stream and
then end it with the structured `server_is_overloaded` error, which the SDK
cannot retry at the HTTP layer. CatMaster retries only that stream error up to
six times, waiting 30, 60, 120, 240, 480, and 600 seconds across every
DeepAgent layer, including CatMaster's explicit `general-purpose` child. Other
model exceptions are not captured by this additional retry.

Do not copy the OAuth store or use one person's profile as the shared identity of a multi-user service.

### Reviewers, images, and multimodal input

`peer_review_models` lists model labels. Each label creates an independent report and therefore adds calls, cost, and latency:

```yaml
peer_review_models:
  - reviewer-a
  - reviewer-b
```

Image generation can bind a dedicated model:

```yaml
image_generation:
  model_label: image-model
  image_config:
    aspect_ratio: "4:3"
```

Image input depends on both profile capability and provider behavior. The runtime defaults image blocks on only for OpenAI, OpenRouter, Anthropic, Gemini, and LangChain providers. Other providers need explicit declaration and a real call. A saved attachment is not proof that it reached the model. Check `multimodal.prepared` when diagnosing.

### Profile selection and offline parsing

The profile path is selected in this order: an explicit code argument, `CATMASTER_LLM_CONFIG`, `configs/llm.yaml`, and finally single-model environment mode if the selected YAML does not exist.

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=<OPENROUTER_MODEL_ID>
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

Parse without calling a model:

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print("models:", sorted(p.models)); print("roles:", p.agents)'
```

Parsing proves only that the profile structure is valid. Verify key, endpoint, model ID, tool calling, and multimodal behavior in a minimal WebUI conversation.

## Literature search and controlled browsing

Provide only the services needed by the deployment:

```bash
export TAVILY_API_KEY="<KEY>"
export SEMANTIC_SCHOLAR_API_KEY="<KEY>"
export OPENALEX_API_KEY="<KEY>"
export NCBI_API_KEY="<KEY>"
export CROSSREF_MAILTO="you@example.org"
```

`TAVILY_API_KEY` is optional for roles using `codex_oauth` or OpenAI Responses:
those roles receive hosted `web_search`. Keep Tavily configured when other
providers need public discovery. The two implementations are not exposed
together under the same tool name, and the same provider resolver is used by
specialists, workers, and self-evolution roles. For CatMaster's function,
`literature.public_web_on_search_failure` controls scholarly-index fallback.
Quota, authentication, rate-limit, and network failures open a circuit for the
current run so later searches do not keep consuming or retrying the failed
Tavily backend. Fallback results identify their actual scholarly backend and do
not claim to be general-web coverage.

The active Literature Review tool surface is authoritative. API keys provide access but do not guarantee full text or correct metadata.

Install the controlled browser with:

```bash
npm install -g agent-browser@0.31.1
agent-browser install
agent-browser doctor --offline --quick
agent-browser mcp --help
```

CatMaster starts the MCP subprocess itself. Do not copy a global Codex MCP configuration into the project. Optional settings include:

```bash
export CATMASTER_AGENT_BROWSER_PROFILE="$HOME/.config/catmaster/browser-profile"
export CATMASTER_AGENT_BROWSER_HEADED=true
```

Keep the profile outside the workspace with private permissions. Users complete institutional login, CAPTCHA, and OTP themselves. Credentials and cookies do not belong in project files.

## Binding and access patterns

Bind a local workstation explicitly to loopback:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

For a personal remote server, keep CatMaster on server-side `127.0.0.1:7991` and create an SSH tunnel from the client:

```bash
ssh -L 7991:127.0.0.1:7991 <USER>@<SERVER>
```

Then open local `http://127.0.0.1:7991`.

A shared service needs a reverse proxy or VPN, TLS, external access control, least-privilege file permissions, logs, and backup. The built-in login provides account isolation and basic registration. It is not a complete internet-facing identity platform: registration is open by default, the application does not terminate TLS, and its cookie should not be the only public security boundary. After provisioning at least one user, start with `--disable-registration` or set `CATMASTER_DISABLE_REGISTRATION=1` to keep login required while rejecting new accounts. The status API then reports `registration_enabled: false`, the frontend hides account creation, and registration endpoints return HTTP 403.

Use `--no-login` only on a trusted machine bound to loopback. It opens the shared `admin` workspace and disables Skill Evolution.

## Remote computation configuration

Chapter 8 explains remote tasks from the user's perspective. Administrators create four private active files from public templates. The `-n` commands preserve existing active files. Merge template changes during an upgrade instead of replacing site configuration:

```bash
cp -n configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml
cp -n configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources.yaml
cp -n configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks.yaml
cp -n configs/dpdispatcher/mlff_backends_template.yaml configs/dpdispatcher/mlff_backends.yaml
```

These files contain host names, usernames, SSH key paths, queues, remote roots, and environment scripts. Git and deployment packaging exclude them. Do not paste active contents into issues, prompts, or shared workspaces.

### Machine, resource, task, and backend

A machine card defines SSH, Slurm or Shell mode, `remote_root`, and base environment. Confirm the host key interactively once, then test BatchMode. The remote root must be writable, and a Slurm machine should expose `sbatch`, `squeue`, and `scancel`.

A resource card binds machine, CPU/GPU, queue, walltime, environment `source_list`, and worker audience. Template core counts and queue names are examples. Preserve audience restrictions.

A task card defines the scientific program, input layout, default resource, boot script, and returned files. The template covers VASP, CP2K, LAMMPS, generic MLFF, MACE training and evaluation, xTB, CREST, and ORCA. Enable only validated tasks.

An MLFF backend card enables MACE, UMA, MatterSim, or ORB-v3 and binds resource, operations, and models. Every model profile declares the exact provider model, official task/domain capabilities, and charge/spin capability; MACE additionally declares its loader, allowed heads, and default head. UMA, MatterSim, and ORB-v3 model keys must exactly match official names rather than CatMaster abbreviations or case aliases. Every backend uses an isolated remote environment. The public template enables MACE `mh-1` and standalone `omol-0`; another backend is exposed only after its dependencies, weights, device, and minimum real case pass.

### Remote environment construction

The remote command environment combines machine `env_setup`, resource `source_list`, an optional submission prepend script, and the task command. Place site modules, conda activation, license variables, and library paths in controlled environment scripts rather than stages or prompts.

DPDispatcher commonly starts a non-interactive shell, so do not assume that it reads the user's `.bashrc`. If GPU nodes require a proxy to reach model repositories, copy and edit `configs/dpdispatcher/env_templates/catmaster_env_proxy.sh` and place it before the provider conda environment script in each applicable GPU resource `source_list`; remove the entry on hosts that need no proxy. Bind the proxy script only to machines that use it. In particular, do not reuse a GPU node's `localhost` proxy script on a different CPU or SSH host.

Before releasing a task, run one inexpensive real case for every enabled engine and verify catalog visibility, environment, result transfer, `status.json`, stdout/stderr, and receipt. `python scripts/remote_execution_smoke.py --list` only lists cases. Other modes submit real work, so do not begin with the entire suite.

## Structure Workbench, JSmol, VESTA, and VASPKIT

The production frontend contains exact-pinned MatterViz/Svelte and lazy Ketcher chunks. They are served from `/static`; no CDN or external font request is required. The server sends a Content Security Policy that permits same-origin chunks, local fonts, data/blob images, and workers used by volume parsing.

JSmol 16.3.13 is a compatibility fallback for OUTCAR vibration and unsupported formats. The launcher installs its pinned assets when the cache is missing. Prewarm a persistent cache for an offline server:

```bash
CATMASTER_JSMOL_CACHE_DIR=/persistent/cache/jsmol \
python scripts/install_jsmol_assets.py
```

A missing JSmol cache affects only those fallback previews. MatterViz-supported structures, the Workbench, LLM calls, and remote-task execution remain available.

After changing frontend dependencies, run `npm run build` under `catmaster/webui/frontend` and verify a periodic structure, a molecule 2D/3D switch, one trajectory frame request, and one volume grid in the deployed base path. Keep the exact versions in `package.json` and the lockfile; do not replace the lazy chunks with CDN scripts.

Set VASPKIT explicitly if needed:

```bash
export CATMASTER_VASPKIT_BIN=/opt/vaspkit/bin/vaspkit
```

For VESTA rendering:

```bash
export CATMASTER_VESTA_BIN=/opt/VESTA/VESTA
export CATMASTER_XVFB_RUN=/usr/bin/xvfb-run
```

Headless servers usually need Xvfb. CatMaster does not distribute VESTA or VASPKIT licenses.

## Pandoc, Chrome, fonts, TeX, and Julia

Markdown PDF needs Pandoc and Chrome or Chromium, plus suitable fonts for CJK content:

```bash
export CATMASTER_PANDOC_BIN=/usr/bin/pandoc
export CATMASTER_CHROME_BIN=/usr/bin/chromium

pandoc --version
chromium --version
fc-match "Noto Sans CJK SC"
```

LaTeX work needs at least `pdflatex`, and bibliography work commonly needs `bibtex`. Visually inspect the PDF after compilation.

PySR may download Julia and precompile on first import. During an online maintenance window:

```bash
python scripts/pysr_julia_smoke.py --fit
```

Install Julia in advance on offline machines and point `PYTHON_JULIACALL_BINDIR` to its `bin` directory.

## Runtime bounds and long output

`recursion_limit`, `max_tool_calls`, and context-compaction thresholds are safety boundaries, not quality sliders. Diagnose actual tool errors, context, and task scope before increasing them.

`configs/tool_output.yaml` keeps a Chat preview of long results and stores full content under `_tool_outputs/`. `configs/tool_policy.yaml` is not the active agent permission surface. Runtime allowlists, task audiences, and Review interruption define effective access.

## Packaging, upgrades, and rollback

`scripts/package_remote_deploy.sh` creates a package without `.git`, private config, keys, user projects, or runtime logs. `scripts/deploy_runtime.sh` synchronizes runtime files on the target. Consult each script's `--help` for current options.

Before upgrading, record the Git commit, conda environment, active LLM profile, four DPDispatcher configurations, launch arguments, and external-program versions. Back up the project root and authentication database, then test conversation, files, structure preview, and one minimal case for every enabled remote engine in a disposable workspace.

Migrate each workspace separately when upgrading from Research Kernel or a
hypothesis campaign. Stop the old writer for that workspace, then run a dry
run:

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python \
  scripts/migrate_research_graph.py /absolute/path/to/workspace
```

The report separates deterministic v3 and v4 campaign imports, v2 or incomplete
Kernel items that need review, and damaged files. After checking the counts,
apply the migration:

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python \
  scripts/migrate_research_graph.py /absolute/path/to/workspace --apply
```

The command returns a rollback manifest path. Legacy files move under
`metadata/legacy_research_state/`, and the new runtime writes only
`metadata/workspace.sqlite`. The batch has a stable in-progress pointer. If the
process stops, another `--apply` resumes that batch. Do not run old and new
writers against the same workspace during a rolling deployment.

Before any new Research Graph work is written, the returned manifest can roll
back the import:

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python \
  scripts/migrate_research_graph.py /absolute/path/to/workspace \
  --rollback metadata/legacy_research_state/<batch>/rollback_manifest.json
```

Rollback removes graphs imported by that batch, restores previous thread
bindings, and moves legacy files back to their original paths. Once researchers
have continued on the new graph, restore the backup into a separate workspace
and merge deliberately instead of overwriting the new scientific state.

Workspace SQLite chooses WAL only on a recognized local filesystem. Network or
unknown filesystems use the rollback journal by default. Set
`CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE=WAL` only after the deployment has
independently verified its storage semantics.

Code rollback must not overwrite user projects. Restore a compatible commit or package together with matching dependencies and config. Do not put project data, private YAML, or secrets into release archives as a rollback mechanism.

## Backup and logs

The default runtime directory is `.runtime/`, with `.runtime/webui.log` as the common log. Shared services need log rotation and should not leave raw-request debugging enabled because it may expose prompts or request content.

A complete backup includes every workspace's `files/` and `metadata/`, `.webui_auth/auth.sqlite` for login deployments, private LLM and DPDispatcher configuration outside Git, and external secret or site-environment backups. Back up when no run is writing and rehearse restoration.
