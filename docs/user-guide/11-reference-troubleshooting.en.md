# 11. Reference and troubleshooting

[Previous](10-deployment-operations.en.md) | [Contents](README.en.md)

This chapter collects common variables, defaults, limits, and diagnosis order.
First classify the problem as startup, model, file, browser, remote
configuration, or scientific execution. Do not replace evidence-based diagnosis
with a full dependency reinstall.

## 11.1 Configuration map

| File | Purpose | Private information |
|---|---|---|
| `requirements/pc-conda.yml` | Single control-plane environment definition | No |
| `.env.example` | Environment checklist, not auto-loaded | No |
| `configs/llm.yaml` | Active LLM profile | Should not contain keys; may contain a private endpoint |
| `configs/llm*.template.yaml` | Provider and role templates | No |
| `configs/tool_output.yaml` | Long-output preview and offload policy | No |
| `configs/tool_policy.yaml` | Compatibility config, not the active specialist authorization source | No |
| `configs/dpdispatcher/*_template.yaml` | Public machine/resource/task/backend templates | No |
| `configs/dpdispatcher/{machines,resources,tasks,mlff_backends}.yaml` | Active remote configuration | Yes |
| `configs/dpdispatcher/env_templates/` | Reference activation scripts | Contains site placeholders |

## 11.2 Common environment variables

### LLM and discovery

| Variable | Purpose |
|---|---|
| `CATMASTER_LLM_CONFIG` | Select YAML profile; default `configs/llm.yaml` |
| `CATMASTER_LLM_PROVIDER` | No-YAML provider or empty-provider fallback |
| `CATMASTER_LLM_MODEL` | No-YAML model or empty-model fallback |
| `CATMASTER_API_KEY_ENV` | Key variable name in no-YAML mode |
| `CATMASTER_BASE_URL` | Endpoint in no-YAML mode or for an empty field |
| `CATMASTER_TEMPERATURE` | Temperature in no-YAML mode or for an empty field |
| `CATMASTER_REASONING_EFFORT` | Effort in no-YAML mode or empty reasoning |
| `OPENAI_API_KEY`, `OPENROUTER_API_KEY` | Provider keys |
| `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY` | Provider keys |
| `TAVILY_API_KEY`, `MP_API_KEY` | Web search and Materials Project |
| `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_API_KEY`, `NCBI_API_KEY` | Literature services |
| `CROSSREF_MAILTO` | Crossref contact address |

### Browser and local helpers

| Variable | Purpose |
|---|---|
| `CATMASTER_AGENT_BROWSER_BIN` | `agent-browser` executable |
| `CATMASTER_AGENT_BROWSER_PROFILE` | Browser profile outside workspaces |
| `CATMASTER_AGENT_BROWSER_AUTO_CONNECT` | Connect to running Chrome |
| `CATMASTER_AGENT_BROWSER_HEADED` | Show the browser window |
| `CATMASTER_AGENT_BROWSER_MAX_OUTPUT` | Controlled browser output cap |
| `CATMASTER_VASPKIT_BIN`, `CATMASTER_VESTA_BIN` | Local helper paths |
| `CATMASTER_XVFB_RUN` | No-DISPLAY rendering wrapper |
| `CATMASTER_PANDOC_BIN`, `CATMASTER_CHROME_BIN` | Markdown PDF tool paths |
| `CATMASTER_JSMOL_CACHE_DIR` | Persistent JSmol cache |

### Runtime and WebUI

| Variable | Purpose |
|---|---|
| `CATMASTER_PROJECT_SPACE_ROOT` | Multi-user project root |
| `CATMASTER_CONDA_ENV` | Conda environment used by the launcher |
| `CATMASTER_HOST`, `CATMASTER_PORT` | WebUI bind address and port |
| `CATMASTER_RUNTIME_DIR` | PID and default-log directory |
| `CATMASTER_WEBUI_LOG`, `CATMASTER_WEBUI_PID` | Log and PID files |
| `CATMASTER_TOOL_OUTPUT_CONFIG` | Tool-output policy |
| `CATMASTER_SELF_EVOLUTION_MODE` | `off`, `observe`, or `auto` |
| `CATMASTER_RECURSION_LIMIT`, `CATMASTER_MAX_TOOL_CALLS` | Primarily no-YAML profile limits |
| `CATMASTER_DEEPAGENT_CONTEXT_TRIGGER_TOKEN_CAP` | No-YAML context-compaction cap |
| `CATMASTER_PRINT_HTTP_RAW_POST` | Raw request debugging, potentially sensitive; false by default |

`CATMASTER_PRINT_HTTP_RAW_POST=true` can place prompts or request data in logs.
Use it briefly in an isolated diagnostic environment.

## 11.3 Addresses and precedence

The launcher resolves CLI arguments before `CATMASTER_*` environment variables,
then script `LOCAL_*` constants, then code fallbacks.

| Launch route | Implicit value |
|---|---|
| `./start_webui.sh` | Embedded `0.0.0.0:7991` |
| `python -m catmaster.webui` | `127.0.0.1:7860` |
| Manual recommendation | Explicit `127.0.0.1:7991` |

For an unreachable page, start with:

```bash
./start_webui.sh --status
tail -n 100 .runtime/webui.log
ss -ltnp | grep 7991
```

## 11.4 WebUI startup failure

Run in the foreground:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991
```

Diagnose the first real traceback:

- `conda is not available`: initialize conda or set the right
  `CATMASTER_CONDA_ENV`.
- `Address already in use`: identify the listener or choose another explicit
  port.
- JSmol download failure: prewarm the cache; if other pages work, treat it as a
  structure-preview problem.
- Missing frontend assets: verify the deployment package; do not replace a full
  runtime package with an incomplete `--include-path` selection.
- Project-root permission denied: fix ownership instead of running as root.

## 11.5 LLM configuration or call failure

Parse without network access:

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print(sorted(p.models)); print(p.agents)'
```

Common problems:

- `Missing API key`: make sure the variable is exported into the process that
  starts the WebUI.
- `.env.local` was sourced but has no effect: use
  `set -a; source .env.local; set +a`.
- A role references an unknown label: fix `agents` or `peer_review_models`.
- Removed field error: delete `tool_calling_profiles`, model-level
  `tool_calling`, or misplaced `extra_body`.
- Provider 400: check model ID, base URL, reasoning shape, and provider options.
- Text answers but no tools: confirm current model tool-schema support and
  inspect the tool card and provider log.
- A long task stops early: inspect `max_tool_calls`, recursion, and the actual
  error before raising limits substantially.

## 11.6 Sign-in and workspace

- Registration fails: use a 3 to 40 character allowed username, a password of at
  least 8 characters, and a fresh challenge.
- Old projects are absent after sign-in: verify `CATMASTER_PROJECT_SPACE_ROOT`
  and username; do not place projects at the wrong root level.
- An old `.catmaster` project is rejected: migrate to `files/` plus `metadata/`.
- Thread history is missing: check `metadata/threads/` and DeepAgent SQLite in
  the restored backup.
- Skill Evolution is absent: make sure the service is not `--no-login` and mode
  is not `off`.

## 11.7 Attachments and previews

- Composer rejects a file: check the 64 MiB browser limit first.
- File is stored but model did not see it: check the 32 MiB inline limit, model
  multimodal capability, and `multimodal.prepared` warnings.
- PDF or Office content is incomplete: check the 50 MiB, 20 page/slide, 60,000
  character, and spreadsheet limits.
- Legacy Office file is stored only: convert to PDF, DOCX, XLSX, or PPTX.
- JSmol is blank: check cache, browser console, and structure format instead of
  restarting a remote task.
- Uploaded file changed unexpectedly: Files overwrites names; restore from an
  external backup.
- `metadata/` was deleted: stop writes and restore a consistent backup. Uploading
  `files/` again cannot reconstruct the thread.

## 11.8 Literature Review

- `agent-browser` unavailable: run `agent-browser doctor --offline --quick`, then
  `agent-browser mcp --help`.
- Sign-in page or CAPTCHA: switch to headed mode for the user; do not repeatedly
  automate it.
- DOI found but no full text: record evidence level, check an authorized
  institutional session, or have the user upload legitimate full text.
- Citation metadata conflict: prioritize DOI/publisher records and the document,
  while recording version differences.
- Local corpus misses a document: inspect ingest manifest, parse status, and
  parser limits.

## 11.9 Remote catalog or connection failure

Diagnose by layer:

1. Do all four active config files exist?
2. Does an active filename mistakenly contain `template`?
3. Does YAML parse, and is a key overwritten by another active file?
4. Does machine SSH work in BatchMode?
5. Is `remote_root` writable?
6. Are resource machine, queue, audience, and `source_list` correct?
7. Are task and backend enabled?
8. Does the worker audience match?

`command not found` or code 127 commonly comes from `machine.env_setup`,
`source_list`, or the task binary. Run `command -v` in the same noninteractive
SSH environment instead of changing a scientific stage to hide the environment
failure.

## 11.10 Remote run or result failure

- Tool call still pending: wait; do not poll the receipt or resubmit.
- SSH disconnected: preserve receipt identities and check the scheduler.
- Scheduler completed but no results: download finished tasks and terminated
  logs; inspect backward files and permissions.
- `status.json` says success but science did not converge: classify it as a
  scientific failure from logs and domain QC.
- Partial batch failure: classify each first-level stage and do not recompute
  successful children.
- Need to stop: WebUI Stop does not cancel the remote job; an administrator uses
  scheduler controls for the receipt's job ID.
- Need remote cleanup: first confirm results, stdout, stderr, and receipt are
  stored locally.

See [Remote machines and execution](08-remote-execution.en.md) for recovery
commands and order.

## 11.11 Current UI limitations

- No historical run selector.
- No thread rename, delete, branch, or retry UI.
- Interrupted state must resume from the message approval card; composer
  `Respond` is not approval resume.
- Monitor overview may represent the current or latest workspace/lane run, not
  exactly the selected thread.
- Files overwrites same-named uploads and deletes recursively and permanently.
- Files exposes `metadata/` without a dedicated protection switch.
- The backend supports safe ZIP extraction, but Files has no extraction switch.
- WebUI Stop does not cancel submitted remote jobs.
- Skill Evolution is account-mode only and loads on the next run.

These limits are listed so users choose the correct path, not as permission to
bypass safety boundaries.

## 11.12 Advanced thread API example

This example is only for a loopback `--no-login` test service. Modern entrypoints
use workspace/thread/artifact APIs; old run APIs are compatibility and diagnostic
paths.

```bash
curl -s http://127.0.0.1:7991/api/bootstrap

THREAD_ID="$(
  curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"title":"CO adsorption","entrypoint":"experiment","permission_mode":"hitl"}' \
    http://127.0.0.1:7991/api/workspaces/admin/threads |
  jq -r '.thread.thread_id'
)"

curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"text":"Inspect structures/slab.vasp and prepare three adsorption structures.","entrypoint":"experiment","permission_mode":"hitl"}' \
  "http://127.0.0.1:7991/api/threads/$THREAD_ID/submit"

curl -N \
  "http://127.0.0.1:7991/api/threads/$THREAD_ID/stream?last_seq=0"
```

Account mode requires the correct session cookie and access context. Do not turn
the no-login example into public automation.

## 11.13 Acceptance checklist

### Local control plane

- [ ] `conda env create/update` succeeds.
- [ ] LLM YAML parses offline.
- [ ] The API key reaches the WebUI process.
- [ ] The WebUI explicitly listens on the expected address and port.
- [ ] Registration, sign-in, and user isolation pass.
- [ ] A workspace has both `files/` and `metadata/`.
- [ ] Thread conversation, SSE, artifacts, and Monitor work.
- [ ] `agent-browser` doctor passes or that route is explicitly disabled.
- [ ] JSmol, PDF, structure, and table previews pass as required.

### Remote execution

- [ ] Four active DPDispatcher configs exist outside Git.
- [ ] SSH, remote root, scheduler, and environment scripts pass.
- [ ] Task/resource/audience/backend catalog matches installed software.
- [ ] `python scripts/remote_execution_smoke.py --list` works.
- [ ] One minimal real case passes for every enabled engine class.
- [ ] Each stage returns status, stdout, stderr, and receipt.
- [ ] Receipt-driven download and failure classification have been rehearsed.

### Operations and security

- [ ] Service is not directly public by default; `--no-login` is loopback only.
- [ ] TLS, VPN, or external access control is configured.
- [ ] Projects, account database, private configs, and secrets are backed up.
- [ ] Upgrade, rollback, and log-retention procedures are tested.
- [ ] Users know Stop does not cancel remote work and completed jobs still need
  scientific QC.
