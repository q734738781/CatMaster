#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/package_remote_deploy.sh [options]

Options:
  --output-dir DIR
      Directory for the generated archive and checksum.
      Default: dist

  --archive-name NAME
      Archive file name. ".tar.gz" is appended when no tar extension is given.
      Default: catmaster_deploy_<YYYYMMDD_HHMMSS>.tar.gz

  --package-root NAME
      Top-level directory name inside the archive.
      Default: CatMaster_Deploy

  --skip-frontend-build
      Do not rebuild catmaster/webui/static from catmaster/webui/frontend.

  --include-tests
      Include tests/ in the archive.

  --include-demos
      Include demos/ in the archive.

  By default, the archive includes only public DPDispatcher template configs
  under configs/dpdispatcher/*_template.yaml. Private deployment configs
  stay excluded.

  --include-path RELPATH
      Add another repository-relative file or directory to the archive.
      May be used more than once.

  --force
      Overwrite an existing archive/checksum with the same name.

  --keep-stage
      Keep the temporary staging directory for inspection.

  --no-verify
      Skip post-package tar/checksum/private-file verification.

  -h, --help
      Show this help.
EOF
}

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    exit 1
  fi
}

absolute_dir() {
  local dir="$1"
  mkdir -p "$dir"
  cd "$dir" && pwd
}

validate_package_root() {
  local name="$1"
  if [[ -z "$name" || "$name" == */* || "$name" == "." || "$name" == ".." ]]; then
    echo "Invalid --package-root value: $name" >&2
    exit 2
  fi
}

validate_extra_path() {
  local rel="$1"
  if [[ -z "$rel" || "$rel" == /* || "$rel" == *".."* ]]; then
    echo "Invalid --include-path value, use a repository-relative path: $rel" >&2
    exit 2
  fi
  if [[ ! -e "$REPO_ROOT/$rel" ]]; then
    echo "Included path does not exist: $rel" >&2
    exit 2
  fi
}

git_commit() {
  if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown"
  else
    echo "unknown"
  fi
}

git_source_status() {
  if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "unknown"
    return
  fi

  local output_rel=""
  if [[ "$OUTPUT_DIR" == "$REPO_ROOT" ]]; then
    output_rel=""
  elif [[ "$OUTPUT_DIR" == "$REPO_ROOT"/* ]]; then
    output_rel="${OUTPUT_DIR#$REPO_ROOT/}"
  fi

  local status=""
  if [[ -n "$output_rel" ]]; then
    status="$(git -C "$REPO_ROOT" status --short --untracked-files=all -- . ":(exclude)$output_rel" 2>/dev/null || true)"
  else
    status="$(git -C "$REPO_ROOT" status --short --untracked-files=all 2>/dev/null || true)"
  fi

  if [[ -n "$status" ]]; then
    echo "dirty"
  else
    echo "clean"
  fi
}

build_frontend() {
  local deploy_cache_dir="${CATMASTER_DEPLOY_CACHE_DIR:-${TMPDIR:-/tmp}/catmaster-deploy-cache}"
  mkdir -p "$deploy_cache_dir"
  if [[ -f "$REPO_ROOT/scripts/install_jsmol_assets.py" ]]; then
    XDG_CACHE_HOME="$deploy_cache_dir" python3 "$REPO_ROOT/scripts/install_jsmol_assets.py" --quiet
  fi

  local frontend_dir="$REPO_ROOT/catmaster/webui/frontend"
  if [[ -f "$frontend_dir/package.json" ]]; then
    require_command npm
    echo "Building WebUI bundle..."
    (cd "$frontend_dir" && XDG_CACHE_HOME="$deploy_cache_dir" npm run build)
    echo
  fi
}

copy_runtime_path() {
  local rel="$1"
  local src="$REPO_ROOT/$rel"
  local dst="$PKG_ROOT/$rel"

  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    rsync "${RSYNC_ARGS[@]}" "$src/" "$dst/"
  elif [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    rsync -a "$src" "$dst"
  else
    echo "Skipping missing path: $rel" >&2
  fi
}

write_deploy_info() {
  cat > "$PKG_ROOT/.deploy_info" <<EOF
source_repo=$REPO_ROOT
source_commit=$(git_commit)
source_status=$(git_source_status)
packaged_at_local=$(date '+%Y-%m-%dT%H:%M:%S%z')
package_profile=runtime-webui-deploy
package_root=$PACKAGE_ROOT_NAME
archive_name=$ARCHIVE_NAME
included_config_templates=configs/dpdispatcher/*_template.yaml,configs/llm*.template.yaml
excluded_private_configs=configs/dpdispatcher/{machines,resources,tasks,mlff_backends}.yaml
excluded_private_files=.env,.sesskey
EOF
}

write_deploy_readme() {
  {
    printf '%s\n' '# CatMaster Remote Deployment'
    printf '\n'
    printf '%s\n' 'This archive contains a runtime-oriented CatMaster checkout with the rebuilt WebUI static bundle. Local secrets, logs, project spaces, caches, node_modules, `.git`, and private active config files are intentionally excluded.'
    printf '\n'
    printf '%s\n' '## 1. Unpack'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' "tar -xzf $ARCHIVE_NAME"
    printf '%s\n' "cd $PACKAGE_ROOT_NAME"
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' '## 2. Create Python environment'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'conda env create -f requirements/pc-conda.yml'
    printf '%s\n' 'conda activate catmaster'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'Do not install `requirements/mace.txt`, `requirements/uma.txt`, `requirements/mattersim.txt`, or `requirements/orb.txt` into the `catmaster` control-plane environment. Create a separate remote environment for each enabled MLFF provider and connect it through the corresponding resource `source_list`; use `configs/dpdispatcher/env_templates/` as the activation-script reference. If the GPU host requires a proxy, copy and edit `catmaster_env_proxy.sh`, then keep it before the provider activation script in `source_list`; remove that entry on hosts that need no proxy.'
    printf '\n'
    printf '%s\n' 'For a local or desktop deployment that uses the Literature Review browser path, install the pinned agent-browser CLI as well:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'npm install -g agent-browser@0.31.1'
    printf '%s\n' 'agent-browser install'
    printf '%s\n' 'agent-browser doctor --offline --quick'
    printf '%s\n' 'agent-browser mcp --help'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'CatMaster starts the MCP subprocess. Browser profiles, cookies, credentials, OTPs, and exported session state are local machine data and are not included in this archive. A headless remote deployment without a user-authenticated browser can still use previously ingested local literature files, but cannot claim institution-authorized browser access.'
    printf '\n'
    printf '%s\n' 'Create a separate environment for FairChem UMA. Do not install UMA into the MACE environment:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'conda create -n catmaster-uma python=3.11 -y'
    printf '%s\n' 'conda activate catmaster-uma'
    printf '%s\n' 'pip install -r requirements/uma.txt'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' '## 3. Add local secrets and remote resources'
    printf '\n'
    printf '%s\n' 'The archive includes public DPDispatcher templates under `configs/dpdispatcher/*_template.yaml`, but excludes deployment-specific active configs: `machines.yaml`, `resources.yaml`, `tasks.yaml`, and `mlff_backends.yaml`.'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml'
    printf '%s\n' 'cp configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources.yaml'
    printf '%s\n' 'cp configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks.yaml'
    printf '%s\n' 'cp configs/dpdispatcher/mlff_backends_template.yaml configs/dpdispatcher/mlff_backends.yaml'
    printf '%s\n' '# choose one LLM template, then edit as needed:'
    printf '%s\n' '# cp configs/llm.template.yaml configs/llm.yaml'
    printf '%s\n' '# cp configs/llm.full.template.yaml configs/llm.yaml'
    printf '%s\n' '# cp configs/llm_codex_oauth.template.yaml configs/llm.yaml'
    printf '%s\n' '# provide configs/llm.yaml or export provider keys such as OPENROUTER_API_KEY'
    printf '%s\n' '# edit configs/dpdispatcher/{machines,resources,tasks,mlff_backends}.yaml for your cluster'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'Keep real API keys and SSH credentials out of the archive. Use environment variables, local ignored files, or machine-level secret management.'
    printf '\n'
    cat <<'EOF'
### Optional Codex OAuth local profile

The standard `configs/llm.template.yaml` profile uses OpenRouter. Codex OAuth is an optional alternative for a single system user and is not enabled unless you copy its template deliberately.

The archive includes `configs/llm_codex_oauth.template.yaml` for this path. It uses the pinned `langchain-openai` Codex OAuth model (`langchain_openai.chat_models.codex._ChatOpenAICodex`) and defaults to:

- provider: `codex_oauth`
- model: `gpt-5.6-sol`
- `provider_options.codex_oauth.chat_kwargs.reasoning.effort: xhigh`
- `provider_options.codex_oauth.chat_kwargs.verbosity: medium`

The OAuth token provider stores credentials under the user home directory. Do not package, commit, or share those credentials, and do not use this profile for shared multi-user hosting.

#### 1. Activate the environment

```bash
conda activate catmaster
python -m pip show langchain-openai
```

#### 2. Login once on the target machine

For a local desktop session:

```bash
python - <<'PY'
from langchain_openai.chatgpt_oauth import login_chatgpt

login_chatgpt()
PY
```

For SSH, headless, or remote machines, use the device-code flow and open the printed verification URL in your browser:

```bash
python - <<'PY'
from langchain_openai.chatgpt_oauth import login_chatgpt_device

login_chatgpt_device()
PY
```

The account must have ChatGPT Plus/Pro or another plan with Codex access. If the upstream Codex consumer endpoint changes, upgrade the pinned `langchain-openai` version after testing the tool-calling smoke workflow.

#### 3. Enable the OAuth LLM profile

```bash
cp configs/llm_codex_oauth.template.yaml configs/llm.yaml
```

The important part of the template is:

```yaml
models:
  codex-oauth-main:
    provider: codex_oauth
    model: gpt-5.6-sol
    provider_options:
      codex_oauth:
        chat_kwargs:
          reasoning:
            effort: xhigh
            summary: auto
          verbosity: medium
```

CatMaster passes Codex-specific knobs through `provider_options.codex_oauth.chat_kwargs` to `_ChatOpenAICodex`. Legacy `text_verbosity` is mapped to `verbosity`, and legacy `system_prompt_mode` is ignored because `_ChatOpenAICodex` lifts `SystemMessage` content into the Responses API `instructions` field itself.

#### 4. Verify config loading without calling the model

```bash
python - <<'PY'
from catmaster.llm.config import LLMProfile

profile = LLMProfile.from_env_or_file("configs/llm.yaml")
cfg = profile.config_for_role("task_runner")
print(cfg.provider)
print(cfg.model)
print(cfg.provider_options)
PY
```

Expected values are `codex_oauth`, `gpt-5.6-sol`, and a `chat_kwargs` mapping containing `reasoning.effort: xhigh` and `verbosity: medium`.

#### 5. Optional live smoke test

This makes a real Codex request and consumes account usage:

```bash
python - <<'PY'
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model

profile = LLMProfile.from_env_or_file("configs/llm.yaml")
model = build_chat_model(profile.config_for_role("task_runner"))
resp = model.invoke("Reply with exactly: codex oauth ok")
print(getattr(resp, "content", resp))
PY
```

#### Troubleshooting

- `No ChatGPT OAuth token found` or auth refresh errors: rerun the `login_chatgpt_device()` command above.
- If you previously logged in with `langchain-codex-oauth`, CatMaster can temporarily read the old `~/.langchain-codex-oauth/auth/openai.json` token when it is still valid, but this is a migration fallback only. Re-login with `langchain_openai.chatgpt_oauth` for refreshable credentials.
- Model or authorization errors: confirm the account has Codex access; if necessary, edit `model: gpt-5.6-sol` to another Codex-supported model.
- Usage-limit errors: wait for the ChatGPT/Codex quota window or switch back to the OpenRouter/API-key profile.
- Never copy the OAuth credential directory into the deployment archive. Re-authenticate per target user/machine.

EOF
    printf '%s\n' '### PySR Julia backend'
    printf '\n'
    printf '%s\n' '`pysr` is installed in the CatMaster environment, but the Julia backend must be available before symbolic-regression tools run. On first import, `juliacall`/`juliapkg` may download Julia and precompile `SymbolicRegression.jl`; do this during deployment, not during an agent run:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'conda activate catmaster'
    printf '%s\n' 'python scripts/pysr_julia_smoke.py --fit'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'For offline or firewalled machines, install Julia yourself and point JuliaCall at it before running the smoke test:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'export PYTHON_JULIACALL_BINDIR=/opt/julia/bin'
    printf '%s\n' 'python scripts/pysr_julia_smoke.py --julia-bindir "$PYTHON_JULIACALL_BINDIR" --fit'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'For UMA, keep `HF_TOKEN` only in the remote UMA source script or a remote secret file, never in task params or staged files. Point `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `TORCH_HOME` to persistent remote cache directories. If compute nodes have no internet access, prewarm the model cache once before enabling `HF_HUB_OFFLINE=1`:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'mkdir -p "$HOME/.config/huggingface"'
    printf '%s\n' 'chmod 700 "$HOME/.config/huggingface"'
    printf '%s\n' "printf '%s' '<hf_token_with_facebook_UMA_access>' > \"\$HOME/.config/huggingface/token\""
    printf '%s\n' 'chmod 600 "$HOME/.config/huggingface/token"'
    printf '%s\n' 'source /path/to/catmaster_env_uma.sh'
    printf '%s\n' 'python - <<'"'"'PY'"'"''
    printf '%s\n' 'from fairchem.core import pretrained_mlip'
    printf '%s\n' 'pretrained_mlip.get_predict_unit("uma-s-1p2", device="cpu")'
    printf '%s\n' 'PY'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' '## 4. Start WebUI'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'mkdir -p ~/catmaster_projects'
    printf '%s\n' 'CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects CATMASTER_HOST=127.0.0.1 CATMASTER_PORT=7991 ./start_webui.sh --start'
    printf '%s\n' './start_webui.sh --status'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'Keep the application on loopback. From your workstation, run `ssh -L 7991:127.0.0.1:7991 <user>@<remote-host>` and open `http://127.0.0.1:7991`. A shared deployment needs a TLS reverse proxy and an external network or identity boundary; never publish `--no-login` mode. After at least one account exists, use `--disable-registration` or `CATMASTER_DISABLE_REGISTRATION=1` to keep login required while closing public signup.'
    printf '\n'
    printf '%s\n' '## 5. Verify remote execution'
    printf '\n'
    printf '%s\n' 'After `configs/dpdispatcher/{machines,resources,tasks,mlff_backends}.yaml` are edited, inspect the catalog and then submit one real minimal smoke job:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'python scripts/remote_execution_smoke.py --list'
    printf '%s\n' 'python scripts/remote_execution_smoke.py --case mace_sp --check-interval 30 --stop-on-failure'
    printf '%s\n' '# broader coverage:'
    printf '%s\n' '# python scripts/remote_execution_smoke.py --suite core --check-interval 30'
    printf '%s\n' '# python scripts/remote_execution_smoke.py --suite all --check-interval 60'
    printf '%s\n' '# UMA is isolated because it needs FairChem, Hugging Face access, and model cache; the suite covers SP and short relax jobs:'
    printf '%s\n' '# python scripts/remote_execution_smoke.py --suite uma --uma-model uma-s-1p2 --uma-task omat --uma-check-interval 60'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'These commands submit actual DPDispatcher calculations. The default report root is `/tmp/catmaster_remote_execution_smoke`.'
    printf '\n'
    printf '%s\n' '## Notes'
    printf '\n'
    printf '%s\n' '- The launcher records the real `python -m catmaster.webui` PID in `.runtime/webui.pid`.'
    printf '%s\n' '- Use a different `CATMASTER_PORT` if another service already owns the default port.'
    printf '%s\n' '- Project outputs stay under `CATMASTER_PROJECT_SPACE_ROOT`, not inside this package unless you choose that path.'
  } > "$PKG_ROOT/DEPLOY_REMOTE.md"
}

verify_archive() {
  local tar_list
  tar -tzf "$ARCHIVE_PATH" >/dev/null
  tar_list="$(tar -tzf "$ARCHIVE_PATH")"

  local private_pattern='(^|/)(\.git|\.env$|\.sesskey|dpdispatcher\.log|node_modules|\.runtime|project_space|workspace)(/|$)'
  local private_hits=""
  local config_hits=""
  local env_hits=""
  private_hits="$(printf '%s\n' "$tar_list" | grep -E "$private_pattern" || true)"
  config_hits="$(printf '%s\n' "$tar_list" \
    | grep -E "(^|/)configs(/|$)" \
    | grep -v -E "^$PACKAGE_ROOT_NAME/configs/?$|^$PACKAGE_ROOT_NAME/configs/dpdispatcher/?$|^$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/?$|^$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/[^/]+\.sh$|^$PACKAGE_ROOT_NAME/configs/dpdispatcher/(machines_template|resources_template|tasks_template|mlff_backends_template)\.yaml$|^$PACKAGE_ROOT_NAME/configs/(llm\.template|llm\.full\.template|llm_codex_oauth\.template|tool_output|tool_policy)\.yaml$" \
    || true)"
  env_hits="$(printf '%s\n' "$tar_list" | grep -E '(^|/)\.env($|\.)' | grep -v -E '(^|/)\.env\.example$' || true)"
  if [[ -n "$private_hits" || -n "$config_hits" || -n "$env_hits" ]]; then
    echo "Archive contains private or runtime-only paths:" >&2
    printf '%s\n' "$private_hits" "$config_hits" "$env_hits" | sed '/^$/d' >&2
    exit 1
  fi

  local required=(
    "$PACKAGE_ROOT_NAME/start_webui.sh"
    "$PACKAGE_ROOT_NAME/DEPLOY_REMOTE.md"
    "$PACKAGE_ROOT_NAME/.deploy_info"
    "$PACKAGE_ROOT_NAME/.env.example"
    "$PACKAGE_ROOT_NAME/scripts/remote_execution_smoke.py"
    "$PACKAGE_ROOT_NAME/catmaster/tools/execution/dpdispatcher_runner.py"
    "$PACKAGE_ROOT_NAME/catmaster/tools/execution/remote_submission.py"
    "$PACKAGE_ROOT_NAME/catmaster/tools/execution/mlff_specs.py"
    "$PACKAGE_ROOT_NAME/catmaster/tools/execution/mlff_stage.py"
    "$PACKAGE_ROOT_NAME/catmaster/remote/cpu/k8s_vasp_boot.py"
    "$PACKAGE_ROOT_NAME/catmaster/remote/mlff/mlff_common.py"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/machines_template.yaml"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/resources_template.yaml"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/tasks_template.yaml"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/mlff_backends_template.yaml"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/catmaster_env_proxy.sh"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/catmaster_env_mace.sh"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/catmaster_env_uma.sh"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/catmaster_env_mattersim.sh"
    "$PACKAGE_ROOT_NAME/configs/dpdispatcher/env_templates/catmaster_env_orb.sh"
    "$PACKAGE_ROOT_NAME/requirements/mace.txt"
    "$PACKAGE_ROOT_NAME/requirements/uma.txt"
    "$PACKAGE_ROOT_NAME/requirements/mattersim.txt"
    "$PACKAGE_ROOT_NAME/requirements/orb.txt"
    "$PACKAGE_ROOT_NAME/configs/llm.template.yaml"
    "$PACKAGE_ROOT_NAME/configs/llm.full.template.yaml"
    "$PACKAGE_ROOT_NAME/configs/llm_codex_oauth.template.yaml"
    "$PACKAGE_ROOT_NAME/catmaster/webui/static/app.js"
    "$PACKAGE_ROOT_NAME/catmaster/webui/static/app.css"
  )
  local rel
  for rel in "${required[@]}"; do
    if ! printf '%s\n' "$tar_list" | grep -Fx "$rel" >/dev/null; then
      echo "Archive is missing required path: $rel" >&2
      exit 1
    fi
  done

  (cd "$OUTPUT_DIR" && sha256sum -c "$ARCHIVE_NAME.sha256")
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$REPO_ROOT/dist"
ARCHIVE_NAME="catmaster_deploy_${TS}.tar.gz"
PACKAGE_ROOT_NAME="CatMaster_Deploy"
SKIP_FRONTEND_BUILD=0
INCLUDE_TESTS=0
INCLUDE_DEMOS=0
KEEP_STAGE=0
VERIFY_ARCHIVE=1
FORCE=0
EXTRA_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --archive-name)
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    --package-root)
      PACKAGE_ROOT_NAME="$2"
      shift 2
      ;;
    --skip-frontend-build)
      SKIP_FRONTEND_BUILD=1
      shift
      ;;
    --include-tests)
      INCLUDE_TESTS=1
      shift
      ;;
    --include-demos)
      INCLUDE_DEMOS=1
      shift
      ;;
    --include-path)
      EXTRA_PATHS+=("$2")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --keep-stage)
      KEEP_STAGE=1
      shift
      ;;
    --no-verify)
      VERIFY_ARCHIVE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

validate_package_root "$PACKAGE_ROOT_NAME"
for rel in "${EXTRA_PATHS[@]}"; do
  validate_extra_path "$rel"
done

case "$ARCHIVE_NAME" in
  *.tar.gz|*.tgz)
    ;;
  *)
    ARCHIVE_NAME="${ARCHIVE_NAME}.tar.gz"
    ;;
esac

OUTPUT_DIR="$(absolute_dir "$OUTPUT_DIR")"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"

if [[ $FORCE -eq 0 && ( -e "$ARCHIVE_PATH" || -e "$CHECKSUM_PATH" ) ]]; then
  echo "Refusing to overwrite existing archive/checksum: $ARCHIVE_PATH" >&2
  echo "Use --force or choose another --archive-name." >&2
  exit 1
fi

require_command rsync
require_command tar
require_command sha256sum

if [[ $SKIP_FRONTEND_BUILD -eq 0 ]]; then
  build_frontend
fi

STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/catmaster_deploy_stage.XXXXXX")"
PKG_ROOT="$STAGE_ROOT/$PACKAGE_ROOT_NAME"
mkdir -p "$PKG_ROOT"

cleanup() {
  if [[ $KEEP_STAGE -eq 0 ]]; then
    rm -rf "$STAGE_ROOT"
  else
    echo "Stage kept: $PKG_ROOT"
  fi
}
trap cleanup EXIT

RSYNC_ARGS=(
  -a
  --exclude='.git/'
  --exclude='.idea/'
  --exclude='.vscode/'
  --exclude='.pytest_cache/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.venv/'
  --exclude='node_modules/'
  --exclude='project_space/'
  --exclude='workspace/'
  --exclude='logs/'
  --exclude='.runtime/'
  --exclude='dpdispatcher.log'
  --exclude='.sesskey'
  --include='.env.example'
  --exclude='.env'
  --exclude='.env.*'
  --exclude='configs/'
  --exclude='configs/llm.yaml'
  --exclude='configs/llm_*.yaml'
  --exclude='llm.yaml'
  --exclude='llm_*.yaml'
  --exclude='configs/*.local.yaml'
  --exclude='*.local.yaml'
  --exclude='configs/dpdispatcher/machines.yaml'
  --exclude='configs/dpdispatcher/resources.yaml'
  --exclude='configs/dpdispatcher/tasks.yaml'
  --exclude='dpdispatcher/machines.yaml'
  --exclude='dpdispatcher/resources.yaml'
  --exclude='dpdispatcher/tasks.yaml'
  --exclude='configs/dpdispatcher/*.local.yaml'
  --exclude='dpdispatcher/*.local.yaml'
  --exclude='POTCAR'
  --exclude='POTCAR.*'
  --exclude='WAVECAR'
  --exclude='CHGCAR'
  --exclude='vasprun.xml'
)

RUNTIME_PATHS=(
  "catmaster"
  "requirements"
  "skills"
  "scripts"
  "docs"
  "configs/dpdispatcher/machines_template.yaml"
  "configs/dpdispatcher/resources_template.yaml"
  "configs/dpdispatcher/tasks_template.yaml"
  "configs/dpdispatcher/mlff_backends_template.yaml"
  "configs/dpdispatcher/env_templates"
  "configs/llm.template.yaml"
  "configs/llm.full.template.yaml"
  "configs/llm_codex_oauth.template.yaml"
  "configs/tool_output.yaml"
  "configs/tool_policy.yaml"
  "main.py"
  "README.md"
  "LICENSE"
  "AGENTS.md"
  ".env.example"
  "start_webui.sh"
)

if [[ $INCLUDE_TESTS -eq 1 ]]; then
  RUNTIME_PATHS+=("tests")
fi
if [[ $INCLUDE_DEMOS -eq 1 ]]; then
  RUNTIME_PATHS+=("demos")
fi
RUNTIME_PATHS+=("${EXTRA_PATHS[@]}")

echo "Source: $REPO_ROOT"
echo "Package root: $PACKAGE_ROOT_NAME"
echo "Output archive: $ARCHIVE_PATH"
echo

for rel in "${RUNTIME_PATHS[@]}"; do
  copy_runtime_path "$rel"
done

write_deploy_info
write_deploy_readme
chmod +x "$PKG_ROOT/start_webui.sh"
bash -n "$PKG_ROOT/start_webui.sh"

tar -C "$STAGE_ROOT" -czf "$ARCHIVE_PATH" "$PACKAGE_ROOT_NAME"
(cd "$OUTPUT_DIR" && sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256")

if [[ $VERIFY_ARCHIVE -eq 1 ]]; then
  verify_archive
fi

echo
echo "Package created."
echo "Archive:  $ARCHIVE_PATH"
echo "Checksum: $CHECKSUM_PATH"
du -h "$ARCHIVE_PATH" "$CHECKSUM_PATH"
