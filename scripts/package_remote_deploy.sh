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
  if [[ -f "$REPO_ROOT/scripts/install_jsmol_assets.py" ]]; then
    python3 "$REPO_ROOT/scripts/install_jsmol_assets.py" --quiet
  fi

  local frontend_dir="$REPO_ROOT/catmaster/webui/frontend"
  if [[ -f "$frontend_dir/package.json" ]]; then
    require_command npm
    echo "Building WebUI bundle..."
    (cd "$frontend_dir" && npm run build)
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
    rsync "${RSYNC_ARGS[@]}" "$src" "$dst"
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
excluded_private_configs=configs/llm.yaml,configs/dpdispatcher/machines.yaml,configs/dpdispatcher/resources.yaml,configs/dpdispatcher/tasks.yaml,.env,.sesskey
EOF
}

write_deploy_readme() {
  {
    printf '%s\n' '# CatMaster Remote Deployment'
    printf '\n'
    printf '%s\n' 'This archive contains a runtime-oriented CatMaster checkout with the rebuilt WebUI static bundle. Local secrets, logs, project spaces, caches, node_modules, `.git`, `configs/llm.yaml`, and active DPDispatcher deployment files (`configs/dpdispatcher/machines.yaml`, `resources.yaml`, `tasks.yaml`) are intentionally excluded.'
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
    printf '%s\n' 'conda create -n catmaster python=3.11 -y'
    printf '%s\n' 'conda activate catmaster'
    printf '%s\n' 'pip install -r requirements/pc.txt'
    printf '%s\n' '# Optional MACE/GPU add-on for machines that execute local or remote GPU boot scripts:'
    printf '%s\n' '# pip install -r requirements/gpu.txt'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' '## 3. Configure local secrets and remote resources'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'cp configs/llm.template.yaml configs/llm.yaml'
    printf '%s\n' '# edit configs/llm.yaml or export provider keys such as OPENROUTER_API_KEY'
    printf '%s\n' 'cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml'
    printf '%s\n' 'cp configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources.yaml'
    printf '%s\n' 'cp configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks.yaml'
    printf '%s\n' '# edit machines.yaml for cluster login/paths/env_setup'
    printf '%s\n' '# edit resources.yaml for queues, source_list, prepend_script, and CPU/GPU counts'
    printf '%s\n' '# edit tasks.yaml only when task commands or resource bindings differ'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'Keep real API keys and SSH credentials out of the archive. Use environment variables, local ignored files, or machine-level secret management.'
    printf '\n'
    printf '%s\n' '## 4. Start WebUI'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'mkdir -p ~/catmaster_projects'
    printf '%s\n' 'CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects CATMASTER_HOST=0.0.0.0 CATMASTER_PORT=7990 ./start_webui.sh --start'
    printf '%s\n' './start_webui.sh --status'
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' 'Open `http://<remote-host>:7990`. If the server uses a firewall, expose the chosen port explicitly.'
    printf '\n'
    printf '%s\n' '## 5. Verify remote execution'
    printf '\n'
    printf '%s\n' 'After `configs/dpdispatcher/{machines,resources,tasks}.yaml` are edited, submit real smoke jobs:'
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' 'python scripts/remote_execution_smoke.py --list'
    printf '%s\n' 'python scripts/remote_execution_smoke.py --suite core --check-interval 30'
    printf '%s\n' '# broader coverage:'
    printf '%s\n' '# python scripts/remote_execution_smoke.py --suite no_cp2k --check-interval 60'
    printf '%s\n' '# use --suite all only after CP2K is configured'
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

  local private_pattern='(^|/)(\.git|\.env$|\.sesskey|dpdispatcher\.log|node_modules|\.runtime|project_space|workspace)(/|$)|(^|/)configs/llm\.yaml$|(^|/)configs/dpdispatcher/(machines|resources|tasks)\.yaml$'
  local private_hits=""
  local env_hits=""
  private_hits="$(printf '%s\n' "$tar_list" | grep -E "$private_pattern" || true)"
  env_hits="$(printf '%s\n' "$tar_list" | grep -E '(^|/)\.env($|\.)' | grep -v -E '(^|/)\.env\.example$' || true)"
  if [[ -n "$private_hits" || -n "$env_hits" ]]; then
    echo "Archive contains private or runtime-only paths:" >&2
    printf '%s\n' "$private_hits" "$env_hits" | sed '/^$/d' >&2
    exit 1
  fi

  local required=(
    "$PACKAGE_ROOT_NAME/start_webui.sh"
    "$PACKAGE_ROOT_NAME/DEPLOY_REMOTE.md"
    "$PACKAGE_ROOT_NAME/.deploy_info"
    "$PACKAGE_ROOT_NAME/.env.example"
    "$PACKAGE_ROOT_NAME/scripts/remote_execution_smoke.py"
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
  --exclude='configs/llm.yaml'
  --exclude='llm.yaml'
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
  "configs"
  "requirements"
  "skills"
  "scripts"
  "docs"
  "main.py"
  "README.md"
  "LICENSE"
  "AGENTS.MD"
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
