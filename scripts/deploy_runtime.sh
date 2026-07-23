#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_runtime.sh [--target DIR] [--project-space-root DIR] [--dry-run] [--no-delete] [--full-repo] [--sync-configs] [--skip-frontend-build] [--sync-start-webui] [--autorun|--no-autorun]

The default runtime sync includes the complete DPDispatcher execution stack:
Python runners, MLFF backend code, remote boot scripts, and provider
requirements. Existing target configs are preserved by default. A new target
without configs is initialized from the source checkout.

Options:
  --target DIR
      Runtime checkout directory to update.
      Default: ../CatMaster_Run (relative to repo root)

  --project-space-root DIR
      Project-space root that runtime WebUI should use.
      Default: <target>/project_space

  --dry-run
      Show rsync changes without writing files.

  --no-delete
      Do not delete files in target that are removed from source.

  --full-repo
      Sync full repository instead of runtime-only paths.

  --sync-configs
      Overwrite the target configs directory from this checkout. By default,
      an existing target configs directory is preserved.

  --skip-frontend-build
      Do not rebuild catmaster/webui/static from catmaster/webui/frontend before deploy.

  --sync-start-webui
      Overwrite target start_webui.sh from this checkout. By default, deploy
      preserves an existing target launcher and only initializes it if missing.

  --autorun
      Start runtime WebUI automatically after deployment (default).

  --no-autorun
      Do not start runtime WebUI automatically after deployment.
EOF
}

verify_runtime_target() {
  local required=(
    "catmaster/tools/execution/dpdispatcher_runner.py"
    "catmaster/tools/execution/remote_submission.py"
    "catmaster/tools/execution/mlff_specs.py"
    "catmaster/tools/execution/mlff_stage.py"
    "catmaster/remote/cpu/k8s_vasp_boot.py"
    "catmaster/remote/mlff/mlff_common.py"
    "configs/dpdispatcher/machines_template.yaml"
    "configs/dpdispatcher/resources_template.yaml"
    "configs/dpdispatcher/tasks_template.yaml"
    "configs/dpdispatcher/mlff_backends_template.yaml"
    "configs/dpdispatcher/env_templates/catmaster_env_proxy.sh"
    "configs/dpdispatcher/env_templates/catmaster_env_mace.sh"
    "configs/dpdispatcher/env_templates/catmaster_env_uma.sh"
    "configs/dpdispatcher/env_templates/catmaster_env_mattersim.sh"
    "configs/dpdispatcher/env_templates/catmaster_env_orb.sh"
    "requirements/mace.txt"
    "requirements/uma.txt"
    "requirements/mattersim.txt"
    "requirements/orb.txt"
    "scripts/remote_execution_smoke.py"
  )
  local missing=()
  local rel
  for rel in "${required[@]}"; do
    if [[ ! -e "$TARGET_DIR/$rel" ]]; then
      missing+=("$rel")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Deployment is missing required DPDispatcher runtime paths:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/../CatMaster_Run"
PROJECT_SPACE_ROOT=""
DRY_RUN=0
NO_DELETE=0
FULL_REPO=0
SYNC_CONFIGS=0
AUTORUN=1
SKIP_FRONTEND_BUILD=0
SYNC_START_WEBUI=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET_DIR="$2"
      shift 2
      ;;
    --project-space-root)
      PROJECT_SPACE_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-delete)
      NO_DELETE=1
      shift
      ;;
    --full-repo)
      FULL_REPO=1
      shift
      ;;
    --sync-configs)
      SYNC_CONFIGS=1
      shift
      ;;
    --skip-frontend-build)
      SKIP_FRONTEND_BUILD=1
      shift
      ;;
    --sync-start-webui)
      SYNC_START_WEBUI=1
      shift
      ;;
    --autorun)
      AUTORUN=1
      shift
      ;;
    --no-autorun)
      AUTORUN=0
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

TARGET_DIR="$(cd "$(dirname "$TARGET_DIR")" && pwd)/$(basename "$TARGET_DIR")"
mkdir -p "$TARGET_DIR"

COPY_CONFIGS=0
CONFIG_ACTION="preserve existing target configs"
if [[ $SYNC_CONFIGS -eq 1 ]]; then
  COPY_CONFIGS=1
  CONFIG_ACTION="sync configs from source (--sync-configs)"
elif [[ ! -d "$TARGET_DIR/configs" ]]; then
  COPY_CONFIGS=1
  CONFIG_ACTION="initialize configs for new target"
fi

if [[ -z "$PROJECT_SPACE_ROOT" ]]; then
  PROJECT_SPACE_ROOT="$TARGET_DIR/project_space"
else
  PROJECT_SPACE_ROOT="$(cd "$(dirname "$PROJECT_SPACE_ROOT")" && pwd)/$(basename "$PROJECT_SPACE_ROOT")"
fi

RSYNC_ARGS=(
  -a
  --human-readable
  --itemize-changes
  --exclude ".git/"
  --exclude ".idea/"
  --exclude ".vscode/"
  --exclude ".pytest_cache/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".venv/"
  --exclude "node_modules/"
  --exclude "project_space/"
  --exclude "workspace/"
  --exclude "dpdispatcher.log"
)

if [[ $NO_DELETE -eq 0 ]]; then
  RSYNC_ARGS+=(--delete)
fi
if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi
if [[ $SYNC_START_WEBUI -eq 0 ]]; then
  RSYNC_ARGS+=(--exclude "/start_webui.sh")
fi

echo "Source: $REPO_ROOT"
echo "Target: $TARGET_DIR"
echo "Project space root for launcher: $PROJECT_SPACE_ROOT"
echo "Configs: $CONFIG_ACTION"
if [[ $SYNC_START_WEBUI -eq 1 ]]; then
  echo "Launcher: sync start_webui.sh from source"
else
  echo "Launcher: preserve target start_webui.sh"
fi
echo

if [[ $DRY_RUN -eq 0 && $SKIP_FRONTEND_BUILD -eq 0 ]]; then
  DEPLOY_CACHE_DIR="${CATMASTER_DEPLOY_CACHE_DIR:-${TMPDIR:-/tmp}/catmaster-deploy-cache}"
  mkdir -p "$DEPLOY_CACHE_DIR"
  if [[ -f "$REPO_ROOT/scripts/install_jsmol_assets.py" ]]; then
    XDG_CACHE_HOME="$DEPLOY_CACHE_DIR" python3 "$REPO_ROOT/scripts/install_jsmol_assets.py" --quiet
  fi
  FRONTEND_DIR="$REPO_ROOT/catmaster/webui/frontend"
  if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
      echo "npm is required to build the WebUI bundle before deploy." >&2
      exit 1
    fi
    echo "Building WebUI bundle..."
    (cd "$FRONTEND_DIR" && XDG_CACHE_HOME="$DEPLOY_CACHE_DIR" npm run build)
    echo
  fi
fi

if [[ $FULL_REPO -eq 1 ]]; then
  if [[ $COPY_CONFIGS -eq 1 ]]; then
    rsync "${RSYNC_ARGS[@]}" "$REPO_ROOT/" "$TARGET_DIR/"
  else
    rsync "${RSYNC_ARGS[@]}" --exclude "/configs/" "$REPO_ROOT/" "$TARGET_DIR/"
  fi
else
  RUNTIME_PATHS=(
    "catmaster"
    "requirements"
    "skills"
    "scripts"
    "docs"
    "main.py"
    "README.md"
    "LICENSE"
    "AGENTS.md"
    ".env.example"
  )
  if [[ $COPY_CONFIGS -eq 1 ]]; then
    RUNTIME_PATHS+=("configs")
  fi
  if [[ $SYNC_START_WEBUI -eq 1 ]]; then
    RUNTIME_PATHS+=("start_webui.sh")
  fi
  for rel in "${RUNTIME_PATHS[@]}"; do
    src="$REPO_ROOT/$rel"
    dst="$TARGET_DIR/$rel"
    if [[ -d "$src" ]]; then
      mkdir -p "$dst"
      rsync "${RSYNC_ARGS[@]}" "$src/" "$dst/"
    elif [[ -f "$src" ]]; then
      mkdir -p "$(dirname "$dst")"
      rsync "${RSYNC_ARGS[@]}" "$src" "$dst"
    fi
  done
fi

if [[ $DRY_RUN -eq 0 ]]; then
  verify_runtime_target
  COMMIT="unknown"
  if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  fi
  DEPLOY_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  cat > "$TARGET_DIR/.deploy_info" <<EOF
source_repo=$REPO_ROOT
source_commit=$COMMIT
deployed_at_utc=$DEPLOY_TIME
deployment_profile=$([[ $FULL_REPO -eq 1 ]] && echo full-repo || echo runtime)
dpdispatcher_sync=runtime-code-requirements
configs_sync=$([[ $COPY_CONFIGS -eq 1 ]] && echo synced || echo preserved)
EOF
  if [[ ! -f "$TARGET_DIR/start_webui.sh" ]]; then
    if [[ -f "$REPO_ROOT/start_webui.sh" ]]; then
      echo "Target has no start_webui.sh; installing default launcher."
      install -m 755 "$REPO_ROOT/start_webui.sh" "$TARGET_DIR/start_webui.sh"
    else
      echo "Missing source launcher: $REPO_ROOT/start_webui.sh" >&2
      exit 1
    fi
  elif [[ $SYNC_START_WEBUI -eq 0 ]]; then
    echo "Preserved existing target launcher: $TARGET_DIR/start_webui.sh"
  fi

  LAUNCHER_CMD=(./start_webui.sh)
  LAUNCHER_DISPLAY="./start_webui.sh"
  if [[ ! -x "$TARGET_DIR/start_webui.sh" ]]; then
    LAUNCHER_CMD=(bash ./start_webui.sh)
    LAUNCHER_DISPLAY="bash ./start_webui.sh"
  fi
  mkdir -p "$PROJECT_SPACE_ROOT"

  echo
  echo "Deploy completed."
  if [[ $AUTORUN -eq 1 ]]; then
    echo "Autorun enabled. Starting WebUI now..."
    cd "$TARGET_DIR"
    if [[ "$PROJECT_SPACE_ROOT" == "$TARGET_DIR/project_space" ]]; then
      "${LAUNCHER_CMD[@]}"
    else
      CATMASTER_PROJECT_SPACE_ROOT="$PROJECT_SPACE_ROOT" "${LAUNCHER_CMD[@]}"
    fi
  else
    echo "Next run command:"
    if [[ "$PROJECT_SPACE_ROOT" == "$TARGET_DIR/project_space" ]]; then
      echo "  cd \"$TARGET_DIR\" && $LAUNCHER_DISPLAY"
    else
      echo "  cd \"$TARGET_DIR\" && CATMASTER_PROJECT_SPACE_ROOT=\"$PROJECT_SPACE_ROOT\" $LAUNCHER_DISPLAY"
    fi
  fi
else
  echo
  echo "Dry-run completed (no files were changed)."
fi
