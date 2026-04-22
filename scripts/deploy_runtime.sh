#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_runtime.sh [--target DIR] [--project-space-root DIR] [--dry-run] [--no-delete] [--full-repo] [--skip-frontend-build] [--autorun|--no-autorun]

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

  --skip-frontend-build
      Do not rebuild catmaster/webui/static from catmaster/webui/frontend before deploy.

  --autorun
      Start runtime WebUI automatically after deployment (default).

  --no-autorun
      Do not start runtime WebUI automatically after deployment.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/../CatMaster_Run"
PROJECT_SPACE_ROOT=""
DRY_RUN=0
NO_DELETE=0
FULL_REPO=0
AUTORUN=1
SKIP_FRONTEND_BUILD=0

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
    --skip-frontend-build)
      SKIP_FRONTEND_BUILD=1
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

echo "Source: $REPO_ROOT"
echo "Target: $TARGET_DIR"
echo "Project space root for launcher: $PROJECT_SPACE_ROOT"
echo

if [[ $DRY_RUN -eq 0 && $SKIP_FRONTEND_BUILD -eq 0 ]]; then
  if [[ -f "$REPO_ROOT/scripts/install_jsmol_assets.py" ]]; then
    python3 "$REPO_ROOT/scripts/install_jsmol_assets.py" --quiet
  fi
  FRONTEND_DIR="$REPO_ROOT/catmaster/webui/frontend"
  if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
      echo "npm is required to build the WebUI bundle before deploy." >&2
      exit 1
    fi
    echo "Building WebUI bundle..."
    (cd "$FRONTEND_DIR" && npm run build)
    echo
  fi
fi

if [[ $FULL_REPO -eq 1 ]]; then
  rsync "${RSYNC_ARGS[@]}" "$REPO_ROOT/" "$TARGET_DIR/"
else
  RUNTIME_PATHS=(
    "catmaster"
    "configs"
    "requirements"
    "skills"
    "scripts"
    "main.py"
    "start_webui.sh"
    "README.md"
    "LICENSE"
  )
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
  COMMIT="unknown"
  if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  fi
  DEPLOY_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  cat > "$TARGET_DIR/.deploy_info" <<EOF
source_repo=$REPO_ROOT
source_commit=$COMMIT
deployed_at_utc=$DEPLOY_TIME
EOF
  chmod +x "$TARGET_DIR/start_webui.sh"
  mkdir -p "$PROJECT_SPACE_ROOT"

  echo
  echo "Deploy completed."
  if [[ $AUTORUN -eq 1 ]]; then
    echo "Autorun enabled. Starting WebUI now..."
    cd "$TARGET_DIR"
    if [[ "$PROJECT_SPACE_ROOT" == "$TARGET_DIR/project_space" ]]; then
      ./start_webui.sh --port 7991
    else
      CATMASTER_PROJECT_SPACE_ROOT="$PROJECT_SPACE_ROOT" ./start_webui.sh --port 7991
    fi
  else
    echo "Next run command:"
    if [[ "$PROJECT_SPACE_ROOT" == "$TARGET_DIR/project_space" ]]; then
      echo "  cd \"$TARGET_DIR\" && ./start_webui.sh --port 7991"
    else
      echo "  cd \"$TARGET_DIR\" && CATMASTER_PROJECT_SPACE_ROOT=\"$PROJECT_SPACE_ROOT\" ./start_webui.sh --port 7991"
    fi
  fi
else
  echo
  echo "Dry-run completed (no files were changed)."
fi
