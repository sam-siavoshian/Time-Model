#!/usr/bin/env bash
# Shared run-directory contract for supported shell runners.

time_model_find_root() {
  local source_dir
  source_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  cd "$source_dir/../.." && pwd
}

time_model_init_run() {
  REPO_ROOT="${REPO_ROOT:-$(time_model_find_root)}"
  RUN_CONTEXT_ARGS=()
  RESUME_RUN_ID=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --run-id)
        if [ "$#" -lt 2 ]; then
          echo "missing value for --run-id" >&2
          return 2
        fi
        RUN_ID="$2"
        shift 2
        ;;
      --resume-run-id)
        if [ "$#" -lt 2 ]; then
          echo "missing value for --resume-run-id" >&2
          return 2
        fi
        RUN_ID="$2"
        RESUME_RUN_ID="$2"
        shift 2
        ;;
      *)
        RUN_CONTEXT_ARGS+=("$1")
        shift
        ;;
    esac
  done

  if [ -z "${RUN_ID:-}" ]; then
    echo "RUN_ID is required. Use --run-id <id>, --resume-run-id <id>, or RUN_ID=<id>." >&2
    return 2
  fi

  RUN_ROOT="${RUN_ROOT:-runs/$RUN_ID}"
  if [ -e "$REPO_ROOT/$RUN_ROOT" ] && [ "${FORCE:-0}" != "1" ] && [ -z "$RESUME_RUN_ID" ]; then
    echo "refusing to reuse existing run directory: $RUN_ROOT (use --resume-run-id $RUN_ID or FORCE=1)" >&2
    return 1
  fi

  DATA_DIR="${DATA_DIR:-$RUN_ROOT/data}"
  LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
  CKPT_DIR="${CKPT_DIR:-$RUN_ROOT/checkpoints}"
  REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports}"

  mkdir -p "$REPO_ROOT/$DATA_DIR" "$REPO_ROOT/$LOG_DIR" "$REPO_ROOT/$CKPT_DIR" "$REPO_ROOT/$REPORT_DIR"

  export REPO_ROOT RUN_ID RUN_ROOT DATA_DIR LOG_DIR CKPT_DIR REPORT_DIR RESUME_RUN_ID
}

time_model_manifest() {
  local script_path="$1"
  local status="${2:-done}"
  local manifest_path="$REPO_ROOT/$RUN_ROOT/manifest.json"
  python3 "$REPO_ROOT/scripts/run_context.py" manifest \
    --run-id "$RUN_ID" \
    --run-root "$RUN_ROOT" \
    --script "$script_path" \
    --status "$status" \
    --out "$manifest_path"
}

time_model_done() {
  local script_path="$1"
  local sentinel="${2:-$RUN_ROOT/ALL_DONE.txt}"
  case "$sentinel" in
    */*) ;;
    *) sentinel="$RUN_ROOT/$sentinel" ;;
  esac
  time_model_manifest "$script_path" "done"
  echo "{\"status\":\"done\",\"run_id\":\"$RUN_ID\",\"time\":\"$(date -Iseconds)\"}" > "$REPO_ROOT/$sentinel"
}
