#!/usr/bin/env bash
# Track A alias: canonical mechanistic CI training/eval.
# This preserves scripts/run_v15_seeds.sh while placing outputs under
# runs/<run_id>/{data,logs,checkpoints,reports}/track_a/.

set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
if [ "$#" -ne 0 ]; then
    echo "unsupported Track A arguments: $*" >&2
    exit 2
fi
cd "$REPO_ROOT"

export DATA_DIR="$RUN_ROOT/data/track_a"
export LOG_DIR="$RUN_ROOT/logs/track_a"
export CKPT_DIR="$RUN_ROOT/checkpoints/track_a"
export REPORT_DIR="$RUN_ROOT/reports/track_a"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR"

bash "$REPO_ROOT/scripts/run_v15_seeds.sh" --resume-run-id "$RUN_ID"
