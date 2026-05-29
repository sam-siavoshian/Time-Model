#!/usr/bin/env bash
# W12: run external tau_sessions benchmark on all 3 reference adapters.
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"

OUT_DIR="${OUT_DIR:-$REPORT_DIR/external}"
EXT_LOG_DIR="${EXT_LOG_DIR:-$LOG_DIR/external}"
BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
CKPT="${CKPT:-release_ckpts/qwen_time_v15s_20260523_141410_seed0.pt}"
mkdir -p "$OUT_DIR" "$EXT_LOG_DIR"

run() {
    local adapter="$1"; shift
    local out="$OUT_DIR/ext_bench_${adapter}.json"
    local log="$EXT_LOG_DIR/ext_bench_${adapter}.log"
    if [ -e "$out" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "refusing to overwrite existing output: $out (set FORCE=1 to replace)" >&2
        return 1
    fi
    uv run python -m eval.external.eval_tau_bench --adapter "$adapter" --base "$BASE" \
        "$@" --out "$out" 2>&1 | tee "$log"
    test -s "$out"
}

run vanilla
run prompt
test -s "$CKPT"
run ci --checkpoint "$CKPT"

{
    echo "RUN_ID=$RUN_ID"
    echo "OUT_DIR=$OUT_DIR"
    echo "ALL DONE $(date)"
} > "$LOG_DIR/ext_bench.done"
time_model_manifest "$SCRIPT_PATH" "done"
echo "all 3 adapters done"
