#!/usr/bin/env bash
# Pre-registered (docs/experiments/current/PREREGISTRATION_v2.md section 2): per-layer probe on
# the clock-heldout checkpoints (seeds 0, 1, 2).
#
# Uses model/qwen_time_probe_within.py — SAME script that produced
# reports/probe_per_layer_v15s_s0.json (within-dist 80/20 random split,
# not OOD). Two conditions:
#   A: trained model (the load-bearing line)
#   B: alpha=0 (chrono off; control)
#
# Output: runs/<run_id>/reports/probe_per_layer_clock_heldout_s{0,1,2}.json
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

for SEED in 0 1 2; do
    CKPT="checkpoints/clock_heldout_s${SEED}.pt"
    OUT="$REPORT_DIR/probe_per_layer_clock_heldout_s${SEED}.json"
    LOG="$LOG_DIR/probe_clock_heldout_s${SEED}.log"

    if [ ! -f "$CKPT" ]; then
        echo "MISSING: $CKPT (skip seed $SEED)" | tee -a "$LOG"
        continue
    fi

    if [ -s "$OUT" ]; then
        echo "DONE already: $OUT (skip)" | tee -a "$LOG"
        continue
    fi

    echo "=== within-dist probe seed $SEED on $CKPT ===" | tee -a "$LOG"
    uv run python -m model.qwen_time_probe_within \
        --checkpoint "$CKPT" --device cuda \
        --timescales "$TIMESCALES" --n-samples 600 --seed 4242 \
        --out "$OUT" 2>&1 | tee -a "$LOG"
done

time_model_done "$SCRIPT_PATH" "probe_clock_heldout.done"
echo "all 3 clock-heldout probes done"
