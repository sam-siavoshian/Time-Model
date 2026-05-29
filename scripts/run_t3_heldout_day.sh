#!/usr/bin/env bash
# W8/Q9: train on Mon-Sat (no Sunday) x 3 seeds. Eval T3 at Sunday tau.
# If T3 weekend signal holds at Sunday, phase channel generalizes; if
# not, T3 was supervised classification on the trained labels, not
# unsupervised phase discovery.
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
for SEED in 0 1 2; do
    DATA="$DATA_DIR/t3_heldout_s${SEED}_18k.jsonl"
    OUT="$CKPT_DIR/t3_heldout_s${SEED}.pt"
    REC="$REPORT_DIR/t3_heldout_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/t3_heldout_s${SEED}.log"
    uv run python -m model.qwen_time_data_t3_holdout \
        --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --seed "$SEED" \
        --timescales "$TIMESCALES" --out "$OUT" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --out "$REC" 2>&1 | tee -a "$STDOUT"
    test -s "$REC"
done
time_model_done "$SCRIPT_PATH" "t3_heldout.done"
