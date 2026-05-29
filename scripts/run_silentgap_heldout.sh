#!/usr/bin/env bash
# W1 fix (secondary): SILENT-GAP held-out.
# Train without ANY SILENT-GAP supervision (only CLOCK + PHASE).
# At eval, T2 measures whether silent-gap acknowledgement generalizes
# from the chrono channel alone.
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
STEPS=${STEPS:-18000}

for SEED in 0 1 2; do
    DATA="$DATA_DIR/silentgap_heldout_s${SEED}_18k.jsonl"
    CKPT="$CKPT_DIR/silentgap_heldout_s${SEED}.pt"
    REC="$REPORT_DIR/silentgap_heldout_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/silentgap_heldout_s${SEED}.log"

    echo "=== seed $SEED: gen held-out data (mix 0.6/0/0.4: NO silent-gap) ===" | tee -a "$STDOUT"
    if [ ! -s "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n "$STEPS" --seed "$SEED" --mix "0.6,0,0.4" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi

    echo "=== seed $SEED: train ===" | tee -a "$STDOUT"
    if [ ! -f "$CKPT" ]; then
        uv run python -m model.qwen_time_train \
            --data "$DATA" --steps "$STEPS" --seed "$SEED" \
            --timescales "$TIMESCALES" --out "$CKPT" 2>&1 | tee -a "$STDOUT"
    fi

    echo "=== seed $SEED: eval (T2 is ZERO-SHOT here) ===" | tee -a "$STDOUT"
    if [ ! -s "$REC" ]; then
        uv run python -m model.qwen_time_check \
            --checkpoint "$CKPT" --timescales "$TIMESCALES" \
            --out "$REC" 2>&1 | tee -a "$STDOUT"
    fi
done

time_model_done "$SCRIPT_PATH" "silentgap_heldout.done"
echo "all 3 seeds done"
