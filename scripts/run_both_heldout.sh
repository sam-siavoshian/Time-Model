#!/usr/bin/env bash
# Pre-registered (docs/experiments/current/PREREGISTRATION_v2.md §3.1): both-held-out training.
# Train without ANY CLOCK supervision AND without ANY SILENT-GAP supervision.
# Only PHASE supervision. Eval the full T1/T1b/T2/T3/T4 protocol.
#
# Pre-registered predictions:
#   T1, T1b -> nan (no clock supervision; readout untrained)
#   T2     -> Delta approx 0 (no silent-gap supervision)
#   T3     -> passes at full-supervision levels (only task seen)
#   T4 mp  -> degraded relative to full supervision (no clock/silent-gap
#             shape to push the channel through), reported descriptively.
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
    DATA="$DATA_DIR/both_heldout_s${SEED}_18k.jsonl"
    CKPT="$CKPT_DIR/both_heldout_s${SEED}.pt"
    REC="$REPORT_DIR/both_heldout_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/both_heldout_s${SEED}.log"

    echo "=== seed $SEED: gen PHASE-only data (mix 0,0,1) ===" | tee -a "$STDOUT"
    if [ ! -s "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n "$STEPS" --seed "$SEED" --mix "0,0,1" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi

    echo "=== seed $SEED: train (chrono + LoRA + FiLM; no CLOCK no SILENT-GAP supervision) ===" | tee -a "$STDOUT"
    if [ ! -f "$CKPT" ]; then
        uv run python -m model.qwen_time_train \
            --data "$DATA" --steps "$STEPS" --seed "$SEED" \
            --timescales "$TIMESCALES" --out "$CKPT" 2>&1 | tee -a "$STDOUT"
    fi

    echo "=== seed $SEED: eval T1-T4 (both T1 and T2 ZERO-SHOT) ===" | tee -a "$STDOUT"
    if [ ! -s "$REC" ]; then
        uv run python -m model.qwen_time_check \
            --checkpoint "$CKPT" --timescales "$TIMESCALES" \
            --out "$REC" 2>&1 | tee -a "$STDOUT"
    fi
done

time_model_done "$SCRIPT_PATH" "both_heldout.done"
echo "all 3 seeds done"
