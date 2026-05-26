#!/usr/bin/env bash
# W1 fix (THE BIG ONE): held-out training distribution.
# Train without ANY CLOCK supervision (only SILENT-GAP + PHASE).
# At eval, ask the model "how long has it been?" -- it has NEVER seen
# this question paired with a duration-string answer in training.
#
# If T1/T1b still pass: STRONG claim. Chrono channel encodes
# generalizable time info, not just supervised label memorization.
# Paper score 4 -> 6.
#
# If T1/T1b collapse: honest reframe. T1/T1b are supervised recall.
# Paper restructure around TPDR + T4 only.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports checkpoints data/processed

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
STEPS=${STEPS:-18000}

for SEED in 0 1 2; do
    DATA="data/processed/clock_heldout_s${SEED}_18k.jsonl"
    CKPT="checkpoints/clock_heldout_s${SEED}.pt"
    REC="reports/clock_heldout_s${SEED}_recall.json"
    STDOUT="logs/clock_heldout_s${SEED}.log"

    echo "=== seed $SEED: gen held-out data (mix 0/0.6/0.4: NO clock) ===" | tee -a "$STDOUT"
    if [ ! -s "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n "$STEPS" --seed "$SEED" --mix "0,0.6,0.4" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi

    echo "=== seed $SEED: train (chrono + LoRA + FiLM, no CLOCK supervision) ===" | tee -a "$STDOUT"
    if [ ! -f "$CKPT" ]; then
        uv run python -m model.qwen_time_train \
            --data "$DATA" --steps "$STEPS" --seed "$SEED" \
            --timescales "$TIMESCALES" --out "$CKPT" 2>&1 | tee -a "$STDOUT"
    fi

    echo "=== seed $SEED: eval T1-T4 (T1/T1b ZERO-SHOT -- never trained on duration-string format) ===" | tee -a "$STDOUT"
    if [ ! -s "$REC" ]; then
        uv run python -m model.qwen_time_check \
            --checkpoint "$CKPT" --timescales "$TIMESCALES" \
            --out "$REC" 2>&1 | tee -a "$STDOUT"
    fi
done

touch logs/clock_heldout.done
echo "all 3 seeds done"
