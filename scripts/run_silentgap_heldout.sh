#!/usr/bin/env bash
# W1 fix (secondary): SILENT-GAP held-out.
# Train without ANY SILENT-GAP supervision (only CLOCK + PHASE).
# At eval, T2 measures whether silent-gap acknowledgement generalizes
# from the chrono channel alone.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports checkpoints data/processed

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
STEPS=${STEPS:-18000}

for SEED in 0 1 2; do
    DATA="data/processed/silentgap_heldout_s${SEED}_18k.jsonl"
    CKPT="checkpoints/silentgap_heldout_s${SEED}.pt"
    REC="reports/silentgap_heldout_s${SEED}_recall.json"
    STDOUT="logs/silentgap_heldout_s${SEED}.log"

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

touch logs/silentgap_heldout.done
echo "all 3 seeds done"
