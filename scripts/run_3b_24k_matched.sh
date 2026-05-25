#!/usr/bin/env bash
# W4: 3B at 24000 steps x 3 seeds. Matched-budget control for the 7B
# scaling claim. If 3B@24k passes T3 bidirectionally on >= 2/3 seeds,
# "7B fixed T3" collapses to "more steps fixed T3".
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs reports checkpoints

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
for SEED in 0 1 2; do
    DATA="data/processed/v15s_seed${SEED}_18k.jsonl"
    OUT="checkpoints/qwen_time_3b_24k_s${SEED}.pt"
    STDOUT="logs/3b_24k_s${SEED}.log"
    if [ ! -f "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps 24000 --seed "$SEED" \
        --timescales "$TIMESCALES" --out "$OUT" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --tag "3b_24k_s${SEED}" 2>&1 | tee -a "$STDOUT"
done
touch logs/3b_24k.done
