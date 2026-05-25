#!/usr/bin/env bash
# W10 (prior review) + sufficiency: chrono channel on, LoRA frozen at zero.
# Tests whether chrono alone (no LoRA surface capacity) can fit T1-T4.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs reports checkpoints
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

# Note: requires trainer to support --freeze-lora flag. If not present,
# the script will error -- add the flag to qwen_time_train.py first.
for SEED in 0 1 2; do
    DATA="data/processed/v15s_seed${SEED}_18k.jsonl"
    OUT="checkpoints/chrono_only_s${SEED}.pt"
    STDOUT="logs/chrono_only_s${SEED}.log"
    if [ ! -f "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --seed "$SEED" \
        --timescales "$TIMESCALES" --freeze-lora \
        --out "$OUT" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --tag "chrono_only_s${SEED}" 2>&1 | tee -a "$STDOUT"
done
touch logs/chrono_only.done
