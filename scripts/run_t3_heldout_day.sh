#!/usr/bin/env bash
# W8/Q9: train on Mon-Sat (no Sunday) x 3 seeds. Eval T3 at Sunday tau.
# If T3 weekend signal holds at Sunday, phase channel generalizes; if
# not, T3 was supervised classification on the trained labels, not
# unsupervised phase discovery.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs reports checkpoints

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
for SEED in 0 1 2; do
    DATA="data/processed/t3_heldout_s${SEED}_18k.jsonl"
    OUT="checkpoints/t3_heldout_s${SEED}.pt"
    STDOUT="logs/t3_heldout_s${SEED}.log"
    uv run python -m model.qwen_time_data_t3_holdout \
        --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --seed "$SEED" \
        --timescales "$TIMESCALES" --out "$OUT" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --tag "t3_heldout_s${SEED}" 2>&1 | tee -a "$STDOUT"
done
touch logs/t3_heldout.done
