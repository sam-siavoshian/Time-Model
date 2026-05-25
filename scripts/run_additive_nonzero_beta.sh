#!/usr/bin/env bash
# W9: additive injection with non-zero beta init (default 0.01).
# Tests whether the reviewer's claim "additive could train with a sane
# non-zero beta init" holds. If this PASSES T1/T2/T3/T4 then the
# "FiLM gating mathematically required" claim further narrows to
# "FiLM gating with this specific init"; if it FAILS then the original
# claim "additive cannot escape the AdaLN-Zero gradient trap" generalizes
# beyond a single init choice.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs reports checkpoints

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
BETA_INIT="${BETA_INIT:-0.01}"

for SEED in 0 1 2; do
    DATA="data/processed/v15s_seed${SEED}_18k.jsonl"
    OUT="checkpoints/additive_nonzero_beta_s${SEED}.pt"
    STDOUT="logs/additive_nonzero_beta_s${SEED}.log"
    if [ ! -f "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --seed "$SEED" \
        --timescales "$TIMESCALES" \
        --injection-type additive \
        --additive-beta-init "$BETA_INIT" \
        --out "$OUT" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --tag "additive_nonzero_beta_s${SEED}" 2>&1 | tee -a "$STDOUT"
done
touch logs/additive_nonzero_beta.done
