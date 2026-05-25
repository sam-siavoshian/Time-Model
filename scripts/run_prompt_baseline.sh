#!/usr/bin/env bash
# W6 baseline: prompt-injected tau. Three seeds at v15-matched budget.
# Same data distribution as v15 but with tau text-prefixed; chrono
# channel forced off via --freeze-alpha so tau MUST come from the prompt.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs reports data/processed

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

for SEED in 0 1 2; do
    DATA="data/processed/prompt_baseline_s${SEED}_18k.jsonl"
    OUT="checkpoints/prompt_baseline_s${SEED}.pt"
    STDOUT="logs/prompt_baseline_s${SEED}.log"
    mkdir -p "$(dirname "$OUT")"

    echo "=== seed $SEED data ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_data_prompt \
        --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"

    echo "=== seed $SEED train ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps "$STEPS" --seed "$SEED" \
        --timescales "$TIMESCALES" --freeze-alpha \
        --out "$OUT" 2>&1 | tee -a "$STDOUT"

    echo "=== seed $SEED check ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --tag "prompt_baseline_s${SEED}" 2>&1 | tee -a "$STDOUT"
done

touch logs/prompt_baseline.done
echo "all 3 seeds done"
