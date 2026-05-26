#!/usr/bin/env bash
# W6 fix: prompt baseline at MATCHED TOTAL TRAINABLE PARAMETERS.
# CI has ~9.5M trainable (LoRA rank-8 + chrono encoder + per-layer FiLM).
# Original prompt baseline has ~4.8M (LoRA only, chrono frozen).
# This version trains prompt baseline with LoRA rank-16 -> ~9.6M params
# matching CI's total trainable budget. Same data distribution
# (qwen_time_data_prompt), same --freeze-alpha (chrono off), same steps.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports checkpoints

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
LORA_RANK=16

for SEED in 0 1 2; do
    DATA="data/processed/prompt_baseline_s${SEED}_18k.jsonl"
    OUT="checkpoints/prompt_baseline_matched_s${SEED}.pt"
    REC="reports/prompt_baseline_matched_injected_s${SEED}_recall.json"
    STDOUT="logs/prompt_baseline_matched_s${SEED}.log"

    if [ ! -s "$DATA" ]; then
        echo "[error] data missing: $DATA" | tee -a "$STDOUT"
        continue
    fi
    if [ ! -f "$OUT" ]; then
        echo "=== seed $SEED train (LoRA rank=$LORA_RANK, freeze-alpha) ===" | tee -a "$STDOUT"
        uv run python -m model.qwen_time_train \
            --data "$DATA" --steps "$STEPS" --seed "$SEED" \
            --timescales "$TIMESCALES" --freeze-alpha \
            --lora-rank "$LORA_RANK" \
            --out "$OUT" 2>&1 | tee -a "$STDOUT"
    fi
    if [ -f "$OUT" ] && [ ! -s "$REC" ]; then
        echo "=== seed $SEED check (--inject-prompt --lora-rank $LORA_RANK) ===" | tee -a "$STDOUT"
        uv run python -m model.qwen_time_check \
            --checkpoint "$OUT" --timescales "$TIMESCALES" \
            --lora-rank "$LORA_RANK" --inject-prompt \
            --out "$REC" 2>&1 | tee -a "$STDOUT"
    fi
done

touch logs/prompt_baseline_matched.done
echo "all 3 seeds done"
