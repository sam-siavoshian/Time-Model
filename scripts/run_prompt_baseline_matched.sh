#!/usr/bin/env bash
# W6 fix: prompt baseline at MATCHED TOTAL TRAINABLE PARAMETERS.
# CI has ~9.5M trainable (LoRA rank-8 + chrono encoder + per-layer FiLM).
# Original prompt baseline has ~4.8M (LoRA only, chrono frozen).
# This version trains prompt baseline with LoRA rank-16 -> ~9.6M params
# matching CI's total trainable budget. Same data distribution
# (qwen_time_data_prompt), same --freeze-alpha (chrono off), same steps.
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

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
LORA_RANK=16

for SEED in 0 1 2; do
    DATA="$DATA_DIR/prompt_baseline_s${SEED}_18k.jsonl"
    OUT="$CKPT_DIR/prompt_baseline_matched_s${SEED}.pt"
    REC="$REPORT_DIR/prompt_baseline_matched_injected_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/prompt_baseline_matched_s${SEED}.log"

    if [ ! -s "$DATA" ]; then
        uv run python -m model.qwen_time_data_prompt \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
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

time_model_done "$SCRIPT_PATH" "prompt_baseline_matched.done"
echo "all 3 seeds done"
