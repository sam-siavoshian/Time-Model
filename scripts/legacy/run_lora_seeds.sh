#!/usr/bin/env bash
# LoRA-only baseline seeds 1 and 2 (seed 0 already done).

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
TAG_BASE="qwen_time_lora_only_seeds_$(date +%Y%m%d_%H%M%S)"

for SEED in 1 2; do
    TAG="${TAG_BASE}_seed${SEED}"
    DATA="data/qwen_time/train_lora_seed${SEED}_18k.jsonl"
    LOG_PATH="logs/${TAG}.jsonl"
    STDOUT="logs/${TAG}_stdout.log"
    CKPT="checkpoints/${TAG}.pt"
    REPORT="reports/${TAG}_recall.json"
    SENT="reports/${TAG}_DONE.txt"
    mkdir -p logs checkpoints reports data/qwen_time

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
        --n 18000 --seed "$SEED" --mix 0.40,0.30,0.30 --out "$DATA" 2>&1 | tee -a "$STDOUT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
        --log-every 200 --log-path "$LOG_PATH" --out "$CKPT" \
        --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$SEED" \
        --freeze-alpha 2>&1 | tee -a "$STDOUT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
        --checkpoint "$CKPT" --device cuda --out "$REPORT" \
        --base "Qwen/Qwen2.5-3B-Instruct" \
        --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"

    echo "{\"status\":\"done\",\"seed\":$SEED}" > "$SENT"
done

echo "{\"status\":\"all_lora_seeds_done\"}" > "reports/${TAG_BASE}_ALL_DONE.txt"
