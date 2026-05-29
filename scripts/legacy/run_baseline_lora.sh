#!/usr/bin/env bash
# LoRA-only baseline: train v15 spec but freeze all chrono alpha gates at 0.
# Chrono encoder + projectors exist but cannot influence residual stream.
# Critical ablation to falsify the architectural claim.

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TAG="qwen_time_lora_only_$(date +%Y%m%d_%H%M%S)"
DATA="data/qwen_time/train_lora_baseline_18k.jsonl"
LOG_PATH="logs/${TAG}.jsonl"
STDOUT="logs/${TAG}_stdout.log"
CKPT_PATH="checkpoints/${TAG}.pt"
REPORT_PATH="reports/${TAG}_recall.json"
SENT="reports/${TAG}_DONE.txt"
mkdir -p logs checkpoints reports data/qwen_time

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

echo "============================================================"
echo "LoRA-only baseline (alpha=0 frozen)"
echo "  tag: $TAG"
echo "============================================================"
date -Iseconds | tee -a "$STDOUT"

PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
    --n 18000 --seed 0 --mix 0.40,0.30,0.30 --out "$DATA" 2>&1 | tee -a "$STDOUT"

PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
    --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
    --log-every 200 --log-path "$LOG_PATH" --out "$CKPT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
    --timescales "$TIMESCALES" --seed 0 \
    --freeze-alpha 2>&1 | tee -a "$STDOUT"

PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
    --checkpoint "$CKPT_PATH" --device cuda --out "$REPORT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"

echo "{\"status\":\"done\",\"time\":\"$(date -Iseconds)\"}" > "$SENT"
echo "DONE." | tee -a "$STDOUT"
