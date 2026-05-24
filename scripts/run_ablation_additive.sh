#!/usr/bin/env bash
# Architectural ablation: additive residual injection (vs FiLM).
# Tests reviewer attack: GazeQwen does additive; CI does FiLM. Comparable?

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TAG="qwen_time_additive_$(date +%Y%m%d_%H%M%S)"
DATA="data/qwen_time/train_v15_18k.jsonl"
CKPT="checkpoints/${TAG}.pt"
LOG_PATH="logs/${TAG}.jsonl"
STDOUT="logs/${TAG}_stdout.log"
REPORT="reports/${TAG}_recall.json"
SENT="reports/${TAG}_DONE.txt"
mkdir -p logs checkpoints reports

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

if [ ! -f "$DATA" ]; then
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
        --n 18000 --seed 0 --mix 0.40,0.30,0.30 --out "$DATA" 2>&1 | tee -a "$STDOUT"
fi

PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
    --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
    --log-every 200 --log-path "$LOG_PATH" --out "$CKPT" \
    --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
    --timescales "$TIMESCALES" --seed 0 \
    --injection-type additive 2>&1 | tee -a "$STDOUT"

PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
    --checkpoint "$CKPT" --device cuda --out "$REPORT" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" --injection-type additive 2>&1 | tee -a "$STDOUT"

echo "{\"status\":\"done\"}" > "$SENT"
