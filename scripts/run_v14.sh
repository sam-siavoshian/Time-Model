#!/usr/bin/env bash
# v14: T3 phase fix via within-phase weekend/weekday balancing.
# v13 added day+week timescales (helped T1b mae & T4 KL but T3 still
# flat). Root cause: within-phase data was 5:2 weekday:weekend.
# v14 forces 50/50 in gen_phase_conversation.

set -uo pipefail
cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

LOG_DIR="logs"
CKPT_DIR="checkpoints"
REPORT_DIR="reports"
TAG="qwen_time_v14_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR"

DATA="data/qwen_time/train_v14_phase_balanced.jsonl"
LOG_PATH="$LOG_DIR/${TAG}.jsonl"
STDOUT="$LOG_DIR/${TAG}_stdout.log"
CKPT_PATH="$CKPT_DIR/${TAG}.pt"
REPORT_PATH="$REPORT_DIR/${TAG}_recall.json"
SENT="$REPORT_DIR/${TAG}_DONE.txt"

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

echo "============================================================"
echo "v14 retrain (T3 fix: 50/50 within-phase + day+week scales)"
echo "  tag:    $TAG"
echo "  scales: $TIMESCALES"
echo "============================================================"
date -Iseconds | tee -a "$STDOUT"

echo "=== STEP 1: GENERATE PHASE-BALANCED DATA ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
    --n 6000 --seed 14 --mix 0.34,0.33,0.33 --out "$DATA" 2>&1 | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== STEP 2: TRAIN ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
    --data "$DATA" --steps 12000 --lr 1e-4 --device cuda \
    --log-every 100 --log-path "$LOG_PATH" --out "$CKPT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
    --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"
TRAIN_RC=${PIPESTATUS[0]}
echo "train exit=$TRAIN_RC at $(date -Iseconds)" | tee -a "$STDOUT"

if [ $TRAIN_RC -ne 0 ]; then
    echo "TRAIN FAILED" | tee -a "$STDOUT"
    echo "{\"status\":\"train_failed\",\"exit_code\":$TRAIN_RC}" > "$SENT"
    exit $TRAIN_RC
fi

echo "" | tee -a "$STDOUT"
echo "=== STEP 3: EVAL ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
    --checkpoint "$CKPT_PATH" --device cuda --out "$REPORT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"
EVAL_RC=${PIPESTATUS[0]}

echo "{\"status\":\"done\",\"train_rc\":$TRAIN_RC,\"eval_rc\":$EVAL_RC,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
echo "DONE $(date -Iseconds). Sentinel: $SENT" | tee -a "$STDOUT"
