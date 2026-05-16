#!/usr/bin/env bash
# Track C end-to-end: train QwenTime then eval. Runs in tmux on Spark.
# Survives independent of local Claude session.

set -uo pipefail
cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export IPCN_ROOT="$HOME/ipcn"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

LOG_DIR="logs"
CKPT_DIR="checkpoints"
REPORT_DIR="reports"
TAG="qwen_time_v10_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR"

LOG_PATH="$LOG_DIR/${TAG}.jsonl"
STDOUT_PATH="$LOG_DIR/${TAG}_stdout.log"
CKPT_PATH="$CKPT_DIR/${TAG}.pt"
REPORT_PATH="$REPORT_DIR/${TAG}_recall.json"
SENTINEL_PATH="$REPORT_DIR/${TAG}_DONE.txt"

echo "============================================================"
echo "Track C v10 launcher"
echo "  tag:     $TAG"
echo "  log:     $LOG_PATH"
echo "  ckpt:    $CKPT_PATH"
echo "  report:  $REPORT_PATH"
echo "  sentinel:$SENTINEL_PATH"
echo "============================================================"
echo "started $(date -Iseconds)"

echo "=== STEP 1: TRAIN ==="
PYTHONPATH=. uv run python3 -m model.qwen_time_train \
    --data data/qwen_time/train_v2.jsonl \
    --steps 12000 \
    --lr 1e-4 \
    --device cuda \
    --log-every 100 \
    --log-path "$LOG_PATH" \
    --out "$CKPT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --chunk-length 512 \
    2>&1 | tee "$STDOUT_PATH"
TRAIN_RC=$?
echo "train exit=$TRAIN_RC at $(date -Iseconds)"

if [ $TRAIN_RC -ne 0 ]; then
    echo "TRAIN FAILED. Skipping eval."
    echo "{\"status\":\"train_failed\",\"exit_code\":$TRAIN_RC,\"time\":\"$(date -Iseconds)\"}" > "$SENTINEL_PATH"
    exit $TRAIN_RC
fi

echo ""
echo "=== STEP 2: EVAL ==="
PYTHONPATH=. uv run python3 -m model.qwen_time_check \
    --checkpoint "$CKPT_PATH" \
    --device cuda \
    --out "$REPORT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    2>&1 | tee -a "$STDOUT_PATH"
EVAL_RC=$?
echo "eval exit=$EVAL_RC at $(date -Iseconds)"

echo "=== STEP 3: SUMMARY ==="
if [ -f "$REPORT_PATH" ]; then
    python3 -c "
import json
with open('$REPORT_PATH') as f:
    r = json.load(f)
s = r.get('summary', {})
print('TRACK C SUMMARY:')
for k, v in s.items():
    print(f'  {k}: {v}')
"
fi
echo "{\"status\":\"done\",\"train_rc\":$TRAIN_RC,\"eval_rc\":$EVAL_RC,\"time\":\"$(date -Iseconds)\",\"report\":\"$REPORT_PATH\"}" > "$SENTINEL_PATH"
echo "DONE $(date -Iseconds). Sentinel: $SENTINEL_PATH"
