#!/usr/bin/env bash
# v15: land all 5 tests in one checkpoint.
# Root cause of v14 T1 dip = data exhaustion: stream_records is
# single-pass, --steps 12000 only saw ~6K records.
# v15 = 18K records, mix 0.40/0.30/0.30 (clock-favored), --steps 18000,
# v13 timescales (incl 86400 day + 604800 week), v14 50/50 phase balance
# (default in gen_phase_conversation since v14).

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TAG="qwen_time_v15_$(date +%Y%m%d_%H%M%S)"
DATA="data/qwen_time/train_v15_18k.jsonl"
LOG_PATH="logs/${TAG}.jsonl"
STDOUT="logs/${TAG}_stdout.log"
CKPT_PATH="checkpoints/${TAG}.pt"
REPORT_PATH="reports/${TAG}_recall.json"
SENT="reports/${TAG}_DONE.txt"
mkdir -p logs checkpoints reports data/qwen_time

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

echo "============================================================"
echo "v15 retrain (final SOTA attempt)"
echo "  tag:    $TAG"
echo "  data:   $DATA (18K records, mix 0.40/0.30/0.30)"
echo "  scales: $TIMESCALES (15 scales incl day+week)"
echo "============================================================"
date -Iseconds | tee -a "$STDOUT"

echo "=== STEP 1: GENERATE 18K BALANCED DATA ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
    --n 18000 --seed 15 --mix 0.40,0.30,0.30 --out "$DATA" 2>&1 | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== STEP 2: TRAIN ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
    --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
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
echo "=== STEP 3: EVAL 5 TESTS ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
    --checkpoint "$CKPT_PATH" --device cuda --out "$REPORT_PATH" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"
EVAL_RC=${PIPESTATUS[0]}

echo "" | tee -a "$STDOUT"
echo "=== STEP 4: GENUINE OOD + MULTI-WEEK T3 ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_check_genuine_ood \
    --checkpoint "$CKPT_PATH" --device cuda \
    --out "reports/${TAG}_genuine_ood.json" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== STEP 5: PRESSURE v2 (n=30, max=256, bootstrap CI) ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_pressure_v2 \
    --checkpoint "$CKPT_PATH" --device cuda \
    --out "reports/${TAG}_pressure_v2.json" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" --max-new 256 2>&1 | tee -a "$STDOUT"

echo "{\"status\":\"done\",\"train_rc\":$TRAIN_RC,\"eval_rc\":$EVAL_RC,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
echo "DONE $(date -Iseconds). Sentinel: $SENT" | tee -a "$STDOUT"
