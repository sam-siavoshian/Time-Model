#!/usr/bin/env bash
# Cross-seed v15: train 3 independent seeds, eval each on full 5-test,
# compute mean +/- std across seeds. Addresses reviewer attack on
# single-seed reporting.

set -uo pipefail
cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

mkdir -p logs checkpoints reports data/qwen_time
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
TAG_BASE="qwen_time_v15s_$(date +%Y%m%d_%H%M%S)"

for SEED in 0 1 2; do
    TAG="${TAG_BASE}_seed${SEED}"
    DATA="data/qwen_time/train_v15s_seed${SEED}_18k.jsonl"
    LOG_PATH="logs/${TAG}.jsonl"
    STDOUT="logs/${TAG}_stdout.log"
    CKPT="checkpoints/${TAG}.pt"
    REPORT="reports/${TAG}_recall.json"
    SENT="reports/${TAG}_DONE.txt"

    echo "============================================================"
    echo "v15 seed $SEED -- tag $TAG"
    echo "============================================================"
    date -Iseconds | tee -a "$STDOUT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
        --n 18000 --seed "$SEED" --mix 0.40,0.30,0.30 --out "$DATA" \
        2>&1 | tee -a "$STDOUT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
        --log-every 200 --log-path "$LOG_PATH" --out "$CKPT" \
        --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$SEED" 2>&1 | tee -a "$STDOUT"
    TRAIN_RC=${PIPESTATUS[0]}

    if [ $TRAIN_RC -ne 0 ]; then
        echo "{\"status\":\"train_failed\",\"seed\":$SEED}" > "$SENT"
        continue
    fi

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
        --checkpoint "$CKPT" --device cuda --out "$REPORT" \
        --base "Qwen/Qwen2.5-3B-Instruct" \
        --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"

    echo "{\"status\":\"done\",\"seed\":$SEED,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
done

echo "{\"status\":\"all_seeds_done\",\"time\":\"$(date -Iseconds)\"}" > "reports/${TAG_BASE}_ALL_DONE.txt"
