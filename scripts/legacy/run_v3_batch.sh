#!/usr/bin/env bash
# V3 batch: longest tail items.
#  1. 7B at 24K steps (was 12K, T1 below threshold)
#  2. L0-only seeds 1+2 (paired with seed 0 already done)
#  3. Additive seeds 1+2 (paired with seed 0 already done)
#  4. Within-distribution probe (train and test tau in [1s, 7d] both)
#  5. Teacher-forced T4 with token-position labels

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

mkdir -p logs checkpoints reports data/qwen_time

#-----------------------------------------------------
# 1. 7B at 24K steps
#-----------------------------------------------------
echo "=== 1/5: 7B at 24K steps ==="
TAG_7B="scale_7b_24k_$(date +%Y%m%d_%H%M%S)"
DATA_7B="data/qwen_time/train_v15_18k.jsonl"
[ -f "$DATA_7B" ] || PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
    --n 18000 --seed 0 --mix 0.40,0.30,0.30 --out "$DATA_7B"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
    --data "$DATA_7B" --steps 24000 --lr 1e-4 --device cuda \
    --log-every 200 --log-path "logs/${TAG_7B}.jsonl" \
    --out "checkpoints/${TAG_7B}.pt" \
    --base "Qwen/Qwen2.5-7B-Instruct" --chunk-length 512 \
    --timescales "$TIMESCALES" --seed 0 \
    > "logs/${TAG_7B}_stdout.log" 2>&1
PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
    --checkpoint "checkpoints/${TAG_7B}.pt" --device cuda \
    --out "reports/${TAG_7B}_recall.json" \
    --base "Qwen/Qwen2.5-7B-Instruct" --timescales "$TIMESCALES" \
    >> "logs/${TAG_7B}_stdout.log" 2>&1
echo "{\"status\":\"done\"}" > "reports/${TAG_7B}_DONE.txt"

#-----------------------------------------------------
# 2. L0-only seeds 1+2
#-----------------------------------------------------
echo "=== 2/5: L0-only seeds 1+2 ==="
for SEED in 1 2; do
    TAG="qwen_time_l0_only_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
    DATA="data/qwen_time/train_l0_seed${SEED}_18k.jsonl"
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
        --n 18000 --seed "$SEED" --mix 0.40,0.30,0.30 --out "$DATA"
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
        --log-every 200 --log-path "logs/${TAG}.jsonl" \
        --out "checkpoints/${TAG}.pt" \
        --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$SEED" \
        --inject-layers "0" > "logs/${TAG}_stdout.log" 2>&1
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
        --checkpoint "checkpoints/${TAG}.pt" --device cuda \
        --out "reports/${TAG}_recall.json" \
        --base "Qwen/Qwen2.5-3B-Instruct" --timescales "$TIMESCALES" \
        --inject-layers "0" >> "logs/${TAG}_stdout.log" 2>&1
    echo "{\"status\":\"done\",\"seed\":$SEED}" > "reports/${TAG}_DONE.txt"
done

#-----------------------------------------------------
# 3. Additive seeds 1+2
#-----------------------------------------------------
echo "=== 3/5: Additive seeds 1+2 ==="
for SEED in 1 2; do
    TAG="qwen_time_additive_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
    DATA="data/qwen_time/train_additive_seed${SEED}_18k.jsonl"
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
        --n 18000 --seed "$SEED" --mix 0.40,0.30,0.30 --out "$DATA"
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
        --log-every 200 --log-path "logs/${TAG}.jsonl" \
        --out "checkpoints/${TAG}.pt" \
        --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$SEED" \
        --injection-type additive > "logs/${TAG}_stdout.log" 2>&1
    PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
        --checkpoint "checkpoints/${TAG}.pt" --device cuda \
        --out "reports/${TAG}_recall.json" \
        --base "Qwen/Qwen2.5-3B-Instruct" --timescales "$TIMESCALES" \
        --injection-type additive >> "logs/${TAG}_stdout.log" 2>&1
    echo "{\"status\":\"done\",\"seed\":$SEED}" > "reports/${TAG}_DONE.txt"
done

#-----------------------------------------------------
# 4. Within-distribution probe (no OOD split)
#-----------------------------------------------------
echo "=== 4/5: Within-distribution probe ==="
PYTHONPATH=. uv run python3 -u -m model.qwen_time_probe_within \
    --checkpoint checkpoints/qwen_time_v15s_20260523_141410_seed0.pt \
    --base "Qwen/Qwen2.5-3B-Instruct" --device cuda \
    --timescales "$TIMESCALES" --n-samples 600 \
    --out reports/probe_within_dist_v15s_seed0.json \
    > logs/probe_within.log 2>&1 || true

#-----------------------------------------------------
# 5. Teacher-forced T4 with token labels
#-----------------------------------------------------
echo "=== 5/5: T4 token-labeled teacher-forced ==="
PYTHONPATH=. uv run python3 -u -m model.qwen_time_t4_labeled \
    --checkpoint checkpoints/qwen_time_v15s_20260523_141410_seed0.pt \
    --base "Qwen/Qwen2.5-3B-Instruct" --device cuda \
    --timescales "$TIMESCALES" \
    --out reports/t4_labeled_v15s_seed0.json \
    > logs/t4_labeled.log 2>&1 || true

echo "{\"status\":\"all_done\"}" > reports/v3_batch_ALL_DONE.txt
echo "V3 BATCH DONE"
