#!/usr/bin/env bash
# Track B: TPS policy CI training/eval.
# Trains on hidden-only REUSE/REFRESH/ASK/SUMMARIZE forced-choice labels and
# evaluates policy behavior separately from Track A CLOCK/SILENT-GAP/PHASE.

set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
if [ "$#" -ne 0 ]; then
    echo "unsupported Track B arguments: $*" >&2
    exit 2
fi
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-8000}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-0}"
SCORING="${SCORING:-logprob}"
RUN_CHRONO_ONLY="${RUN_CHRONO_ONLY:-0}"

TRACK_DATA_DIR="$DATA_DIR/track_b"
TRACK_LOG_DIR="$LOG_DIR/track_b"
TRACK_CKPT_DIR="$CKPT_DIR/track_b"
TRACK_REPORT_DIR="$REPORT_DIR/track_b"
mkdir -p "$TRACK_DATA_DIR" "$TRACK_LOG_DIR" "$TRACK_CKPT_DIR" "$TRACK_REPORT_DIR"

fresh_path() {
    local path="$1"
    if [ -e "$path" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "refusing to overwrite existing artifact: $path (set FORCE=1 to replace)" >&2
        exit 1
    fi
}

ITEMS="$TRACK_DATA_DIR/tps_items.jsonl"
fresh_path "$ITEMS"
PYTHONPATH=. uv run python -m eval.tps.benchmark --out "$ITEMS"
test -s "$ITEMS"

REPORTS=()

VANILLA_REPORT="$TRACK_REPORT_DIR/vanilla_policy.json"
fresh_path "$VANILLA_REPORT"
PYTHONPATH=. uv run python -m eval.tps.run_tps \
    --adapter vanilla --items "$ITEMS" --out "$VANILLA_REPORT" \
    --base-model "$BASE_MODEL" --scoring "$SCORING" \
    2>&1 | tee "$TRACK_LOG_DIR/vanilla_eval_stdout.log"
test -s "$VANILLA_REPORT"
REPORTS+=("$VANILLA_REPORT")

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
for SEED in "${SEED_ARRAY[@]}"; do
    TAG="ci_policy_s${SEED}"
    TRAIN="$TRACK_DATA_DIR/tps_policy_train_seed${SEED}.jsonl"
    LOG_PATH="$TRACK_LOG_DIR/${TAG}.jsonl"
    STDOUT="$TRACK_LOG_DIR/${TAG}_stdout.log"
    CKPT="$TRACK_CKPT_DIR/${TAG}.pt"
    REPORT="$TRACK_REPORT_DIR/${TAG}.json"
    SENT="$TRACK_REPORT_DIR/${TAG}_DONE.txt"
    for path in "$TRAIN" "$LOG_PATH" "$STDOUT" "$CKPT" "$REPORT" "$SENT"; do
        fresh_path "$path"
    done

    echo "============================================================"
    echo "Track B CI policy seed $SEED -- tag $TAG"
    echo "============================================================"
    date -Iseconds | tee -a "$STDOUT"

    PYTHONPATH=. uv run python -m eval.tps.training_data \
        --out "$TRAIN" --seed "$SEED" --split train \
        2>&1 | tee -a "$STDOUT"
    test -s "$TRAIN"

    PYTHONPATH=. uv run python -u -m model.qwen_time_train \
        --data "$TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
        --log-every 200 --log-path "$LOG_PATH" --out "$CKPT" \
        --base "$BASE_MODEL" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$SEED" \
        2>&1 | tee -a "$STDOUT"
    test -s "$CKPT"

    PYTHONPATH=. uv run python -m eval.tps.run_tps \
        --adapter ci --checkpoint "$CKPT" --items "$ITEMS" --out "$REPORT" \
        --base-model "$BASE_MODEL" --scoring "$SCORING" \
        2>&1 | tee -a "$STDOUT"
    test -s "$REPORT"

    echo "{\"status\":\"done\",\"track\":\"B\",\"seed\":$SEED,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
    REPORTS+=("$REPORT")
done

# Matched negative control: same Track B labels with chrono gates frozen.
LORA_SEED="${SEED_ARRAY[0]}"
LORA_TAG="lora_only_policy_s${LORA_SEED}"
LORA_TRAIN="$TRACK_DATA_DIR/tps_policy_train_seed${LORA_SEED}.jsonl"
LORA_LOG="$TRACK_LOG_DIR/${LORA_TAG}.jsonl"
LORA_STDOUT="$TRACK_LOG_DIR/${LORA_TAG}_stdout.log"
LORA_CKPT="$TRACK_CKPT_DIR/${LORA_TAG}.pt"
LORA_REPORT="$TRACK_REPORT_DIR/${LORA_TAG}.json"
for path in "$LORA_LOG" "$LORA_STDOUT" "$LORA_CKPT" "$LORA_REPORT"; do
    fresh_path "$path"
done
PYTHONPATH=. uv run python -u -m model.qwen_time_train \
    --data "$LORA_TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
    --log-every 200 --log-path "$LORA_LOG" --out "$LORA_CKPT" \
    --base "$BASE_MODEL" --chunk-length 512 \
    --timescales "$TIMESCALES" --seed "$LORA_SEED" --freeze-alpha \
    2>&1 | tee -a "$LORA_STDOUT"
test -s "$LORA_CKPT"
PYTHONPATH=. uv run python -m eval.tps.run_tps \
    --adapter ci --checkpoint "$LORA_CKPT" --items "$ITEMS" --out "$LORA_REPORT" \
    --base-model "$BASE_MODEL" --scoring "$SCORING" \
    2>&1 | tee -a "$LORA_STDOUT"
test -s "$LORA_REPORT"
REPORTS+=("$LORA_REPORT")

if [ "$RUN_CHRONO_ONLY" = "1" ]; then
    CHRONO_TAG="chrono_only_policy_s${LORA_SEED}"
    CHRONO_LOG="$TRACK_LOG_DIR/${CHRONO_TAG}.jsonl"
    CHRONO_STDOUT="$TRACK_LOG_DIR/${CHRONO_TAG}_stdout.log"
    CHRONO_CKPT="$TRACK_CKPT_DIR/${CHRONO_TAG}.pt"
    CHRONO_REPORT="$TRACK_REPORT_DIR/${CHRONO_TAG}.json"
    for path in "$CHRONO_LOG" "$CHRONO_STDOUT" "$CHRONO_CKPT" "$CHRONO_REPORT"; do
        fresh_path "$path"
    done
    PYTHONPATH=. uv run python -u -m model.qwen_time_train \
        --data "$LORA_TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
        --log-every 200 --log-path "$CHRONO_LOG" --out "$CHRONO_CKPT" \
        --base "$BASE_MODEL" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$LORA_SEED" --freeze-lora \
        2>&1 | tee -a "$CHRONO_STDOUT"
    test -s "$CHRONO_CKPT"
    PYTHONPATH=. uv run python -m eval.tps.run_tps \
        --adapter ci --checkpoint "$CHRONO_CKPT" --items "$ITEMS" --out "$CHRONO_REPORT" \
        --base-model "$BASE_MODEL" --scoring "$SCORING" \
        2>&1 | tee -a "$CHRONO_STDOUT"
    test -s "$CHRONO_REPORT"
    REPORTS+=("$CHRONO_REPORT")
fi

HEADLINE="$TRACK_REPORT_DIR/headline.json"
fresh_path "$HEADLINE"
PYTHONPATH=. uv run python eval/tps/analyze.py \
    --inputs "${REPORTS[@]}" \
    --out "$HEADLINE"
test -s "$HEADLINE"

RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" SCRIPT_PATH="$SCRIPT_PATH" \
TRACK_DATA_DIR="$TRACK_DATA_DIR" TRACK_LOG_DIR="$TRACK_LOG_DIR" \
TRACK_CKPT_DIR="$TRACK_CKPT_DIR" TRACK_REPORT_DIR="$TRACK_REPORT_DIR" \
STEPS="$STEPS" LR="$LR" SEEDS="$SEEDS" SCORING="$SCORING" \
python3 - "${REPORTS[@]}" <<'PY' > "$RUN_ROOT/manifest.json"
import json
import os
import sys

print(json.dumps({
    "run_id": os.environ["RUN_ID"],
    "run_root": os.environ["RUN_ROOT"],
    "track": "B_policy",
    "reports": sys.argv[1:],
    "data_dir": os.environ["TRACK_DATA_DIR"],
    "log_dir": os.environ["TRACK_LOG_DIR"],
    "checkpoint_dir": os.environ["TRACK_CKPT_DIR"],
    "report_dir": os.environ["TRACK_REPORT_DIR"],
    "steps": int(os.environ["STEPS"]),
    "lr": float(os.environ["LR"]),
    "seeds": [int(x) for x in os.environ["SEEDS"].split(",") if x],
    "scoring": os.environ["SCORING"],
    "created_from": os.environ["SCRIPT_PATH"],
}, indent=2))
PY

echo "{\"status\":\"track_b_done\",\"run_id\":\"$RUN_ID\",\"time\":\"$(date -Iseconds)\"}" > "$RUN_ROOT/TRACK_B_DONE.txt"
