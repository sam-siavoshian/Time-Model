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
SAFE="${SAFE:-0}"
if [ "$SAFE" = "1" ]; then
    : "${STEPS:=800}"
    : "${SEEDS:=0}"
    : "${RUN_CHRONO_ONLY:=0}"
    : "${RUN_LORA_ONLY:=0}"
    : "${SKIP_VANILLA_EVAL:=1}"
    : "${EVAL_LIMIT:=648}"
    : "${CHUNK_LENGTH:=256}"
    : "${SAVE_EVERY:=100}"
    : "${EMPTY_CACHE_EVERY:=1}"
    : "${LOG_EVERY:=50}"
else
    : "${STEPS:=8000}"
    : "${SEEDS:=0}"
    : "${RUN_CHRONO_ONLY:=0}"
    : "${RUN_LORA_ONLY:=1}"
    : "${SKIP_VANILLA_EVAL:=0}"
    : "${EVAL_LIMIT:=0}"
    : "${CHUNK_LENGTH:=512}"
    : "${SAVE_EVERY:=0}"
    : "${EMPTY_CACHE_EVERY:=0}"
    : "${LOG_EVERY:=200}"
fi
LR="${LR:-1e-4}"
SCORING="${SCORING:-logprob}"
TRACK_B_INIT_FROM_TRACK_A="${TRACK_B_INIT_FROM_TRACK_A:-0}"
TRACK_A_CKPT_DIR="${TRACK_A_CKPT_DIR:-release_ckpts}"
TRACK_A_CKPT_PREFIX="${TRACK_A_CKPT_PREFIX:-qwen_time_v15s_20260523_141410_seed}"
WATCHDOG_INTERVAL_SEC="${WATCHDOG_INTERVAL_SEC:-5}"
WATCHDOG_REQUIRE_ISOLATED="${WATCHDOG_REQUIRE_ISOLATED:-1}"

TRACK_DATA_DIR="$DATA_DIR/track_b"
TRACK_LOG_DIR="$LOG_DIR/track_b"
TRACK_CKPT_DIR="$CKPT_DIR/track_b"
TRACK_REPORT_DIR="$REPORT_DIR/track_b"
for path in "$TRACK_DATA_DIR" "$TRACK_LOG_DIR" "$TRACK_CKPT_DIR" "$TRACK_REPORT_DIR"; do
    case "$path" in
        */track_a|*/track_a/*)
            echo "refusing Track B run with Track A path: $path" >&2
            exit 2
            ;;
    esac
done
mkdir -p "$TRACK_DATA_DIR" "$TRACK_LOG_DIR" "$TRACK_CKPT_DIR" "$TRACK_REPORT_DIR"

start_memory_watchdog() {
    if [ -z "${WATCHDOG_MEM_GB:-}" ]; then
        return 0
    fi
    local threshold_kb pgid log_path
    threshold_kb=$((WATCHDOG_MEM_GB * 1024 * 1024))
    pgid="$(ps -o pgid= "$$" | tr -d ' ')"
    if [ "$WATCHDOG_REQUIRE_ISOLATED" = "1" ] && [ "$pgid" != "$$" ]; then
        echo "refusing RAM watchdog without isolated process group: pid=$$ pgid=$pgid" >&2
        exit 2
    fi
    log_path="$RUN_ROOT/watchdog.log"
    (
        echo "$(date -Iseconds) RAM watchdog active: threshold=${WATCHDOG_MEM_GB}GiB interval=${WATCHDOG_INTERVAL_SEC}s pgid=$pgid" >> "$log_path"
        while true; do
            local avail_kb
            avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
            if [ -n "$avail_kb" ] && [ "$avail_kb" -le "$threshold_kb" ]; then
                echo "$(date -Iseconds) RAM watchdog killing pgid=$pgid MemAvailableKiB=$avail_kb thresholdKiB=$threshold_kb" >> "$log_path"
                echo "{\"status\":\"watchdog_killed\",\"run_id\":\"$RUN_ID\",\"mem_available_kib\":$avail_kb,\"threshold_kib\":$threshold_kb,\"time\":\"$(date -Iseconds)\"}" > "$RUN_ROOT/WATCHDOG_KILLED.txt"
                kill -TERM "-$pgid" 2>/dev/null || true
                sleep 10
                kill -KILL "-$pgid" 2>/dev/null || true
                exit 0
            fi
            sleep "$WATCHDOG_INTERVAL_SEC"
        done
    ) &
    WATCHDOG_PID=$!
    trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT
}

start_memory_watchdog

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

EVAL_LIMIT_ARGS=()
if [ "$EVAL_LIMIT" -gt 0 ]; then
    EVAL_LIMIT_ARGS=(--limit "$EVAL_LIMIT")
fi

if [ "$SKIP_VANILLA_EVAL" != "1" ]; then
    VANILLA_REPORT="$TRACK_REPORT_DIR/vanilla_policy.json"
    fresh_path "$VANILLA_REPORT"
    PYTHONPATH=. uv run python -m eval.tps.run_tps \
        --adapter vanilla --items "$ITEMS" --out "$VANILLA_REPORT" \
        --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
        "${EVAL_LIMIT_ARGS[@]}" \
        2>&1 | tee "$TRACK_LOG_DIR/vanilla_eval_stdout.log"
    test -s "$VANILLA_REPORT"
    REPORTS+=("$VANILLA_REPORT")
else
    echo "SAFE/SKIP_VANILLA_EVAL: skipping pre-training vanilla TPS eval"
fi

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
	    INIT_ARGS=()
	    if [ "$TRACK_B_INIT_FROM_TRACK_A" = "1" ]; then
	        INIT_CKPT="$TRACK_A_CKPT_DIR/${TRACK_A_CKPT_PREFIX}${SEED}.pt"
	        if [ ! -s "$INIT_CKPT" ]; then
	            echo "missing Track A init checkpoint for seed $SEED: $INIT_CKPT" >&2
	            exit 1
	        fi
	        INIT_ARGS=(--init-checkpoint "$INIT_CKPT")
	        echo "Track B fine-tune seed $SEED initialized from Track A checkpoint $INIT_CKPT" | tee -a "$STDOUT"
	    fi

	    PYTHONPATH=. uv run python -m eval.tps.training_data \
	        --out "$TRAIN" --seed "$SEED" --split train \
        2>&1 | tee -a "$STDOUT"
    test -s "$TRAIN"

	    PYTHONPATH=. uv run python -u -m model.qwen_time_train \
	        --data "$TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
		        --log-every "$LOG_EVERY" --log-path "$LOG_PATH" --out "$CKPT" \
		        --base "$BASE_MODEL" --chunk-length "$CHUNK_LENGTH" --save-every "$SAVE_EVERY" \
		        --empty-cache-every "$EMPTY_CACHE_EVERY" \
		        --timescales "$TIMESCALES" --seed "$SEED" "${INIT_ARGS[@]}" \
		        2>&1 | tee -a "$STDOUT"
    test -s "$CKPT"

	    PYTHONPATH=. uv run python -m eval.tps.run_tps \
	        --adapter ci --checkpoint "$CKPT" --items "$ITEMS" --out "$REPORT" \
	        --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
	        "${EVAL_LIMIT_ARGS[@]}" \
	        2>&1 | tee -a "$STDOUT"
    test -s "$REPORT"

    echo "{\"status\":\"done\",\"track\":\"B\",\"seed\":$SEED,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
    REPORTS+=("$REPORT")
done

# Matched negative control: same Track B labels with chrono gates frozen.
LORA_SEED="${SEED_ARRAY[0]}"
LORA_TRAIN="$TRACK_DATA_DIR/tps_policy_train_seed${LORA_SEED}.jsonl"

if [ "$RUN_LORA_ONLY" = "1" ]; then
    LORA_TAG="lora_only_policy_s${LORA_SEED}"
    LORA_LOG="$TRACK_LOG_DIR/${LORA_TAG}.jsonl"
    LORA_STDOUT="$TRACK_LOG_DIR/${LORA_TAG}_stdout.log"
    LORA_CKPT="$TRACK_CKPT_DIR/${LORA_TAG}.pt"
    LORA_REPORT="$TRACK_REPORT_DIR/${LORA_TAG}.json"
    for path in "$LORA_LOG" "$LORA_STDOUT" "$LORA_CKPT" "$LORA_REPORT"; do
        fresh_path "$path"
    done
    LORA_INIT_ARGS=()
    if [ "$TRACK_B_INIT_FROM_TRACK_A" = "1" ]; then
        LORA_INIT_CKPT="$TRACK_A_CKPT_DIR/${TRACK_A_CKPT_PREFIX}${LORA_SEED}.pt"
        if [ ! -s "$LORA_INIT_CKPT" ]; then
            echo "missing Track A init checkpoint for LoRA control seed $LORA_SEED: $LORA_INIT_CKPT" >&2
            exit 1
        fi
        LORA_INIT_ARGS=(--init-checkpoint "$LORA_INIT_CKPT")
        echo "Track B LoRA-only control seed $LORA_SEED initialized from Track A checkpoint $LORA_INIT_CKPT" | tee -a "$LORA_STDOUT"
    fi
    PYTHONPATH=. uv run python -u -m model.qwen_time_train \
        --data "$LORA_TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
        --log-every "$LOG_EVERY" --log-path "$LORA_LOG" --out "$LORA_CKPT" \
        --base "$BASE_MODEL" --chunk-length "$CHUNK_LENGTH" --save-every "$SAVE_EVERY" \
        --empty-cache-every "$EMPTY_CACHE_EVERY" \
        --timescales "$TIMESCALES" --seed "$LORA_SEED" --freeze-alpha "${LORA_INIT_ARGS[@]}" \
        2>&1 | tee -a "$LORA_STDOUT"
    test -s "$LORA_CKPT"
    PYTHONPATH=. uv run python -m eval.tps.run_tps \
        --adapter ci --checkpoint "$LORA_CKPT" --items "$ITEMS" --out "$LORA_REPORT" \
        --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
        "${EVAL_LIMIT_ARGS[@]}" \
        2>&1 | tee -a "$LORA_STDOUT"
    test -s "$LORA_REPORT"
    REPORTS+=("$LORA_REPORT")
fi

if [ "$RUN_CHRONO_ONLY" = "1" ]; then
    CHRONO_TAG="chrono_only_policy_s${LORA_SEED}"
    CHRONO_LOG="$TRACK_LOG_DIR/${CHRONO_TAG}.jsonl"
    CHRONO_STDOUT="$TRACK_LOG_DIR/${CHRONO_TAG}_stdout.log"
    CHRONO_CKPT="$TRACK_CKPT_DIR/${CHRONO_TAG}.pt"
    CHRONO_REPORT="$TRACK_REPORT_DIR/${CHRONO_TAG}.json"
	    for path in "$CHRONO_LOG" "$CHRONO_STDOUT" "$CHRONO_CKPT" "$CHRONO_REPORT"; do
	        fresh_path "$path"
	    done
	    CHRONO_INIT_ARGS=()
	    if [ "$TRACK_B_INIT_FROM_TRACK_A" = "1" ]; then
	        CHRONO_INIT_CKPT="$TRACK_A_CKPT_DIR/${TRACK_A_CKPT_PREFIX}${LORA_SEED}.pt"
	        if [ ! -s "$CHRONO_INIT_CKPT" ]; then
	            echo "missing Track A init checkpoint for chrono control seed $LORA_SEED: $CHRONO_INIT_CKPT" >&2
	            exit 1
	        fi
	        CHRONO_INIT_ARGS=(--init-checkpoint "$CHRONO_INIT_CKPT")
	        echo "Track B chrono-only control seed $LORA_SEED initialized from Track A checkpoint $CHRONO_INIT_CKPT" | tee -a "$CHRONO_STDOUT"
	    fi
	    PYTHONPATH=. uv run python -u -m model.qwen_time_train \
	        --data "$LORA_TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
	        --log-every "$LOG_EVERY" --log-path "$CHRONO_LOG" --out "$CHRONO_CKPT" \
	        --base "$BASE_MODEL" --chunk-length "$CHUNK_LENGTH" --save-every "$SAVE_EVERY" \
	        --empty-cache-every "$EMPTY_CACHE_EVERY" \
	        --timescales "$TIMESCALES" --seed "$LORA_SEED" --freeze-lora "${CHRONO_INIT_ARGS[@]}" \
        2>&1 | tee -a "$CHRONO_STDOUT"
    test -s "$CHRONO_CKPT"
	    PYTHONPATH=. uv run python -m eval.tps.run_tps \
	        --adapter ci --checkpoint "$CHRONO_CKPT" --items "$ITEMS" --out "$CHRONO_REPORT" \
	        --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
	        "${EVAL_LIMIT_ARGS[@]}" \
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
	STEPS="$STEPS" LR="$LR" SEEDS="$SEEDS" SCORING="$SCORING" SAFE="$SAFE" \
	CHUNK_LENGTH="$CHUNK_LENGTH" SAVE_EVERY="$SAVE_EVERY" EVAL_LIMIT="$EVAL_LIMIT" \
	EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
	SKIP_VANILLA_EVAL="$SKIP_VANILLA_EVAL" RUN_LORA_ONLY="$RUN_LORA_ONLY" \
	TRACK_B_INIT_FROM_TRACK_A="$TRACK_B_INIT_FROM_TRACK_A" TRACK_A_CKPT_DIR="$TRACK_A_CKPT_DIR" \
	TRACK_A_CKPT_PREFIX="$TRACK_A_CKPT_PREFIX" \
	python3 - "${REPORTS[@]}" <<'PY' > "$RUN_ROOT/manifest.json"
import json
import os
import sys

print(json.dumps({
    "run_id": os.environ["RUN_ID"],
    "run_root": os.environ["RUN_ROOT"],
    "track": "B_policy",
    "model_family": "track_b_policy_adapters",
    "reports": sys.argv[1:],
    "data_dir": os.environ["TRACK_DATA_DIR"],
    "log_dir": os.environ["TRACK_LOG_DIR"],
    "checkpoint_dir": os.environ["TRACK_CKPT_DIR"],
    "report_dir": os.environ["TRACK_REPORT_DIR"],
	    "steps": int(os.environ["STEPS"]),
	    "lr": float(os.environ["LR"]),
	    "seeds": [int(x) for x in os.environ["SEEDS"].split(",") if x],
	    "scoring": os.environ["SCORING"],
	    "safe": os.environ["SAFE"] == "1",
	    "chunk_length": int(os.environ["CHUNK_LENGTH"]),
	    "save_every": int(os.environ["SAVE_EVERY"]),
	    "empty_cache_every": int(os.environ["EMPTY_CACHE_EVERY"]),
	    "eval_limit": int(os.environ["EVAL_LIMIT"]),
	    "skip_vanilla_eval": os.environ["SKIP_VANILLA_EVAL"] == "1",
	    "run_lora_only": os.environ["RUN_LORA_ONLY"] == "1",
	    "init_from_track_a": os.environ["TRACK_B_INIT_FROM_TRACK_A"] == "1",
	    "track_a_checkpoint_dir": os.environ["TRACK_A_CKPT_DIR"],
	    "track_a_checkpoint_prefix": os.environ["TRACK_A_CKPT_PREFIX"],
	    "created_from": os.environ["SCRIPT_PATH"],
	}, indent=2))
PY

echo "{\"status\":\"track_b_done\",\"run_id\":\"$RUN_ID\",\"time\":\"$(date -Iseconds)\"}" > "$RUN_ROOT/TRACK_B_DONE.txt"
