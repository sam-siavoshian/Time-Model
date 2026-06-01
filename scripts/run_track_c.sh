#!/usr/bin/env bash
# Track C: compositional temporal-state reasoning.

set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
if [ "$#" -ne 0 ]; then
    echo "unsupported Track C arguments: $*" >&2
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
    : "${DATA_PROFILE:=safe}"
    : "${EVAL_LIMIT:=200}"
	    : "${RUN_PROMPT_LORA:=0}"
	    : "${RUN_LORA_ONLY:=0}"
	    : "${SKIP_VANILLA_EVAL:=0}"
	    : "${CHUNK_LENGTH:=256}"
	    : "${SAVE_EVERY:=100}"
	    : "${SELECT_BY_VAL:=1}"
	    : "${VAL_LIMIT:=200}"
	    : "${EMPTY_CACHE_EVERY:=1}"
	    : "${LOG_EVERY:=50}"
else
    : "${STEPS:=8000}"
    : "${SEEDS:=0,1,2}"
    : "${DATA_PROFILE:=full}"
    : "${EVAL_LIMIT:=0}"
    : "${RUN_PROMPT_LORA:=1}"
    : "${RUN_LORA_ONLY:=1}"
    : "${SKIP_VANILLA_EVAL:=0}"
	    : "${CHUNK_LENGTH:=512}"
	    : "${SAVE_EVERY:=500}"
	    : "${SELECT_BY_VAL:=1}"
	    : "${VAL_LIMIT:=0}"
	    : "${EMPTY_CACHE_EVERY:=10}"
	    : "${LOG_EVERY:=200}"
fi
LR="${LR:-1e-4}"
SCORING="${SCORING:-logprob}"
WATCHDOG_INTERVAL_SEC="${WATCHDOG_INTERVAL_SEC:-5}"
WATCHDOG_REQUIRE_ISOLATED="${WATCHDOG_REQUIRE_ISOLATED:-1}"

TRACK_DATA_DIR="$DATA_DIR/track_c"
TRACK_LOG_DIR="$LOG_DIR/track_c"
TRACK_CKPT_DIR="$CKPT_DIR/track_c"
TRACK_REPORT_DIR="$REPORT_DIR/track_c"
PRED_DIR="$TRACK_REPORT_DIR/predictions"
TABLE_DIR="$TRACK_REPORT_DIR/tables"
FIG_DIR="$TRACK_REPORT_DIR/figures"
for path in "$TRACK_DATA_DIR" "$TRACK_LOG_DIR" "$TRACK_CKPT_DIR" "$TRACK_REPORT_DIR"; do
    case "$path" in
        */track_a|*/track_a/*|*/track_b|*/track_b/*)
            echo "refusing Track C run with non-Track-C path: $path" >&2
            exit 2
            ;;
    esac
done
mkdir -p "$TRACK_DATA_DIR" "$TRACK_LOG_DIR" "$TRACK_CKPT_DIR" "$PRED_DIR" "$TABLE_DIR" "$FIG_DIR"

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

fresh_path() {
    local path="$1"
    if [ -e "$path" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "refusing to overwrite existing artifact: $path (set FORCE=1 to replace)" >&2
        exit 1
    fi
}

eval_limit_args=()
if [ "$EVAL_LIMIT" -gt 0 ]; then
    eval_limit_args=(--limit "$EVAL_LIMIT")
fi
val_limit_args=()
if [ "$VAL_LIMIT" -gt 0 ]; then
    val_limit_args=(--limit "$VAL_LIMIT")
fi

start_memory_watchdog

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
EVAL_SPLITS=(standard_test heldout_template heldout_duration heldout_composition heldout_family)
CONDITIONS=(ci_hidden_time no_time_control shuffled_time_control prompt_timestamp both_agree conflict)
BASELINE_REPORTS=()

for SEED in "${SEED_ARRAY[@]}"; do
    echo "============================================================"
    echo "Track C seed $SEED"
    echo "============================================================"
    PYTHONPATH=. uv run python -m eval.track_c.generate \
        --out-dir "$TRACK_DATA_DIR" --seed "$SEED" --profile "$DATA_PROFILE"

    TRAIN_ITEMS="$TRACK_DATA_DIR/track_c_seed${SEED}_train.jsonl"
    TRAIN_HIDDEN="$TRACK_DATA_DIR/track_c_train_seed${SEED}_hidden_only.jsonl"
    fresh_path "$TRAIN_HIDDEN"
    PYTHONPATH=. uv run python -m eval.track_c.training_data \
        --items "$TRAIN_ITEMS" --out "$TRAIN_HIDDEN" --seed "$SEED" --condition hidden_only

    TAG="ci_track_c_s${SEED}"
    LOG_PATH="$TRACK_LOG_DIR/${TAG}.jsonl"
    STDOUT="$TRACK_LOG_DIR/${TAG}_stdout.log"
    CKPT="$TRACK_CKPT_DIR/${TAG}.pt"
    for path in "$LOG_PATH" "$STDOUT" "$CKPT"; do
        fresh_path "$path"
    done
    PYTHONPATH=. uv run python -u -m model.qwen_time_train \
        --data "$TRAIN_HIDDEN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
        --log-every "$LOG_EVERY" --log-path "$LOG_PATH" --out "$CKPT" \
        --base "$BASE_MODEL" --chunk-length "$CHUNK_LENGTH" --save-every "$SAVE_EVERY" \
        --loss-mode forced_choice \
        --empty-cache-every "$EMPTY_CACHE_EVERY" \
        --timescales "$TIMESCALES" --seed "$SEED" \
        2>&1 | tee -a "$STDOUT"
    test -s "$CKPT"
    SELECTED_CKPT="$CKPT"
    if [ "$SELECT_BY_VAL" = "1" ]; then
        VAL_ITEMS="$TRACK_DATA_DIR/track_c_seed${SEED}_val.jsonl"
        VAL_PRED_DIR="$TRACK_REPORT_DIR/validation/${TAG}"
        mkdir -p "$VAL_PRED_DIR"
        CANDIDATES=()
        for CANDIDATE in "$TRACK_CKPT_DIR/${TAG}.step"*.pt; do
            if [ -e "$CANDIDATE" ]; then
                CANDIDATES+=("$CANDIDATE")
            fi
        done
        CANDIDATES+=("$CKPT")
        for CANDIDATE in "${CANDIDATES[@]}"; do
            CANDIDATE_NAME="$(basename "$CANDIDATE" .pt)"
            VAL_REPORT="$VAL_PRED_DIR/${CANDIDATE_NAME}_val.json"
            fresh_path "$VAL_REPORT"
            PYTHONPATH=. uv run python -m eval.track_c.run_track_c \
                --adapter ci --checkpoint "$CANDIDATE" --items "$VAL_ITEMS" --out "$VAL_REPORT" \
                --condition ci_hidden_time --model-tag "$TAG" \
                --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
                "${val_limit_args[@]}" \
                2>&1 | tee -a "$STDOUT"
        done
        SELECT_JSON="$TRACK_REPORT_DIR/${TAG}_checkpoint_selection.json"
        fresh_path "$SELECT_JSON"
        SELECTED_REL="$(PYTHONPATH=. uv run python -m eval.track_c.select_checkpoint \
            --inputs "$VAL_PRED_DIR"/*.json --metric balanced_acc --out "$SELECT_JSON" | tail -n 1)"
        if [[ "$SELECTED_REL" = /* ]]; then
            SELECTED_CKPT="$SELECTED_REL"
        else
            SELECTED_CKPT="$REPO_ROOT/$SELECTED_REL"
        fi
        test -s "$SELECTED_CKPT"
        echo "Selected validation checkpoint for $TAG: $SELECTED_CKPT" | tee -a "$STDOUT"
    fi

    for SPLIT in "${EVAL_SPLITS[@]}"; do
        ITEMS="$TRACK_DATA_DIR/track_c_seed${SEED}_${SPLIT}.jsonl"
        for CONDITION in "${CONDITIONS[@]}"; do
            REPORT="$PRED_DIR/${TAG}_${CONDITION}_${SPLIT}.json"
            fresh_path "$REPORT"
            PYTHONPATH=. uv run python -m eval.track_c.run_track_c \
                --adapter ci --checkpoint "$SELECTED_CKPT" --items "$ITEMS" --out "$REPORT" \
                --condition "$CONDITION" --model-tag "$TAG" \
                --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
                "${eval_limit_args[@]}" \
                2>&1 | tee -a "$STDOUT"
        done
    done

    BASELINE_REPORT="$TRACK_REPORT_DIR/linear_baselines_s${SEED}.json"
    fresh_path "$BASELINE_REPORT"
    PYTHONPATH=. uv run python -m eval.track_c.baselines \
        --train "$TRAIN_ITEMS" \
        --eval \
        "$TRACK_DATA_DIR/track_c_seed${SEED}_standard_test.jsonl" \
        "$TRACK_DATA_DIR/track_c_seed${SEED}_heldout_template.jsonl" \
        "$TRACK_DATA_DIR/track_c_seed${SEED}_heldout_duration.jsonl" \
        "$TRACK_DATA_DIR/track_c_seed${SEED}_heldout_composition.jsonl" \
        "$TRACK_DATA_DIR/track_c_seed${SEED}_heldout_family.jsonl" \
        --out "$BASELINE_REPORT"
    BASELINE_REPORTS+=("$BASELINE_REPORT")
done

if [ "$SKIP_VANILLA_EVAL" != "1" ]; then
    for CONTROL_SEED in "${SEED_ARRAY[@]}"; do
        for SPLIT in "${EVAL_SPLITS[@]}"; do
            ITEMS="$TRACK_DATA_DIR/track_c_seed${CONTROL_SEED}_${SPLIT}.jsonl"
            for CONDITION in no_time_control prompt_timestamp; do
                REPORT="$PRED_DIR/vanilla_s${CONTROL_SEED}_${CONDITION}_${SPLIT}.json"
                fresh_path "$REPORT"
                PYTHONPATH=. uv run python -m eval.track_c.run_track_c \
                    --adapter vanilla --items "$ITEMS" --out "$REPORT" \
                    --condition "$CONDITION" --model-tag "vanilla_s${CONTROL_SEED}" \
                    --base-model "$BASE_MODEL" --scoring "$SCORING" \
                    "${eval_limit_args[@]}"
            done
        done
    done
fi

if [ "$RUN_LORA_ONLY" = "1" ]; then
    for CONTROL_SEED in "${SEED_ARRAY[@]}"; do
        LORA_TAG="lora_only_track_c_s${CONTROL_SEED}"
        LORA_LOG="$TRACK_LOG_DIR/${LORA_TAG}.jsonl"
        LORA_STDOUT="$TRACK_LOG_DIR/${LORA_TAG}_stdout.log"
        LORA_CKPT="$TRACK_CKPT_DIR/${LORA_TAG}.pt"
        for path in "$LORA_LOG" "$LORA_STDOUT" "$LORA_CKPT"; do
            fresh_path "$path"
        done
        PYTHONPATH=. uv run python -u -m model.qwen_time_train \
            --data "$TRACK_DATA_DIR/track_c_train_seed${CONTROL_SEED}_hidden_only.jsonl" \
            --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
            --log-every "$LOG_EVERY" --log-path "$LORA_LOG" --out "$LORA_CKPT" \
            --base "$BASE_MODEL" --chunk-length "$CHUNK_LENGTH" --save-every "$SAVE_EVERY" \
            --loss-mode forced_choice \
            --empty-cache-every "$EMPTY_CACHE_EVERY" \
            --timescales "$TIMESCALES" --seed "$CONTROL_SEED" --freeze-alpha \
            2>&1 | tee -a "$LORA_STDOUT"
        test -s "$LORA_CKPT"
        for SPLIT in "${EVAL_SPLITS[@]}"; do
            REPORT="$PRED_DIR/${LORA_TAG}_ci_hidden_time_${SPLIT}.json"
            fresh_path "$REPORT"
            PYTHONPATH=. uv run python -m eval.track_c.run_track_c \
                --adapter ci --checkpoint "$LORA_CKPT" \
                --items "$TRACK_DATA_DIR/track_c_seed${CONTROL_SEED}_${SPLIT}.jsonl" \
                --out "$REPORT" --condition ci_hidden_time --model-tag "$LORA_TAG" \
                --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
                "${eval_limit_args[@]}"
        done
    done
fi

if [ "$RUN_PROMPT_LORA" = "1" ]; then
    for CONTROL_SEED in "${SEED_ARRAY[@]}"; do
        PROMPT_TRAIN="$TRACK_DATA_DIR/track_c_train_seed${CONTROL_SEED}_prompt_timestamp.jsonl"
        fresh_path "$PROMPT_TRAIN"
        PYTHONPATH=. uv run python -m eval.track_c.training_data \
            --items "$TRACK_DATA_DIR/track_c_seed${CONTROL_SEED}_train.jsonl" \
            --out "$PROMPT_TRAIN" --seed "$CONTROL_SEED" --condition prompt_timestamp
        PROMPT_TAG="prompt_lora_track_c_s${CONTROL_SEED}"
        PROMPT_LOG="$TRACK_LOG_DIR/${PROMPT_TAG}.jsonl"
        PROMPT_STDOUT="$TRACK_LOG_DIR/${PROMPT_TAG}_stdout.log"
        PROMPT_CKPT="$TRACK_CKPT_DIR/${PROMPT_TAG}.pt"
        for path in "$PROMPT_LOG" "$PROMPT_STDOUT" "$PROMPT_CKPT"; do
            fresh_path "$path"
        done
        PYTHONPATH=. uv run python -u -m model.qwen_time_train \
            --data "$PROMPT_TRAIN" --steps "$STEPS" --lr "$LR" --device "$DEVICE" \
            --log-every "$LOG_EVERY" --log-path "$PROMPT_LOG" --out "$PROMPT_CKPT" \
            --base "$BASE_MODEL" --chunk-length "$CHUNK_LENGTH" --save-every "$SAVE_EVERY" \
            --loss-mode forced_choice \
            --empty-cache-every "$EMPTY_CACHE_EVERY" \
            --timescales "$TIMESCALES" --seed "$CONTROL_SEED" --freeze-alpha \
            2>&1 | tee -a "$PROMPT_STDOUT"
        test -s "$PROMPT_CKPT"
        for SPLIT in "${EVAL_SPLITS[@]}"; do
            REPORT="$PRED_DIR/${PROMPT_TAG}_prompt_timestamp_${SPLIT}.json"
            fresh_path "$REPORT"
            PYTHONPATH=. uv run python -m eval.track_c.run_track_c \
                --adapter ci --checkpoint "$PROMPT_CKPT" \
                --items "$TRACK_DATA_DIR/track_c_seed${CONTROL_SEED}_${SPLIT}.jsonl" \
                --out "$REPORT" --condition prompt_timestamp --model-tag "$PROMPT_TAG" \
                --base-model "$BASE_MODEL" --scoring "$SCORING" --chunk-length "$CHUNK_LENGTH" \
                "${eval_limit_args[@]}"
        done
    done
fi

HEADLINE="$TRACK_REPORT_DIR/headline.json"
fresh_path "$HEADLINE"
PYTHONPATH=. uv run python -m eval.track_c.analyze \
    --input-glob "$PRED_DIR/*.json" \
    --baselines "${BASELINE_REPORTS[@]}" \
    --out "$HEADLINE"
test -s "$HEADLINE"

PYTHONPATH=. uv run python -m eval.track_c.tables --headline "$HEADLINE" --out-dir "$TABLE_DIR"
PYTHONPATH=. uv run python -m eval.track_c.plots --headline "$HEADLINE" --out-dir "$FIG_DIR"

RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" SCRIPT_PATH="$SCRIPT_PATH" \
TRACK_DATA_DIR="$TRACK_DATA_DIR" TRACK_LOG_DIR="$TRACK_LOG_DIR" \
TRACK_CKPT_DIR="$TRACK_CKPT_DIR" TRACK_REPORT_DIR="$TRACK_REPORT_DIR" \
STEPS="$STEPS" LR="$LR" SEEDS="$SEEDS" SCORING="$SCORING" SAFE="$SAFE" \
DATA_PROFILE="$DATA_PROFILE" CHUNK_LENGTH="$CHUNK_LENGTH" SAVE_EVERY="$SAVE_EVERY" \
EVAL_LIMIT="$EVAL_LIMIT" VAL_LIMIT="$VAL_LIMIT" SELECT_BY_VAL="$SELECT_BY_VAL" \
EMPTY_CACHE_EVERY="$EMPTY_CACHE_EVERY" \
RUN_PROMPT_LORA="$RUN_PROMPT_LORA" RUN_LORA_ONLY="$RUN_LORA_ONLY" \
python3 - <<'PY' > "$RUN_ROOT/manifest.json"
import json
import os

print(json.dumps({
    "run_id": os.environ["RUN_ID"],
    "run_root": os.environ["RUN_ROOT"],
    "track": "C_compositional_temporal_state",
    "model_family": "track_c_policy_adapters",
    "data_dir": os.environ["TRACK_DATA_DIR"],
    "log_dir": os.environ["TRACK_LOG_DIR"],
    "checkpoint_dir": os.environ["TRACK_CKPT_DIR"],
    "report_dir": os.environ["TRACK_REPORT_DIR"],
    "steps": int(os.environ["STEPS"]),
    "lr": float(os.environ["LR"]),
    "seeds": [int(x) for x in os.environ["SEEDS"].split(",") if x],
    "scoring": os.environ["SCORING"],
    "safe": os.environ["SAFE"] == "1",
    "data_profile": os.environ["DATA_PROFILE"],
    "chunk_length": int(os.environ["CHUNK_LENGTH"]),
    "save_every": int(os.environ["SAVE_EVERY"]),
    "select_by_val": os.environ["SELECT_BY_VAL"] == "1",
    "val_limit": int(os.environ["VAL_LIMIT"]),
    "empty_cache_every": int(os.environ["EMPTY_CACHE_EVERY"]),
    "eval_limit": int(os.environ["EVAL_LIMIT"]),
    "run_prompt_lora": os.environ["RUN_PROMPT_LORA"] == "1",
    "run_lora_only": os.environ["RUN_LORA_ONLY"] == "1",
    "created_from": os.environ["SCRIPT_PATH"],
}, indent=2))
PY

echo "{\"status\":\"track_c_done\",\"run_id\":\"$RUN_ID\",\"time\":\"$(date -Iseconds)\"}" > "$RUN_ROOT/TRACK_C_DONE.txt"
echo "TRACK C COMPLETE: $RUN_ID"
