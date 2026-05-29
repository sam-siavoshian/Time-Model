#!/usr/bin/env bash
# W10 (prior review) + sufficiency: chrono channel on, LoRA frozen at zero.
# Tests whether chrono alone (no LoRA surface capacity) can fit T1-T4.
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

# Note: requires trainer to support --freeze-lora flag. If not present,
# the script will error -- add the flag to qwen_time_train.py first.
for SEED in 0 1 2; do
    DATA="$DATA_DIR/v15s_seed${SEED}_18k.jsonl"
    OUT="$CKPT_DIR/chrono_only_s${SEED}.pt"
    REC="$REPORT_DIR/chrono_only_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/chrono_only_s${SEED}.log"
    if [ ! -f "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --seed "$SEED" \
        --timescales "$TIMESCALES" --freeze-lora \
        --out "$OUT" 2>&1 | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --out "$REC" 2>&1 | tee -a "$STDOUT"
    test -s "$REC"
done
time_model_done "$SCRIPT_PATH" "chrono_only.done"
