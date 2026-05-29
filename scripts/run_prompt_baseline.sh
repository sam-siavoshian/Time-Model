#!/usr/bin/env bash
# W6 baseline: prompt-injected tau. Three seeds at v15-matched budget.
# Same data distribution as v15 but with tau text-prefixed; chrono
# channel forced off via --freeze-alpha so tau MUST come from the prompt.
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

for SEED in 0 1 2; do
    DATA="$DATA_DIR/prompt_baseline_s${SEED}_18k.jsonl"
    OUT="$CKPT_DIR/prompt_baseline_s${SEED}.pt"
    REC="$REPORT_DIR/prompt_baseline_injected_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/prompt_baseline_s${SEED}.log"

    echo "=== seed $SEED data ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_data_prompt \
        --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"

    echo "=== seed $SEED train ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_train \
        --data "$DATA" --steps "$STEPS" --seed "$SEED" \
        --timescales "$TIMESCALES" --freeze-alpha \
        --out "$OUT" 2>&1 | tee -a "$STDOUT"

    echo "=== seed $SEED check ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check \
        --ckpt "$OUT" --inject-prompt --out "$REC" 2>&1 | tee -a "$STDOUT"
    test -s "$REC"
done

time_model_done "$SCRIPT_PATH" "prompt_baseline.done"
echo "all 3 seeds done"
