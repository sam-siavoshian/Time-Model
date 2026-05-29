#!/usr/bin/env bash
# Phase B: alternative prompt-format baselines for W6/reviewer concern.
# Trains prompt baselines with ISO-timestamp + natural-language formats.
# Each seed 0 only initially (rapid eval). If first run shows
# meaningfully different result from [elapsed: X] baseline, expand to n=3.
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

# === ISO TIMESTAMP FORMAT ===
DATA_ISO="$DATA_DIR/prompt_iso_s0_18k.jsonl"
CKPT_ISO="$CKPT_DIR/prompt_iso_s0.pt"
REC_ISO="$REPORT_DIR/prompt_iso_s0_recall.json"
LOG_ISO="$LOG_DIR/prompt_iso_s0.log"
echo "=== ISO timestamp format train ==="
[ ! -s "$DATA_ISO" ] && uv run python -m model.qwen_time_data_prompt_iso --n "$STEPS" --seed 0 --out "$DATA_ISO" 2>&1 | tee "$LOG_ISO"
if [ ! -f "$CKPT_ISO" ]; then
    uv run python -m model.qwen_time_train --data "$DATA_ISO" --steps "$STEPS" --seed 0 \
        --timescales "$TIMESCALES" --freeze-alpha --out "$CKPT_ISO" 2>&1 | tee -a "$LOG_ISO"
fi
# Eval with --inject-prompt + tell check it's ISO format
if [ ! -s "$REC_ISO" ]; then
    uv run python -m model.qwen_time_check --checkpoint "$CKPT_ISO" --timescales "$TIMESCALES" \
        --inject-prompt --prompt-format iso --out "$REC_ISO" 2>&1 | tee -a "$LOG_ISO"
fi

# === NATURAL LANGUAGE FORMAT ===
DATA_NL="$DATA_DIR/prompt_nl_s0_18k.jsonl"
CKPT_NL="$CKPT_DIR/prompt_nl_s0.pt"
REC_NL="$REPORT_DIR/prompt_nl_s0_recall.json"
LOG_NL="$LOG_DIR/prompt_nl_s0.log"
echo "=== Natural language format train ==="
[ ! -s "$DATA_NL" ] && uv run python -m model.qwen_time_data_prompt_nl --n "$STEPS" --seed 0 --out "$DATA_NL" 2>&1 | tee "$LOG_NL"
if [ ! -f "$CKPT_NL" ]; then
    uv run python -m model.qwen_time_train --data "$DATA_NL" --steps "$STEPS" --seed 0 \
        --timescales "$TIMESCALES" --freeze-alpha --out "$CKPT_NL" 2>&1 | tee -a "$LOG_NL"
fi
if [ ! -s "$REC_NL" ]; then
    uv run python -m model.qwen_time_check --checkpoint "$CKPT_NL" --timescales "$TIMESCALES" \
        --inject-prompt --prompt-format nl --out "$REC_NL" 2>&1 | tee -a "$LOG_NL"
fi

test -s "$REC_ISO"
test -s "$REC_NL"
time_model_done "$SCRIPT_PATH" "prompt_baselines_alt.done"
echo "all alternative prompt baselines done"
