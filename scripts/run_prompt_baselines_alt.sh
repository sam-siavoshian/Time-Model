#!/usr/bin/env bash
# Phase B: alternative prompt-format baselines for W6/reviewer concern.
# Trains prompt baselines with ISO-timestamp + natural-language formats.
# Each seed 0 only initially (rapid eval). If first run shows
# meaningfully different result from [elapsed: X] baseline, expand to n=3.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports checkpoints data/processed

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

# === ISO TIMESTAMP FORMAT ===
DATA_ISO="data/processed/prompt_iso_s0_18k.jsonl"
CKPT_ISO="checkpoints/prompt_iso_s0.pt"
REC_ISO="reports/prompt_iso_s0_recall.json"
echo "=== ISO timestamp format train ==="
[ ! -s "$DATA_ISO" ] && uv run python -m model.qwen_time_data_prompt_iso --n "$STEPS" --seed 0 --out "$DATA_ISO" 2>&1 | tee logs/prompt_iso_s0.log
if [ ! -f "$CKPT_ISO" ]; then
    uv run python -m model.qwen_time_train --data "$DATA_ISO" --steps "$STEPS" --seed 0 \
        --timescales "$TIMESCALES" --freeze-alpha --out "$CKPT_ISO" 2>&1 | tee -a logs/prompt_iso_s0.log
fi
# Eval with --inject-prompt + tell check it's ISO format
if [ ! -s "$REC_ISO" ]; then
    uv run python -m model.qwen_time_check --checkpoint "$CKPT_ISO" --timescales "$TIMESCALES" \
        --inject-prompt --prompt-format iso --out "$REC_ISO" 2>&1 | tee -a logs/prompt_iso_s0.log
fi

# === NATURAL LANGUAGE FORMAT ===
DATA_NL="data/processed/prompt_nl_s0_18k.jsonl"
CKPT_NL="checkpoints/prompt_nl_s0.pt"
REC_NL="reports/prompt_nl_s0_recall.json"
echo "=== Natural language format train ==="
[ ! -s "$DATA_NL" ] && uv run python -m model.qwen_time_data_prompt_nl --n "$STEPS" --seed 0 --out "$DATA_NL" 2>&1 | tee logs/prompt_nl_s0.log
if [ ! -f "$CKPT_NL" ]; then
    uv run python -m model.qwen_time_train --data "$DATA_NL" --steps "$STEPS" --seed 0 \
        --timescales "$TIMESCALES" --freeze-alpha --out "$CKPT_NL" 2>&1 | tee -a logs/prompt_nl_s0.log
fi
if [ ! -s "$REC_NL" ]; then
    uv run python -m model.qwen_time_check --checkpoint "$CKPT_NL" --timescales "$TIMESCALES" \
        --inject-prompt --prompt-format nl --out "$REC_NL" 2>&1 | tee -a logs/prompt_nl_s0.log
fi

touch logs/prompt_baselines_alt.done
echo "all alternative prompt baselines done"
