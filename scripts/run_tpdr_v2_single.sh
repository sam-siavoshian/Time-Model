#!/usr/bin/env bash
# Single-pair TPDR runner for parallel execution.
# Usage: bash scripts/run_tpdr_v2_single.sh <CI_SEED> <PR_SEED>
#
# Same pre-reg parameters as scripts/run_tpdr_v2_prereg.sh
# (200 scen, 10 tau, max_new=150, greedy) but only runs ONE seed pair.
# Used to parallelize the headline + cross-seed pairs once GPU clears.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports/tpdr_crossseed

CI_SEED=${1:?Usage: $0 CI_SEED PR_SEED}
PR_SEED=${2:?Usage: $0 CI_SEED PR_SEED}
OUT="reports/tpdr_crossseed/tpdr_v2_seed${CI_SEED}_pair${CI_SEED}.json"
LOG="logs/tpdr_v2_seed${CI_SEED}_pair${CI_SEED}.log"

if [ -s "$OUT" ]; then
    echo "[skip] $OUT already exists"; exit 0
fi

CI_CKPT="release_ckpts/qwen_time_v15s_20260523_141410_seed${CI_SEED}.pt"
PR_CKPT="checkpoints/prompt_baseline_s${PR_SEED}.pt"

if [ ! -f "$CI_CKPT" ]; then echo "MISSING ci ckpt: $CI_CKPT"; exit 1; fi
if [ ! -f "$PR_CKPT" ]; then echo "MISSING prompt ckpt: $PR_CKPT"; exit 1; fi

LABEL=$([ "$CI_SEED" = "0" ] && echo "HEADLINE" || echo "CROSS-SEED")
echo "=== [$LABEL] CI seed $CI_SEED, prompt seed $PR_SEED ===" | tee -a "$LOG"
echo "  CI ckpt:    $CI_CKPT" | tee -a "$LOG"
echo "  prompt ckpt: $PR_CKPT" | tee -a "$LOG"
echo "  out:        $OUT" | tee -a "$LOG"

stdbuf -oL -eL uv run python -u eval/tpdr/run_tpdr.py \
    --device cuda --n-scenarios 200 --n-tau 10 --max-new 150 \
    --ci-ckpt "$CI_CKPT" --prompt-ckpt "$PR_CKPT" --out "$OUT" \
    2>&1 | stdbuf -oL -eL tee -a "$LOG"

if [ -s "$OUT" ]; then
    echo "[done] $OUT written" | tee -a "$LOG"
else
    echo "[FAIL] $OUT was not written" | tee -a "$LOG"
    exit 1
fi
