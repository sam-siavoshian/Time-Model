#!/usr/bin/env bash
# Pre-registered (PREREGISTRATION_v2.md sections 1.1-1.10) TPDR replication.
# Anchor commit: 80ddafc.
#
# Runs the 3 pre-reg pairs: (ci0,pr0) HEADLINE, (ci1,pr1) CROSS-SEED,
# (ci2,pr2) CROSS-SEED. 200 scenarios each, 10 tau, greedy, max_new=150.
# Output: reports/tpdr_crossseed/tpdr_v2_seed{S}_pair{S}.json
#
# Memory: each run uses ~12 GiB GPU and ~6 GiB system RAM.
# Coordinate with concurrent Spark jobs; run sequentially if free RAM
# is tight.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports/tpdr_crossseed

# Pre-reg pairs (CI_SEED, PROMPT_SEED): headline + 2 cross-seed.
PAIRS=("0,0" "1,1" "2,2")

for SPEC in "${PAIRS[@]}"; do
    CI_SEED=$(echo "$SPEC" | cut -d, -f1)
    PR_SEED=$(echo "$SPEC" | cut -d, -f2)
    OUT="reports/tpdr_crossseed/tpdr_v2_seed${CI_SEED}_pair${CI_SEED}.json"
    LOG="logs/tpdr_v2_seed${CI_SEED}_pair${CI_SEED}.log"

    if [ -s "$OUT" ]; then
        echo "[skip] $OUT already exists" | tee -a "$LOG"
        continue
    fi

    CI_CKPT="release_ckpts/qwen_time_v15s_20260523_141410_seed${CI_SEED}.pt"
    PR_CKPT="checkpoints/prompt_baseline_s${PR_SEED}.pt"

    if [ ! -f "$CI_CKPT" ]; then
        echo "MISSING ci ckpt: $CI_CKPT" | tee -a "$LOG"; continue
    fi
    if [ ! -f "$PR_CKPT" ]; then
        echo "MISSING prompt ckpt: $PR_CKPT" | tee -a "$LOG"; continue
    fi

    LABEL=$([ "$CI_SEED" = "0" ] && echo "HEADLINE" || echo "CROSS-SEED")
    echo "=== [$LABEL] CI seed $CI_SEED, prompt seed $PR_SEED ===" | tee -a "$LOG"
    echo "  CI ckpt:    $CI_CKPT" | tee -a "$LOG"
    echo "  prompt ckpt: $PR_CKPT" | tee -a "$LOG"
    echo "  out:        $OUT" | tee -a "$LOG"

    # python -u + stdbuf -oL force per-line flush of stdout under heavy
    # GPU contention; required to see scenario-level progress without
    # waiting for adapter completion. Does NOT change result content.
    stdbuf -oL -eL uv run python -u eval/tpdr/run_tpdr.py \
        --device cuda --n-scenarios 200 --n-tau 10 --max-new 150 \
        --ci-ckpt "$CI_CKPT" --prompt-ckpt "$PR_CKPT" --out "$OUT" \
        2>&1 | stdbuf -oL -eL tee -a "$LOG"

    if [ -s "$OUT" ]; then
        echo "[done] $OUT written" | tee -a "$LOG"
    else
        echo "[FAIL] $OUT was not written; aborting queue" | tee -a "$LOG"
        exit 1
    fi
done

touch logs/tpdr_v2_prereg.done
echo "all 3 pre-reg pairs done"
