#!/usr/bin/env bash
# Pre-registered (PREREGISTRATION_v2.md section 2): per-layer probe on
# the clock-heldout checkpoints (seeds 0, 1, 2).
#
# Same probe code as reports/probe_per_layer_v15s_s0.json
# (model/qwen_time_probe.py). Three conditions per checkpoint:
#   A: trained model (the load-bearing line)
#   B: alpha=0 (chrono off)
#   C: shuffled tau labels
#
# Output: reports/probe_per_layer_clock_heldout_s{0,1,2}.json
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

for SEED in 0 1 2; do
    CKPT="checkpoints/clock_heldout_s${SEED}.pt"
    OUT="reports/probe_per_layer_clock_heldout_s${SEED}.json"
    LOG="logs/probe_clock_heldout_s${SEED}.log"

    if [ ! -f "$CKPT" ]; then
        echo "MISSING: $CKPT (skip seed $SEED)" | tee -a "$LOG"
        continue
    fi

    if [ -s "$OUT" ]; then
        echo "DONE already: $OUT (skip)" | tee -a "$LOG"
        continue
    fi

    echo "=== probe seed $SEED on $CKPT ===" | tee -a "$LOG"
    uv run python -m model.qwen_time_probe \
        --checkpoint "$CKPT" --device cuda \
        --timescales "$TIMESCALES" --n-samples 400 --seed 4242 \
        --out "$OUT" 2>&1 | tee -a "$LOG"
done

touch logs/probe_clock_heldout.done
echo "all 3 clock-heldout probes done"
