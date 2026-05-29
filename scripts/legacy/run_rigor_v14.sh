#!/usr/bin/env bash
# Reviewer-rigor reruns against existing v14 checkpoint (no retrain).
# Targets the 3 weakest claims found in audit:
#   1. Pressure n=30, max_new=256, bootstrap CI on P2 delta
#   2. T1b genuine OOD on tau in [7d, 28d] (truly held out)
#   3. T3 multi-week on Sat at weeks 1-4 (tau-memorization vs phase-encoding)

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

CKPT="${1:-checkpoints/qwen_time_v14_20260523_033916.pt}"
TAG_PREFIX="${2:-v14_rigor}"
TAG="${TAG_PREFIX}_$(date +%Y%m%d_%H%M%S)"
STDOUT="logs/${TAG}.log"
SENT="reports/${TAG}_DONE.txt"
mkdir -p logs reports

# v14 used v13's 15-scale timescale list
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

echo "============================================================"
echo "Rigor reruns against $CKPT"
echo "  tag: $TAG"
echo "============================================================"
date -Iseconds | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== 1/3 PRESSURE v2 (n=30 prompts, max_new=256, bootstrap CI) ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_pressure_v2 \
    --checkpoint "$CKPT" --device cuda \
    --out "reports/${TAG}_pressure_v2.json" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" --max-new 256 2>&1 | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== 2-3/3 GENUINE OOD + MULTI-WEEK T3 ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -u -m model.qwen_time_check_genuine_ood \
    --checkpoint "$CKPT" --device cuda \
    --out "reports/${TAG}_genuine_ood.json" \
    --base "Qwen/Qwen2.5-3B-Instruct" \
    --timescales "$TIMESCALES" --t1b-lo-days 7 --t1b-hi-days 28 \
    --t1b-n 30 2>&1 | tee -a "$STDOUT"

echo "{\"status\":\"done\",\"tag\":\"$TAG\",\"time\":\"$(date -Iseconds)\"}" > "$SENT"
echo "DONE $(date -Iseconds). Sentinel: $SENT" | tee -a "$STDOUT"
