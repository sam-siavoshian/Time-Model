#!/usr/bin/env bash
# Disproof suite: linear probe + falsification + behavioral pressure.
# All against existing v11 checkpoint. NO retraining.
# Survives SSH disconnect via tmux + nohup.

set -uo pipefail
cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
export IPCN_ROOT="$HOME/ipcn"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

CKPT="${1:-checkpoints/qwen_time_v11.pt}"
BASE="${2:-Qwen/Qwen2.5-3B-Instruct}"
TAG="disproof_$(date +%Y%m%d_%H%M%S)"

LOG_DIR="logs"
REPORT_DIR="reports"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

STDOUT="$LOG_DIR/${TAG}.log"
SENT="$REPORT_DIR/${TAG}_DONE.txt"

echo "============================================================"
echo "Disproof suite"
echo "  tag:    $TAG"
echo "  ckpt:   $CKPT"
echo "  base:   $BASE"
echo "  log:    $STDOUT"
echo "============================================================"
date -Iseconds | tee -a "$STDOUT"

if [ ! -f "$CKPT" ]; then
    echo "FATAL: checkpoint not found: $CKPT" | tee -a "$STDOUT"
    echo "{\"status\":\"no_ckpt\"}" > "$SENT"
    exit 1
fi

echo "=== 1/3 LINEAR PROBE ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -m model.qwen_time_probe \
    --checkpoint "$CKPT" --base "$BASE" --device cuda \
    --out "$REPORT_DIR/${TAG}_probe.json" \
    --n-samples 400 \
    2>&1 | tee -a "$STDOUT"
PROBE_RC=${PIPESTATUS[0]}
echo "probe exit=$PROBE_RC" | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== 2/3 FALSIFICATION ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -m model.qwen_time_falsify \
    --checkpoint "$CKPT" --base "$BASE" --device cuda \
    --out "$REPORT_DIR/${TAG}_falsify.json" \
    2>&1 | tee -a "$STDOUT"
FALS_RC=${PIPESTATUS[0]}
echo "falsify exit=$FALS_RC" | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== 3/3 BEHAVIORAL PRESSURE ===" | tee -a "$STDOUT"
PYTHONPATH=. uv run python3 -m model.qwen_time_pressure \
    --checkpoint "$CKPT" --base "$BASE" --device cuda \
    --out "$REPORT_DIR/${TAG}_pressure.json" \
    2>&1 | tee -a "$STDOUT"
PRES_RC=${PIPESTATUS[0]}
echo "pressure exit=$PRES_RC" | tee -a "$STDOUT"

echo "" | tee -a "$STDOUT"
echo "=== SUMMARY ===" | tee -a "$STDOUT"
python3 - <<PY 2>&1 | tee -a "$STDOUT"
import json, os
tag = "$TAG"
for name in ["probe", "falsify", "pressure"]:
    path = f"reports/{tag}_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        v = r.get("verdict", {})
        print(f"--- {name} ---")
        for k, vv in v.items():
            print(f"  {k}: {vv}")
    else:
        print(f"--- {name} --- MISSING")
PY

echo "{\"status\":\"done\",\"tag\":\"$TAG\",\"probe_rc\":$PROBE_RC,\"falsify_rc\":$FALS_RC,\"pressure_rc\":$PRES_RC,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
echo "DONE. Sentinel: $SENT" | tee -a "$STDOUT"
