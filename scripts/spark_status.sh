#!/usr/bin/env bash
# Quick health check for a running IPCN training session on Spark.
# Run on the Spark box. Prints:
#   - tmux session state
#   - latest training step (from log)
#   - GPU memory usage
#   - recent alerts
#   - last 5 log lines
#
# Usage: bash scripts/spark_status.sh <session_name>

set -uo pipefail
ROOT="${IPCN_ROOT:-$HOME/ipcn}"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

SESSION="${1:-}"
if [ -z "$SESSION" ]; then
    echo "usage: bash scripts/spark_status.sh <session_name>"
    echo "available sessions:"
    tmux ls 2>/dev/null | grep -E "^ipcn_" || echo "  (none)"
    exit 1
fi

echo "============================================================"
echo "IPCN status: $SESSION"
echo "  generated: $(date -Iseconds)"
echo "============================================================"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "TMUX: session NOT FOUND -- training may have crashed or completed"
    echo ""
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "TMUX: $SESSION is alive ($(tmux list-windows -t $SESSION 2>/dev/null | wc -l) windows)"
fi

LOG=$(ls -t logs/phase*_${SESSION}.jsonl 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    echo ""
    echo "LOG: $LOG ($(wc -l < "$LOG") lines, $(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG") bytes)"
    LAST_STEP=$(tail -200 "$LOG" | grep -oE '"step": [0-9]+' | tail -1 | awk -F': ' '{print $2}')
    LAST_LM=$(tail -200 "$LOG" | grep -oE '"lm_loss": [0-9.eE+-]+' | tail -1 | awk -F': ' '{print $2}')
    LAST_GRAD=$(tail -200 "$LOG" | grep -oE '"grad_norm": [0-9.eE+-]+' | tail -1 | awk -F': ' '{print $2}')
    LAST_MEM=$(tail -200 "$LOG" | grep -oE '"memory_norm": [0-9.eE+-]+' | tail -1 | awk -F': ' '{print $2}')
    echo "  last step:     ${LAST_STEP:-n/a}"
    echo "  last LM loss:  ${LAST_LM:-n/a}"
    echo "  last grad:     ${LAST_GRAD:-n/a}"
    echo "  last mem_norm: ${LAST_MEM:-n/a}"
    # Completion sentinel
    if tail -3 "$LOG" | grep -q '"event": "training_complete"'; then
        echo "  STATUS: training_complete sentinel found"
    fi
else
    echo "LOG: no log file found for session $SESSION"
fi

echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv 2>/dev/null | head -3 || echo "  (nvidia-smi unavailable)"

echo ""
echo "ALERTS (last 10):"
ALERT=$(ls -t reports/alerts_${SESSION}.jsonl 2>/dev/null | head -1)
if [ -z "$ALERT" ]; then
    ALERT=$(ls -t reports/alerts*.jsonl 2>/dev/null | head -1)
fi
if [ -n "$ALERT" ]; then
    tail -10 "$ALERT" 2>/dev/null | python3 -c "
import json, sys
for line in sys.stdin:
    try:
        rec = json.loads(line)
        t = rec.get('time', 0)
        import datetime
        ts = datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
        kind = rec.get('kind', '?')
        extras = {k: v for k, v in rec.items() if k not in ('time', 'kind')}
        print(f'  [{ts}] {kind}: {json.dumps(extras)[:160]}')
    except Exception:
        pass
" 2>/dev/null || tail -10 "$ALERT"
else
    echo "  (no alert file yet)"
fi

echo ""
echo "RECENT LOG TAIL (last 5 records):"
if [ -n "$LOG" ]; then
    tail -5 "$LOG" 2>/dev/null | python3 -c "
import json, sys
for line in sys.stdin:
    try:
        rec = json.loads(line)
        if rec.get('event') == 'consolidation':
            print(f'  [consol step={rec.get(\"step\")}] committed={rec.get(\"committed\")} drift={rec.get(\"lm_drift_kl\", 0):.4f}')
            continue
        if 'lm_loss' in rec:
            print(f'  step={rec.get(\"step\"):>6} | LM={rec.get(\"lm_loss\", 0):7.4f} | grad={rec.get(\"grad_norm\", 0):6.2f} | mem={rec.get(\"memory_norm\", 0):6.1f}')
    except Exception:
        pass
"
fi

echo ""
echo "DISK: $(df -h "$ROOT" | tail -1 | awk '{print $4}') free"
echo "============================================================"
