#!/usr/bin/env bash
# Spark training launcher. Wraps training + safety + monitor in a 3-pane
# tmux session. Auto-resumes from the latest checkpoint and handles OOM
# by retrying with a smaller chunk_length until success.
#
# Usage (on Spark, inside a tmux session you own):
#   bash scripts/spark_launch.sh [--phase N] [--steps N] [--batch N]
#
# This script is NOT designed to be run on macOS. It assumes:
#   - $HOME/ipcn is the repo root
#   - uv is on PATH
#   - CUDA-capable GPU is visible
#   - You created your OWN tmux session (this script writes to the current
#     session via separate windows, NOT the existing 4 long-running ones).

set -euo pipefail

ROOT="${IPCN_ROOT:-$HOME/ipcn}"
cd "$ROOT"
export IPCN_ROOT="$ROOT"
export PATH="$HOME/.local/bin:$PATH"

PHASE=0
STEPS=100000
LOG_EVERY=25
CKPT_EVERY=500
DEVICE="cuda"
USE_REAL_TAU="--use-real-tau"
EXTRA_ARGS=""
RESUME=""
SESSION_NAME="ipcn_train_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --log-every) LOG_EVERY="$2"; shift 2;;
        --ckpt-every) CKPT_EVERY="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --no-real-tau) USE_REAL_TAU=""; shift;;
        --resume) RESUME="--resume $2"; shift 2;;
        --session-name) SESSION_NAME="$2"; shift 2;;
        --extra) EXTRA_ARGS="$2"; shift 2;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

mkdir -p logs checkpoints reports

# Auto-resume from most recent checkpoint if not explicitly set.
# `set -e + pipefail` makes `ls X 2>/dev/null | head` abort on empty glob;
# wrap in a subshell that always succeeds.
if [ -z "$RESUME" ]; then
    LATEST=$( ( ls -t checkpoints/phase${PHASE}_*.pt 2>/dev/null || true ) | head -1 )
    if [ -n "${LATEST:-}" ]; then
        echo "Auto-resuming from $LATEST"
        RESUME="--resume $LATEST"
    fi
fi

LOG_PATH="logs/phase${PHASE}_${SESSION_NAME}.jsonl"
OUT_CKPT="checkpoints/phase${PHASE}_${SESSION_NAME}_final.pt"
ALERT_PATH="reports/alerts_${SESSION_NAME}.jsonl"

echo "============================================================"
echo "IPCN Spark launcher"
echo "============================================================"
echo "ROOT:        $ROOT"
echo "PHASE:       $PHASE"
echo "STEPS:       $STEPS"
echo "DEVICE:      $DEVICE"
echo "REAL_TAU:    ${USE_REAL_TAU:-DISABLED}"
echo "LOG:         $LOG_PATH"
echo "OUT_CKPT:    $OUT_CKPT"
echo "ALERT:       $ALERT_PATH"
echo "RESUME:      ${RESUME:-NONE}"
echo "SESSION:     $SESSION_NAME"
echo "============================================================"

# Preflight gate. Run it for visibility. Git-state FAIL is expected on
# Spark (no .git). Hard-fail only on REAL prereq misses. All pipeline
# stages here are wrapped to never propagate non-zero under pipefail.
set +e
PYTHONPATH=. uv run python3 scripts/preflight.py ${DEVICE:+--cuda} > /tmp/preflight_check.txt 2>&1
PREFLIGHT_RC=$?
tail -20 /tmp/preflight_check.txt
HARD_FAILS=$( { grep '\[FAIL\]' /tmp/preflight_check.txt 2>/dev/null || true; } | { grep -v 'git state' 2>/dev/null || true; } | wc -l | tr -d ' ')
set -e
if [ "${HARD_FAILS:-0}" -gt 0 ]; then
    echo ""
    echo "preflight reported $HARD_FAILS non-git failure(s); aborting"
    grep '\[FAIL\]' /tmp/preflight_check.txt || true
    exit 1
fi
echo "preflight: non-git failures=$HARD_FAILS, total rc=$PREFLIGHT_RC (proceeding)"

# Spawn 3-pane tmux session: training | safety | monitor.
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "session $SESSION_NAME already exists; refusing to clobber"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n train
tmux send-keys -t "$SESSION_NAME:train" \
    "bash scripts/spark_train_with_oom_guard.sh --phase $PHASE --steps $STEPS --log-every $LOG_EVERY --ckpt-every $CKPT_EVERY --device $DEVICE --log-path $LOG_PATH --out-ckpt $OUT_CKPT $USE_REAL_TAU $RESUME $EXTRA_ARGS 2>&1 | tee logs/${SESSION_NAME}_stdout.log" C-m

# Wait a moment so the training PID file exists.
sleep 2

tmux split-window -h -t "$SESSION_NAME:train"
tmux send-keys -t "$SESSION_NAME:train.1" \
    "sleep 5; while [ ! -f /tmp/ipcn_train_${SESSION_NAME}.pid ]; do sleep 2; done; PID=\$(cat /tmp/ipcn_train_${SESSION_NAME}.pid); echo \"safety attaching to PID \$PID\"; PYTHONPATH=. uv run python3 -m scripts.safety --pid \$PID --log $LOG_PATH --alert-path $ALERT_PATH --stall-secs 1200 --start-stall-secs 900" C-m

tmux split-window -v -t "$SESSION_NAME:train.1"
tmux send-keys -t "$SESSION_NAME:train.2" \
    "sleep 8; PYTHONPATH=. uv run python3 -m scripts.monitor $LOG_PATH --tail --alert-path $ALERT_PATH" C-m

echo ""
echo "tmux session '$SESSION_NAME' started with 3 panes."
echo "  attach:  tmux attach -t $SESSION_NAME"
echo "  detach:  Ctrl-b d"
echo "  kill:    tmux kill-session -t $SESSION_NAME"
echo "  status:  bash scripts/spark_status.sh $SESSION_NAME"
