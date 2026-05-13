#!/usr/bin/env bash
# Launch IPCN Phase N training in a tmux session.
#
# Layout: 3 panes
#   pane 0: training (run_phase)
#   pane 1: safety watchdog (kills training on NaN / explosion / disk-full)
#   pane 2: live monitor (rolling LM avg, gradient norm, alerts)
#
# Usage on Spark:
#   bash scripts/tmux_launch.sh phase0 100000 cuda
#   bash scripts/tmux_launch.sh phase1 50000  cuda checkpoints/phase0.pt
#
# After launch:
#   tmux attach -t ipcn

set -eu

PHASE="${1:-phase0}"
STEPS="${2:-100000}"
DEVICE="${3:-cuda}"
RESUME="${4:-}"

SESSION="ipcn"
PHASE_NUM="${PHASE#phase}"
CKPT_OUT="checkpoints/${PHASE}.pt"
LOG_PATH="logs/${PHASE}.jsonl"
CONS_FLAG=""
TAU_FLAG=""
if [ "$PHASE_NUM" != "0" ]; then
    CONS_FLAG="--enable-consolidation"
fi
TAU_FLAG="--use-real-tau"

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
fi

# Kill prior session if exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create new session, pane 0 = training
tmux new-session -d -s "$SESSION" -n "training" \
    "uv run python -m model.run_phase \
        --phase $PHASE_NUM \
        --steps $STEPS \
        --device $DEVICE \
        $TAU_FLAG $CONS_FLAG $RESUME_FLAG \
        --ckpt-every 10000 \
        --out-ckpt $CKPT_OUT \
        --log-path $LOG_PATH \
        --log-every 100; \
     echo TRAINING_DONE; read"

# Wait briefly for training to start, then capture pid for safety watchdog
sleep 8

# Pane 1: safety watchdog
TRAIN_PID=$(pgrep -f "run_phase --phase $PHASE_NUM" | head -1)
if [ -z "$TRAIN_PID" ]; then
    echo "Could not find training PID. Exiting."
    exit 1
fi

tmux split-window -t "$SESSION":0 -v \
    "uv run python -m scripts.safety \
        --pid $TRAIN_PID \
        --log $LOG_PATH \
        --check-every 30 \
        --lm-explode-factor 10 \
        --min-free-gb 10; \
     echo SAFETY_DONE; read"

# Pane 2: live monitor
tmux split-window -t "$SESSION":0 -h \
    "sleep 15 && uv run python -m scripts.monitor $LOG_PATH --tail --window 50; \
     echo MONITOR_DONE; read"

tmux select-layout -t "$SESSION" tiled
echo ""
echo "Launched tmux session '$SESSION'"
echo "  pane 0: training (pid $TRAIN_PID)"
echo "  pane 1: safety watchdog"
echo "  pane 2: live monitor"
echo ""
echo "Attach with: tmux attach -t $SESSION"
echo "Detach with: Ctrl+b d"
echo "Logs:        $LOG_PATH"
echo "Checkpoint:  $CKPT_OUT  (intermediates: checkpoints/${PHASE}_step*.pt)"
