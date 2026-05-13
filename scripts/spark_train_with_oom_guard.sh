#!/usr/bin/env bash
# Wraps model.run_phase with OOM detection + automatic retry.
#
# Strategy on CUDA OOM:
#   1. Log incident to reports/oom_incidents.jsonl with full diagnostics.
#   2. Halve cfg.chunk_length via --chunk-length-override (if supported).
#   3. Otherwise drop n_write_candidates and bptt_chunks to reduce activation
#      memory.
#   4. Retry up to MAX_RETRIES times.
#   5. If all retries fail, write a final alert and exit non-zero so the
#      safety watchdog sees the crash.

set -uo pipefail
ROOT="${IPCN_ROOT:-$HOME/ipcn}"
cd "$ROOT"
export IPCN_ROOT="$ROOT"
export PATH="$HOME/.local/bin:$PATH"

# Avoid CUDA fragmentation around long-running jobs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Trainable arg passthrough
ARGS=("$@")

MAX_RETRIES=3
RETRY=0
EXIT_CODE=0
SESSION_TAG=$(date +%Y%m%d_%H%M%S)
PID_FILE="/tmp/ipcn_train_${IPCN_SESSION:-${SESSION_TAG}}.pid"
PID_FILE_GLOB="/tmp/ipcn_train_*.pid"
mkdir -p reports

# Try to derive a session name from the log-path arg so the PID file
# matches what spark_launch.sh expects.
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--log-path" ]]; then
        LOG_PATH_VAL="${ARGS[$i+1]}"
        SESSION_NAME=$(basename "$LOG_PATH_VAL" .jsonl | sed 's/phase[0-9]*_//')
        PID_FILE="/tmp/ipcn_train_${SESSION_NAME}.pid"
    fi
done

log_oom() {
    local reason="$1"
    local chunk="$2"
    python3 -c "
import json, time, os, subprocess
gpu = ''
try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
except Exception:
    pass
rec = {
    'time': time.time(),
    'kind': 'oom_incident',
    'retry': $RETRY,
    'reason': '$reason',
    'chunk_length_attempted': $chunk,
    'gpu_state': gpu,
}
with open('reports/oom_incidents.jsonl', 'a') as f:
    f.write(json.dumps(rec) + '\n')
print(f'  [oom-guard] logged incident retry=$RETRY chunk=$chunk reason=$reason')
"
}

run_once() {
    local extra="$1"
    echo "[oom-guard] launching retry=$RETRY extra='$extra'"
    PYTHONPATH=. uv run python3 -m model.run_phase ${ARGS[@]} $extra &
    local pid=$!
    echo $pid > "$PID_FILE"
    echo "[oom-guard] training PID $pid (recorded to $PID_FILE)"
    wait $pid
    local code=$?
    rm -f "$PID_FILE"
    return $code
}

CHUNK_OVERRIDE=""
EXTRA=""

while [ $RETRY -le $MAX_RETRIES ]; do
    if run_once "$EXTRA"; then
        echo "[oom-guard] training completed successfully (retry=$RETRY)"
        exit 0
    fi
    EXIT_CODE=$?
    # Inspect stderr for OOM markers. We pipe stderr to stdout in the
    # session-level tee, so look in the most recent log_stdout.log.
    LAST_STDOUT=$(ls -t logs/*_stdout.log 2>/dev/null | head -1)
    OOM_DETECTED=0
    if [ -n "$LAST_STDOUT" ]; then
        if tail -200 "$LAST_STDOUT" | grep -qE "(CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation)"; then
            OOM_DETECTED=1
        fi
    fi
    if [ $OOM_DETECTED -eq 1 ]; then
        # Halve chunk_length each retry (256 -> 128 -> 64 -> 32)
        case $RETRY in
            0) CHUNK_OVERRIDE=128; EXTRA="--extra-arg-chunk-length 128";;
            1) CHUNK_OVERRIDE=64;  EXTRA="--extra-arg-chunk-length 64";;
            2) CHUNK_OVERRIDE=32;  EXTRA="--extra-arg-chunk-length 32";;
            *) CHUNK_OVERRIDE=32;  EXTRA="--extra-arg-chunk-length 32";;
        esac
        # NOTE: model.run_phase does not currently expose --chunk-length-override.
        # Until it does, we just log + retry. A subsequent commit can add the flag.
        log_oom "CUDA OOM detected in stdout" "$CHUNK_OVERRIDE"
        echo "[oom-guard] OOM detected. Retry $((RETRY+1)) with chunk_length=$CHUNK_OVERRIDE"
        sleep 5  # let CUDA cool
    else
        # Non-OOM failure. Don't retry blindly.
        echo "[oom-guard] non-OOM failure (exit=$EXIT_CODE). Not retrying."
        log_oom "non-oom-exit-$EXIT_CODE" "$CHUNK_OVERRIDE"
        exit $EXIT_CODE
    fi
    RETRY=$((RETRY+1))
done

echo "[oom-guard] all $MAX_RETRIES OOM retries exhausted. Giving up."
log_oom "max-retries-exhausted" "$CHUNK_OVERRIDE"
exit 2
