#!/usr/bin/env bash
# Run from the local Mac. Polls the Spark box via SSH for training health.
# Pulls back: training status, GPU usage, alerts.jsonl tail, log tail.
#
# Usage:
#   bash scripts/spark_remote_status.sh [session_name]
#
# Requires:
#   - sshpass installed
#   - Spark password in macOS keychain under
#     service=ssh-omarramadan-100.122.27.75 account=omarramadan
#
# Designed to be invoked from a /loop on the Mac so the assistant can
# wake up periodically and check on Spark training health.

set -uo pipefail

SPARK_IP="100.122.27.75"
SPARK_USER="omarramadan"
SESSION="${1:-}"

PASS=$(security find-generic-password -a "$SPARK_USER" -s "ssh-${SPARK_USER}-${SPARK_IP}" -w 2>/dev/null)
if [ -z "$PASS" ]; then
    echo "ERROR: no keychain entry for ssh-${SPARK_USER}-${SPARK_IP}"
    exit 1
fi

REMOTE_CMD="cd ~/ipcn"
if [ -n "$SESSION" ]; then
    REMOTE_CMD="$REMOTE_CMD && bash scripts/spark_status.sh '$SESSION'"
else
    REMOTE_CMD="$REMOTE_CMD && echo '--- running ipcn sessions ---' && (tmux ls 2>/dev/null | grep ^ipcn_ || echo '(none)') && echo '' && echo '--- GPU ---' && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv 2>/dev/null | head -3 && echo '' && echo '--- recent alerts ---' && (tail -5 reports/alerts*.jsonl 2>/dev/null || echo '(no alerts)') && echo '' && echo '--- recent log tail ---' && (LATEST=\$(ls -t logs/*.jsonl 2>/dev/null | head -1); if [ -n \"\$LATEST\" ]; then echo \"log: \$LATEST\"; tail -5 \"\$LATEST\"; else echo '(no logs)'; fi)"
fi

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$SPARK_USER@$SPARK_IP" "$REMOTE_CMD" 2>&1
