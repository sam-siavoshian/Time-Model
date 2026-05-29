#!/usr/bin/env bash
# Chain launcher: wait until prompt_nl_s0 + ia3_only_s2 trainings free
# their GPU memory, then start scripts/run_both_heldout.sh.
#
# Used so the both-heldout queue doesn't OOM when launched alongside
# the other concurrent jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "[chain] waiting for prompt_nl_s0.pt checkpoint to land..."
while [ ! -f checkpoints/prompt_nl_s0.pt ]; do sleep 120; done
echo "[chain] prompt_nl_s0.pt landed"

echo "[chain] waiting for ia3_only_s2.pt checkpoint to land..."
while [ ! -f checkpoints/ia3_only_s2.pt ]; do sleep 120; done
echo "[chain] ia3_only_s2.pt landed"

echo "[chain] GPU should have ~14 GB freed; starting both_heldout queue"
bash scripts/run_both_heldout.sh 2>&1 | stdbuf -oL -eL tee logs/both_heldout_chain.log

echo "[chain] DONE"
