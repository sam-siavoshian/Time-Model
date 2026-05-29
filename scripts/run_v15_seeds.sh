#!/usr/bin/env bash
# Cross-seed v15: train 3 independent seeds, eval each on full 5-test,
# compute mean +/- std across seeds. Addresses reviewer attack on
# single-seed reporting.

set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/run_context.sh"
time_model_init_run "$@"
set -- "${RUN_CONTEXT_ARGS[@]}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"
TAG_BASE="$RUN_ID"
DATA_DIR="$DATA_DIR/qwen_time"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR" "$DATA_DIR"
REPORTS=()

for SEED in 0 1 2; do
    TAG="${TAG_BASE}_seed${SEED}"
    DATA="${DATA_DIR}/train_v15s_seed${SEED}_18k.jsonl"
    LOG_PATH="$LOG_DIR/${TAG}.jsonl"
    STDOUT="$LOG_DIR/${TAG}_stdout.log"
    CKPT="$CKPT_DIR/${TAG}.pt"
    REPORT="$REPORT_DIR/${TAG}_recall.json"
    SENT="$REPORT_DIR/${TAG}_DONE.txt"
    for path in "$DATA" "$CKPT" "$REPORT" "$SENT"; do
        if [ -e "$path" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "refusing to overwrite existing artifact: $path (set FORCE=1 to replace)" >&2
            exit 1
        fi
    done

    echo "============================================================"
    echo "v15 seed $SEED -- tag $TAG"
    echo "============================================================"
    date -Iseconds | tee -a "$STDOUT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_data \
        --n 18000 --seed "$SEED" --mix 0.40,0.30,0.30 --out "$DATA" \
        2>&1 | tee -a "$STDOUT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_train \
        --data "$DATA" --steps 18000 --lr 1e-4 --device cuda \
        --log-every 200 --log-path "$LOG_PATH" --out "$CKPT" \
        --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 512 \
        --timescales "$TIMESCALES" --seed "$SEED" 2>&1 | tee -a "$STDOUT"
    test -s "$CKPT"

    PYTHONPATH=. uv run python3 -u -m model.qwen_time_check \
        --checkpoint "$CKPT" --device cuda --out "$REPORT" \
        --base "Qwen/Qwen2.5-3B-Instruct" \
        --timescales "$TIMESCALES" 2>&1 | tee -a "$STDOUT"
    test -s "$REPORT"

    echo "{\"status\":\"done\",\"seed\":$SEED,\"time\":\"$(date -Iseconds)\"}" > "$SENT"
    REPORTS+=("$REPORT")
done

PYTHONPATH=. uv run python3 scripts/aggregate_seeds.py \
    --inputs "${REPORTS[@]}" \
    --out "$REPORT_DIR/${TAG_BASE}_aggregate.json"

RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" DATA_DIR="$DATA_DIR" CKPT_DIR="$CKPT_DIR" python3 - "${REPORTS[@]}" <<'PY' > "$RUN_ROOT/manifest.json"
import json
import os
import sys
from pathlib import Path

reports = sys.argv[1:]
root = Path.cwd()
print(json.dumps({
    "run_id": os.environ["RUN_ID"],
    "run_root": os.environ["RUN_ROOT"],
    "reports": reports,
    "data_dir": os.environ["DATA_DIR"],
    "checkpoints": [str(Path(os.environ["CKPT_DIR"]) / (Path(p).name.replace("_recall.json", ".pt"))) for p in reports],
    "created_from": str(root / "scripts" / "run_v15_seeds.sh"),
}, indent=2))
PY

echo "{\"status\":\"all_seeds_done\",\"run_id\":\"$RUN_ID\",\"time\":\"$(date -Iseconds)\"}" > "$RUN_ROOT/ALL_DONE.txt"
