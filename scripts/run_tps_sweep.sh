#!/usr/bin/env bash
# TPS full sweep launcher (Spark). Sequential per-adapter to limit GPU
# contention with Omar's training jobs. Each adapter writes its own
# runs/<run_id>/reports/tps/<name>.json. A sentinel file marks completion.

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

OUT_DIR="${OUT_DIR:-$REPORT_DIR/tps}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.18}"
mkdir -p "$OUT_DIR"
SENT="$OUT_DIR/SWEEP_DONE.txt"
rm -f "$SENT"

ITEMS="${ITEMS:-data/tps/items.jsonl}"
RELEASE="${RELEASE_DIR:-release_ckpts}"
CK="${CHECKPOINT_DIR:-checkpoints}"
test -s "$ITEMS"
test -s "$RELEASE/qwen_time_v15s_20260523_141410_seed0.pt"
test -s "$RELEASE/qwen_time_v15s_20260523_141410_seed1.pt"
test -s "$RELEASE/qwen_time_v15s_20260523_141410_seed2.pt"
test -s "$CK/chrono_only_s0.pt"

run() {
  local tag="$1"; shift
  local out="$OUT_DIR/$tag.json"
  local log="$OUT_DIR/${tag}.log"
  if [ -e "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "refusing to overwrite existing output: $out (set FORCE=1 to replace)" >&2
    return 1
  fi
  echo
  echo "============================================================"
  echo "  TPS RUN: $tag"
  echo "  args: $*"
  echo "  out:  $out"
  echo "============================================================"
  uv run python -m eval.tps.run_tps \
      --items "$ITEMS" \
      --out "$out" \
      --memory-fraction "$MEMORY_FRACTION" \
      --progress-every 200 \
      "$@" 2>&1 | tee "$log"
  test -s "$out"
}

# 1. vanilla floor.
run vanilla --adapter vanilla

# 2. CI v15s seed 0 (HEADLINE).
run ci_v15s_s0 --adapter ci --checkpoint "$RELEASE/qwen_time_v15s_20260523_141410_seed0.pt"

# 3. chrono_only seed 0 (necessity check; CI without LoRA).
run chrono_only_s0 --adapter ci --checkpoint "$CK/chrono_only_s0.pt"

# 4. CI v15s seed 1.
run ci_v15s_s1 --adapter ci --checkpoint "$RELEASE/qwen_time_v15s_20260523_141410_seed1.pt"

# 5. CI v15s seed 2.
run ci_v15s_s2 --adapter ci --checkpoint "$RELEASE/qwen_time_v15s_20260523_141410_seed2.pt"

uv run python eval/tps/analyze.py \
  --inputs \
  "$OUT_DIR/vanilla.json" \
  "$OUT_DIR/ci_v15s_s0.json" \
  "$OUT_DIR/chrono_only_s0.json" \
  "$OUT_DIR/ci_v15s_s1.json" \
  "$OUT_DIR/ci_v15s_s2.json" \
  --out "$OUT_DIR/headline.json" 2>&1 | tee "$OUT_DIR/analyze.log"
test -s "$OUT_DIR/headline.json"

{
  echo "RUN_ID=$RUN_ID"
  echo "OUT_DIR=$OUT_DIR"
  echo "ALL DONE $(date)"
} > "$SENT"
time_model_manifest "$SCRIPT_PATH" "done"
echo "SWEEP COMPLETE"
