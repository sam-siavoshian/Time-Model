#!/usr/bin/env bash
# TPS full sweep launcher (Spark). Sequential per-adapter to limit GPU
# contention with Omar's training jobs. Each adapter writes its own
# reports/tps/<name>.json. A sentinel file marks completion.

set -uo pipefail
cd /home/omarramadan/ipcn
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

OUT_DIR="reports/tps"
mkdir -p "$OUT_DIR"
SENT="$OUT_DIR/SWEEP_DONE.txt"
rm -f "$SENT"

ITEMS="data/tps/items.jsonl"
RELEASE="/home/omarramadan/ipcn/release_ckpts"
CK="/home/omarramadan/ipcn/checkpoints"

run() {
  local tag="$1"; shift
  local out="$OUT_DIR/$tag.json"
  echo
  echo "============================================================"
  echo "  TPS RUN: $tag"
  echo "  args: $*"
  echo "  out:  $out"
  echo "============================================================"
  uv run python -m eval.tps.run_tps \
      --items "$ITEMS" \
      --out "$out" \
      --memory-fraction 0.18 \
      --progress-every 200 \
      "$@" 2>&1 | tee -a "$OUT_DIR/${tag}.log"
}

# 1. vanilla floor.
run vanilla --adapter vanilla

# 2. prompt timestamp baseline (Qwen base + [elapsed:X] prefix).
run prompt --adapter prompt

# 3. CI v15s seed 0 (HEADLINE).
run ci_v15s_s0 --adapter ci --checkpoint "$RELEASE/qwen_time_v15s_20260523_141410_seed0.pt"

# 4. chrono_only seed 0 (necessity check — CI without LoRA).
run chrono_only_s0 --adapter ci --checkpoint "$CK/chrono_only_s0.pt"

# 5. CI v15s seed 1.
run ci_v15s_s1 --adapter ci --checkpoint "$RELEASE/qwen_time_v15s_20260523_141410_seed1.pt"

# 6. CI v15s seed 2.
run ci_v15s_s2 --adapter ci --checkpoint "$RELEASE/qwen_time_v15s_20260523_141410_seed2.pt"

date > "$SENT"
echo "ALL DONE $(date)" >> "$SENT"
echo "SWEEP COMPLETE"
