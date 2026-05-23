#!/usr/bin/env bash
# End-to-end smoke for Track C (chronometric injection).
# Runs on CPU with tiny step counts. Verifies the v15 pipeline works
# end-to-end before paying for GPU time.
#
# Old Track A version of this script called model.eval_all and
# model.run_phase, both deleted in the 2026-05-23 cleanup. This
# version uses the v15 pipeline.

set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

rm -rf checkpoints/e2e_smoke logs/e2e_smoke reports/e2e_smoke
mkdir -p checkpoints/e2e_smoke logs/e2e_smoke reports/e2e_smoke data/qwen_time

DATA=data/qwen_time/train_smoke.jsonl
CKPT=checkpoints/e2e_smoke/smoke.pt
LOG=logs/e2e_smoke/smoke.jsonl
REPORT=reports/e2e_smoke/smoke_recall.json

echo "================================================================"
echo "1/3 Generate 200 conversations (seed=0, 0.40/0.30/0.30 mix) ..."
echo "================================================================"
uv run python3 -m model.qwen_time_data \
  --n 200 --seed 0 --mix 0.40,0.30,0.30 --out "$DATA"

echo ""
echo "================================================================"
echo "2/3 Train QwenTime for 20 steps on CPU ..."
echo "(Real training is 18000 steps on GPU; this is a pipeline smoke.)"
echo "================================================================"
uv run python3 -m model.qwen_time_train \
  --data "$DATA" --steps 20 --lr 1e-4 --device cpu \
  --log-every 5 --log-path "$LOG" --out "$CKPT" \
  --base "Qwen/Qwen2.5-3B-Instruct" --chunk-length 256

echo ""
echo "================================================================"
echo "3/3 Eval the 5-test battery on CPU ..."
echo "(Numbers will be terrible -- this is a smoke, not a result.)"
echo "================================================================"
uv run python3 -m model.qwen_time_check \
  --checkpoint "$CKPT" --device cpu --out "$REPORT" \
  --base "Qwen/Qwen2.5-3B-Instruct"

echo ""
echo "================================================================"
echo "E2E SMOKE PASSED."
echo "================================================================"
ls -la reports/e2e_smoke/
echo ""
echo "Summary:"
uv run python3 -c "
import json
with open('$REPORT') as f:
    r = json.load(f)
import json as _j
print(_j.dumps(r.get('summary', {}), indent=2))
"
