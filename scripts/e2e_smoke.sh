#!/usr/bin/env bash
# End-to-end smoke: Phase 0 -> Phase 1 -> eval at all 3 checkpoints.
# Run on CPU, small step counts, just to verify the pipeline works.
# After this passes, Spark runs are high-confidence.

set -eu

ROOT="/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model"
cd "$ROOT"

# Clean prior outputs
rm -rf checkpoints/e2e_smoke logs/e2e_smoke reports/e2e_smoke
mkdir -p checkpoints/e2e_smoke logs/e2e_smoke reports/e2e_smoke

echo "================================================================"
echo "Step 1/5: Eval baseline (untrained) ..."
echo "================================================================"
uv run python -m model.eval_all \
  --n-trials 5 \
  --out reports/e2e_smoke/00_baseline.md \
  --device cpu

echo ""
echo "================================================================"
echo "Step 2/5: Phase 0 training (100 steps) ..."
echo "================================================================"
uv run python -m model.run_phase \
  --phase 0 --steps 100 \
  --log-every 25 \
  --log-path logs/e2e_smoke/phase0.jsonl \
  --out-ckpt checkpoints/e2e_smoke/phase0.pt \
  --use-real-tau \
  --device cpu

echo ""
echo "================================================================"
echo "Step 3/5: Eval Phase 0 checkpoint ..."
echo "================================================================"
uv run python -m model.eval_all \
  --checkpoint checkpoints/e2e_smoke/phase0.pt \
  --n-trials 5 \
  --out reports/e2e_smoke/01_phase0.md \
  --device cpu

echo ""
echo "================================================================"
echo "Step 4/5: Phase 1 training (100 steps, consolidation enabled) ..."
echo "================================================================"
uv run python -m model.run_phase \
  --phase 1 --steps 100 \
  --resume checkpoints/e2e_smoke/phase0.pt \
  --enable-consolidation \
  --cons-freq 25 \
  --tau-cons-override 0.0 \
  --log-every 25 \
  --log-path logs/e2e_smoke/phase1.jsonl \
  --out-ckpt checkpoints/e2e_smoke/phase1.pt \
  --device cpu

echo ""
echo "================================================================"
echo "Step 5/5: Eval Phase 1 checkpoint + H4 CTI (pre=phase0, post=phase1) ..."
echo "================================================================"
uv run python -m model.eval_all \
  --checkpoint checkpoints/e2e_smoke/phase1.pt \
  --pre-checkpoint checkpoints/e2e_smoke/phase0.pt \
  --n-trials 5 \
  --out reports/e2e_smoke/02_phase1.md \
  --device cpu

echo ""
echo "================================================================"
echo "E2E SMOKE PASSED. Reports:"
echo "================================================================"
ls -la reports/e2e_smoke/

echo ""
echo "Summary deltas (D_0, KL, perplexity):"
for r in reports/e2e_smoke/*.md; do
  echo "--- $r ---"
  head -16 "$r" | tail -10
done
