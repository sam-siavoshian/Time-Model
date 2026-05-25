#!/usr/bin/env bash
# W12: run external tau_sessions benchmark on all 3 reference adapters.
set -euo pipefail
cd "$HOME/Time-Model" 2>/dev/null || cd "$HOME/Desktop/Time-Model" 2>/dev/null || cd "$HOME/ipcn"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs reports

BASE="Qwen/Qwen2.5-3B-Instruct"
CKPT="release_ckpts/qwen_time_v15s_20260523_141410_seed0.pt"

uv run python -m eval.external.eval_tau_bench --adapter vanilla --base "$BASE" \
    --out reports/ext_bench_vanilla.json 2>&1 | tee logs/ext_bench_vanilla.log
uv run python -m eval.external.eval_tau_bench --adapter prompt --base "$BASE" \
    --out reports/ext_bench_prompt.json 2>&1 | tee logs/ext_bench_prompt.log
uv run python -m eval.external.eval_tau_bench --adapter ci --base "$BASE" \
    --checkpoint "$CKPT" \
    --out reports/ext_bench_ci.json 2>&1 | tee logs/ext_bench_ci.log
touch logs/ext_bench.done
echo "all 3 adapters done"
