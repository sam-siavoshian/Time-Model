#!/usr/bin/env bash
# W9 fix: IA3 (Liu et al. 2022) PEFT baseline at sibling-method comparison.
# Replaces LoRA with IA3 multiplicative scaling vectors on k_proj, v_proj,
# and FFN up_proj. Trainable params ~405K for Qwen 2.5 3B (vs LoRA rank-8
# ~4.8M). NOTE: IA3 has fewer params than LoRA at default; we use IA3
# WITH FiLM chrono channel intact (testing whether IA3 surface capacity
# is sufficient when chrono channel is present). For matched-param IA3
# baseline (chrono off), pair with --freeze-alpha.
#
# Three configurations:
#   ia3_with_chrono_s{0,1,2}: IA3 + CI chrono channel (chrono on)
#   ia3_only_s{0,1,2}: IA3 + chrono frozen (PEFT-only baseline, no CI)
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

STEPS=${STEPS:-18000}
TIMESCALES="2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800"

# Config 1: IA3 + chrono active (PEFT-swap ablation)
for SEED in 0 1 2; do
    DATA="$DATA_DIR/v15s_seed${SEED}_18k.jsonl"
    OUT="$CKPT_DIR/ia3_with_chrono_s${SEED}.pt"
    REC="$REPORT_DIR/ia3_with_chrono_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/ia3_with_chrono_s${SEED}.log"
    if [ -f "$OUT" ]; then continue; fi
    if [ ! -s "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi
    echo "=== seed $SEED: IA3 + chrono train ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_train --data "$DATA" --steps "$STEPS" --seed "$SEED" \
        --timescales "$TIMESCALES" --use-ia3 --out "$OUT" 2>&1 | tee -a "$STDOUT"
    echo "=== eval ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check --checkpoint "$OUT" --timescales "$TIMESCALES" \
        --use-ia3 --out "$REC" 2>&1 | tee -a "$STDOUT"
done

# Config 2: IA3 only, chrono frozen (PEFT-only no-CI baseline)
for SEED in 0 1 2; do
    DATA="$DATA_DIR/v15s_seed${SEED}_18k.jsonl"
    OUT="$CKPT_DIR/ia3_only_s${SEED}.pt"
    REC="$REPORT_DIR/ia3_only_s${SEED}_recall.json"
    STDOUT="$LOG_DIR/ia3_only_s${SEED}.log"
    if [ -f "$OUT" ]; then continue; fi
    if [ ! -s "$DATA" ]; then
        uv run python -m model.qwen_time_data \
            --n 18000 --seed "$SEED" --out "$DATA" 2>&1 | tee -a "$STDOUT"
    fi
    echo "=== seed $SEED: IA3 only (chrono frozen) train ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_train --data "$DATA" --steps "$STEPS" --seed "$SEED" \
        --timescales "$TIMESCALES" --use-ia3 --freeze-alpha --out "$OUT" 2>&1 | tee -a "$STDOUT"
    echo "=== eval ===" | tee -a "$STDOUT"
    uv run python -m model.qwen_time_check --checkpoint "$OUT" --timescales "$TIMESCALES" \
        --use-ia3 --out "$REC" 2>&1 | tee -a "$STDOUT"
done

time_model_done "$SCRIPT_PATH" "ia3_baseline.done"
echo "all 6 ckpts done (3 ia3+chrono, 3 ia3-only)"
