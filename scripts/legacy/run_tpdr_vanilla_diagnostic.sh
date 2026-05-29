#!/usr/bin/env bash
# Legacy pre-registered TPDR vanilla diagnostic.
# Current preregistration path: docs/experiments/current/PREREGISTRATION_v2.md §1.8.
# 10-scenario subset, two configurations:
#   (a) Current configuration (chat template, greedy, max_new=150)
#   (b) Stripped configuration (raw prompt, greedy, max_new=150)
# Decision rule: if (b) varies while (a) is identical-template, the
# vanilla numbers in the original 50-scen sweep were misconfigured.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs reports

# (a) Current configuration: re-use run_tpdr.py with n-scenarios=10.
echo "=== (a) current-config diagnostic ==="
uv run python eval/tpdr/run_tpdr.py \
    --device cuda --n-scenarios 10 --n-tau 10 --max-new 150 \
    --out reports/tpdr_vanilla_diagnostic_curconf.json \
    2>&1 | tee logs/tpdr_vanilla_diagnostic_curconf.log

# (b) Stripped configuration: separate script that strips chat template.
echo "=== (b) stripped-config diagnostic ==="
uv run python eval/tpdr/run_tpdr_stripped.py \
    --device cuda --n-scenarios 10 --n-tau 10 --max-new 150 \
    --out reports/tpdr_vanilla_diagnostic_strip.json \
    2>&1 | tee logs/tpdr_vanilla_diagnostic_strip.log

touch logs/tpdr_vanilla_diagnostic.done
echo "diagnostic done"
