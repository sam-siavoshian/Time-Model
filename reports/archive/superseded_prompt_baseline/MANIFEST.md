# Superseded Prompt Baselines

These files are retained only to explain past reporting drift.

| File | Original path | Reason archived | Current successor |
|---|---|---|---|
| `prompt.json` | `reports/tps/prompt.json` | TPS standalone PromptAdapter run was contaminated by prompt double-prefixing. | Use `reports/tps/vanilla.json` under the `prompt_only` condition. |
| `prompt.log` | `reports/tps/prompt.log` | Log for the contaminated TPS standalone prompt run. | `reports/tps/headline.json` |
| `prompt_baseline_s0_recall.json` | `reports/prompt_baseline_s0_recall.json` | Prompt-baseline eval was superseded by injected prompt-time runs. | `reports/prompt_baseline_injected_s0_recall.json` |
