# Reproducibility Checklist

Canonical manuscript: `paper/main.tex`.

Canonical evidence index: `ARTIFACT.md` and
`reports/current/manifest.json`.

This file is intentionally short. Older lab-notebook and handoff material is
archived under `docs/history/`; older IPCN, TPDR v1, prompt-baseline, and
pre-v15 model artifacts are archived under `reports/archive/`.

## Environment

- Python: `>=3.11,<3.14` from `pyproject.toml`.
- Package manager: `uv`.
- Runtime dependencies are declared in `pyproject.toml`; analysis dependencies
  include `scipy` and `scikit-learn`.
- Base model: `Qwen/Qwen2.5-3B-Instruct`, fetched from Hugging Face and not
  redistributed in this repo.
- Checkpoints, when used, contain trainable adapter/chrono weights only and are
  loaded on top of the frozen Qwen base.

## Canonical v15 Configuration

- Base model: `Qwen/Qwen2.5-3B-Instruct`.
- Training data: 18,000 synthetic conversations per seed.
- Mix: `0.40,0.30,0.30` for CLOCK, SILENT-GAP, PHASE.
- Seeds: `0,1,2`.
- Chronometric encoder: 15 scale divisors,
  `{2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800}`.
- Encoder dimension: `31 = 15 sin + 15 cos + log1p(tau)`.
- Implemented formula: `sin(tau / T_k)`, `cos(tau / T_k)`, `log1p(tau)`.
  There is no `2*pi` multiplier in code.
- Injection: AdaLN-Zero-style FiLM on every decoder layer except the final
  layer, so Qwen 2.5 3B uses 35 injected layers out of 36.
- PEFT surface: LoRA rank 8 on attention projections plus `lm_head`.

The code source of truth for these defaults is `model/qwen_time.py`.

## Reproduction Commands

Fresh cross-seed run, with outputs isolated under `runs/<RUN_ID>/`:

```bash
bash scripts/run_v15_seeds.sh --run-id qwen_time_v15s_rerun
```

Aggregate explicit run outputs:

```bash
uv run python scripts/aggregate_seeds.py \
  --inputs \
  runs/qwen_time_v15s_rerun/reports/qwen_time_v15s_rerun_seed0_recall.json \
  runs/qwen_time_v15s_rerun/reports/qwen_time_v15s_rerun_seed1_recall.json \
  runs/qwen_time_v15s_rerun/reports/qwen_time_v15s_rerun_seed2_recall.json \
  --out runs/qwen_time_v15s_rerun/reports/qwen_time_v15s_rerun_aggregate.json
```

TPS sweep, isolated under `runs/<RUN_ID>/`:

```bash
bash scripts/run_tps_sweep.sh --run-id tps_rerun
```

External `tau_sessions` benchmark:

```bash
bash scripts/run_external_bench.sh --run-id external_tau_rerun
```

## Current Evidence

Use `reports/current/manifest.json` before citing a result. It records:

- current v15 cross-seed reports;
- probe and CLOCK-heldout reports;
- TPS negative policy-transfer reports;
- TPDR v2 seed-pair-dependent response-shape reports;
- external `tau_sessions` reports;
- ablations and causal controls;
- superseded, contaminated, and historical outputs.

## Known Negative Results

- Prompt-injected elapsed time beats CI on the current `tau_sessions`
  composite and adaptive length correlation.
- TPS is negative for CI v15s zero-shot policy transfer.
- TPDR v2 confirms only the headline seed pair and is not stable across all
  seed pairs.
- The repo does not claim subjective time perception or autonomous tracking
  between forward passes.

## Validation

Run unit and manifest checks:

```bash
uv run pytest
```

If `uv` is unavailable in the environment:

```bash
python3 -m pytest
```
