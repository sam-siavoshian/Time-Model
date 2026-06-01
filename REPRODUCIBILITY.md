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

## Training Tracks

- Track A Mechanistic CI: `CLOCK`, `SILENT-GAP`, and `PHASE`. This is the
  current paper evidence track.
- Track B Policy CI: TPS forced-choice labels `REUSE`, `REFRESH`, `ASK`, and
  `SUMMARIZE`. This is a separate policy-training track; the current run shows
  hidden-time policy control with weak held-out-family transfer.

See `docs/TRAINING_TRACKS.md` for the track map and output layout.

## Reproduction Commands

Fresh Track A cross-seed run, with outputs isolated under `runs/<RUN_ID>/`:

```bash
bash scripts/run_track_a_mechanistic.sh --run-id ci_track_a_v15s
```

Aggregate explicit run outputs:

```bash
uv run python scripts/aggregate_seeds.py \
  --inputs \
  runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_seed0_recall.json \
  runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_seed1_recall.json \
  runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_seed2_recall.json \
  --out runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_aggregate.json
```

Track B TPS policy-training run:

```bash
bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy
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
- Track B policy-trained TPS reports;
- Track B full-safe provenance bundle under
  `reports/track_b_policy_full_safe_20260531/`, including raw eval reports,
  train/eval JSONL, structured training logs, run manifest, done marker, and
  watchdog log;
- Track A and Track B final adapter checkpoints under `release_ckpts/`
  with SHA256 hashes in `release_ckpts/SHA256SUMS`;
- TPDR v2 seed-pair-dependent response-shape reports;
- external `tau_sessions` reports;
- ablations and causal controls;
- superseded, contaminated, and historical outputs.

## Known Negative Results

- Prompt-injected elapsed time beats CI on the current `tau_sessions`
  composite and adaptive length correlation.
- TPS is negative for CI v15s zero-shot policy transfer.
- Track B policy training is positive for hidden-time action control but weak on
  held-out-family transfer, so broad TPS generalization is not claimed.
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
