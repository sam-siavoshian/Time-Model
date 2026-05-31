# Training Tracks

This repo now separates chronometric-injection training into two non-overlapping
tracks.

## Track A: Mechanistic CI

- Data families: `clock`, `silent_gap`, `phase`.
- Canonical runner: `scripts/run_v15_seeds.sh`.
- Alias runner: `scripts/run_track_a_mechanistic.sh`.
- Claim type: residual-stream elapsed-time transmission, causal gate use, and
  bounded in-distribution behavior.
- Paper status: this is the current supported evidence track for
  `paper/main.tex`.

Run:

```bash
bash scripts/run_track_a_mechanistic.sh --run-id ci_track_a_v15s
```

Track A writes under:

- `runs/<run_id>/data/track_a/`
- `runs/<run_id>/logs/track_a/`
- `runs/<run_id>/checkpoints/track_a/`
- `runs/<run_id>/reports/track_a/`

## Track B: Policy CI

- Data families: TPS forced-choice policy items.
- Labels: `REUSE`, `REFRESH`, `ASK`, `SUMMARIZE`, trained as single-letter
  answers `A`, `B`, `C`, or `D`.
- Canonical runner: `scripts/run_track_b_policy.sh`.
- Claim type: hidden elapsed-time control over downstream action selection.
- Paper status: current with caveat. The completed
  `ci_track_b_policy_full_safe_20260531` run supports policy-trained hidden-time action
  control, but not broad held-out-family generalization.

Run:

```bash
bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy
```

Fine-tune Track B from released Track A seed-matched checkpoints while still
writing separate Track B outputs:

```bash
TRACK_B_INIT_FROM_TRACK_A=1 SEEDS=0,1,2 RUN_CHRONO_ONLY=1 \
  bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy_from_track_a
```

Safe smoke restart profile for local desktop GPUs:

```bash
SAFE=1 WATCHDOG_MEM_GB=3 WATCHDOG_REQUIRE_ISOLATED=1 \
  bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy_safe_smoke
```

Safe mode defaults to one independent Track B seed, 800 steps, periodic
checkpoints every 100 steps, no pre-training vanilla eval, no controls, and a
limited post-training TPS eval. Run it in an isolated process group, for example
with `setsid`, when enabling the watchdog.

Track B writes under:

- `runs/<run_id>/data/track_b/`
- `runs/<run_id>/logs/track_b/`
- `runs/<run_id>/checkpoints/track_b/`
- `runs/<run_id>/reports/track_b/`

The default Track B data generator excludes held-out templates 8-11 and the
held-out `market_data` family from training. Evaluation still includes held-in,
held-out-template, held-out-family, prompt-only, hidden-only, both-agree, and
conflict conditions.

The prompt-timestamp baseline is represented by the `prompt_only` condition in
the vanilla TPS report: the elapsed time is visible in the prompt and the CI
scalar is zero.

Current Track B evidence:

- Canonical paper summary: `reports/track_b_policy_headline.json`.
- Run-local reports:
  `runs/ci_track_b_policy_full_safe_20260531/reports/track_b/headline.json`.
- Run-local checkpoints:
  `runs/ci_track_b_policy_full_safe_20260531/checkpoints/track_b/`.
- CI policy cross-seed headline: overall `0.756+/-0.023`,
  hidden-only `0.871+/-0.030`, monotonicity `0.691+/-0.027`,
  held-out-family `0.391+/-0.084`.

Interpretation: Track B is evidence that direct policy supervision can make the
hidden elapsed-time scalar affect action choice. It is not evidence that Track A
zero-shot transfers to TPS, and it should not be cited as broad domain
generalization because the held-out `market_data` family is weak.

Generate only the Track B train data:

```bash
python -m eval.tps.training_data \
  --out runs/<run_id>/data/track_b/tps_train_seed0.jsonl \
  --seed 0 \
  --split train
```

## Track C: Combined Training

There is intentionally no default combined Track A+B runner. Combining CLOCK /
SILENT-GAP / PHASE with TPS policy labels is a later Track C experiment after
Track B has stronger validation.

## Track B v2 Contract

The next Track B experiment should improve cohesion rather than only repeat the
first run:

- Keep Track A and Track B checkpoints separate on disk.
- Prefer a seed-matched Track-A-initialized fine-tune as the cohesive bridge
  experiment, using `TRACK_B_INIT_FROM_TRACK_A=1`.
- Add leave-one-family-out evaluation across every TPS family, not only the
  current fixed `market_data` holdout.
- Select checkpoints using a validation split before reporting the test split.
- Preserve negative controls: vanilla, LoRA-only policy, chrono-only policy,
  prompt-only, hidden-only, both-agree, and prompt/scalar conflict.
- Report both action accuracy and monotonicity of the family-specific
  long-delay action, with seed-level rows and cross-seed mean/std.
- Treat broad generalization as unsupported unless leave-one-family-out results
  are stable across families.
