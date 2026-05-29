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
- Paper status: experimental until policy-trained results exist.

Run:

```bash
bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy
```

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
Track B has been validated independently.
