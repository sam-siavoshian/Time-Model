# Track B Policy Documentation

Track B is the TPS policy-training track. It is intentionally separate from
Track A mechanistic CI.

## Current Run

- Run ID: `ci_track_b_policy_full_safe_20260531`
- Runner: `scripts/run_track_b_policy.sh`
- Canonical summary copied for the paper package:
  `reports/track_b_policy_headline.json`
- Run-local summary:
  `runs/ci_track_b_policy_full_safe_20260531/reports/track_b/headline.json`
- Checkpoints:
  `runs/ci_track_b_policy_full_safe_20260531/checkpoints/track_b/`

Track B checkpoints are policy adapters trained on TPS forced-choice labels.
They are not Track A v15s mechanistic checkpoints and should not be used as
evidence for zero-shot TPS transfer.

## Result Summary

| Model | Overall | Hidden-only | H/O template | H/O family | Scalar-follow | Monotonicity |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla policy | 0.401 | 0.418 | 0.366 | 0.181 | 0.377 | -0.000 |
| LoRA-only policy s0 | 0.581 | 0.574 | 0.574 | 0.155 | 0.585 | 0.000 |
| chrono-only policy s0 | 0.690 | 0.796 | 0.639 | 0.384 | 0.765 | 0.635 |
| CI policy seeds 0-2 | 0.756+/-0.023 | 0.871+/-0.030 | 0.713+/-0.020 | 0.391+/-0.084 | 0.871+/-0.029 | 0.691+/-0.027 |

Interpretation: direct TPS supervision makes the hidden elapsed-time scalar
usable for action choice. The clean Track B claim is hidden-time policy control
under task-specific supervision.

Caveat: held-out-family transfer to `market_data` is weak for CI policy
adapters. Track B therefore does not establish broad domain generalization.

## Segregation Rules

- Track A data stays under `runs/<run_id>/data/track_a/`.
- Track A checkpoints stay under `runs/<run_id>/checkpoints/track_a/`.
- Track A reports stay under `runs/<run_id>/reports/track_a/`.
- Track B data stays under `runs/<run_id>/data/track_b/`.
- Track B checkpoints stay under `runs/<run_id>/checkpoints/track_b/`.
- Track B reports stay under `runs/<run_id>/reports/track_b/`.
- Do not write Track B checkpoints into `release_ckpts/` unless they are
  explicitly released as policy adapters with Track B labels.

## Next Experiment

Track B v2 should not just rerun the same fixed `market_data` holdout. The next
cohesive experiment should:

- fine-tune from seed-matched Track A checkpoints with
  `TRACK_B_INIT_FROM_TRACK_A=1`, while writing all new artifacts under
  `runs/<run_id>/{data,logs,checkpoints,reports}/track_b/`;
- run leave-one-family-out folds across all TPS families;
- train and report seeds 0, 1, and 2 for each headline adapter family;
- include validation-based checkpoint selection;
- keep vanilla, LoRA-only, chrono-only, prompt-only, hidden-only, both-agree,
  and conflict controls;
- report action accuracy, hidden-only accuracy, scalar-follow, prompt-follow,
  held-out-template accuracy, held-out-family accuracy, and monotonicity;
- claim broad generalization only if family-level holdouts are stable.

## Safe Local Runs

Use the safe profile before any full local restart:

```bash
SAFE=1 WATCHDOG_MEM_GB=3 WATCHDOG_REQUIRE_ISOLATED=1 \
  bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy_safe_smoke
```

Safe mode runs one independent Track B seed, skips the pre-training vanilla eval,
uses 800 steps, writes periodic checkpoints every 100 steps, limits evaluation,
and disables optional control adapters. The watchdog kills the isolated run
process group when `MemAvailable` falls to 3 GiB or below.
