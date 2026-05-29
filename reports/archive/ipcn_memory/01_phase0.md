# IPCN Eval Report

- Checkpoint: `checkpoints/e2e_smoke/phase0.pt`
- Train step: 100
- Generated: 2026-05-13T07:14:36.615605+00:00
- Trials per test: 5

## Pre-registered predictions

| # | Metric | Result | Threshold | Pass |
|---|---|---|---|---|
| H1 | D_0 mean | 0.5166 | > 0.1 | PASS |
| H2 | probe_acc max(H0,H1) | 0.5000 | >= 0.80 | FAIL |
| H3 | ablation order A0..A6 | -- | -- | DEFERRED |
| H4 | CTI mean | -- | -- | SKIPPED |
| H5 | Acc(evolve) - Acc(static) | 0.0000 | >= 0.15 | FAIL |
| H6 | KL(real, ablated) | 0.000000 | nonzero | ZERO |
| H7 | KL_amb / KL_exp | 0.0000 / 0.0000 | >= 0.5 / <= 0.1 | FAIL |

## Raw results

### H1
```json
{
  "n_trials": 5,
  "D0_mean": 0.5165615081787109,
  "D0_std": 0.17529265582561493,
  "D0_min": 0.28621944785118103,
  "D0_max": 0.7018408179283142,
  "pass_threshold": 0.1,
  "passes": true
}
```

### H2
```json
{
  "n_pairs": 5,
  "n_train_examples": 8,
  "n_test_examples": 2,
  "probe_acc_H0": 0.5,
  "probe_acc_H1": 0.5,
  "threshold": 0.8,
  "passes_H0": false,
  "passes_H1": false,
  "passes_either": false
}
```

### H3
```json
{
  "status": "deferred",
  "note": "H3 requires training A0-A6 variants. Run separately via run_phase --ablation."
}
```

### H4
```json
{
  "status": "skipped",
  "note": "H4 requires --pre-checkpoint (model BEFORE consolidation) plus current checkpoint."
}
```

### H5
```json
{
  "n_streams": 5,
  "Acc_evolve": 0.0,
  "Acc_static": 0.0,
  "gap": 0.0,
  "threshold": 0.15,
  "passes": false
}
```

### H6
```json
{
  "n_pairs": 5,
  "KL_real_vs_ablated_mean": 0.0,
  "threshold_min": 0.1,
  "passes_distinct": false,
  "note": "Approximate: full spec measures answer-accuracy delta; we use KL on output distribution."
}
```

### H7
```json
{
  "n_pairs": 5,
  "KL_amb_mean": 5.8059068578586445e-06,
  "KL_amb_std": 7.963436279040448e-06,
  "KL_exp_mean": 7.032338089629775e-06,
  "KL_exp_std": 8.859687779607614e-06,
  "threshold_amb": 0.5,
  "threshold_exp": 0.1,
  "passes_amb": false,
  "passes_exp": true,
  "passes_overall": false
}
```
