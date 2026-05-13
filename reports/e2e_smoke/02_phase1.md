# IPCN Eval Report

- Checkpoint: `checkpoints/e2e_smoke/phase1.pt`
- Train step: 200
- Generated: 2026-05-13T07:16:33.276651+00:00
- Trials per test: 5

## Pre-registered predictions

| # | Metric | Result | Threshold | Pass |
|---|---|---|---|---|
| H1 | D_0 mean | 0.5161 | > 0.1 | PASS |
| H2 | probe_acc max(H0,H1) | 0.5000 | >= 0.80 | FAIL |
| H3 | ablation order A0..A6 | -- | -- | DEFERRED |
| H4 | CTI mean | 0.0000 | > 0.70 | FAIL |
| H5 | Acc(evolve) - Acc(static) | 0.0000 | >= 0.15 | FAIL |
| H6 | KL(real, ablated) | 0.000000 | nonzero | ZERO |
| H7 | KL_amb / KL_exp | 0.0000 / 0.0000 | >= 0.5 / <= 0.1 | FAIL |

## Raw results

### H1
```json
{
  "n_trials": 5,
  "D0_mean": 0.5160619616508484,
  "D0_std": 0.1749030500650406,
  "D0_min": 0.28496426343917847,
  "D0_max": 0.7004066705703735,
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
  "n_rules": 4,
  "CTI_mean": 0.0,
  "Acc_pre_with": 0.0,
  "Acc_pre_without": 0.0,
  "Acc_post_with": 0.0,
  "Acc_post_without": 0.0,
  "threshold": 0.7,
  "passes": false
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
  "KL_amb_mean": 5.792565571027808e-06,
  "KL_amb_std": 7.913927447661723e-06,
  "KL_exp_mean": 7.003692462603794e-06,
  "KL_exp_std": 8.799315666152465e-06,
  "threshold_amb": 0.5,
  "threshold_exp": 0.1,
  "passes_amb": false,
  "passes_exp": true,
  "passes_overall": false
}
```
