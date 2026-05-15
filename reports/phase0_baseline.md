# IPCN Eval Report

- Checkpoint: `checkpoints/phase0_ipcn_phase0_initial_final.pt`
- Train step: 50000
- Generated: 2026-05-14T01:41:55.628375+00:00
- Trials per test: 20

## Pre-registered predictions

| # | Metric | Result | Threshold | Pass |
|---|---|---|---|---|
| H1 | D_0 mean | 0.0858 | > 0.1 | FAIL |
| H2 | probe_acc max(H0,H1) | 0.5000 | >= 0.80 | FAIL |
| H3 | ablation order A0..A6 | -- | -- | DEFERRED |
| H4 | CTI mean | -- | -- | SKIPPED |
| H5 | Acc(evolve) - Acc(static) | 0.0000 | >= 0.15 | FAIL |
| H6 | KL(real, ablated) | 0.000000 | nonzero | ZERO |
| H7 | KL_amb / KL_exp | 0.0339 / 0.0167 | >= 0.5 / <= 0.1 | FAIL |

## Prefix Integrity (SPEC.tex §16.4)

| Condition | Accuracy |
|---|---|
| correct | 0.0000 |
| zero | 0.0000 |
| shuffled | 0.0000 |
| adversarial | 0.0000 |

Expected: correct >= shuffled > adversarial; zero < correct

## Diagnostic metrics (SPEC.tex §17.2)

- H_P (prefix entropy): 4.6414
- H_u (slot util entropy): 5.5452 (max if uniform: 5.5452)
- D_Omega (adapter drift): 8.9433

## Raw results

### H1
```json
{
  "n_trials": 20,
  "D0_mean": 0.08581115305423737,
  "D0_std": 0.023560991510748863,
  "D0_min": 0.046417880803346634,
  "D0_max": 0.13015231490135193,
  "pass_threshold": 0.1,
  "passes": false
}
```

### H2
```json
{
  "n_pairs": 20,
  "n_train_examples": 30,
  "n_test_examples": 10,
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
  "n_streams": 10,
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
  "n_pairs": 20,
  "KL_real_vs_ablated_mean": 0.0,
  "threshold_min": 0.1,
  "passes_distinct": false,
  "note": "Approximate: full spec measures answer-accuracy delta; we use KL on output distribution."
}
```

### H7
```json
{
  "n_pairs": 20,
  "KL_amb_mean": 0.03393766029803373,
  "KL_amb_std": 0.07012042469166126,
  "KL_exp_mean": 0.01666440271146712,
  "KL_exp_std": 0.024771612293042294,
  "threshold_amb": 0.5,
  "threshold_exp": 0.1,
  "passes_amb": false,
  "passes_exp": true,
  "passes_overall": false
}
```

### PrefixIntegrity
```json
{
  "n_pairs": 20,
  "acc_correct": 0,
  "acc_zero": 0,
  "acc_shuffled": 0,
  "acc_adversarial": 0,
  "expected_ordering": "correct >= shuffled > adversarial; zero < correct"
}
```

### DiagMetrics
```json
{
  "prefix_entropy_H_P": 4.641361236572266,
  "slot_util_entropy_H_u": 5.545178413391113,
  "slot_util_entropy_max_if_uniform": 5.545177444479562,
  "adapter_drift_D_Omega": 8.943341886857764
}
```
