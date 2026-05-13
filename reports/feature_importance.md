# Feature Importance Report

- Checkpoint: `untrained baseline`
- Batches used: 20

## Loss term contribution (% of total)

| Loss | Fraction | Sum |
|---|---|---|
| L_lm | 99.81% | 221.6355 |
| L_precision | 0.09% | 0.2006 |
| L_chrono | 0.07% | 0.1620 |
| L_slot_util | 0.02% | 0.0555 |
| L_pre_influence | 0.00% | 0.0000 |
| L_diversity | 0.00% | 0.0000 |

## Gradient magnitude per top-level group

| Group | Mean |abs grad| |
|---|---|
| pfc | 2.864670e-03 |
| broadcast | 2.402359e-03 |
| core | 1.578283e-03 |
| memory | 0.000000e+00 |

## Slot utilization

```json
{
  "top5_mass": 0.625,
  "bottom5_mass": 0.625,
  "entropy": 5.545178413391113,
  "max_entropy_if_uniform": 5.545177459716797,
  "top10_slots": [
    128,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "n_zero_slots": 0
}
```