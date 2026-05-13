# Ablation summary

- Cache: `data/tokenized/ambiguity/train`
- Steps per variant: 30
- Seed: 42

| Variant | trainable params | wall (s) | final LM (last 10) | D_0 mean |
|---|---|---|---|---|
| A0 | 99,914,378 | 18.9 | 2.5420 | 0.0000 |
| A2 | 99,914,378 | 20.0 | 2.5479 | 0.2873 |
| A3 | 99,914,378 | 21.1 | 2.5374 | 0.3459 |
| A4 | 99,914,378 | 26.2 | 2.5374 | 0.3459 |
| A5 | 101,077,642 | 32.1 | 2.5671 | 0.3503 |
| A6 | 101,298,826 | 29.4 | 2.5533 | 0.3421 |

## H3 ordering check (D_0 mean)
Expected: A0 < A2 < A3 < A4 < A5 < A6
Observed: A0=0.000 < A2=0.287 < A3=0.346 < A4=0.346 < A5=0.350 < A6=0.342