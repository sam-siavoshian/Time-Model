"""Within-distribution probe: train AND test tau both in [1s, 7d].

Reviewer attack on §24.7.14: probe v4/v5 used OOD split (train tau ≤ 1e5,
test tau > 1e5), which produces -2.4 to -143 R^2 because ridge cannot
linearly extrapolate sinusoidal features. Within-dist random 80/20 split
removes the extrapolation requirement; if chrono encodes tau linearly
in the residual stream, R^2 should be high.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable
from model.qwen_time_probe import (
    PROMPT, hidden_states_for_tau, ridge_fit_predict, r2_score,
    collect_dataset, freeze_alpha,
)


def probe_per_layer_within(X, y, train_mask, test_mask):
    n_layers = X.shape[1]
    out = {}
    for li in range(n_layers):
        X_tr = X[train_mask, li]
        X_te = X[test_mask, li]
        y_tr = y[train_mask]
        y_te = y[test_mask]
        try:
            pred = ridge_fit_predict(X_tr, y_tr, X_te)
            out[li] = r2_score(y_te, pred)
        except Exception as e:
            print(f"  L{li}: error {type(e).__name__}: {e}")
            out[li] = float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--timescales", type=str, default="")
    args = p.parse_args()

    rng = random.Random(args.seed)
    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
            for _ in range(args.n_samples)]
    y = torch.tensor([math.log10(t) for t in taus], dtype=torch.float32)

    # Random 80/20 split (within distribution)
    n = args.n_samples
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_te = max(50, n // 5)
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[perm[:n_te]] = True
    train_mask = ~test_mask
    print(f"  Within-dist split: train={train_mask.sum().item()} test={test_mask.sum().item()}")

    print("\n=== CONDITION A: v15 trained (within-dist) ===")
    X_A = collect_dataset(model, taus, args.device)
    r2_A = probe_per_layer_within(X_A, y, train_mask, test_mask)
    for li, r in sorted(r2_A.items()):
        if r == r and (r > 0.3 or li < 5):
            bar = "#" * max(0, min(40, int(r * 40)))
            print(f"    L{li:2d}: R^2={r:+.3f}  {bar}")
    best_A = max(r2_A.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1e18)
    print(f"  BEST: L{best_A[0]} R^2={best_A[1]:.3f}")

    print("\n=== CONDITION B: alpha=0 (within-dist) ===")
    freeze_alpha(model)
    X_B = collect_dataset(model, taus, args.device)
    r2_B = probe_per_layer_within(X_B, y, train_mask, test_mask)
    best_B = max(r2_B.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1e18)
    print(f"  BEST: L{best_B[0]} R^2={best_B[1]:.3f}")

    verdict = {
        "A_trained_best_r2": float(best_A[1]),
        "A_trained_best_layer": int(best_A[0]),
        "B_alpha_off_best_r2": float(best_B[1]),
        "A_minus_B_gap": float(best_A[1]) - float(best_B[1]),
        "PASS_within_dist_linear_axis":
            best_A[1] > 0.6 and best_B[1] < 0.2,
    }
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "verdict": verdict,
            "condition_A_trained": {str(k): v for k, v in r2_A.items()},
            "condition_B_alpha_off": {str(k): v for k, v in r2_B.items()},
        }, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
