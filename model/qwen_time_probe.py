"""Linear probe: decode log(tau) from each Qwen layer's hidden state.

Mechanistic figure for the paper. If a deep layer's hidden state encodes
tau as a continuous variable, a linear regression should recover log(tau)
on held-out values with high R^2. If the model only template-matches,
hidden states would not carry a continuous tau axis -- probe R^2 would
collapse to near zero.

Procedure:
  1. Sample N=1000 tau values log-uniform in [1s, 7d]. Disjoint split:
       - probe-train tau in [1, 1e5]
       - probe-test (OOD) tau in [1e5, 7*86400] = [27.7h, 7d]
     OOD split is the load-bearing test. If probe overfits to a finite
     set of training tau buckets, OOD R^2 collapses.
  2. For each tau, run forward on a neutral prompt with
     output_hidden_states=True. Capture last-token hidden state at every
     layer (n_layers+1 incl. embedding).
  3. Fit ridge regression per layer: log10(tau) ~ W @ h + b.
     Closed form: w = (X^T X + lam*I)^-1 X^T y.
  4. Report R^2 per layer on held-out OOD tau.

Falsifier conditions also run:
  A. v11 ckpt loaded (full model)            -> expected: high R^2 deep
  B. zero alpha (chrono off at eval)         -> expected: R^2 ~ 0
  C. shuffled tau labels at probe-fit        -> expected: R^2 ~ 0

If A is high AND B is near-zero AND C is near-zero, evidence is strong
that tau is encoded as a continuous variable in the hidden states by
the trained alpha/gamma/beta projectors, not template-matched in lm_head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable


PROMPT = (
    "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@torch.no_grad()
def hidden_states_for_tau(model: QwenTime, prompt: str, tau_t: float,
                          device: str) -> list:
    """Return list of per-layer last-token hidden states for given tau."""
    tok = model.tokenizer
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    chi = model.chrono(torch.tensor(tau_t, device=device, dtype=torch.float32))
    chi_t = chi.squeeze(0) if chi.dim() > 1 else chi
    model._chi_stack.append(chi_t)
    try:
        out = model.base(input_ids=ids, return_dict=True,
                         output_hidden_states=True)
    finally:
        model._chi_stack.pop()
    # tuple of (n_layers+1) tensors, each (B=1, L, d_model)
    return [h[0, -1].float().cpu() for h in out.hidden_states]


def ridge_fit_predict(X_tr: torch.Tensor, y_tr: torch.Tensor,
                      X_te: torch.Tensor,
                      lams=(1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)) -> torch.Tensor:
    """Closed-form ridge with feature standardization and CV-picked lambda.

    Critical: d_model=2048 features and ~few-hundred train samples is
    severely underdetermined. Without proper scaling + regularization,
    ridge degenerates to OLS and test R^2 explodes to large negatives.
    Fix: standardize each feature to unit variance, then sweep lam on a
    held-out 20% sub-validation split, refit on full train with winner.
    """
    d = X_tr.shape[1]
    x_mean = X_tr.mean(dim=0, keepdim=True)
    x_std = X_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
    Xc = (X_tr - x_mean) / x_std
    Xt = (X_te - x_mean) / x_std
    y_mean = y_tr.mean()
    yc = y_tr - y_mean

    n = Xc.shape[0]
    if n < 8:
        # Too few samples for CV; use the largest lam
        best_lam = lams[-1]
    else:
        g = torch.Generator(); g.manual_seed(0)
        perm = torch.randperm(n, generator=g)
        n_val = max(2, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        Xv_tr, Xv_va = Xc[tr_idx], Xc[val_idx]
        yv_tr, yv_va = yc[tr_idx], yc[val_idx]
        I = torch.eye(d, dtype=Xc.dtype)
        best_r2 = -1e18
        best_lam = lams[-1]
        for lam in lams:
            try:
                A = Xv_tr.T @ Xv_tr + lam * I
                w = torch.linalg.solve(A, Xv_tr.T @ yv_tr)
                pred = Xv_va @ w
                ss_res = ((yv_va - pred) ** 2).sum().item()
                ss_tot = ((yv_va - yv_va.mean()) ** 2).sum().item() + 1e-12
                r2 = 1.0 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_lam = lam
            except Exception:
                continue

    # Refit on full standardized train with best lambda
    I = torch.eye(d, dtype=Xc.dtype)
    A = Xc.T @ Xc + best_lam * I
    w = torch.linalg.solve(A, Xc.T @ yc)
    return Xt @ w + y_mean


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().item()
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def collect_dataset(model: QwenTime, taus: list, device: str) -> torch.Tensor:
    """Returns tensor of shape (N, n_layers+1, d_model)."""
    rows = []
    for i, tau in enumerate(taus):
        hs = hidden_states_for_tau(model, PROMPT, tau, device)
        rows.append(torch.stack(hs))  # (n_layers+1, d_model)
        if (i + 1) % 50 == 0:
            print(f"  collected {i+1}/{len(taus)}")
    return torch.stack(rows)  # (N, n_layers+1, d_model)


def probe_per_layer(X: torch.Tensor, y: torch.Tensor,
                    train_mask: torch.Tensor, test_mask: torch.Tensor,
                    lam: float = 1e-2) -> dict:
    """X: (N, L, d). y: (N,). Returns {layer: r2}."""
    n_layers = X.shape[1]
    out = {}
    for li in range(n_layers):
        X_tr = X[train_mask, li]
        X_te = X[test_mask, li]
        y_tr = y[train_mask]
        y_te = y[test_mask]
        try:
            y_pred = ridge_fit_predict(X_tr, y_tr, X_te, lam=lam)
            out[li] = r2_score(y_te, y_pred)
        except Exception as e:
            out[li] = float("nan")
    return out


def freeze_alpha(model: QwenTime) -> None:
    """Falsifier B: zero out all alpha gates -> chrono injection becomes
    pure identity. If trained model encodes tau in hidden states, this
    should kill the probe signal entirely."""
    for inj in model.chrono_injectors.values():
        with torch.no_grad():
            inj.alpha.zero_()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/qwen_time_v11_probe.json")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    rng = random.Random(args.seed)

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading ckpt {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    # Sample tau values log-uniform across [1s, 7d]
    taus = []
    for _ in range(args.n_samples):
        taus.append(math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0))))
    y = torch.tensor([math.log10(t) for t in taus], dtype=torch.float32)

    # train/test split by tau cutoff (OOD)
    cutoff = math.log10(1e5)
    train_mask = y <= cutoff
    test_mask = y > cutoff
    print(f"  split: train={train_mask.sum().item()} (tau<=1e5), "
          f"test={test_mask.sum().item()} (tau>1e5 OOD)")

    # Condition A: trained model
    print("\n=== CONDITION A: v11 trained model ===")
    t0 = time.time()
    X_A = collect_dataset(model, taus, args.device)
    print(f"  collected in {time.time()-t0:.1f}s, shape={tuple(X_A.shape)}")
    r2_A = probe_per_layer(X_A, y, train_mask, test_mask)
    print(f"  per-layer OOD R^2 (trained):")
    for li, r in sorted(r2_A.items()):
        bar = "#" * max(0, int(r * 40))
        print(f"    L{li:3d}: R^2={r:+.3f}  {bar}")
    best_layer_A = max(r2_A.items(), key=lambda kv: kv[1])
    print(f"  BEST: L{best_layer_A[0]} R^2={best_layer_A[1]:.3f}")

    # Condition B: zero alpha (chrono off)
    print("\n=== CONDITION B: alpha=0 (chrono injection OFF) ===")
    freeze_alpha(model)
    X_B = collect_dataset(model, taus, args.device)
    r2_B = probe_per_layer(X_B, y, train_mask, test_mask)
    print(f"  per-layer OOD R^2 (chrono OFF):")
    for li, r in sorted(r2_B.items()):
        if li % 4 == 0 or r > 0.1:
            print(f"    L{li:3d}: R^2={r:+.3f}")
    best_layer_B = max(r2_B.items(), key=lambda kv: kv[1])
    print(f"  BEST: L{best_layer_B[0]} R^2={best_layer_B[1]:.3f}")

    # Condition C: shuffled labels (sanity / negative ctrl on the probe)
    print("\n=== CONDITION C: shuffled tau labels (probe sanity) ===")
    perm = torch.randperm(y.shape[0])
    y_shuf = y[perm]
    r2_C = probe_per_layer(X_A, y_shuf, train_mask, test_mask)
    print(f"  per-layer OOD R^2 (shuffled labels):")
    for li, r in sorted(r2_C.items()):
        if li % 8 == 0 or r > 0.1:
            print(f"    L{li:3d}: R^2={r:+.3f}")
    best_layer_C = max(r2_C.items(), key=lambda kv: kv[1])
    print(f"  BEST: L{best_layer_C[0]} R^2={best_layer_C[1]:.3f}")

    # Verdict
    a_best = best_layer_A[1]
    b_best = best_layer_B[1]
    c_best = best_layer_C[1]
    verdict = {
        "A_trained_best_r2": a_best,
        "A_trained_best_layer": best_layer_A[0],
        "B_alpha_off_best_r2": b_best,
        "C_shuffled_best_r2": c_best,
        "A_minus_B_gap": a_best - b_best,
        "A_minus_C_gap": a_best - c_best,
        "PASS_continuous_time_axis": (a_best > 0.6 and b_best < 0.2 and c_best < 0.2),
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
            "condition_C_shuffled_labels": {str(k): v for k, v in r2_C.items()},
            "n_train": int(train_mask.sum().item()),
            "n_test": int(test_mask.sum().item()),
            "cutoff_log10_tau": cutoff,
        }, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
