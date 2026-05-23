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
                      lams=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)) -> torch.Tensor:
    """SVD-based ridge with CV-picked lambda, float64 for stability.

    d_model=2048 hidden-state features with n_train ~ 516 is
    severely underdetermined. Naive `(X^T X + lam I)^-1 X^T y` in
    float32 produces NaN due to numerical singularities. Fix:
    one SVD of standardized X in float64; ridge then becomes
        w(lam) = V * (S / (S^2 + lam)) * U^T y
    Sweep lam, pick on a 20% val split, refit on full train.
    """
    Xtr = X_tr.double()
    Xte = X_te.double()
    ytr = y_tr.double()

    x_mean = Xtr.mean(dim=0, keepdim=True)
    x_std = Xtr.std(dim=0, keepdim=True).clamp(min=1e-6)
    Xc = (Xtr - x_mean) / x_std
    Xt = (Xte - x_mean) / x_std
    y_mean = ytr.mean()
    yc = ytr - y_mean

    n = Xc.shape[0]

    def _ridge_via_svd(X, y, lam):
        # X: (n, d). For underdetermined (n < d), use reduced SVD.
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # w = V * (S / (S^2 + lam)) * U^T y
        UTy = U.T @ y
        coeff = S / (S * S + lam)
        return Vh.T @ (coeff * UTy)

    # Pick lam via held-out 20% val
    if n < 8:
        best_lam = 1.0
    else:
        g = torch.Generator(); g.manual_seed(0)
        perm = torch.randperm(n, generator=g)
        n_val = max(2, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        Xv_tr, Xv_va = Xc[tr_idx], Xc[val_idx]
        yv_tr, yv_va = yc[tr_idx], yc[val_idx]
        # SVD once on val-train subset
        U, S, Vh = torch.linalg.svd(Xv_tr, full_matrices=False)
        UTy = U.T @ yv_tr
        ss_tot_va = ((yv_va - yv_va.mean()) ** 2).sum().item() + 1e-12
        best_r2 = -1e18
        best_lam = 1.0
        for lam in lams:
            coeff = S / (S * S + lam)
            w = Vh.T @ (coeff * UTy)
            pred = Xv_va @ w
            ss_res = ((yv_va - pred) ** 2).sum().item()
            r2 = 1.0 - ss_res / ss_tot_va
            if r2 == r2 and r2 > best_r2:
                best_r2 = r2
                best_lam = lam

    # Refit on full train with best lam
    w = _ridge_via_svd(Xc, yc, best_lam)
    pred = Xt @ w + y_mean
    # Clamp predictions to the y_train support to prevent ridge
    # underdetermined solutions from producing wild constants on
    # OOD test (the -143 R^2 floor in probe_v4 was such a constant).
    y_min, y_max = ytr.min(), ytr.max()
    pred = pred.clamp(min=y_min, max=y_max)
    return pred.float()


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
                    train_mask: torch.Tensor, test_mask: torch.Tensor) -> dict:
    """X: (N, L, d). y: (N,). Returns {layer: r2}."""
    n_layers = X.shape[1]
    out = {}
    for li in range(n_layers):
        X_tr = X[train_mask, li]
        X_te = X[test_mask, li]
        y_tr = y[train_mask]
        y_te = y[test_mask]
        try:
            y_pred = ridge_fit_predict(X_tr, y_tr, X_te)
            out[li] = r2_score(y_te, y_pred)
        except Exception as e:
            print(f"    L{li}: solver error: {type(e).__name__}: {e}")
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
        if r != r:  # NaN
            print(f"    L{li:3d}: R^2=NaN")
            continue
        bar = "#" * max(0, min(60, int(r * 40)))
        print(f"    L{li:3d}: R^2={r:+.3f}  {bar}")
    def _key(kv):
        v = kv[1]
        return v if v == v else -1e18
    best_layer_A = max(r2_A.items(), key=_key)
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
    best_layer_B = max(r2_B.items(), key=_key)
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
    best_layer_C = max(r2_C.items(), key=_key)
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
