"""Nonlinear MLP probe across all Qwen layers.

Linear probe (model/qwen_time_probe.py v4) found tau encoded as a linear
axis only in shallow layers L1-L3 (R^2 = 0.43, 0.28, 0.10). Deeper
layers had negative linear R^2 -- the time axis got nonlinearly warped.

This probe fits a 1-hidden-layer MLP (256 hidden units, ReLU) per layer
to decode log10(tau) from the last-token hidden state. If tau is
genuinely represented throughout the network (just as nonlinear
features rather than a linear axis), MLP R^2 should rise above 0.5
at deeper layers.

Three conditions same as linear probe:
  A. v11 trained
  B. alpha=0 (chrono off)
  C. shuffled labels (sanity)

OOD split: train tau <= 1e5 s, test tau > 1e5 s.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable


PROMPT = (
    "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@torch.no_grad()
def hidden_states_for_tau(model: QwenTime, prompt: str, tau_t: float,
                          device: str) -> list:
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
    return [h[0, -1].float().cpu() for h in out.hidden_states]


def collect_dataset(model: QwenTime, taus: list, device: str) -> torch.Tensor:
    rows = []
    for i, tau in enumerate(taus):
        hs = hidden_states_for_tau(model, PROMPT, tau, device)
        rows.append(torch.stack(hs))
        if (i + 1) % 50 == 0:
            print(f"  collected {i+1}/{len(taus)}")
    return torch.stack(rows)


class MLPProbe(nn.Module):
    """Heavily-regularized small MLP: bottleneck projects 2048->d_bottle
    BEFORE the nonlinearity, then d_hidden=32 ReLU. Far fewer parameters
    than a vanilla 2048->256 MLP; survives n_train ~ 500."""

    def __init__(self, d_in: int, d_bottle: int = 64, d_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_bottle),
            nn.LayerNorm(d_bottle),
            nn.Dropout(0.3),
            nn.Linear(d_bottle, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp_probe(X_tr: torch.Tensor, y_tr: torch.Tensor,
                  X_te: torch.Tensor, y_te: torch.Tensor,
                  device: str, epochs: int = 400,
                  d_bottle: int = 64, d_hidden: int = 32,
                  lr: float = 1e-3, weight_decay: float = 1e-2) -> tuple:
    """Train MLP probe. Returns (best_test_r2, best_epoch)."""
    d_in = X_tr.shape[1]
    x_mean = X_tr.mean(0, keepdim=True)
    x_std = X_tr.std(0, keepdim=True).clamp(min=1e-6)
    X_tr_n = (X_tr - x_mean) / x_std
    X_te_n = (X_te - x_mean) / x_std
    y_mean = y_tr.mean()
    y_tr_c = y_tr - y_mean
    y_te_c = y_te - y_mean

    X_tr_n = X_tr_n.to(device)
    X_te_n = X_te_n.to(device)
    y_tr_c = y_tr_c.to(device)
    y_te_c = y_te_c.to(device)

    n = X_tr_n.shape[0]
    g = torch.Generator(); g.manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(8, n // 7)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Xv_tr, Xv_va = X_tr_n[tr_idx], X_tr_n[val_idx]
    yv_tr, yv_va = y_tr_c[tr_idx], y_tr_c[val_idx]

    ss_tot_te = ((y_te_c - y_te_c.mean()) ** 2).sum().item() + 1e-12
    ss_tot_va = ((yv_va - yv_va.mean()) ** 2).sum().item() + 1e-12

    best_val_r2 = -1e18
    best_test_r2 = float("nan")
    best_epoch = -1
    patience = 60
    bad = 0

    probe = MLPProbe(d_in, d_bottle=d_bottle, d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    bs = 128
    for ep in range(epochs):
        probe.train(True)
        pp = torch.randperm(Xv_tr.shape[0])
        for i in range(0, Xv_tr.shape[0], bs):
            idx = pp[i:i+bs]
            xb = Xv_tr[idx]
            yb = yv_tr[idx]
            pred = probe(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.train(False)
        with torch.no_grad():
            v_pred = probe(Xv_va)
            v_r2 = 1.0 - ((yv_va - v_pred) ** 2).sum().item() / ss_tot_va
            t_pred = probe(X_te_n)
            t_r2 = 1.0 - ((y_te_c - t_pred) ** 2).sum().item() / ss_tot_te
        if v_r2 > best_val_r2:
            best_val_r2 = v_r2
            best_test_r2 = t_r2
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_test_r2, best_epoch


def probe_per_layer_mlp(X: torch.Tensor, y: torch.Tensor,
                        train_mask: torch.Tensor, test_mask: torch.Tensor,
                        device: str) -> dict:
    n_layers = X.shape[1]
    out = {}
    for li in range(n_layers):
        X_tr = X[train_mask, li].float()
        X_te = X[test_mask, li].float()
        y_tr = y[train_mask]
        y_te = y[test_mask]
        try:
            r2, ep = fit_mlp_probe(X_tr, y_tr, X_te, y_te, device=device)
            out[li] = float(r2)
        except Exception as e:
            print(f"    L{li}: error {type(e).__name__}: {e}")
            out[li] = float("nan")
        if li % 4 == 0:
            print(f"    L{li:2d}: MLP R^2={out[li]:+.3f}")
    return out


def freeze_alpha(model: QwenTime) -> None:
    for inj in model.chrono_injectors.values():
        with torch.no_grad():
            inj.alpha.zero_()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/qwen_time_v11_probe_mlp.json")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n-samples", type=int, default=600)
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

    taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
            for _ in range(args.n_samples)]
    y = torch.tensor([math.log10(t) for t in taus], dtype=torch.float32)

    cutoff = math.log10(1e5)
    train_mask = y <= cutoff
    test_mask = y > cutoff
    print(f"  train={train_mask.sum().item()} test_OOD={test_mask.sum().item()}")

    print("\n=== CONDITION A: v11 trained ===")
    t0 = time.time()
    X_A = collect_dataset(model, taus, args.device)
    print(f"  collected in {time.time()-t0:.1f}s, shape={tuple(X_A.shape)}")
    r2_A = probe_per_layer_mlp(X_A, y, train_mask, test_mask, args.device)
    print(f"\n  per-layer OOD MLP R^2 (trained):")
    for li, r in sorted(r2_A.items()):
        bar = "#" * max(0, min(60, int(r * 40)))
        rs = f"{r:+.3f}" if r == r else "NaN"
        print(f"    L{li:3d}: R^2={rs}  {bar}")
    best_A = max(r2_A.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1e18)
    print(f"  BEST: L{best_A[0]} R^2={best_A[1]:.3f}")

    print("\n=== CONDITION B: alpha=0 (chrono OFF) ===")
    freeze_alpha(model)
    X_B = collect_dataset(model, taus, args.device)
    r2_B = probe_per_layer_mlp(X_B, y, train_mask, test_mask, args.device)
    best_B = max(r2_B.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1e18)
    print(f"  BEST: L{best_B[0]} R^2={best_B[1]:.3f}")

    print("\n=== CONDITION C: shuffled labels ===")
    perm = torch.randperm(y.shape[0])
    y_shuf = y[perm]
    r2_C = probe_per_layer_mlp(X_A, y_shuf, train_mask, test_mask, args.device)
    best_C = max(r2_C.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1e18)
    print(f"  BEST: L{best_C[0]} R^2={best_C[1]:.3f}")

    verdict = {
        "A_trained_best_r2": float(best_A[1]),
        "A_trained_best_layer": int(best_A[0]),
        "B_alpha_off_best_r2": float(best_B[1]),
        "B_alpha_off_best_layer": int(best_B[0]),
        "C_shuffled_best_r2": float(best_C[1]),
        "A_minus_B_gap": float(best_A[1]) - float(best_B[1]),
        "A_minus_C_gap": float(best_A[1]) - float(best_C[1]),
        "PASS_nonlinear_time_axis": (best_A[1] > 0.5 and best_B[1] < 0.2 and best_C[1] < 0.2),
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
        }, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
