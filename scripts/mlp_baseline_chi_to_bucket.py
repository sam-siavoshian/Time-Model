"""W8 baseline: how well can a tiny MLP map chi(tau) -> bucket string
(via duration index) compared to the full CI architecture on T1b?

If a small MLP on chi(tau) already gives near-perfect r on this task,
then T1/T1b are largely a learned-tokenizer problem and the LLM is not
contributing much beyond a 148-class classifier head.

Output: reports/mlp_baseline_chi_to_bucket.json
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TIMESCALES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384,
              65536, 86400, 604800)


def chi(tau: float) -> np.ndarray:
    parts = []
    for T in TIMESCALES:
        w = 2 * math.pi / T
        parts.append(math.sin(w * tau))
        parts.append(math.cos(w * tau))
    parts.append(math.log1p(tau))
    return np.array(parts, dtype=np.float32)


def fmt_seconds(secs: float) -> str:
    if secs < 60:
        n = int(round(secs))
        return f"{n} second{'s' if n != 1 else ''}"
    if secs < 3600:
        n = int(round(secs / 60))
        return f"{n} minute{'s' if n != 1 else ''}"
    if secs < 86400:
        n = int(round(secs / 3600))
        return f"about {n} hour{'s' if n != 1 else ''}"
    n = int(round(secs / 86400))
    return f"about {n} day{'s' if n != 1 else ''}"


def parse_bucket(s: str) -> float:
    """Inverse of fmt_seconds: parse bucket string back to seconds."""
    parts = s.split()
    if "second" in s:
        return float(parts[0])
    if "minute" in s:
        return float(parts[0]) * 60
    if "hour" in s:
        return float(parts[1]) * 3600
    if "day" in s:
        return float(parts[1]) * 86400
    return float("nan")


def build_bucket_index(n_train: int = 30000, seed: int = 0):
    """Enumerate every bucket string the formatter can emit over [1s, 7d]."""
    buckets = set()
    rng = random.Random(seed)
    for _ in range(n_train):
        tau = math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
        buckets.add(fmt_seconds(tau))
    # also enumerate the exact integer-rounded boundaries
    for k in range(1, 60):
        buckets.add(f"{k} second{'s' if k != 1 else ''}")
    for k in range(1, 60):
        buckets.add(f"{k} minute{'s' if k != 1 else ''}")
    for k in range(1, 24):
        buckets.add(f"about {k} hour{'s' if k != 1 else ''}")
    for k in range(1, 8):
        buckets.add(f"about {k} day{'s' if k != 1 else ''}")
    return sorted(buckets, key=parse_bucket)


def sample(n: int, seed: int) -> tuple[list[float], list[str]]:
    rng = random.Random(seed)
    taus, labels = [], []
    for _ in range(n):
        tau = math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
        taus.append(tau)
        labels.append(fmt_seconds(tau))
    return taus, labels


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def pearson(a: list[float], b: list[float]) -> float:
    aa = np.array(a, dtype=np.float64)
    bb = np.array(b, dtype=np.float64)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def log_mae(a: list[float], b: list[float]) -> float:
    return float(np.mean(np.abs(np.log10(np.array(a)) - np.log10(np.array(b)))))


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("building bucket index...")
    buckets = build_bucket_index()
    b2i = {b: i for i, b in enumerate(buckets)}
    print(f"  {len(buckets)} unique bucket strings")

    print("sampling 30k train + 24 T1b-equivalent eval points...")
    train_taus, train_labels = sample(30000, seed=1)
    eval_taus, eval_labels = sample(24, seed=2)

    Xtr = np.stack([chi(t) for t in train_taus])
    Ytr = np.array([b2i[l] for l in train_labels], dtype=np.int64)
    Xev = np.stack([chi(t) for t in eval_taus])
    Yev = np.array([b2i[l] for l in eval_labels], dtype=np.int64)

    # train tiny MLP
    print("training MLP (chi(tau) -> bucket)...")
    model = MLP(in_dim=31, n_classes=len(buckets)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Xev_t = torch.from_numpy(Xev).to(DEVICE)

    n_epochs = 200
    bs = 256
    n = len(Xtr)
    for ep in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            logits = model(Xtr_t[b])
            loss = loss_fn(logits, Ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                preds = model(Xev_t).argmax(-1).cpu().numpy()
                acc = (preds == Yev).mean()
                pred_taus = [parse_bucket(buckets[i]) for i in preds]
                print(f"  ep{ep+1} loss={loss.item():.4f} eval-acc={acc:.3f} r={pearson(eval_taus, pred_taus):.4f}")

    # final eval
    with torch.no_grad():
        preds = model(Xev_t).argmax(-1).cpu().numpy()
    pred_taus = [parse_bucket(buckets[i]) for i in preds]
    acc = float((preds == Yev).mean())
    r = pearson(eval_taus, pred_taus)
    mae = log_mae(eval_taus, pred_taus)

    # linear baseline (no hidden layer): just chi -> bucket
    print("\nlinear baseline (chi -> bucket, no hidden)...")
    lin = nn.Linear(31, len(buckets)).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    for ep in range(200):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            loss = loss_fn(lin(Xtr_t[b]), Ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        preds_lin = lin(Xev_t).argmax(-1).cpu().numpy()
    pred_taus_lin = [parse_bucket(buckets[i]) for i in preds_lin]
    acc_lin = float((preds_lin == Yev).mean())
    r_lin = pearson(eval_taus, pred_taus_lin)
    mae_lin = log_mae(eval_taus, pred_taus_lin)

    out = {
        "task": "MLP on chi(tau) -> bucket (matched to T1b protocol)",
        "n_train": 30000, "n_eval": 24, "n_buckets": len(buckets),
        "tau_range_seconds": [1.0, 7 * 86400.0],
        "mlp_2x128": {
            "eval_acc": acc, "pearson_r": r, "log10_mae": mae,
        },
        "linear": {
            "eval_acc": acc_lin, "pearson_r": r_lin, "log10_mae": mae_lin,
        },
        "ci_v15_cross_seed_for_comparison": {
            "pearson_r": 0.993, "pearson_r_std": 0.003,
            "log10_mae": 0.044, "log10_mae_std": 0.010,
            "note": "from reports/v15_cross_seed_aggregate.json T1b",
        },
        "interpretation": (
            "MLP / linear baseline numbers shown for the question 'does the chrono encoder "
            "alone (no LLM) already solve T1b?' If the MLP/linear gets r near 1 and log-MAE "
            "near 0.044, then T1/T1b are largely a learned-tokenizer task and the LLM "
            "contributes little. Lower numbers here = the LLM is contributing real work."
        ),
    }
    Path("reports/mlp_baseline_chi_to_bucket.json").write_text(json.dumps(out, indent=2))
    print(f"\nMLP r={r:.4f}, log-MAE={mae:.4f}, acc={acc:.3f}")
    print(f"Linear r={r_lin:.4f}, log-MAE={mae_lin:.4f}, acc={acc_lin:.3f}")
    print(f"CI v15 cross-seed r=0.993, log-MAE=0.044")
    print("saved -> reports/mlp_baseline_chi_to_bucket.json")


if __name__ == "__main__":
    main()
