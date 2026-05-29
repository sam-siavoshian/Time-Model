"""Generate paper figures from JSON reports.

Outputs PNGs into figures/:
  fig1_probe_r2_by_layer.png  -- linear + MLP probe R^2 per layer, three conditions
  fig2_t1_ood_scatter.png     -- v11 OOD tau predictions vs true tau (log-log)
  fig3_pressure_lengths.png   -- response length under three pressure conditions
  fig4_alpha_flip_scatter.png -- E (alpha sign flipped) vs A (normal) Pearson r demo
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def _load(name: str):
    p = REP / name
    if not p.exists():
        p = REP / "archive" / "model_versions_v10_v14" / name
    if not p.exists():
        print(f"  missing: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def fig_probe_r2_by_layer():
    """Linear probe (v4) + MLP probe (if exists), 3 conditions, per layer."""
    lin = _load("probe_v4_20260522_200113.json")
    if lin is None:
        return
    # Optional: most recent MLP probe
    mlp_files = sorted(REP.glob("probe_mlp_*.json"))
    mlp = None
    if mlp_files:
        with open(mlp_files[-1]) as f:
            mlp = json.load(f)

    fig, axes = plt.subplots(1, 2 if mlp else 1, figsize=(12 if mlp else 7, 5))
    if not mlp:
        axes = [axes]

    def _plot(ax, data, title):
        for cond, color, label in [
            ("condition_A_trained", "C0", "A: v11 trained"),
            ("condition_B_alpha_off", "C3", "B: alpha=0 (chrono off)"),
            ("condition_C_shuffled_labels", "C7", "C: shuffled labels"),
        ]:
            d = data.get(cond, {})
            xs = sorted(int(k) for k in d.keys())
            ys = [d[str(li)] for li in xs]
            # Clip extreme negatives for visibility
            ys = [max(-2.0, y) if y == y else None for y in ys]
            ax.plot(xs, ys, "o-", label=label, color=color, alpha=0.85, markersize=4)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.axhline(0.5, color="green", linestyle="--", linewidth=0.5,
                   alpha=0.4, label="R^2 = 0.5 threshold")
        ax.set_xlabel("Layer index (0 = embedding output)")
        ax.set_ylabel("OOD R^2")
        ax.set_title(title)
        ax.set_ylim(-2.2, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0], lin, "Linear ridge probe (v4)")
    if mlp:
        _plot(axes[1], mlp, "Nonlinear MLP probe")

    plt.tight_layout()
    out = FIG / "fig1_probe_r2_by_layer.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def fig_t1_ood_scatter():
    """v11 OOD scatter: true tau vs predicted tau (log-log)."""
    r = _load("disproof_20260522_224016_falsify.json")
    if r is None:
        return
    A = r["A_normal"]
    exs = A.get("examples", [])
    # examples are dicts with true_tau, eval_tau, resp, parsed
    true_t = [e["true_tau"] for e in exs if e.get("parsed", None) and e["parsed"] > 0]
    pred_t = [e["parsed"] for e in exs if e.get("parsed", None) and e["parsed"] > 0]

    # Augment with original v11 T1b examples if available
    v11 = _load("qwen_time_v10_20260516_032348_recall.json")
    if v11:
        for tau, pred in v11.get("t1b", {}).get("samples", []):
            true_t.append(tau)
            pred_t.append(pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_t, pred_t, alpha=0.6, s=50, color="C0", label="prediction")
    lo = max(1, min(min(true_t), min(pred_t)) / 2)
    hi = max(max(true_t), max(pred_t)) * 2
    plt.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x (perfect)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("true tau (seconds, log)")
    plt.ylabel("predicted tau (seconds, log)")
    plt.title(f"T1b OOD: predicted vs true tau (Pearson r={A.get('pearson_r',0):.3f})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out = FIG / "fig2_t1_ood_scatter.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def fig_pressure_lengths():
    """P1/P2/P3 length deltas as grouped bars."""
    r = _load("disproof_20260522_224016_pressure.json")
    if r is None:
        return
    P1 = r["P1"]
    P2 = r["P2"]
    P3 = r["P3"]
    conds = ["P1\ntext+chrono", "P2\nchrono alone", "P3\nalpha=0 + text"]
    short = [P1["mean_short"], P2["mean_short"], P3["mean_short"]]
    long = [P1["mean_long"], P2["mean_long"], P3["mean_long"]]
    import numpy as np
    x = np.arange(len(conds))
    w = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - w/2, short, w, label="short tau (30 s)", color="C3", alpha=0.85)
    plt.bar(x + w/2, long, w, label="long tau (3600 s)", color="C0", alpha=0.85)
    for i, (s, l) in enumerate(zip(short, long)):
        plt.text(i, max(s, l) + 2, f"Δ={l-s:+.1f}", ha="center", fontsize=9)
    plt.xticks(x, conds)
    plt.ylabel("Mean response length (tokens)")
    plt.title("Behavioral pressure: chrono signal contribution OOD")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    out = FIG / "fig3_pressure_lengths.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def fig_alpha_flip():
    """Bar of Pearson r across 5 falsify conditions."""
    r = _load("disproof_20260522_224016_falsify.json")
    if r is None:
        return
    conds = ["A: normal", "B: alpha=0", "C: random tau", "D: tau=0",
             "E: alpha flipped"]
    rs = [r["A_normal"]["pearson_r"], r["B_alpha_off"]["pearson_r"],
          r["C_random_tau"]["pearson_r"], r["D_tau_zero"]["pearson_r"],
          r["E_alpha_flipped"]["pearson_r"]]
    plt.figure(figsize=(8, 5))
    colors = ["C2", "C7", "C7", "C7", "C3"]
    bars = plt.bar(conds, rs, color=colors, alpha=0.85)
    for bar, r_ in zip(bars, rs):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 h + (0.05 if h >= 0 else -0.08),
                 f"{r_:+.3f}", ha="center", fontsize=10)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axhline(0.7, color="green", linestyle="--", linewidth=0.5,
                alpha=0.5, label="PASS gate (>=0.7)")
    plt.axhline(-0.7, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.ylim(-1.15, 1.15)
    plt.ylabel("Pearson r (T1 OOD)")
    plt.title("Falsification: r under 5 causal interventions")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=10)
    out = FIG / "fig4_alpha_flip_scatter.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def main():
    print("Generating figures...")
    fig_probe_r2_by_layer()
    fig_t1_ood_scatter()
    fig_pressure_lengths()
    fig_alpha_flip()
    print("Done.")


if __name__ == "__main__":
    main()
