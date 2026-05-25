"""Figure 10: 3B vs 7B scaling comparison on the five pre-registered tests.

Reads reports/v15_cross_seed_aggregate.json for the 3B cross-seed mean/std
and reports/scale_7b_24k_*_recall.json for the 7B single-seed numbers.
Visual proof that the architecture scales without degradation up to 7B and
that the T3 fragility on 3B (one of three seeds mode-collapses) is resolved
at 7B (bidirectional pass).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CROSS = Path("reports/v15_cross_seed_aggregate.json")
SCALE_7B = Path("reports/scale_7b_24k_20260524_180844_recall.json")
DST = Path("figures/fig10_scaling.png")


METRICS = [
    ("T1 clock r", "T1_clock_pearson_r", "T1_clock_pearson_r"),
    ("T1b OOD r", "T1b_ood_pearson_r", "T1b_ood_pearson_r"),
    ("T2 silent-gap", "T2_ack_delta", "T2_ack_delta"),
    ("T3 weekday signal", "T3_weekday_signal", "T3_weekday_signal"),
    ("T3 weekend signal", "T3_weekend_signal", "T3_weekend_signal"),
    ("T4 first-pos KL", "T4_mean_pairwise_kl", "T4_mean_pairwise_kl"),
]


def main() -> None:
    cross = json.loads(CROSS.read_text())["aggregate"]
    seven = json.loads(SCALE_7B.read_text())["summary"]

    fig, ax = plt.subplots(figsize=(12, 5.0))

    labels, m_3b, s_3b, m_7b = [], [], [], []
    for label, key_3b, key_7b in METRICS:
        m = cross[key_3b]
        labels.append(label)
        m_3b.append(m["mean"])
        s_3b.append(m["std"])
        m_7b.append(float(seven[key_7b]))

    x = np.arange(len(labels))
    w = 0.36

    bars_3b = ax.bar(x - w/2, m_3b, w, yerr=s_3b, capsize=6,
                     color="#3498db", edgecolor="black", linewidth=0.7,
                     label="Qwen 2.5 3B (n=3, mean +/- std)",
                     error_kw={"ecolor": "black", "lw": 1.2})
    bars_7b = ax.bar(x + w/2, m_7b, w,
                     color="#c0392b", edgecolor="black", linewidth=0.7,
                     label="Qwen 2.5 7B (single seed, 24K steps)")

    # value labels
    for i, (v, s) in enumerate(zip(m_3b, s_3b)):
        ax.text(i - w/2, v + s + 0.03, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color="#1f618d")
    for i, v in enumerate(m_7b):
        ax.text(i + w/2, v + 0.03, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color="#7b241c",
                fontweight="bold")

    # T3 highlight: small green arrows on the two T3 columns showing improvement
    for idx in (3, 4):
        v3b = m_3b[idx]
        v7b = m_7b[idx]
        if v7b > v3b + 0.2:
            ax.annotate("",
                        xy=(idx + w/2, v7b - 0.05),
                        xytext=(idx - w/2, v3b + 0.10),
                        arrowprops=dict(arrowstyle="->", lw=1.3,
                                        color="#229954"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("metric value", fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.set_title("3B vs 7B: chronometric injection scales without degradation",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10,
              bbox_to_anchor=(0.01, 0.99))

    # caption below the two T3 columns (inside plot, no overlap with legend)
    ax.text(3.5, 0.85,
            "T3 phase fragility resolved at scale:\n"
            "3B = 1 of 3 seeds passes weekday; 7B = bidirectional 1.00 / 1.00",
            ha="center", va="top", fontsize=8.5, color="#229954",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#229954", alpha=0.85))

    fig.tight_layout()
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
