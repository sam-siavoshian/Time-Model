"""Figure 9: cross-seed (n=3) variance bars for the five pre-registered tests.

Reads reports/v15_cross_seed_aggregate.json and shows mean +/- std per metric.
Visual proof of reproducibility for the main result. Bars colored by status
(green pass, yellow partial, red fail) so the reader sees at a glance which
tests are saturated, which are noisy, and which mode-collapse on some seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SRC = Path("reports/v15_cross_seed_aggregate.json")
DST = Path("figures/fig9_cross_seed_variance.png")


METRICS = [
    ("T1\nclock r", "T1_clock_pearson_r", 0.8, "higher is better"),
    ("T1b\nOOD r", "T1b_ood_pearson_r", 0.7, "higher is better"),
    ("T2\nsilent-gap delta", "T2_ack_delta", 0.5, "higher is better"),
    ("T4 first-pos\nKL", "T4_mean_pairwise_kl", 0.05, "higher is better"),
]

# T1b log-MAE separately (lower better) on a second axis
LOG_MAE_KEY = "T1b_ood_log_mae"
LOG_MAE_THRESHOLD = 0.5


def color_from(mean: float, threshold: float) -> str:
    if mean >= threshold * 1.5:
        return "#27ae60"   # solid pass
    if mean >= threshold:
        return "#f1c40f"   # partial
    return "#c0392b"        # fail


def main() -> None:
    d = json.loads(SRC.read_text())["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [3.2, 1]})

    # LEFT: higher-is-better metrics
    ax = axes[0]
    labels, means, stds, vals_list, thresholds, colors = [], [], [], [], [], []
    for label, key, thr, _ in METRICS:
        m = d[key]
        labels.append(label)
        means.append(m["mean"])
        stds.append(m["std"])
        vals_list.append(m["vals"])
        thresholds.append(thr)
        colors.append(color_from(m["mean"], thr))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  edgecolor="black", linewidth=0.7,
                  error_kw={"ecolor": "black", "lw": 1.4})

    # threshold dashes per metric - drawn ABOVE bars with white bbox so they
    # are not buried inside the bar fill
    for i, thr in enumerate(thresholds):
        ax.hlines(thr, i - 0.45, i + 0.45, colors="black",
                  linestyles="--", lw=1.4, alpha=1.0, zorder=4)
        ax.text(i, thr - 0.045, f"thr {thr}", fontsize=8, va="top",
                ha="center", color="black", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="black", lw=0.5, alpha=0.95))

    # individual seed dots
    for i, vals in enumerate(vals_list):
        for v in vals:
            ax.plot([i - 0.10, i + 0.10], [v, v], color="black",
                    alpha=0.75, lw=0.8)

    # value labels above bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.04, f"{m:.3f} +/- {s:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("metric value", fontsize=10)
    ax.set_ylim(0, 1.30)
    ax.set_title("v15 cross-seed (n=3) - higher is better metrics",
                 fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # RIGHT: T1b log-MAE separately (lower better)
    ax2 = axes[1]
    mae = d[LOG_MAE_KEY]
    mean_mae, std_mae = mae["mean"], mae["std"]
    bar_color = "#27ae60" if mean_mae < LOG_MAE_THRESHOLD * 0.5 else "#f1c40f"
    ax2.bar([0], [mean_mae], yerr=[std_mae], capsize=8, color=bar_color,
            edgecolor="black", linewidth=0.7,
            error_kw={"ecolor": "black", "lw": 1.4}, width=0.5)
    ax2.hlines(LOG_MAE_THRESHOLD, -0.30, 0.30, colors="black",
               linestyles="--", lw=1.4, alpha=1.0, zorder=4)
    ax2.text(0, LOG_MAE_THRESHOLD - 0.025, f"thr {LOG_MAE_THRESHOLD}",
             fontsize=8, va="top", ha="center", fontweight="bold", zorder=5,
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="black", lw=0.5, alpha=0.95))
    for v in mae["vals"]:
        ax2.plot([-0.12, 0.12], [v, v], color="black", alpha=0.75, lw=0.8)
    ax2.text(0, mean_mae + std_mae + 0.025,
             f"{mean_mae:.3f} +/- {std_mae:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["T1b\nlog-MAE\n(lower is better)"], fontsize=9.5)
    ax2.set_ylabel("log10-MAE", fontsize=10)
    ax2.set_ylim(0, max(0.6, mean_mae + std_mae + 0.15))
    ax2.set_title("T1b precision", fontsize=11, pad=8)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # global legend
    legend = [
        plt.Rectangle((0, 0), 1, 1, color="#27ae60",
                      label="passes pre-registered threshold by >=1.5x"),
        plt.Rectangle((0, 0), 1, 1, color="#f1c40f", label="passes threshold"),
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="fails threshold"),
        plt.Line2D([0], [0], color="black", lw=1.0, linestyle="--",
                   label="pre-registered threshold"),
        plt.Line2D([0], [0], color="black", lw=0.8,
                   label="individual seed value"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    fig.suptitle("v15 cross-seed (n=3) reproducibility on pre-registered tests",
                 fontsize=12.5, fontweight="bold", y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
