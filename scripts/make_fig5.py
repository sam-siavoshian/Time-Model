"""Fig 5: per-version × test heatmap.

Cells: pass = green, fail = red. Cell text = numeric value.
Black border on best-in-column. Tells cross-version story at a glance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def _load(name):
    p = REP / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    # Collect version -> summary
    versions = {
        "v11 (3B)": _load("qwen_time_v10_20260516_032348_recall.json"),
        "v12": _load("qwen_time_v12_20260523_004930_recall.json"),
        "v13": _load("qwen_time_v13_20260523_032242_recall.json"),
        "v14": _load("qwen_time_v14_20260523_033916_recall.json"),
        "7B": _load("scale_7b_20260523_010310_recall.json"),
    }

    # Try v15 single-anchor if it exists
    for f in sorted(REP.glob("qwen_time_v15_*_recall.json")):
        versions["v15 (single seed)"] = json.load(open(f))
        break
    # v15 cross-seed: synthesize a virtual row from the aggregate
    agg_file = REP / "v15_cross_seed_aggregate.json"
    if agg_file.exists():
        with open(agg_file) as af:
            agg = json.load(af)
        a = agg.get("aggregate", {})
        def _m(k):
            v = a.get(k, {})
            return v.get("mean", float("nan"))
        synth = {"summary": {
            "T1_clock_pearson_r": _m("T1_clock_pearson_r"),
            "T1b_ood_pearson_r": _m("T1b_ood_pearson_r"),
            "T1b_ood_log_mae": _m("T1b_ood_log_mae"),
            "T2_ack_delta": _m("T2_ack_delta"),
            "T3_weekend_signal": _m("T3_weekend_signal"),
            "T3_weekday_signal": _m("T3_weekday_signal"),
            "T4_mean_pairwise_kl": _m("T4_mean_pairwise_kl"),
        }}
        versions["v15 cross-seed (n=3)"] = synth

    tests = ["T1", "T1b r", "T1b mae", "T2", "T3", "T4"]
    thresholds = {"T1": 0.8, "T1b r": 0.7, "T1b mae": 0.5,
                  "T2": 0.5, "T3": 0.3, "T4": 0.05}
    # T1b mae passes if BELOW threshold
    inverted = {"T1b mae"}

    rows = []
    labels = []
    pass_mat = []
    for vname, data in versions.items():
        if data is None:
            continue
        s = data.get("summary", {})
        t1 = s.get("T1_clock_pearson_r", float('nan'))
        t1b_r = s.get("T1b_ood_pearson_r", float('nan'))
        t1b_mae = s.get("T1b_ood_log_mae", float('nan'))
        t2 = s.get("T2_ack_delta", float('nan'))
        t3 = max(s.get("T3_weekday_signal", 0), s.get("T3_weekend_signal", 0))
        t4 = s.get("T4_mean_pairwise_kl", float('nan'))
        row = [t1, t1b_r, t1b_mae, t2, t3, t4]
        rows.append(row)
        labels.append(vname)
        pass_row = []
        for ti, val in enumerate(row):
            name = tests[ti]
            thr = thresholds[name]
            if val != val:
                pass_row.append(0)
            elif name in inverted:
                pass_row.append(1 if val < thr else 0)
            else:
                pass_row.append(1 if val >= thr else 0)
        pass_mat.append(pass_row)

    if not rows:
        print("No data found, exiting")
        return

    rows = np.array(rows)
    pass_mat = np.array(pass_mat)

    # Find best per column (ignoring NaN)
    best_idx = []
    for ci in range(rows.shape[1]):
        col = rows[:, ci]
        name = tests[ci]
        if np.all(np.isnan(col)):
            best_idx.append(-1)
            continue
        if name in inverted:
            bi = int(np.nanargmin(col))
        else:
            bi = int(np.nanargmax(col))
        best_idx.append(bi)

    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.7 * len(labels)))
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(tests) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_aspect("equal")

    for ri in range(len(labels)):
        for ci in range(len(tests)):
            val = rows[ri, ci]
            if val != val:
                color = "#bbbbbb"
                text = "—"
            elif pass_mat[ri, ci]:
                color = "#7CC79C"
                text = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
            else:
                color = "#E89090"
                text = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
            ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                       facecolor=color, edgecolor="white"))
            # Black border on best-in-column
            if best_idx[ci] == ri:
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                           facecolor="none",
                                           edgecolor="black", linewidth=3))
            ax.text(ci, ri, text, ha="center", va="center",
                    fontsize=11, fontweight="bold")

    ax.set_title("Fig 5. Per-version performance on five pre-registered tests.\n"
                 "Green = PASS, red = FAIL, black border = best-in-column.",
                 fontsize=11)
    pass_patch = mpatches.Patch(color="#7CC79C", label="PASS (meets pre-reg threshold)")
    fail_patch = mpatches.Patch(color="#E89090", label="FAIL")
    ax.legend(handles=[pass_patch, fail_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    out = FIG / "fig5_per_version_tests.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")
    print()
    print("Versions shown:", labels)
    print("Best per column:", [labels[i] if i >= 0 else "—" for i in best_idx])


if __name__ == "__main__":
    main()
