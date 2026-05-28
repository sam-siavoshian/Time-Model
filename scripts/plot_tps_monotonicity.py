"""Plot fig_tps_monotonicity.png from reports/tps/*.json.

For each adapter, plot P(REFRESH) (over hidden_only items) as a function
of log_10(tau). One line per adapter. CI should be monotonic; vanilla
should be flat.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"


def main() -> int:
    in_glob = sys.argv[1] if len(sys.argv) > 1 else "reports/tps/*.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "paper/figures/fig_tps_monotonicity.png"

    files = sorted(glob.glob(in_glob))
    files = [
        f for f in files
        if os.path.basename(f) not in {"headline.json", "baselines.json"}
        and not os.path.basename(f).startswith("SMOKE_")
    ]

    # Display order + colors.
    order = ["vanilla", "prompt", "chrono_only_s0", "ci_v15s_s0", "ci_v15s_s1", "ci_v15s_s2"]
    color = {
        "vanilla": "#999999",
        "prompt": "#4477AA",
        "chrono_only_s0": "#EE6677",
        "ci_v15s_s0": "#228833",
        "ci_v15s_s1": "#66CC99",
        "ci_v15s_s2": "#AAEE99",
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        if tag not in order:
            continue
        data = json.load(open(f))
        # Group by tau_ci_s within hidden_only items.
        by_tau: dict[int, list[bool]] = defaultdict(list)
        for r in data["results"]:
            if r["condition"] != "hidden_only":
                continue
            if r["tau_ci_s"] is None or r["tau_ci_s"] <= 0:
                continue
            by_tau[int(r["tau_ci_s"])].append(r["action"] == "REFRESH")
        xs = sorted(by_tau.keys())
        ys = [sum(by_tau[x]) / len(by_tau[x]) for x in xs]
        ax.plot(
            [math.log10(x) for x in xs], ys,
            marker="o", color=color.get(tag, "#000000"), label=tag.replace("_", "-"),
        )

    ax.set_xlabel(r"$\log_{10}\tau$ (seconds)")
    ax.set_ylabel(r"$P(\mathrm{REFRESH})$ on hidden-only items")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", frameon=True)
    ax.set_title("Temporal Policy Switching: refresh rate vs hidden elapsed time")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
