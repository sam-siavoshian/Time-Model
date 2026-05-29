"""Figure 6: per-layer |alpha| magnitude with dominant subset highlighted.

Reads reports/alpha_norms_v15s_seed0.json (output of qwen_time_alpha_norms.py).
Writes figures/fig6_alpha_norm_per_layer.png.

Used by historical drafts to visualize that mid-deep layers L19-L28 dominate
the chrono signal and that top-8 inversion (vs bottom-8) is what collapses
the alpha-flip correlation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SRC = Path("reports/alpha_norms_v15s_seed0.json")
DST = Path("figures/fig6_alpha_norm_per_layer.png")
TOP_K = 8


def main() -> None:
    payload = json.loads(SRC.read_text())
    raw = payload["per_layer_mean_abs_alpha"]
    layers = sorted(int(k) for k in raw)
    values = np.array([raw[str(li)] for li in layers], dtype=float)

    order = np.argsort(values)[::-1]
    top_set = set(int(layers[i]) for i in order[:TOP_K])
    bot_set = set(int(layers[i]) for i in order[-TOP_K:])

    colors = []
    for li in layers:
        if li in top_set:
            colors.append("#c0392b")
        elif li in bot_set:
            colors.append("#7f8c8d")
        else:
            colors.append("#2c3e50")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(layers, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"mean $|\alpha|$ (FiLM gate magnitude)")
    ax.set_title(
        r"Per-layer chrono dominance (v15 seed 0). Red = top-8, grey = bottom-8."
    )
    ax.set_xticks(layers[::2])
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label=f"top-{TOP_K} dominant"),
        plt.Rectangle((0, 0), 1, 1, color="#2c3e50", label="middle"),
        plt.Rectangle((0, 0), 1, 1, color="#7f8c8d", label=f"bottom-{TOP_K}"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=180, bbox_inches="tight")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
