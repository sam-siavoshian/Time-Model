"""Figure 0: Chronometric Injection architecture schematic. v2 layout.

Bottom-to-top flow with the chrono channel as a single bus on the left.
Layers shown as a compact vertical stack with one FiLM badge per layer.
No overlapping bounding boxes, legend in its own row.

Writes figures/fig0_architecture.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


DST = Path("figures/fig0_architecture.png")

FROZEN = "#2c3e50"
TRAINABLE = "#c0392b"
CHRONO = "#16a085"
FLOW = "#34495e"


def box(ax, x, y, w, h, text, color, *, subtext=None, fontsize=10, italic_sub=True):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=color, edgecolor="black", lw=1.0,
    ))
    if subtext is None:
        ax.text(x + w/2, y + h/2, text, color="white", ha="center",
                va="center", fontsize=fontsize, fontweight="bold")
    else:
        ax.text(x + w/2, y + h*0.66, text, color="white", ha="center",
                va="center", fontsize=fontsize, fontweight="bold")
        style = "italic" if italic_sub else "normal"
        ax.text(x + w/2, y + h*0.28, subtext, color="white", ha="center",
                va="center", fontsize=fontsize - 2.5, style=style)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.4, alpha=1.0, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.22,head_length=0.35",
                    lw=lw, color=color, alpha=alpha,
                    connectionstyle=f"arc3,rad={rad}",
                ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.set_title(
        "Chronometric Injection: per-layer AdaLN-Zero FiLM of real elapsed time on a frozen LLM",
        fontsize=12.5, pad=12, fontweight="bold",
    )

    # --- LEFT COLUMN: chronometric channel (green) ---
    box(ax, 0.4, 5.2, 2.2, 0.9,
        "tau  (real seconds)", CHRONO,
        subtext="wall clock since session start", fontsize=10)
    box(ax, 0.4, 3.5, 2.2, 1.4,
        "chrono encoder", CHRONO,
        subtext="27-dim: 13 sin + 13 cos + log(1+tau)",
        fontsize=11, italic_sub=False)
    arrow(ax, 1.5, 5.18, 1.5, 4.93, color=CHRONO, lw=1.6)

    # chi_t label
    ax.text(1.5, 3.30, r"$\chi_t \in \mathbb{R}^{27}$",
            color=CHRONO, ha="center", va="top", fontsize=11, fontweight="bold")

    # chi bus vertical down to layer column
    arrow(ax, 1.5, 3.05, 1.5, 1.05, color=CHRONO, lw=2.2, alpha=0.55)
    ax.text(1.5, 2.05, "chi bus", color=CHRONO, ha="center",
            va="center", fontsize=8, style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

    # --- BOTTOM: prompt input ---
    box(ax, 3.6, 0.3, 2.4, 0.9,
        "prompt tokens", "#f39c12",
        subtext="(BPE ids)", fontsize=10)

    # embedding
    box(ax, 6.5, 0.3, 2.0, 0.9,
        "embedding", FROZEN,
        subtext="frozen", fontsize=10)
    arrow(ax, 6.0, 0.75, 6.5, 0.75, color=FLOW)

    # --- CENTER COLUMN: layer stack with FiLM gates on left ---
    layer_x = 6.5
    layer_w = 3.6
    layers = [("L0", 1.55), ("L1", 2.40), ("...", 3.25), ("L34", 4.10),
              ("L35", 4.95)]
    for label, y in layers:
        if label == "...":
            ax.text(layer_x + layer_w/2, y + 0.15, "...", color=FROZEN,
                    ha="center", va="center", fontsize=14, fontweight="bold")
            continue
        # layer body
        ax.add_patch(mpatches.FancyBboxPatch(
            (layer_x, y), layer_w, 0.7,
            boxstyle="round,pad=0.04,rounding_size=0.09",
            facecolor=FROZEN, edgecolor="black", lw=0.9,
        ))
        ax.text(layer_x + 0.18, y + 0.35, label, color="white",
                ha="left", va="center", fontsize=9.5, fontweight="bold")
        ax.text(layer_x + layer_w/2 + 0.2, y + 0.50, "frozen attn + MLP",
                color="white", ha="center", va="center", fontsize=8)
        ax.text(layer_x + layer_w/2 + 0.2, y + 0.18, "+ LoRA rank 8 (trainable)",
                color="#f1c40f", ha="center", va="center", fontsize=7.2,
                style="italic")
        # FiLM badge on the left of the layer
        bx, by, bw, bh = layer_x - 1.85, y + 0.10, 1.65, 0.50
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=TRAINABLE, edgecolor="black", lw=0.8,
        ))
        ax.text(bx + bw/2, by + bh*0.60,
                f"FiLM({label})", color="white",
                ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(bx + bw/2, by + bh*0.20,
                r"$\gamma_\ell,\ \beta_\ell,\ \alpha_\ell$",
                color="white", ha="center", va="center", fontsize=7.5)
        # connector from chi bus into FiLM gate (short horizontal)
        arrow(ax, 1.5, by + bh/2, bx, by + bh/2,
              color=CHRONO, lw=0.9, alpha=0.55)
        # connector from FiLM gate into layer body
        arrow(ax, bx + bw, by + bh/2, layer_x, by + bh/2,
              color=TRAINABLE, lw=0.9)

    # arrow embedding -> L0
    arrow(ax, 7.5, 1.20, 7.5, 1.55, color=FLOW, lw=1.5)
    # arrow L0 -> L1 -> ... -> L35 (single curved bar on right edge)
    for i in range(4):
        y_from = 1.55 + 0.7 + i * 0.85
        y_to = y_from + 0.15
        arrow(ax, layer_x + layer_w, y_from - 0.05,
              layer_x + layer_w, y_to + 0.1, color=FLOW, lw=1.2)
    # arrow L35 -> lm_head
    arrow(ax, 7.5, 5.65, 7.5, 6.05, color=FLOW, lw=1.5)

    # lm_head + next-token (top right)
    box(ax, 6.5, 6.05, 2.0, 0.85,
        "lm_head", FROZEN,
        subtext="frozen + LoRA rank 8", fontsize=10)
    box(ax, 9.4, 6.05, 2.3, 0.85,
        "next-token", "#f39c12",
        subtext="tau-conditional", fontsize=10)
    arrow(ax, 8.5, 6.48, 9.4, 6.48, color=FLOW, lw=1.5)

    # --- FAR RIGHT: equation + init (shifted further right so it doesn't
    # touch the layer column) ---
    eq_box_x, eq_box_y, eq_w, eq_h = 10.25, 2.2, 1.7, 3.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (eq_box_x, eq_box_y), eq_w, eq_h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor="#ecf0f1", edgecolor="black", lw=0.8,
    ))
    ax.text(eq_box_x + eq_w/2, eq_box_y + eq_h - 0.30,
            "per-layer rule", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(eq_box_x + eq_w/2, eq_box_y + eq_h - 0.95,
            r"$h' = h + \alpha_\ell\,(\gamma_\ell(\chi_t)\cdot h + \beta_\ell(\chi_t))$",
            ha="center", va="center", fontsize=10, color=TRAINABLE)
    ax.text(eq_box_x + eq_w/2, eq_box_y + eq_h - 1.55,
            "AdaLN-Zero init", ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    ax.text(eq_box_x + eq_w/2, eq_box_y + eq_h - 2.05,
            r"$\alpha_\ell = 0,\ \gamma_\ell = 1,\ \beta_\ell = 0$",
            ha="center", va="center", fontsize=9.5, color=TRAINABLE)
    ax.text(eq_box_x + eq_w/2, eq_box_y + 0.30,
            "step 0 = identity\nalpha gradient = h",
            ha="center", va="center", fontsize=8, color="black",
            style="italic")

    # --- LEGEND row at very bottom, outside content area ---
    legend = [
        mpatches.Patch(color=FROZEN, label="frozen (no gradient)"),
        mpatches.Patch(color=TRAINABLE, label="trainable (~36 M params: FiLM + LoRA)"),
        mpatches.Patch(color=CHRONO, label="chronometric channel (encoder + chi bus)"),
        mpatches.Patch(color="#f39c12", label="external IO (tokens)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=4, frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
