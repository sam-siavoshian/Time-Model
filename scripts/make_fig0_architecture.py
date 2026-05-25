"""Figure 0: Chronometric Injection architecture schematic.

A frozen Qwen 2.5 3B with per-layer AdaLN-Zero FiLM injection of a 27-dim
sinusoidal+log encoding of real elapsed seconds tau. ~36 M trainable
parameters (LoRA on attention + lm_head plus per-layer FiLM projectors and
per-layer alpha scalars). Base weights are frozen.

Writes figures/fig0_architecture.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


DST = Path("figures/fig0_architecture.png")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    frozen = "#2c3e50"
    trainable = "#c0392b"
    chrono = "#16a085"
    bg = "#ecf0f1"

    # tau input
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 5.5), 1.8, 0.8,
        boxstyle="round,pad=0.05", facecolor=chrono, edgecolor="black", lw=1.2,
    ))
    ax.text(1.1, 5.9, "tau  (real seconds)", color="white", ha="center",
            va="center", fontsize=10, fontweight="bold")
    ax.text(1.1, 5.65, "wall clock", color="white", ha="center", va="center",
            fontsize=8, style="italic")

    # chrono encoder
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 3.9), 1.8, 1.2,
        boxstyle="round,pad=0.05", facecolor=chrono, edgecolor="black", lw=1.2,
    ))
    ax.text(1.1, 4.7, "chrono", color="white", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(1.1, 4.4, "encoder", color="white", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(1.1, 4.05, "27-dim\n(13 sin + 13 cos + log)",
            color="white", ha="center", va="center", fontsize=7.5)

    # arrow tau -> chrono
    ax.annotate("", xy=(1.1, 5.15), xytext=(1.1, 5.45),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

    # chi_t output label
    ax.annotate("chi_t", xy=(2.0, 4.5), xytext=(2.2, 4.5),
                fontsize=9, color=chrono, fontweight="bold", va="center")

    # token input
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 2.0), 1.8, 0.8,
        boxstyle="round,pad=0.05", facecolor="#f39c12", edgecolor="black", lw=1.2,
    ))
    ax.text(1.1, 2.4, "prompt tokens", color="white", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(1.1, 2.15, "(BPE ids)", color="white", ha="center", va="center",
            fontsize=8, style="italic")

    # embedding (frozen base)
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.0, 2.0), 1.8, 0.8,
        boxstyle="round,pad=0.05", facecolor=frozen, edgecolor="black", lw=1.2,
    ))
    ax.text(3.9, 2.4, "embedding", color="white", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(3.9, 2.15, "(frozen)", color="white", ha="center", va="center",
            fontsize=8, style="italic")

    # arrows prompt -> embedding -> first layer
    ax.annotate("", xy=(3.0, 2.4), xytext=(2.0, 2.4),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

    # decoder layers (36 layers shown as 3 stacked boxes with ellipsis)
    layer_x = 5.5
    layer_w = 4.2
    for i, (y, label) in enumerate([(0.5, "L35"), (1.5, "..."), (2.6, "L1"), (3.5, "L0")]):
        ax.add_patch(mpatches.FancyBboxPatch(
            (layer_x, y), layer_w, 0.7,
            boxstyle="round,pad=0.04", facecolor=frozen, edgecolor="black", lw=1.0,
        ))
        ax.text(layer_x + 0.15, y + 0.35, label, color="white", ha="left",
                va="center", fontsize=9, fontweight="bold")
        if label not in ("...",):
            ax.text(layer_x + layer_w/2 + 0.1, y + 0.5, "frozen attn + MLP", color="white",
                    ha="center", va="center", fontsize=8)
            # LoRA badge
            ax.add_patch(mpatches.FancyBboxPatch(
                (layer_x + layer_w - 1.0, y + 0.08), 0.85, 0.25,
                boxstyle="round,pad=0.02", facecolor=trainable, edgecolor="black", lw=0.7,
            ))
            ax.text(layer_x + layer_w - 0.58, y + 0.20, "+LoRA r=8", color="white",
                    ha="center", va="center", fontsize=6.5, fontweight="bold")
            # FiLM gate badge on the LEFT of layer
            ax.add_patch(mpatches.FancyBboxPatch(
                (layer_x - 1.3, y + 0.15), 1.15, 0.4,
                boxstyle="round,pad=0.02", facecolor=trainable, edgecolor="black", lw=0.8,
            ))
            ax.text(layer_x - 0.72, y + 0.35, f"FiLM alpha_{label[1:]}", color="white",
                    ha="center", va="center", fontsize=7, fontweight="bold")
            # chrono feed arrow
            ax.annotate("", xy=(layer_x - 1.3, y + 0.35), xytext=(2.1, 4.5),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color=chrono,
                                        connectionstyle=f"arc3,rad={-0.18 - 0.05*i}",
                                        alpha=0.7))

    # arrow embedding -> L0
    ax.annotate("", xy=(layer_x, 3.85), xytext=(4.8, 2.4),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

    # lm_head
    ax.add_patch(mpatches.FancyBboxPatch(
        (layer_x + 1.2, 5.5), 1.8, 0.8,
        boxstyle="round,pad=0.05", facecolor=frozen, edgecolor="black", lw=1.2,
    ))
    ax.text(layer_x + 2.1, 5.9, "lm_head", color="white", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(layer_x + 2.1, 5.65, "(frozen, +LoRA r=8)", color="white", ha="center",
            va="center", fontsize=7.5, style="italic")
    # arrow L0 -> lm_head
    ax.annotate("", xy=(layer_x + 2.1, 5.5), xytext=(layer_x + 2.1, 4.25),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

    # output
    ax.add_patch(mpatches.FancyBboxPatch(
        (layer_x + 1.2, 4.0), 1.8, 0.8,
        boxstyle="round,pad=0.05", facecolor="#f39c12", edgecolor="black", lw=1.2,
    ))
    ax.text(layer_x + 2.1, 4.4, "next-token", color="white", ha="center",
            va="center", fontsize=10, fontweight="bold")
    ax.text(layer_x + 2.1, 4.15, "(tau-conditional)", color="white", ha="center",
            va="center", fontsize=8, style="italic")

    # legend
    legend = [
        mpatches.Patch(color=frozen, label="frozen (no gradient)"),
        mpatches.Patch(color=trainable, label="trainable (~36 M params)"),
        mpatches.Patch(color=chrono, label="chronometric channel"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, frameon=False,
              bbox_to_anchor=(0.99, 0.01))

    # equation inset
    eq = r"$h' = h + \alpha_\ell \cdot (\gamma_\ell(\chi)\,h + \beta_\ell(\chi))$"
    init = r"AdaLN-Zero init: $\alpha_\ell = 0,\ \gamma_\ell = 1,\ \beta_\ell = 0$"
    ax.text(0.2, 0.6, eq, fontsize=11, color=trainable, fontweight="bold")
    ax.text(0.2, 0.20, init, fontsize=9, color="black", style="italic")

    ax.set_title(
        "Chronometric Injection: per-layer AdaLN-Zero FiLM of real elapsed time on a frozen LLM",
        fontsize=12, pad=10
    )

    fig.tight_layout()
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
