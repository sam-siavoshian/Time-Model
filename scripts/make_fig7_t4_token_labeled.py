"""Figure 7: T4 teacher-forced KL per output position, with token labels.

Reads reports/t4_labeled_v15s_seed0.json. Two side-by-side panels:
left = clock prompt (chrono signal should fire on number+unit tokens),
right = hello control prompt (chrono signal should be silent).

Used in PAPER.md Section 24.7.16 to visualize that the multi-position KL
pattern is mechanistically faithful — the chrono signal lands precisely
on the tokens that should depend on tau.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


SRC = Path("reports/t4_labeled_v15s_seed0.json")
DST = Path("figures/fig7_t4_token_labeled.png")

NUMBER_HINT = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
UNIT_HINT = ("second", "minute", "hour", "day", "week", "Welcome", "Hi")


def classify(tok: str) -> str:
    s = tok.strip()
    if not s:
        return "number"  # the bare space between scaffolding + number
    if any(ch in s for ch in NUMBER_HINT):
        return "number"
    if any(h in tok for h in UNIT_HINT):
        return "unit"
    return "scaffolding"


COLOR = {
    "number": "#c0392b",
    "unit": "#e67e22",
    "scaffolding": "#95a5a6",
}


def main() -> None:
    payload = json.loads(SRC.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, (prompt_name, prompt_label) in zip(
        axes, [("clock", "Clock prompt: \"How long has it been since we started?\""),
               ("hello", "Control prompt: \"Hello.\"")]
    ):
        r = payload[prompt_name]
        tokens = r["tokens"]
        kls = r["kls_per_pos"]
        n = len(tokens)
        colors = [COLOR[classify(t)] for t in tokens]

        bars = ax.bar(range(n), kls, color=colors, edgecolor="white",
                      linewidth=0.6)

        # value labels above each non-trivial bar
        for i, (k, t) in enumerate(zip(kls, tokens)):
            if k > 0.5:
                ax.text(i, k + 0.6, f"{k:.1f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        # token labels under x axis
        ax.set_xticks(range(n))
        clean = [repr(t)[1:-1][:8] for t in tokens]  # strip quotes, trim long
        ax.set_xticklabels(clean, rotation=35, ha="right", fontsize=8)

        ax.set_title(prompt_label, fontsize=11)
        ax.set_xlabel("decoded output position", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 26)

    axes[0].set_ylabel(r"teacher-forced KL across $\tau \in \{15\mathrm{s},\,1\mathrm{h},\,1\mathrm{d}\}$",
                       fontsize=10)

    # shared legend
    legend = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR["number"], label="number / digit token"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR["unit"], label="time-unit token"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR["scaffolding"], label="scaffolding token"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=10)

    fig.suptitle("", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
