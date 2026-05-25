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

DIGIT_CHARS = set("0123456789")
UNIT_HINT = ("second", "minute", "hour", "day", "week")


def classify(tok: str) -> str:
    s = tok.strip()
    if s and all(c in DIGIT_CHARS for c in s):
        return "digit"            # pure digit token
    if not s:
        return "number-prefix"    # the bare space that precedes the number
    if any(h in tok.lower() for h in UNIT_HINT):
        return "unit"
    return "scaffolding"


COLOR = {
    "digit": "#c0392b",
    "number-prefix": "#e74c3c",
    "unit": "#e67e22",
    "scaffolding": "#95a5a6",
}

LEGEND_LABEL = {
    "digit": "digit token (1, 4, ...)",
    "number-prefix": "space-before-number",
    "unit": "time-unit token (seconds, hours, ...)",
    "scaffolding": "scaffolding token (It, has, ., ...)",
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
        plt.Rectangle((0, 0), 1, 1, color=COLOR[k], label=LEGEND_LABEL[k])
        for k in ("digit", "number-prefix", "unit", "scaffolding")
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=9.5)

    fig.suptitle("", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    DST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DST, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
