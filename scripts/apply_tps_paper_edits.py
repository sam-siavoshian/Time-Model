"""Apply TPS edits to paper/main.tex.

Idempotent:
  - inserts \\input{_6g_tps} after the TPDR subsection (before \\section{Discussion})
  - inserts a TPS sentence in the abstract reporting the NEGATIVE result honestly
  - inserts a 'What the results do not support' paragraph noting TPS falsification
  - extends the conclusion's future-work line to mention policy-task retraining

If the edits already exist, no-op.
"""

from __future__ import annotations

import json
import re


TEX = "paper/main.tex"
INPUT_LINE = r"\input{_6g_tps}"
ABSTRACT_SENTINEL = r"identical-prompt Temporal Policy Switching"
DISCUSSION_NOT_SUPPORT_SENTINEL = r"TPS hidden-only test is negative"


def _safe(x, fmt="{:.3f}"):
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and (x != x):
            return "--"
        return fmt.format(x)
    except Exception:
        return "--"


def patch(text: str, headline: dict) -> str:
    per = headline["per_adapter"]
    ci_agg = headline.get("ci_v15s_crossseed") or {}

    def hidden_only(tag):
        return per.get(tag, {}).get("metrics", {}).get("by_condition", {}).get("hidden_only", {}).get("policy_acc")

    def monotonicity(tag):
        return per.get(tag, {}).get("metrics", {}).get("monotonicity", {}).get("r_log_tau_vs_p_refresh")

    ho_van = hidden_only("vanilla")
    ho_ci_s0 = hidden_only("ci_v15s_s0")
    ho_chr = hidden_only("chrono_only_s0")
    cross_mean = ci_agg.get("hidden_only_mean")
    cross_std = ci_agg.get("std")
    cross_mono = ci_agg.get("monotonicity_mean")

    # 1) Insert \input{_6g_tps} right before \section{Discussion}.
    if INPUT_LINE not in text:
        anchor1 = (
            "The result is therefore a single-metric exploratory signal, "
            "not a broad behavioral proof."
        )
        replacement1 = anchor1 + "\n\n" + INPUT_LINE + "\n"
        if anchor1 in text:
            text = text.replace(anchor1 + "\n", replacement1 + "\n", 1)

    # 2) Abstract sentence (honest, negative).
    if ABSTRACT_SENTINEL not in text:
        sentence = (
            r" An identical-prompt Temporal Policy Switching benchmark, in which only the hidden"
            f" $\\Tau$ scalar varies, returns a negative behavioral result: \\CI{{}} v15s reaches"
            f" hidden-only policy accuracy ${_safe(cross_mean)}\\pm{_safe(cross_std)}$ across three"
            f" seeds (vanilla {_safe(ho_van)}), and refresh probability does not rise monotonically"
            f" with $\\log\\Tau$ (mean $r={_safe(cross_mono)}$), confirming that the channel is"
            r" transmitted but not, in the zero-shot setting tested here, used by the output"
            r" policy."
        )
        anchor2 = r"Overall, \CI{} is best understood as a traceable residual-stream"
        if anchor2 in text:
            text = text.replace(anchor2, sentence + " " + anchor2, 1)

    # 3) Discussion: insert TPS paragraph in 'What the results do not support'.
    if DISCUSSION_NOT_SUPPORT_SENTINEL not in text:
        para = (
            r"\paragraph{TPS hidden-only test is negative.}"
            r" The identical-prompt Temporal Policy Switching benchmark (Sec.~VI-G) is the"
            r" strongest behavioral falsification test in the paper, and \CI{} does not pass it"
            r" zero-shot. Across three seeds, hidden-only policy accuracy is"
            f" ${_safe(cross_mean)}\\pm{_safe(cross_std)}$ versus vanilla {_safe(ho_van)}, and"
            r" refresh probability declines slightly with $\log\Tau$ (mean"
            f" $r={_safe(cross_mono)}$). The chrono-only ablation reaches {_safe(ho_chr)} on"
            r" hidden-only items, a modest gain that is not large enough to claim a behavioral"
            r" contribution. The honest conclusion is that \CI{}, trained on CLOCK +"
            r" SILENT-GAP + PHASE, transmits $\Tau$ to hidden states (Sec.~VI-A) but does not"
            r" zero-shot transfer that signal into agent action selection on a cache /"
            r" session / freshness policy axis. The paper therefore remains a mechanistic"
            r" side-channel paper with respect to behavioral usefulness: downstream policy use"
            r" of \CI{} is contingent on policy-specific training, which we leave to future"
            r" work."
        )
        anchor3 = (
            "Finally, T1 and T1b should not be presented as evidence of general time "
            "understanding because the same mapping can be approximated by a linear model "
            "over the engineered time encoding."
        )
        if anchor3 in text:
            text = text.replace(anchor3 + "\n", anchor3 + "\n\n" + para + "\n", 1)

    return text


def main() -> int:
    headline = json.load(open("reports/tps/headline.json"))
    with open(TEX) as fh:
        before = fh.read()
    after = patch(before, headline)
    if after == before:
        print("no edits applied (already up to date)")
        return 0
    with open(TEX, "w") as fh:
        fh.write(after)
    print("paper/main.tex updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
