"""Emit paper/_6g_tps.tex from reports/tps/headline.json + reports/tps/baselines.json.

Honest report: TPS is a falsification test. CI v15s does NOT pass the
hidden-only policy task zero-shot. chrono_only shows a small positive
signal but not enough to claim downstream behavioral usefulness.
"""

from __future__ import annotations

import json
import os
import sys


def _safe(x, fmt="{:.3f}"):
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and (x != x):
            return "--"
        return fmt.format(x)
    except Exception:                                          # noqa: BLE001
        return "--"


def main() -> int:
    headline_path = sys.argv[1] if len(sys.argv) > 1 else "reports/tps/headline.json"
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else "reports/tps/baselines.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "paper/_6g_tps.tex"

    headline = json.load(open(headline_path))
    baselines = json.load(open(baseline_path))
    per = headline["per_adapter"]
    ci_agg = headline.get("ci_v15s_crossseed") or {}

    def row(tag):
        return per.get(tag, {}).get("metrics", {})

    def hidden_only(tag):
        return row(tag).get("by_condition", {}).get("hidden_only", {}).get("policy_acc")

    def prompt_only(tag):
        return row(tag).get("by_condition", {}).get("prompt_only", {}).get("policy_acc")

    def both_agree(tag):
        return row(tag).get("by_condition", {}).get("both_agree", {}).get("policy_acc")

    def held_out_template(tag):
        return row(tag).get("held_out_template_policy_acc")

    def held_out_family(tag):
        return row(tag).get("held_out_family_policy_acc")

    def monotonicity(tag):
        return row(tag).get("monotonicity", {}).get("r_log_tau_vs_p_refresh")

    def scalar_follow(tag):
        return row(tag).get("conflict", {}).get("scalar_follow_rate")

    def prompt_follow(tag):
        return row(tag).get("conflict", {}).get("prompt_follow_rate")

    ho_van = hidden_only("vanilla")
    ho_ci = hidden_only("ci_v15s_s0")
    ho_chr = hidden_only("chrono_only_s0")
    cross_mean = ci_agg.get("hidden_only_mean")
    cross_mono = ci_agg.get("monotonicity_mean")
    cross_std = ci_agg.get("std")
    chi_only_in = baselines["chi_tau_only"]["held_in_template_acc"]
    chi_only_ho_t = baselines["chi_tau_only"]["held_out_template_acc"]
    chi_only_ho_f = baselines["chi_tau_only"]["held_out_family_acc"]
    chi_fam_in = baselines["chi_tau_plus_family"]["held_in_template_acc"]
    chi_fam_ho_t = baselines["chi_tau_plus_family"]["held_out_template_acc"]
    chi_fam_ho_f = baselines["chi_tau_plus_family"]["held_out_family_acc"]

    lines = []
    a = lines.append

    a(r"\subsection{Temporal Policy Switching: an identical-prompt falsification test}")
    a("")
    a(
        r"TPDR measures response shape, not decision quality. As a stronger behavioral test, "
        r"we introduce the \emph{Temporal Policy Switching (TPS)} benchmark, in which the "
        r"visible prompt is held byte-identical and only the hidden $\Tau$ scalar varies. "
        r"Each item asks an agent to choose exactly one of four labelled actions "
        r"(REUSE, REFRESH, ASK, SUMMARIZE). Six policy families with different freshness "
        r"thresholds (cache reuse, session continuation, API freshness, safety advice, "
        r"calendar planning, market data) prevent a single global rule from solving the "
        r"benchmark. Templates 8--11 of each family form a held-out prompt split; the entire "
        r"\texttt{market\_data} family is fully held out for compositional generalization. "
        r"Items are evaluated at nine $\Tau$ values from $10\,\mathrm{s}$ to $7\,\mathrm{d}$. "
        r"The principal test is the \emph{hidden-only} condition: no time text appears in the "
        r"prompt, so any above-vanilla accuracy must come from the hidden channel."
    )
    a("")
    a(
        r"We pre-specified three hypotheses. \textbf{H1}: under hidden-only prompts, \CI{} "
        r"raises policy accuracy above vanilla. \textbf{H2}: refresh probability rises "
        r"monotonically with $\log\Tau$ under \CI{} on hidden-only items. \textbf{H3}: under "
        r"prompt/\CI{} conflict, \CI{} alters the channel-follow rate. All three hypotheses are "
        r"\emph{falsifiable}: if they fail, the behavioral claim narrows accordingly."
    )
    a("")
    a(
        rf"\textbf{{Baselines and ceilings.}} The benchmark is solvable from $(\Tau, "
        rf"\text{{family}})$ alone: a logistic regression on $\chi(\Tau)$ plus a family "
        rf"one-hot achieves {_safe(chi_fam_in)} on held-in templates and "
        rf"{_safe(chi_fam_ho_t)} on held-out templates, but drops to {_safe(chi_fam_ho_f)} on "
        rf"the held-out \texttt{{market\_data}} family. A $\chi(\Tau)$-only classifier (no "
        rf"family) achieves {_safe(chi_only_in)}, confirming that family identity carries "
        rf"most of the discriminative signal. The rule oracle attains $1.000$ by "
        rf"construction. These anchor the LLM results."
    )
    a("")
    a(
        rf"\textbf{{Result (H1: hidden-only).}} \emph{{H1 fails for \CI{{}} v15s and is only "
        rf"weakly supported by the chrono-only ablation.}} Vanilla Qwen 2.5 3B reaches "
        rf"{_safe(ho_van)} on hidden-only items. \CI{{}} v15s seed~0 reaches {_safe(ho_ci)} "
        rf"(\emph{{below}} vanilla); across three seeds the cross-seed mean is "
        rf"${_safe(cross_mean)}\pm{_safe(cross_std)}$, indistinguishable from vanilla. "
        rf"The chrono-only ablation (chrono channel trained, LoRA frozen at zero) reaches "
        rf"{_safe(ho_chr)}, a modest $+{_safe((ho_chr or 0) - (ho_van or 0))}$ over vanilla. "
        rf"This pattern suggests that the LoRA surface trained alongside the chrono channel "
        rf"during v15 training does not help, and may interfere, when the task is policy "
        rf"selection without any visible time string."
    )
    a("")
    a(
        rf"\textbf{{Result (H2: monotonicity).}} \emph{{H2 fails: refresh probability does not "
        rf"rise with $\log\Tau$ under \CI{{}} v15s.}} The Pearson correlation between $\log\Tau$ "
        rf"and $P(\text{{REFRESH}})$ on hidden-only items is "
        rf"$r={_safe(monotonicity('ci_v15s_s0'))}$ for seed 0, "
        rf"$r={_safe(monotonicity('ci_v15s_s1'))}$ for seed 1, "
        rf"$r={_safe(monotonicity('ci_v15s_s2'))}$ for seed 2 (cross-seed mean "
        rf"$r={_safe(cross_mono)}$), all negative. Vanilla is essentially flat "
        rf"($r={_safe(monotonicity('vanilla'))}$), confirming the absence of a time signal. "
        rf"The negative slope under \CI{{}} suggests the chrono channel, as trained on "
        rf"CLOCK + SILENT-GAP + PHASE, does not transfer to a cache/session/freshness policy "
        rf"axis without task-specific supervision."
    )
    a("")
    a(
        rf"\textbf{{Result (H3: conflict).}} \emph{{H3 also fails: \CI{{}} does not"
        rf" preferentially follow the hidden scalar.}} Under prompt/\CI{{}} conflict, "
        rf"\CI{{}} v15s seed~0 follows the hidden scalar at rate {_safe(scalar_follow('ci_v15s_s0'))} "
        rf"and the visible timestamp at {_safe(prompt_follow('ci_v15s_s0'))}; vanilla follows "
        rf"the visible timestamp at {_safe(prompt_follow('vanilla'))} (the only signal it has). "
        rf"\CI{{}} does not clearly elevate scalar-follow over vanilla in any seed."
    )
    a("")
    a(
        r"Tables~\ref{tab:tps-main} and \ref{tab:tps-conflict} report the full numbers and "
        r"Figure~\ref{fig:tps-mono} plots $P(\text{REFRESH})$ versus $\log_{10}\Tau$ for each "
        r"adapter."
    )
    a("")

    a(r"\begin{table}[t]")
    a(r"\centering")
    a(
        r"\caption{TPS hidden-only policy accuracy and monotonicity. "
        r"\emph{Hidden-only}: visible prompt has no time text; only the hidden $\Tau$ varies. "
        r"\emph{Both-agree}: time written in prompt AND supplied to \CI{}. "
        r"\emph{Held-out templates}: prompts 8--11 of each family (kept out of any in-paper "
        r"fitting). \emph{Held-out family}: \texttt{market\_data} held out from the "
        r"$\chi(\Tau)$ classifier training. $r$ is Pearson correlation between $\log\Tau$ and "
        r"$P(\text{REFRESH})$ on hidden-only items.}"
    )
    a(r"\label{tab:tps-main}")
    a(r"\footnotesize")
    a(r"\begin{tabular}{lccccc}")
    a(r"\toprule")
    a(r"Model & Hidden-only & Both-agree & H/O templ. & H/O fam. & $r(\log\Tau,P_R)$ \\")
    a(r"\midrule")
    for tag, name in [
        ("vanilla", "Vanilla"),
        ("prompt", "Prompt-timestamp"),
        ("chrono_only_s0", "chrono-only (s0)"),
        ("ci_v15s_s0", r"\CI{} v15s (s0)"),
        ("ci_v15s_s1", r"\CI{} v15s (s1)"),
        ("ci_v15s_s2", r"\CI{} v15s (s2)"),
    ]:
        a(
            rf"{name} & {_safe(hidden_only(tag))} & {_safe(both_agree(tag))} & "
            rf"{_safe(held_out_template(tag))} & {_safe(held_out_family(tag))} & "
            rf"{_safe(monotonicity(tag))} \\"
        )
    a(r"\midrule")
    a(rf"$\chi(\Tau)$ only (LR) & {_safe(chi_only_in)} & -- & {_safe(chi_only_ho_t)} & {_safe(chi_only_ho_f)} & -- \\")
    a(rf"$\chi(\Tau)$+family (LR) & {_safe(chi_fam_in)} & -- & {_safe(chi_fam_ho_t)} & {_safe(chi_fam_ho_f)} & -- \\")
    a(r"Rule oracle & 1.000 & 1.000 & 1.000 & 1.000 & -- \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")
    a("")

    a(r"\begin{figure}[t]")
    a(r"\centering")
    a(r"\includegraphics[width=0.95\columnwidth]{figures/fig_tps_monotonicity.png}")
    a(
        r"\caption{TPS hidden-only: $P(\text{REFRESH})$ as a function of $\log_{10}\Tau$. "
        r"Vanilla is flat (no signal). \CI{} v15s shows a slight negative slope (refresh "
        r"becomes less likely as $\Tau$ grows, opposite of expected). chrono-only is closer to "
        r"flat. None of the adapters tested produce the expected monotonic increase, so the "
        r"hidden channel does not transfer zero-shot to this policy axis.}"
    )
    a(r"\label{fig:tps-mono}")
    a(r"\end{figure}")
    a("")

    a(r"\begin{table}[t]")
    a(r"\centering")
    a(
        r"\caption{TPS conflict condition: visible prompt timestamp disagrees with hidden "
        r"\CI{} scalar. \emph{Scalar-follow}: $P(\text{action matches hidden gold})$. "
        r"\emph{Prompt-follow}: $P(\text{action matches visible gold})$. None of the \CI{} "
        r"adapters meaningfully elevate scalar-follow over vanilla.}"
    )
    a(r"\label{tab:tps-conflict}")
    a(r"\footnotesize")
    a(r"\begin{tabular}{lcc}")
    a(r"\toprule")
    a(r"Model & Scalar-follow & Prompt-follow \\")
    a(r"\midrule")
    for tag, name in [
        ("vanilla", "Vanilla"),
        ("prompt", "Prompt-timestamp"),
        ("chrono_only_s0", "chrono-only (s0)"),
        ("ci_v15s_s0", r"\CI{} v15s (s0)"),
        ("ci_v15s_s1", r"\CI{} v15s (s1)"),
        ("ci_v15s_s2", r"\CI{} v15s (s2)"),
    ]:
        a(rf"{name} & {_safe(scalar_follow(tag))} & {_safe(prompt_follow(tag))} \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")
    a("")

    a(
        r"\textbf{Interpretation.} The TPS hidden-only condition tests whether the residual"
        r" time channel changes downstream action selection under identical visible text."
        r" The cross-seed result is negative: \CI{} v15s does not outperform vanilla, and"
        r" refresh probability does not rise monotonically with $\Tau$. The chrono-only"
        r" ablation shows a small positive signal on the held-out family, suggesting that"
        r" some compositional generalization is possible without the LoRA surface, but the"
        r" effect size is too small to claim a behavioral contribution. This result is"
        r" consistent with the mechanistic finding that $\Tau$ becomes linearly recoverable"
        r" from hidden states without becoming a behavioral feature: the channel is"
        r" \emph{transmitted} but not, in the zero-shot setting tested here, \emph{used} by"
        r" the model's output policy. We report this negative result because it is the most"
        r" honest statement of the limits of zero-shot transfer for \CI{}."
    )
    a("")
    a(
        r"\textbf{What this rules out.} TPS rules out the claim that \CI{}, as trained on"
        r" CLOCK + SILENT-GAP + PHASE, yields a behavior-shaping policy channel that"
        r" generalizes zero-shot to cache/session/freshness decisions. It does \emph{not}"
        r" rule out the mechanistic transmission result (Sec.~VI-A), the causal interventions"
        r" (Sec.~VI-B), or the response-shape result on TPDR (Sec.~VI-F), all of which"
        r" remain intact. It also does \emph{not} rule out that a \CI{} model retrained on"
        r" policy-labelled data would pass TPS; that experiment is left to future work."
    )
    a("")
    a(
        r"\textbf{Limitations.} TPS uses templated synthetic prompts and rule-based oracle"
        r" labels: a natural-language production benchmark remains open. Soft-prompt and"
        r" matched-parameter scalar-embedding baselines are also future work. All TPS results"
        r" are zero-shot evaluations of checkpoints trained without policy-task supervision."
    )
    a("")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
