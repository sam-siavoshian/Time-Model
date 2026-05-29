"""Emit paper/_6g_tps.tex from reports/tps/headline.json + reports/tps/baselines.json.

Honest report: TPS is a falsification test. CI v15s does NOT pass the
hidden-only policy task zero-shot. chrono_only shows a small positive
signal but not enough to claim downstream behavioral usefulness.

Note (post-review fix): the standalone PromptAdapter run was double-prefixed
(benchmark.py renders [elapsed: X] for prompt_only/conflict items AND
PromptAdapter prepended its own copy on every call), so it is omitted from
the paper tables. The clean prompt-text baseline is vanilla evaluated on
the prompt_only condition; vanilla's per-condition breakdown surfaces this
directly. The conflict scoring drops 108 fake conflicts in market_data
(threshold 10s equals the short-prompt tau, so gold_scalar == gold_prompt).
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

    def by_cond_acc(tag, cond):
        return row(tag).get("by_condition", {}).get(cond, {}).get("policy_acc")

    def hidden_only(tag):
        return by_cond_acc(tag, "hidden_only")

    def prompt_only(tag):
        return by_cond_acc(tag, "prompt_only")

    def both_agree(tag):
        return by_cond_acc(tag, "both_agree")

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

    def n_real(tag):
        return row(tag).get("conflict", {}).get("n_real")

    ho_van = hidden_only("vanilla")
    po_van = prompt_only("vanilla")
    ho_ci = hidden_only("ci_v15s_s0")
    ho_chr = hidden_only("chrono_only_s0")
    cross_mean = ci_agg.get("hidden_only_mean")
    cross_std = ci_agg.get("std")
    cross_mono = ci_agg.get("monotonicity_mean")

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
        r"prompt, so any above-vanilla accuracy must come from the hidden channel. The "
        r"\emph{prompt\_only} condition writes the time directly into the prompt (with the "
        r"chrono channel zeroed) and serves as the prompt-text baseline."
    )
    a("")
    a(
        r"We pre-specified three hypotheses. \textbf{H1}: under hidden-only prompts, \CI{} "
        r"raises policy accuracy above vanilla. \textbf{H2}: refresh probability rises "
        r"monotonically with $\log\Tau$ under \CI{} on hidden-only items. \textbf{H3}: under "
        r"prompt/\CI{} conflict, \CI{} alters the channel-follow rate. All three hypotheses "
        r"are \emph{falsifiable}: if they fail, the behavioral claim narrows accordingly."
    )
    a("")
    a(
        rf"\textbf{{Baselines and ceilings.}} The benchmark is solvable from $(\Tau, "
        rf"\text{{family}})$ alone: a logistic regression on $\chi(\Tau)$ plus a family "
        rf"one-hot achieves {_safe(chi_fam_in)} on held-in templates and "
        rf"{_safe(chi_fam_ho_t)} on held-out templates, but drops to {_safe(chi_fam_ho_f)} on "
        rf"the held-out \texttt{{market\_data}} family. A $\chi(\Tau)$-only classifier (no "
        rf"family) achieves {_safe(chi_only_in)}, confirming that family identity carries "
        rf"most of the discriminative signal. The rule oracle attains $1.000$ by construction. "
        rf"These anchor the LLM results."
    )
    a("")
    a(
        rf"\textbf{{Result (H1: hidden-only).}} \emph{{H1 fails for \CI{{}} v15s and is only "
        rf"weakly supported by the chrono-only ablation.}} Vanilla Qwen 2.5 3B reaches "
        rf"{_safe(ho_van)} on hidden-only items. \CI{{}} v15s seed~0 reaches {_safe(ho_ci)} "
        rf"(\emph{{below}} vanilla); across three seeds the cross-seed mean is "
        rf"${_safe(cross_mean)}\pm{_safe(cross_std)}$, indistinguishable from vanilla. The "
        rf"chrono-only ablation (chrono trained, LoRA frozen at zero) reaches {_safe(ho_chr)}, "
        rf"a modest $+{_safe((ho_chr or 0) - (ho_van or 0))}$ over vanilla. This pattern "
        rf"suggests that the LoRA surface trained alongside the chrono channel during v15 "
        rf"training does not help, and may interfere, when the task is policy selection "
        rf"without any visible time string."
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
        rf"The negative slope under \CI{{}} indicates the chrono channel as trained on "
        rf"CLOCK + SILENT-GAP + PHASE does not transfer to a cache/session/freshness policy "
        rf"axis without task-specific supervision."
    )
    a("")
    a(
        rf"\textbf{{Result (H3: conflict, inconclusive).}} The original conflict generation "
        rf"contained 108 \emph{{fake conflict}} items (in the \texttt{{market\_data}} family the "
        rf"threshold of $10\,\mathrm{{s}}$ equals the short-side prompt $\Tau$, so "
        rf"$\text{{gold}}_{{\text{{scalar}}}}{{=}}\text{{gold}}_{{\text{{prompt}}}}$). We exclude "
        rf"these from the conflict tables. The corrected conflict rates "
        rf"(Table~\ref{{tab:tps-conflict}}; $n={_safe(n_real('ci_v15s_s0'), '{:d}')}$ real "
        rf"conflicts per adapter) show vanilla following the visible prompt at "
        rf"{_safe(prompt_follow('vanilla'))} (the only signal it has) and \CI{{}} v15s seed~0 "
        rf"following the hidden scalar at {_safe(scalar_follow('ci_v15s_s0'))} versus following "
        rf"the prompt at {_safe(prompt_follow('ci_v15s_s0'))}. Cross-seed scalar-follow under "
        rf"\CI{{}} v15s does not exceed vanilla {_safe(scalar_follow('vanilla'))}, so H3 is best "
        rf"reported as inconclusive rather than passing."
    )
    a("")
    a(
        r"Tables~\ref{tab:tps-main} and \ref{tab:tps-conflict} report the full numbers and "
        r"Figure~\ref{fig:tps-mono} plots $P(\text{REFRESH})$ versus $\log_{10}\Tau$ per adapter."
    )
    a("")

    a(r"\begin{table}[t]")
    a(r"\centering")
    a(
        r"\caption{TPS policy accuracy. \emph{Hidden-only}: visible prompt has no time text; "
        r"only the hidden $\Tau$ varies. \emph{Prompt-only}: time text in prompt, chrono "
        r"channel zeroed (Qwen-with-time-in-prompt). \emph{H/O templ.}: prompts 8--11 of each "
        r"family. \emph{H/O fam.}: \texttt{market\_data} held out from the $\chi(\Tau)$ "
        r"classifier training. $r$ is Pearson correlation between $\log\Tau$ and "
        r"$P(\text{REFRESH})$ on hidden-only items.}"
    )
    a(r"\label{tab:tps-main}")
    a(r"\footnotesize")
    a(r"\begin{tabular}{lccccc}")
    a(r"\toprule")
    a(r"Model & Hidden-only & Prompt-only & H/O templ. & H/O fam. & $r(\log\Tau,P_R)$ \\")
    a(r"\midrule")
    for tag, name in [
        ("vanilla", "Vanilla"),
        ("chrono_only_s0", "chrono-only (s0)"),
        ("ci_v15s_s0", r"\CI{} v15s (s0)"),
        ("ci_v15s_s1", r"\CI{} v15s (s1)"),
        ("ci_v15s_s2", r"\CI{} v15s (s2)"),
    ]:
        a(
            rf"{name} & {_safe(hidden_only(tag))} & {_safe(prompt_only(tag))} & "
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
        rf"\caption{{TPS conflict condition (108 \emph{{fake conflict}} items excluded; $n="
        rf"{_safe(n_real('ci_v15s_s0'), '{:d}')}$ per adapter). \emph{{Scalar-follow}}: "
        rf"$P(\text{{action matches hidden gold}})$. \emph{{Prompt-follow}}: "
        rf"$P(\text{{action matches visible gold}})$. \CI{{}} v15s scalar-follow does not "
        rf"exceed vanilla's, so H3 is inconclusive.}}"
    )
    a(r"\label{tab:tps-conflict}")
    a(r"\footnotesize")
    a(r"\begin{tabular}{lcc}")
    a(r"\toprule")
    a(r"Model & Scalar-follow & Prompt-follow \\")
    a(r"\midrule")
    for tag, name in [
        ("vanilla", "Vanilla"),
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
        r"\textbf{Interpretation.} TPS hidden-only tests whether the residual time channel "
        r"changes downstream action selection under identical visible text. The cross-seed "
        r"result is negative: \CI{} v15s does not outperform vanilla, and refresh probability "
        r"does not rise monotonically with $\Tau$. The chrono-only ablation shows a small "
        r"positive signal on the held-out family, suggesting that some compositional "
        r"generalization is possible without the LoRA surface, but the effect size is too "
        r"small to claim a behavioral contribution. This result is consistent with the "
        r"mechanistic finding that $\Tau$ becomes linearly recoverable from hidden states "
        r"without becoming a behavioral feature: the channel is \emph{transmitted} but not, "
        r"in the zero-shot setting tested here, \emph{used} by the model's output policy. We "
        r"report this negative result because it is the most honest statement of the limits "
        r"of zero-shot transfer for \CI{}."
    )
    a("")
    a(
        r"\textbf{What this rules out.} TPS rules out the claim that \CI{}, as trained on "
        r"CLOCK + SILENT-GAP + PHASE, yields a behavior-shaping policy channel that "
        r"generalizes zero-shot to cache/session/freshness decisions. It does \emph{not} rule "
        r"out the mechanistic transmission result (Sec.~VI-A), the causal interventions "
        r"(Sec.~VI-B), or the response-shape result on TPDR (Sec.~VI-F), all of which remain "
        r"intact. It also does \emph{not} rule out that a \CI{} model retrained on "
        r"policy-labelled data would pass TPS; that experiment is left to future work."
    )
    a("")
    a(
        r"\textbf{Limitations.} TPS uses templated synthetic prompts and rule-based oracle "
        r"labels; a natural-language production benchmark remains open. The conflict numbers "
        r"in Table~\ref{tab:tps-conflict} exclude 108 fake-conflict items in "
        r"\texttt{market\_data} where the family threshold equalled the short-side prompt "
        r"$\Tau$. Several reviewer-requested baselines (prompt-text + LoRA, soft-prompt / "
        r"prefix tuning, matched-parameter scalar embedding, TPS-specific LoRA-only) are "
        r"deferred to future work. All TPS results are zero-shot evaluations of checkpoints "
        r"trained without policy-task supervision."
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
