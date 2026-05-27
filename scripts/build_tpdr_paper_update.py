"""Generate the section 9 update paragraph + Table 6b for the paper,
using analyze_v2.py output.

Pre-registered in PREREGISTRATION_v2.md section 1. Reads
reports/tpdr_v2_headline.json (produced by eval/tpdr/analyze_v2.py)
and emits a LaTeX fragment to paper/_9_update.tex.

Decision rule per pre-reg section 1.10:
  - CONFIRM if n_paired>=25 AND p_one<=0.005
  - NULL if n_paired>=25 AND p_one>0.005
  - INCONCLUSIVE if n_paired<25
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def main():
    headline_path = REPORTS / "tpdr_v2_headline.json"
    if not headline_path.exists():
        print(f"MISSING: {headline_path}")
        print("Run eval/tpdr/analyze_v2.py first.")
        return

    d = json.loads(headline_path.read_text())
    pairs = d.get("pairs", {})
    h = pairs.get("seed0", {})
    if h.get("status") == "not run":
        print("seed0 sweep not run yet")
        return

    metrics = h.get("metrics", {})
    primary = metrics.get("deliberative_score", {})
    n = primary.get("n_paired", 0)
    t = primary.get("t", float("nan"))
    df = primary.get("df", 0)
    p_one = primary.get("p_one", float("nan"))
    p_two = primary.get("p_two", float("nan"))
    md = primary.get("mean_diff", float("nan"))
    ci_lo = primary.get("ci95_lo", float("nan"))
    ci_hi = primary.get("ci95_hi", float("nan"))

    if n < 25:
        verdict = "INCONCLUSIVE"
        verdict_text = (
            "The pre-registered test was underpowered at this scenario "
            "set ($n_{\\text{paired}} = $%d$ < 25$). We report the result "
            "as inconclusive and do not headline it as confirmatory "
            "evidence of differential behavioral modulation." % n)
    elif p_one <= 0.005:
        verdict = "CONFIRM"
        verdict_text = (
            "The pre-registered headline survives the planned analysis: "
            "the paired-$t$ on the per-scenario rank correlation of "
            "\\texttt{deliberative\\_score} against $\\log\\tau$, "
            "CI vs.\\ Prompt, is $t = %.3f$ on $df = %d$ with one-tailed "
            "$p = %.3e$ at $\\alpha = 0.005$, $n_{\\text{paired}} = %d$. "
            "Mean diff $%.4f$ with 95\\%% CI $[%.4f, %.4f]$." %
            (t, df, p_one, n, md, ci_lo, ci_hi))
    else:
        verdict = "NULL"
        verdict_text = (
            "The pre-registered headline does not reach the pre-specified "
            "threshold: paired-$t$ $t = %.3f$ on $df = %d$, one-tailed "
            "$p = %.3e > 0.005$, $n_{\\text{paired}} = %d$. Per "
            "PREREGISTRATION\\_v2.md \\S 1.10, the architectural claim "
            "that rests on TPDR is withdrawn for this analysis; "
            "the paper's surviving contribution is the mechanistic "
            "interpretability evidence in Sections~\\ref{sec:mechanism} "
            "and \\ref{sec:critical-controls}." % (t, df, p_one, n))

    # Secondary endpoints (Holm)
    sec_rows = []
    for m in ["length_chars", "length_words", "urgency_score",
              "hedge_score", "conditional_clauses", "imperative_count"]:
        em = metrics.get(m, {})
        sec_rows.append(
            f"    \\texttt{{{m.replace('_', r'\_')}}} & "
            f"{em.get('n_paired', 0)} & "
            f"${em.get('t', float('nan')):.3f}$ & "
            f"${em.get('p_two', float('nan')):.3e}$ & "
            f"${em.get('holm_p', float('nan')):.3e}$ \\\\")

    out = []
    out.append("\\paragraph{Pre-registered 200-scenario headline (v2 revision).} "
               "The pre-registered TPDR replication (PREREGISTRATION\\_v2.md "
               "\\S 1, anchor \\texttt{80ddafc}). Sample: all 200 scenarios "
               "as committed at the anchor, with the 7 metrics and their "
               "lexicons frozen. Primary endpoint: one-tailed paired-$t$ "
               "on per-scenario $r(\\text{deliberative score}, \\log\\tau)$ "
               "CI minus Prompt, at $\\alpha = 0.005$. Inclusion: scenarios "
               f"where neither adapter is constant across $\\tau$. Verdict: "
               f"\\textbf{{{verdict}}}. {verdict_text}")
    out.append("")
    out.append("\\begin{table}[!t]")
    out.append("  \\caption{Pre-registered TPDR v2 replication: secondary endpoints (Holm-corrected family $m=6$, $\\alpha = 0.05$ family-wise, two-tailed paired-$t$). Headline pair $(\\text{ci}_0, \\text{prompt}_0)$.}")
    out.append("  \\label{tab:tpdr-v2-secondary}")
    out.append("  \\centering")
    out.append("  \\footnotesize")
    out.append("  \\begin{tabular}{lrrrr}")
    out.append("    \\toprule")
    out.append("    \\tabhead{Metric} & \\tabhead{$n_{\\text{paired}}$} & \\tabhead{$t$} & \\tabhead{raw $p$} & \\tabhead{Holm $p$} \\\\")
    out.append("    \\midrule")
    out.extend(sec_rows)
    out.append("    \\bottomrule")
    out.append("  \\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    # Cross-seed table
    cross_rows = []
    for s in [1, 2]:
        cs = pairs.get(f"seed{s}", {})
        if cs.get("status") == "not run":
            cross_rows.append(f"    seed pair $({s},{s})$ & --- & not run & --- & --- \\\\")
            continue
        m = cs.get("metrics", {}).get("deliberative_score", {})
        cross_rows.append(
            f"    seed pair $({s},{s})$ & "
            f"{m.get('n_paired', 0)} & "
            f"${m.get('t', float('nan')):.3f}$ & "
            f"${m.get('df', 0)}$ & "
            f"${m.get('p_one', float('nan')):.3e}$ \\\\")

    out.append("\\begin{table}[!t]")
    out.append("  \\caption{Cross-seed replication of the pre-registered TPDR primary endpoint on \\texttt{deliberative\\_score}. Per pre-reg, the headline is seed pair $(0,0)$; seed pairs $(1,1)$ and $(2,2)$ are cross-seed replications reported alongside.}")
    out.append("  \\label{tab:tpdr-v2-crossseed}")
    out.append("  \\centering")
    out.append("  \\footnotesize")
    out.append("  \\begin{tabular}{lrrrr}")
    out.append("    \\toprule")
    out.append("    \\tabhead{Pair} & \\tabhead{$n_{\\text{paired}}$} & \\tabhead{$t$} & \\tabhead{df} & \\tabhead{1-tailed $p$} \\\\")
    out.append("    \\midrule")
    out.append(f"    seed pair $(0,0)$ \\textbf{{(HEADLINE)}} & {n} & ${t:.3f}$ & ${df}$ & ${p_one:.3e}$ \\\\")
    out.extend(cross_rows)
    out.append("    \\bottomrule")
    out.append("  \\end{tabular}")
    out.append("\\end{table}")

    out_path = ROOT / "paper" / "_9_update.tex"
    out_path.write_text("\n".join(out) + "\n")
    print(f"saved -> {out_path}")
    print(f"\nVerdict: {verdict}")
    print(f"  n_paired: {n}")
    print(f"  t: {t:.3f}")
    print(f"  df: {df}")
    print(f"  p_one: {p_one:.3e}")


if __name__ == "__main__":
    main()
