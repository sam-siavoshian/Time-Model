"""Pre-registered TPDR v2 analysis script.

Reads the saved sweep JSONs:
  reports/tpdr_crossseed/tpdr_v2_seed{0,1,2}_pair{0,1,2}.json
where each file holds the sweep for ci_seed=k, prompt_seed=k.

Computes the pre-registered primary endpoint:
  - one-tailed paired Student's t on per-scenario (r_ci - r_prompt)
    differences on the deliberative_score metric, at alpha=0.005.
  - Headline pair: (ci_seed=0, prompt_seed=0).
  - Cross-seed replications: (1,1) and (2,2).

Plus the pre-registered secondary endpoints (Holm-Bonferroni over the
remaining 6 metrics, two-tailed paired-t, family-wise alpha=0.05).

Writes the consolidated result to reports/tpdr_v2_headline.json.

Inclusion rule per scenario per metric (PREREGISTRATION_v2.md section 1.3):
  scenario contributes a pair iff the per-tau series for that metric has
  non-zero variance on BOTH adapters; otherwise Pearson r is nan and
  the scenario is excluded.

Usage:
  uv run python eval/tpdr/analyze_v2.py \\
    --sweep-dir reports/tpdr_crossseed \\
    --pattern 'tpdr_v2_seed{seed}_pair{seed}.json' \\
    --out reports/tpdr_v2_headline.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from statistics import mean, stdev
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy import stats as sstats
from eval.tpdr.metrics import pearson_safe

PRIMARY_METRIC = "deliberative_score"
ALL_METRICS = [
    "length_chars", "length_words", "urgency_score",
    "deliberative_score", "hedge_score", "conditional_clauses",
    "imperative_count",
]
SECONDARY_METRICS = [m for m in ALL_METRICS if m != PRIMARY_METRIC]
ALPHA_PRIMARY = 0.005
ALPHA_SECONDARY_FAMILY = 0.05


def per_scenario_r(per_scenario_results, taus, metric):
    """Return list of per-scenario Pearson r for metric vs log(tau).
    nan if zero-variance.
    """
    logtaus = [math.log(t) for t in taus]
    rs = []
    for sc in per_scenario_results:
        vals = [m[metric] for m in sc["metrics_per_tau"]]
        rs.append(pearson_safe(logtaus, vals))
    return rs


def paired_t(diffs, alternative="two-sided"):
    """Paired-t on a list of diffs (already per-scenario CI - prompt).
    Drops NaNs. Returns dict with t, df, p_two, p_one, n_paired, mean,
    sd, ci95.
    """
    clean = [d for d in diffs if not math.isnan(d)]
    n = len(clean)
    if n < 2:
        return {"n_paired": n, "t": float("nan"), "df": max(n - 1, 0),
                "p_two": float("nan"), "p_one": float("nan"),
                "mean_diff": float("nan"), "sd_diff": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    res = sstats.ttest_1samp(clean, 0.0, alternative="two-sided")
    t = float(res.statistic); df = n - 1
    p_two = float(res.pvalue)
    if alternative == "less":
        p_one = p_two / 2 if t < 0 else 1 - p_two / 2
    elif alternative == "greater":
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    else:
        p_one = p_two
    m = mean(clean); sd = stdev(clean) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t_crit = sstats.t.ppf(0.975, df) if df > 0 else float("nan")
    ci_lo = m - t_crit * se if df > 0 else float("nan")
    ci_hi = m + t_crit * se if df > 0 else float("nan")
    return {"n_paired": n, "t": t, "df": df, "p_two": p_two, "p_one": p_one,
            "mean_diff": m, "sd_diff": sd, "ci95_lo": ci_lo, "ci95_hi": ci_hi}


def holm_correct(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down. Returns list of dicts with
    {raw_p, holm_p, reject} matching input order.
    """
    indexed = list(enumerate(pvalues))
    indexed.sort(key=lambda x: x[1])
    m = len(pvalues)
    holm = [None] * m
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)
        holm[orig_idx] = adj
        prev = adj
    return [{"raw_p": pvalues[i], "holm_p": holm[i],
             "reject": holm[i] <= alpha} for i in range(m)]


def analyze_pair(ci_data, prompt_data, taus, label):
    out = {"label": label, "metrics": {}}
    for metric in ALL_METRICS:
        r_ci = per_scenario_r(ci_data, taus, metric)
        r_pr = per_scenario_r(prompt_data, taus, metric)
        diffs = []
        for rc, rp in zip(r_ci, r_pr):
            if math.isnan(rc) or math.isnan(rp):
                diffs.append(float("nan"))
            else:
                diffs.append(rc - rp)
        if metric == PRIMARY_METRIC:
            res = paired_t(diffs, alternative="less")
            res["alpha"] = ALPHA_PRIMARY
            res["alternative"] = "less (one-tailed, H1: mu_diff < 0)"
        else:
            res = paired_t(diffs, alternative="two-sided")
            res["alternative"] = "two-sided"
        res["per_scenario_r_ci"] = r_ci
        res["per_scenario_r_prompt"] = r_pr
        res["per_scenario_diff"] = diffs
        out["metrics"][metric] = res

    secondary_p = [out["metrics"][m]["p_two"] for m in SECONDARY_METRICS]
    holm = holm_correct(secondary_p, alpha=ALPHA_SECONDARY_FAMILY)
    for i, m in enumerate(SECONDARY_METRICS):
        out["metrics"][m]["holm_p"] = holm[i]["holm_p"]
        out["metrics"][m]["holm_reject_at_0.05"] = holm[i]["reject"]
    return out


def load_pair(sweep_dir, pattern, seed):
    fpath = Path(sweep_dir) / pattern.format(seed=seed)
    if not fpath.exists():
        return None, None, None
    d = json.loads(fpath.read_text())
    return (d["adapters"]["ci"]["per_scenario_results"],
            d["adapters"]["prompt"]["per_scenario_results"],
            d["taus"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="reports/tpdr_crossseed")
    ap.add_argument("--pattern", default="tpdr_v2_seed{seed}_pair{seed}.json")
    ap.add_argument("--out", default="reports/tpdr_v2_headline.json")
    args = ap.parse_args()

    full = {"pre_registration": "PREREGISTRATION_v2.md",
            "primary_metric": PRIMARY_METRIC,
            "primary_alpha": ALPHA_PRIMARY,
            "primary_alternative": "less (one-tailed, H1: mu_diff < 0)",
            "secondary_metrics": SECONDARY_METRICS,
            "secondary_correction": "Holm-Bonferroni step-down (m=6)",
            "secondary_alpha_family": ALPHA_SECONDARY_FAMILY,
            "pairs": {}}

    for seed in [0, 1, 2]:
        ci, pr, taus = load_pair(args.sweep_dir, args.pattern, seed)
        if ci is None:
            full["pairs"][f"seed{seed}"] = {"status": "not run"}
            continue
        label = ("HEADLINE" if seed == 0 else "CROSS-SEED REPLICATION")
        full["pairs"][f"seed{seed}"] = analyze_pair(ci, pr, taus, label)

    headline = full["pairs"].get("seed0", {})
    if headline.get("metrics", {}).get(PRIMARY_METRIC):
        h = headline["metrics"][PRIMARY_METRIC]
        n = h["n_paired"]; p1 = h["p_one"]; t = h["t"]; df = h["df"]
        if n < 25:
            full["headline_decision"] = (
                f"INCONCLUSIVE (n_paired={n} < 25, underpowered).")
        elif p1 <= ALPHA_PRIMARY:
            full["headline_decision"] = (
                f"CONFIRM at alpha={ALPHA_PRIMARY}: t={t:.3f} df={df} "
                f"one-tailed p={p1:.3e} n_paired={n}.")
        else:
            full["headline_decision"] = (
                f"NULL at alpha={ALPHA_PRIMARY}: t={t:.3f} df={df} "
                f"one-tailed p={p1:.3e} n_paired={n}.")
    else:
        full["headline_decision"] = "pending: seed0 sweep not yet saved"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(full, indent=2, default=str))
    print(f"saved -> {args.out}")
    print(full["headline_decision"])


if __name__ == "__main__":
    main()
