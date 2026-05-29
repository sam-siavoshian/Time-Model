"""Aggregate T1-T4 metrics across seeds for the 4 ablation conditions
in Table 3 (docs/experiments/current/PREREGISTRATION_v2.md section 3.4).

Outputs:
  - mean +/- std per metric per condition
  - LaTeX row strings ready for paste into paper/main.tex
"""
import json, statistics, sys
from pathlib import Path

CONDS = {
    "chrono_only":           "Row 1b: chrono-only ($\\alpha$ trainable, LoRA frozen)",
    "additive_nonzero_beta": "Row 2b: additive ($\\beta$-bias init = 0.01)",
    "ia3_with_chrono":       "Row 8:  IA3 + chrono active",
    "ia3_only":              "Row 8a: IA3-only ($\\alpha$ frozen, chrono off)",
}

def load(path):
    p = Path(path)
    return json.load(p.open()) if p.exists() else None

def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return float("nan"), float("nan"), 0
    if len(xs) == 1: return xs[0], 0.0, 1
    return statistics.mean(xs), statistics.stdev(xs), len(xs)

def fmt(m, s, n):
    if n == 0: return "nan"
    if n == 1: return f"{m:.3f} (n=1)"
    return f"{m:.3f} +/- {s:.3f}"

def fmt_tex(m, s, n):
    if n == 0: return "nan"
    if n == 1: return f"${m:.3f}$ ({{\\tiny $n{{=}}1$}})"
    return f"${m:.3f} \\pm {s:.3f}$"

def main():
    rows = {}
    for cond, label in CONDS.items():
        files = [f"reports/{cond}_s{i}_recall.json" for i in (0, 1, 2)]
        seeds = [load(f) for f in files]
        n_seeds = sum(1 for d in seeds if d is not None)
        seeds = [d for d in seeds if d]
        t1   = [d["summary"]["T1_clock_pearson_r"]      for d in seeds]
        t1b  = [d["summary"]["T1b_ood_pearson_r"]       for d in seeds]
        t1bm = [d["summary"]["T1b_ood_log_mae"]         for d in seeds]
        t2   = [d["summary"]["T2_ack_delta"]            for d in seeds]
        t3_w = [d["summary"]["T3_weekday_signal"]       for d in seeds]
        t3_e = [d["summary"]["T3_weekend_signal"]       for d in seeds]
        t4   = [d["summary"]["T4_mean_pairwise_kl"]     for d in seeds]
        rows[cond] = {
            "label": label, "n": n_seeds,
            "T1": fmt(*msd(t1)),    "T1_tex": fmt_tex(*msd(t1)),
            "T1b": fmt(*msd(t1b)),  "T1b_tex": fmt_tex(*msd(t1b)),
            "T1b_mae": fmt(*msd(t1bm)), "T1b_mae_tex": fmt_tex(*msd(t1bm)),
            "T2": fmt(*msd(t2)),    "T2_tex": fmt_tex(*msd(t2)),
            "T3 wkdy": fmt(*msd(t3_w)),
            "T3 wknd": fmt(*msd(t3_e)),
            "T4": fmt(*msd(t4)),    "T4_tex": fmt_tex(*msd(t4)),
            "T3_pass_count": f"wkdy {sum(1 for x in t3_w if x > 0.5)}/{n_seeds}, wknd {sum(1 for x in t3_e if x > 0.5)}/{n_seeds}",
        }

    print("=" * 70)
    for cond, r in rows.items():
        print(f"\n--- {r['label']} (n={r['n']}) ---")
        for k, v in r.items():
            if k.endswith("_tex"): continue
            if k in ("label", "n"): continue
            print(f"  {k:14s}: {v}")

    print("\n" + "=" * 70)
    print("LaTeX rows (T1, T1b r, T1b log-MAE, T2, T3, T4 first-pos):\n")
    for cond, r in rows.items():
        t3 = r["T3_pass_count"].replace("wkdy ", "").replace(", wknd", ",")
        print(f"    {r['label']:50s} & {r['T1_tex']} & {r['T1b_tex']} & {r['T1b_mae_tex']} & {r['T2_tex']} & {t3} & {r['T4_tex']} \\\\")

if __name__ == "__main__":
    main()
