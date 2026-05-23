"""Compute bootstrap 95% CIs on existing falsify + probe results.

Editorial improvement -- no rerun, just adds uncertainty quantification
to the existing JSON outputs. Addresses reviewer attack on missing CIs.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path


def pearson(pred, truth):
    if len(pred) < 4:
        return float("nan")
    mp, mt = statistics.mean(pred), statistics.mean(truth)
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, truth))
    denom = (sum((p - mp) ** 2 for p in pred) *
             sum((t - mt) ** 2 for t in truth)) ** 0.5
    return num / denom if denom > 0 else 0.0


def bootstrap_pearson(pred, truth, n_boot=5000, seed=0):
    rng = random.Random(seed)
    n = len(pred)
    rs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rs.append(pearson([pred[i] for i in idx], [truth[i] for i in idx]))
    rs.sort()
    return {"r": pearson(pred, truth),
            "ci_low": rs[int(0.025 * n_boot)],
            "ci_high": rs[int(0.975 * n_boot)],
            "n_samples": n, "n_boot": n_boot}


def bootstrap_mean(xs, n_boot=5000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": n}
    means = []
    for _ in range(n_boot):
        means.append(statistics.mean([xs[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    return {"mean": statistics.mean(xs),
            "ci_low": means[int(0.025 * n_boot)],
            "ci_high": means[int(0.975 * n_boot)],
            "n": n, "n_boot": n_boot}


def main():
    root = Path(__file__).resolve().parent.parent
    out = {}

    # Falsify Pearson r values from disproof
    f = root / "reports" / "disproof_20260522_224016_falsify.json"
    if f.exists():
        with open(f) as g:
            r = json.load(g)
        falsify = {}
        for cond_name in ["A_normal", "B_alpha_off", "C_random_tau",
                          "D_tau_zero", "E_alpha_flipped"]:
            cond = r[cond_name]
            exs = cond.get("examples", [])
            pred = [e["parsed"] for e in exs
                    if e.get("parsed") and e["parsed"] > 0]
            truth = [e["true_tau"] for e in exs
                     if e.get("parsed") and e["parsed"] > 0]
            if len(pred) >= 4:
                falsify[cond_name] = bootstrap_pearson(pred, truth)
            else:
                falsify[cond_name] = {"r": cond.get("pearson_r"),
                                      "ci_low": None, "ci_high": None,
                                      "n_samples": len(pred),
                                      "note": "examples truncated in original JSON"}
        out["falsify_bootstrap_pearson_CIs"] = falsify

    # Pressure deltas
    p = root / "reports" / "disproof_20260522_224016_pressure.json"
    if p.exists():
        with open(p) as g:
            r = json.load(g)
        pressure = {}
        for cond in ["P1", "P2", "P3"]:
            exs = r[cond].get("examples", [])
            diffs = [e["long_tokens"] - e["short_tokens"] for e in exs]
            pressure[cond] = bootstrap_mean(diffs)
            pressure[cond]["per_prompt_diffs"] = diffs
        out["pressure_bootstrap_CIs"] = pressure

    # v11 T1 OOD
    v11 = root / "reports" / "qwen_time_v10_20260516_032348_recall.json"
    if v11.exists():
        with open(v11) as g:
            r = json.load(g)
        t1b = r.get("t1b", {})
        samples = t1b.get("samples", [])
        if samples:
            truth = [s[0] for s in samples]
            pred = [s[1] for s in samples]
            out["v11_T1b_OOD_bootstrap"] = bootstrap_pearson(pred, truth)

    out_path = root / "reports" / "bootstrap_CIs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")
    print()
    print("=== Bootstrap CI summary ===")
    for section, data in out.items():
        print(f"\n{section}:")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    r_str = v.get('r', v.get('mean', '?'))
                    if isinstance(r_str, (int, float)):
                        r_str = f"{r_str:+.4f}"
                    lo = v.get('ci_low', '?')
                    hi = v.get('ci_high', '?')
                    lo_str = f"{lo:+.4f}" if isinstance(lo, (int, float)) else str(lo)
                    hi_str = f"{hi:+.4f}" if isinstance(hi, (int, float)) else str(hi)
                    print(f"  {k}: r/mean={r_str}  95%CI=[{lo_str}, {hi_str}]  n={v.get('n_samples', v.get('n', '?'))}")


if __name__ == "__main__":
    main()
