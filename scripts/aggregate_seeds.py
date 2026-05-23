"""Aggregate cross-seed v15 results: mean +- std across seeds 0, 1, 2."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    reps = sorted((root / "reports").glob("qwen_time_v15s_*_seed*_recall.json"))
    if not reps:
        print("No v15s seed reports yet")
        return
    print(f"Found {len(reps)} seed reports:")
    per_seed = {}
    for p in reps:
        with open(p) as f:
            r = json.load(f)
        s = r.get("summary", {})
        seed = int(p.stem.split("_seed")[-1].split("_")[0])
        per_seed[seed] = s
        print(f"  seed {seed}: T1={s.get('T1_clock_pearson_r',0):.3f}  "
              f"T1b={s.get('T1b_ood_pearson_r',0):.3f}/mae={s.get('T1b_ood_log_mae',0):.3f}  "
              f"T2={s.get('T2_ack_delta',0):.2f}  "
              f"T3={max(s.get('T3_weekday_signal',0), s.get('T3_weekend_signal',0)):.2f}  "
              f"T4={s.get('T4_mean_pairwise_kl',0):.3f}  "
              f"T4mp={s.get('T4_mean_pairwise_kl_multi_pos', s.get('T4_mean_pairwise_kl', 0)):.3f}")

    keys = [
        ("T1_clock_pearson_r", "T1"),
        ("T1b_ood_pearson_r", "T1b r"),
        ("T1b_ood_log_mae", "T1b log_mae"),
        ("T2_ack_delta", "T2"),
        ("T3_weekend_signal", "T3 weekend"),
        ("T3_weekday_signal", "T3 weekday"),
        ("T4_mean_pairwise_kl", "T4 (first pos)"),
        ("T4_mean_pairwise_kl_multi_pos", "T4 multi-pos"),
    ]

    print("\nCross-seed aggregate (n={} seeds):".format(len(per_seed)))
    agg = {}
    for k, name in keys:
        vals = [per_seed[s].get(k) for s in per_seed if per_seed[s].get(k) is not None]
        if len(vals) >= 2:
            m = statistics.mean(vals)
            sd = statistics.stdev(vals)
            agg[k] = {"mean": m, "std": sd, "n": len(vals), "vals": vals}
            print(f"  {name:20s}: {m:+.4f} +/- {sd:.4f} (n={len(vals)})  vals={vals}")
        elif vals:
            agg[k] = {"mean": vals[0], "std": None, "n": 1, "vals": vals}
            print(f"  {name:20s}: {vals[0]:+.4f} (n=1, no std)")

    out = root / "reports" / "v15_cross_seed_aggregate.json"
    with open(out, "w") as f:
        json.dump({"per_seed": per_seed, "aggregate": agg, "n_seeds": len(per_seed)},
                  f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
