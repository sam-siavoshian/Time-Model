"""Aggregate cross-seed v15 results: mean +- std across seeds 0, 1, 2."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


SEED_RE = re.compile(r"_seed(?P<seed>\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--inputs",
        nargs="+",
        help="Explicit seed recall JSON files. Required to avoid mixing stale reports.",
    )
    src.add_argument(
        "--run-id",
        help="Run id prefix, e.g. qwen_time_v15s_20260523_141410. "
             "Expands only runs/<run-id>/reports/<run-id>_seed*_recall.json.",
    )
    ap.add_argument(
        "--out",
        help="Output aggregate JSON. Defaults to runs/<run-id>/reports/<run-id>_aggregate.json "
             "for --run-id; required with --inputs.",
    )
    ap.add_argument(
        "--expected-seeds",
        default="0,1,2",
        help="Comma-separated seed ids that must be present exactly once.",
    )
    return ap.parse_args()


def seed_from_path(path: Path) -> int:
    match = SEED_RE.search(path.stem)
    if not match:
        raise ValueError(f"could not parse seed id from {path}")
    return int(match.group("seed"))


def display_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    expected = {int(s) for s in args.expected_seeds.split(",") if s.strip()}

    if args.inputs:
        reps = [Path(p) for p in args.inputs]
        if args.out is None:
            raise SystemExit("--out is required when using --inputs")
    else:
        run_reports = root / "runs" / args.run_id / "reports"
        if not run_reports.is_dir():
            raise SystemExit(f"run reports directory does not exist: {run_reports}")
        reps = sorted(run_reports.glob(f"{args.run_id}_seed*_recall.json"))

    if not reps:
        raise SystemExit("no seed reports matched the explicit input selection")

    reps = [p if p.is_absolute() else root / p for p in reps]
    missing_files = [str(p) for p in reps if not p.is_file()]
    if missing_files:
        raise SystemExit("missing input files:\n  " + "\n  ".join(missing_files))

    print(f"Found {len(reps)} seed reports:")
    per_seed = {}
    for p in reps:
        with open(p) as f:
            r = json.load(f)
        s = r.get("summary", {})
        seed = seed_from_path(p)
        if seed in per_seed:
            raise SystemExit(f"duplicate report for seed {seed}: {p}")
        per_seed[seed] = s
        print(f"  seed {seed}: T1={s.get('T1_clock_pearson_r',0):.3f}  "
              f"T1b={s.get('T1b_ood_pearson_r',0):.3f}/mae={s.get('T1b_ood_log_mae',0):.3f}  "
              f"T2={s.get('T2_ack_delta',0):.2f}  "
              f"T3={max(s.get('T3_weekday_signal',0), s.get('T3_weekend_signal',0)):.2f}  "
              f"T4={s.get('T4_mean_pairwise_kl',0):.3f}  "
              f"T4mp={s.get('T4_mean_pairwise_kl_multi_pos', s.get('T4_mean_pairwise_kl', 0)):.3f}")

    found = set(per_seed)
    if found != expected:
        raise SystemExit(f"expected seeds {sorted(expected)}, found {sorted(found)}")

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

    out = Path(args.out) if args.out else root / "runs" / args.run_id / "reports" / f"{args.run_id}_aggregate.json"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"input_files": [display_path(p, root) for p in reps],
                   "per_seed": per_seed,
                   "aggregate": agg,
                   "n_seeds": len(per_seed)},
                  f, indent=2)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
