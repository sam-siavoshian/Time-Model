"""TPS analysis.

Reads an explicit list or run directory of reports/tps/<adapter>.json files, computes:
  - per-adapter policy accuracy (overall, per condition, per family, held-in vs held-out)
  - balanced accuracy across the 4 action classes
  - monotonicity r = corr(log_tau, P_REFRESH) within hidden_only condition
  - conflict scalar-follow rate vs prompt-follow rate vs abstain rate
  - cross-seed mean/std for v15s seeds

Writes a consolidated headline JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_run(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def policy_acc(rows: Iterable[dict], gold_key: str = "gold_scalar") -> tuple[float, int]:
    rows = list(rows)
    if not rows:
        return float("nan"), 0
    n_correct = sum(1 for r in rows if r["action"] == r[gold_key])
    return n_correct / len(rows), len(rows)


def balanced_acc(rows: Iterable[dict], gold_key: str = "gold_scalar") -> tuple[float, int]:
    rows = list(rows)
    if not rows:
        return float("nan"), 0
    by_class: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_class[r[gold_key]].append(r["action"] == r[gold_key])
    if not by_class:
        return float("nan"), len(rows)
    per_class = [sum(v) / len(v) for v in by_class.values()]
    return sum(per_class) / len(per_class), len(rows)


def corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def monotonicity(rows: list[dict]) -> dict[str, float]:
    """Within hidden_only items, correlate log_tau with indicator(action == 'REFRESH')."""
    rows = [r for r in rows if r["condition"] == "hidden_only" and r["tau_ci_s"] > 0]
    if not rows:
        return {"r_log_tau_vs_p_refresh": float("nan"), "n": 0}
    xs = [math.log(float(r["tau_ci_s"])) for r in rows]
    ys = [1.0 if r["action"] == "REFRESH" else 0.0 for r in rows]
    return {
        "r_log_tau_vs_p_refresh": corr(xs, ys),
        "n": len(rows),
        "p_refresh_overall": sum(ys) / len(ys),
    }


def conflict_breakdown(rows: list[dict]) -> dict[str, Any]:
    """Conflict scoring with two corrections from review:

    1. Filter out 'fake conflict' items where gold_scalar == gold_prompt
       (happens in market_data because its threshold equals the short-side
       tau_prompt of 10s, so both golds collapse to REFRESH).
    2. 'Abstain' should not double-count ASK/SUMMARIZE when those ARE the
       gold action for the family (safety -> ASK, session -> SUMMARIZE).
       Restrict to items whose gold_scalar AND gold_prompt are both in
       {REUSE, REFRESH}.
    """
    rows = [r for r in rows if r["condition"] in ("conflict_ps_cl", "conflict_pl_cs")]
    real = [r for r in rows if r["gold_prompt"] is not None and r["gold_prompt"] != r["gold_scalar"]]
    if not real:
        return {"n_raw": len(rows), "n_real": 0}
    scalar_follow = sum(1 for r in real if r["action"] == r["gold_scalar"])
    prompt_follow = sum(1 for r in real if r["action"] == r["gold_prompt"])
    # Restrict abstain to items where ASK/SUMMARIZE are not the gold answer.
    abstain_eligible = [
        r for r in real
        if r["gold_scalar"] in ("REUSE", "REFRESH") and r["gold_prompt"] in ("REUSE", "REFRESH")
    ]
    abstain = sum(1 for r in abstain_eligible if r["action"] in ("ASK", "SUMMARIZE"))
    return {
        "n_raw": len(rows),
        "n_real": len(real),
        "n_fake_conflicts_excluded": len(rows) - len(real),
        "scalar_follow_rate": scalar_follow / len(real),
        "prompt_follow_rate": prompt_follow / len(real),
        "abstain_rate_eligible_only": (abstain / len(abstain_eligible)) if abstain_eligible else float("nan"),
        "n_abstain_eligible": len(abstain_eligible),
    }


def split_metrics(rows: list[dict]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["overall_policy_acc"], metrics["overall_n"] = policy_acc(rows)
    metrics["overall_balanced_acc"], _ = balanced_acc(rows)
    by_cond: dict[str, Any] = {}
    for cond in ("hidden_only", "prompt_only", "both_agree", "conflict_ps_cl", "conflict_pl_cs"):
        cond_rows = [r for r in rows if r["condition"] == cond]
        acc, n = policy_acc(cond_rows)
        bacc, _ = balanced_acc(cond_rows)
        by_cond[cond] = {"policy_acc": acc, "balanced_acc": bacc, "n": n}
    metrics["by_condition"] = by_cond
    by_fam: dict[str, Any] = {}
    for fam in sorted(set(r["family"] for r in rows)):
        fam_rows = [r for r in rows if r["family"] == fam]
        acc, n = policy_acc(fam_rows)
        bacc, _ = balanced_acc(fam_rows)
        by_fam[fam] = {"policy_acc": acc, "balanced_acc": bacc, "n": n}
    metrics["by_family"] = by_fam
    # Held-out splits.
    metrics["held_in_template_policy_acc"], metrics["held_in_template_n"] = policy_acc(
        [r for r in rows if not r["held_out_template"]]
    )
    metrics["held_out_template_policy_acc"], metrics["held_out_template_n"] = policy_acc(
        [r for r in rows if r["held_out_template"]]
    )
    metrics["held_in_family_policy_acc"], metrics["held_in_family_n"] = policy_acc(
        [r for r in rows if not r["held_out_family"]]
    )
    metrics["held_out_family_policy_acc"], metrics["held_out_family_n"] = policy_acc(
        [r for r in rows if r["held_out_family"]]
    )
    metrics["monotonicity"] = monotonicity(rows)
    metrics["conflict"] = conflict_breakdown(rows)
    return metrics


def analyze_run(path: str) -> dict[str, Any]:
    run = load_run(path)
    rows = run["results"]
    return {
        "sweep_id": run["sweep_id"],
        "adapter": run["adapter"],
        "checkpoint": run.get("checkpoint"),
        "n_items": run["n_items"],
        "elapsed_sec": run["elapsed_sec"],
        "metrics": split_metrics(rows),
    }


def aggregate_seeds(runs: list[dict]) -> dict[str, float]:
    accs = [r["metrics"]["overall_policy_acc"] for r in runs]
    n = len(accs)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    hidden_accs = [r["metrics"]["by_condition"]["hidden_only"]["policy_acc"] for r in runs]
    monotonicities = [r["metrics"]["monotonicity"]["r_log_tau_vs_p_refresh"] for r in runs]

    def mean_std(vals: list[float]) -> tuple[float, float]:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1) if len(vals) > 1 else 0.0
        return mean, math.sqrt(var)

    mean, std = mean_std(accs)
    hidden_mean, hidden_std = mean_std(hidden_accs)
    mono_mean, mono_std = mean_std(monotonicities)
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "hidden_only_mean": hidden_mean,
        "hidden_only_std": hidden_std,
        "monotonicity_mean": mono_mean,
        "monotonicity_std": mono_std,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", nargs="+",
                     help="Explicit TPS run JSON files to analyze.")
    src.add_argument("--run-dir",
                     help="Directory containing exactly the TPS run JSONs for one run.")
    src.add_argument("--run-id",
                     help="Run id under runs/<run-id>/reports/tps/.")
    src.add_argument("--input-glob",
                     help="Explicit glob for compatibility. Prefer --inputs or --run-dir.")
    ap.add_argument("--out",
                    help="Output headline JSON. Defaults to <run-dir>/headline.json for "
                         "--run-dir/--run-id; required for --inputs/--input-glob.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.inputs:
        files = args.inputs
        if args.out is None:
            raise SystemExit("--out is required when using --inputs")
    elif args.run_id:
        run_dir = root / "runs" / args.run_id / "reports" / "tps"
        if not run_dir.is_dir():
            raise SystemExit(f"run TPS report directory does not exist: {run_dir}")
        files = sorted(str(p) for p in run_dir.glob("*.json"))
        if args.out is None:
            args.out = str(run_dir / "headline.json")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        files = sorted(str(p) for p in run_dir.glob("*.json"))
        if args.out is None:
            args.out = str(run_dir / "headline.json")
    else:
        files = sorted(glob.glob(args.input_glob))
        if args.out is None:
            raise SystemExit("--out is required when using --input-glob")

    files = [str(Path(f) if Path(f).is_absolute() else root / f) for f in files]
    files = [f for f in files if os.path.basename(f) not in {"headline.json", "baselines.json"}
             and not os.path.basename(f).startswith("SMOKE_")
             and not os.path.basename(f).startswith("_PARTIAL_")]
    if not files:
        raise SystemExit("no TPS run files matched the explicit input selection")

    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise SystemExit("missing input files:\n  " + "\n  ".join(missing))

    per_adapter: dict[str, Any] = {}
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        if tag in per_adapter:
            raise SystemExit(f"duplicate TPS tag from input files: {tag}")
        per_adapter[tag] = analyze_run(f)

    ci_seeds = [v for k, v in per_adapter.items() if k.startswith("ci_v15s_s")]
    ci_agg = aggregate_seeds(ci_seeds) if ci_seeds else None

    headline = {
        "input_files": [str(Path(f).relative_to(root)) if Path(f).is_relative_to(root) else f
                        for f in files],
        "per_adapter": per_adapter,
        "ci_v15s_crossseed": ci_agg,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(headline, fh, indent=2)
    print(f"wrote {args.out}")

    # Console summary.
    print("\n=== TPS HEADLINE ===")
    for tag, info in per_adapter.items():
        m = info["metrics"]
        print(f"{tag}: policy_acc={m['overall_policy_acc']:.3f}  balanced={m['overall_balanced_acc']:.3f}  "
              f"hidden_only={m['by_condition']['hidden_only']['policy_acc']:.3f}  "
              f"r(logtau,Pref)={m['monotonicity']['r_log_tau_vs_p_refresh']:.3f}  "
              f"conflict.scalar_follow={m['conflict'].get('scalar_follow_rate', float('nan')):.3f}")
    if ci_agg:
        print(f"\nCI v15s cross-seed (n={ci_agg['n']}): mean={ci_agg['mean']:.3f}+-{ci_agg['std']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
