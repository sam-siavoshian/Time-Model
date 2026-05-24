"""tau_sessions external benchmark runner.

Loads the dataset built by generate_tau_sessions.py, runs a chosen
adapter on every session (or a sampled subset for a quick smoke), and
emits a JSON report under reports/external_tau_bench_<adapter>.json.

Scoring (per eval_protocol):
  exact_match    -- staleness yes/no. Score in {0, 1}.
  mae            -- duration_recall log-MAE on parsed seconds. Lower is
                    better. We report log10(pred) - log10(gt) absolute
                    error so a 2x miss = 0.30 and an order-of-magnitude
                    miss = 1.0. NaN parses are counted as misses with
                    log_err = NaN, then dropped from MAE and surfaced
                    as parse_fail_rate.
  len_elasticity -- adaptive. Per bucket we take median response length
                    in characters. Aggregate elasticity is the Pearson
                    correlation between log(tau) and log(length) across
                    all 100 sessions. Positive correlation = model
                    follows the deadline.

Confidence intervals: nonparametric bootstrap, B = 1000 by default,
seed pinned to 42 inside the runner so the same input deterministically
produces the same CI.

Reports include:
  per-bucket and per-task metrics with bootstrap CIs
  aggregate score (composite of the three protocols)
  full session-level rows (so anyone can recompute)

License: MIT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

# Reuse the parser from the internal check harness so MAE numbers are
# directly comparable across the paper.
from model.qwen_time_check import parse_duration_to_seconds

from .adapters import load_adapter


DEFAULT_DATA = "eval/external/datasets/tau_sessions.jsonl"


# -- I/O --------------------------------------------------------------------

def load_sessions(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -- scoring primitives -----------------------------------------------------

def score_exact_match(prediction: str, ground_truth: str) -> int:
    """Lowercase + token-level match of 'yes'/'no' anywhere in the first
    short response window. Conservative: requires a word-boundary token
    so 'yesterday' does not match 'yes'."""
    import re
    p = prediction.strip().lower()
    g = ground_truth.strip().lower()
    if g not in {"yes", "no"}:
        return int(p == g)
    # Search for the FIRST yes/no token; mismatch otherwise.
    m = re.search(r"\b(yes|no)\b", p)
    if not m:
        return 0
    return int(m.group(1) == g)


def score_log_mae(prediction: str, gt_seconds: float) -> float:
    """Absolute log10 error between parsed-seconds and ground truth.
    Returns NaN if the prediction fails to parse."""
    pred_s = parse_duration_to_seconds(prediction)
    if pred_s != pred_s or pred_s <= 0:                      # NaN check
        return float("nan")
    return abs(math.log10(pred_s) - math.log10(max(gt_seconds, 1e-9)))


def pearson(xs: list[float], ys: list[float]) -> float:
    """Numerically stable Pearson r. Returns 0.0 when either input has
    zero variance (matches the convention used elsewhere in this repo)."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2)
    if denom == 0.0:
        return 0.0
    return num / denom


# -- bootstrap --------------------------------------------------------------

def bootstrap_ci(values: list[float], stat_fn, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for stat_fn(values).
    Returns (lo, hi). When values is empty, returns (nan, nan)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            boots.append(float(stat_fn(sample)))
        except (ValueError, ZeroDivisionError):
            continue
    if not boots:
        return (float("nan"), float("nan"))
    boots.sort()
    lo_idx = int(alpha / 2 * len(boots))
    hi_idx = int((1 - alpha / 2) * len(boots)) - 1
    return (boots[lo_idx], boots[hi_idx])


def bootstrap_pairs_ci(pairs: list[tuple[float, float]], stat_fn,
                       n_boot: int = 1000, seed: int = 42,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap CI for a statistic over (x, y) pairs (e.g. Pearson r)."""
    if len(pairs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        xs = [s[0] for s in sample]; ys = [s[1] for s in sample]
        try:
            boots.append(float(stat_fn(xs, ys)))
        except (ValueError, ZeroDivisionError):
            continue
    if not boots:
        return (float("nan"), float("nan"))
    boots.sort()
    lo_idx = int(alpha / 2 * len(boots))
    hi_idx = int((1 - alpha / 2) * len(boots)) - 1
    return (boots[lo_idx], boots[hi_idx])


# -- runner -----------------------------------------------------------------

def run(adapter_name: str, sessions: list[dict], adapter_kwargs: dict,
        progress_every: int = 25) -> list[dict]:
    """Drive the adapter over every session, returning per-session rows.

    Each row contains:
      session_id, prompt, ground_truth, prediction,
      tau_seconds, tau_bucket, task_type, eval_protocol,
      score (per-protocol), parsed_seconds (mae only),
      response_length (adaptive only)
    """
    adapter = load_adapter(adapter_name, **adapter_kwargs)
    rows: list[dict] = []
    try:
        t0 = time.time()
        for i, s in enumerate(sessions):
            prompt = s["prompt"]
            tau = float(s["tau_seconds"])
            try:
                pred = adapter.generate(prompt, tau)
            except Exception as e:                            # noqa: BLE001
                pred = f"<<adapter_error: {type(e).__name__}: {e}>>"

            row: dict[str, Any] = {
                "session_id": s["session_id"],
                "tau_bucket": s["tau_bucket"],
                "tau_seconds": tau,
                "task_type": s["task_type"],
                "eval_protocol": s["eval_protocol"],
                "prompt": prompt,
                "ground_truth": s.get("ground_truth", ""),
                "prediction": pred,
            }
            if s["eval_protocol"] == "exact_match":
                row["score"] = score_exact_match(pred, s["ground_truth"])
            elif s["eval_protocol"] == "mae":
                gt = float(s["extra"]["gt_seconds"])
                err = score_log_mae(pred, gt)
                row["log_abs_err"] = err
                parsed = parse_duration_to_seconds(pred)
                row["parsed_seconds"] = (None if parsed != parsed else parsed)
            elif s["eval_protocol"] == "len_elasticity":
                row["response_length"] = len(pred)
            rows.append(row)

            if (i + 1) % progress_every == 0 or (i + 1) == len(sessions):
                dt = time.time() - t0
                print(f"  [{adapter_name}] {i + 1}/{len(sessions)} done ({dt:.1f}s)",
                      file=sys.stderr, flush=True)
    finally:
        adapter.cleanup()
    return rows


# -- aggregation ------------------------------------------------------------

def aggregate(rows: list[dict], n_boot: int = 1000) -> dict:
    """Compute per-bucket and per-task metrics with bootstrap CIs."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["tau_bucket"]].append(r)
        by_task[r["task_type"]].append(r)

    def staleness_acc(rs: list[dict]) -> dict:
        scores = [r["score"] for r in rs if r["task_type"] == "staleness"]
        if not scores:
            return {"n": 0, "accuracy": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan")}
        lo, hi = bootstrap_ci(scores, statistics.mean, n_boot=n_boot)
        return {"n": len(scores), "accuracy": statistics.mean(scores),
                "ci_lo": lo, "ci_hi": hi}

    def duration_mae(rs: list[dict]) -> dict:
        errs = [r["log_abs_err"] for r in rs
                if r["task_type"] == "duration_recall"
                and r.get("log_abs_err") is not None
                and r["log_abs_err"] == r["log_abs_err"]]   # not NaN
        total_dr = sum(1 for r in rs if r["task_type"] == "duration_recall")
        if not errs:
            return {"n": 0, "n_total": total_dr,
                    "parse_fail_rate": 1.0 if total_dr else float("nan"),
                    "log_mae": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan")}
        lo, hi = bootstrap_ci(errs, statistics.mean, n_boot=n_boot)
        parse_fails = total_dr - len(errs)
        return {"n": len(errs), "n_total": total_dr,
                "parse_fail_rate": parse_fails / total_dr if total_dr else 0.0,
                "log_mae": statistics.mean(errs),
                "ci_lo": lo, "ci_hi": hi}

    def adaptive_elasticity(rs: list[dict]) -> dict:
        pairs = [(math.log10(max(r["tau_seconds"], 1e-9)),
                  math.log10(max(r["response_length"], 1)))
                 for r in rs if r["task_type"] == "adaptive"
                 and r.get("response_length") is not None]
        if len(pairs) < 2:
            return {"n": len(pairs), "pearson_r": float("nan"),
                    "median_length": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan")}
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        r = pearson(xs, ys)
        lo, hi = bootstrap_pairs_ci(pairs, pearson, n_boot=n_boot)
        lengths = [10 ** p[1] for p in pairs]
        return {"n": len(pairs), "pearson_r": r,
                "median_length": statistics.median(lengths),
                "ci_lo": lo, "ci_hi": hi}

    per_bucket = {b: {
        "n": len(rs),
        "staleness": staleness_acc(rs),
        "duration_recall": duration_mae(rs),
        "adaptive": adaptive_elasticity(rs),
    } for b, rs in by_bucket.items()}

    per_task = {
        "staleness": staleness_acc(rows),
        "duration_recall": duration_mae(rows),
        "adaptive": adaptive_elasticity(rows),
    }

    # Composite score for one number that summarizes "tau-awareness":
    #   - staleness accuracy (higher = better, 0..1)
    #   - 1 - duration_log_mae (clamped 0..1)
    #   - adaptive Pearson r clamped to [0, 1]
    # Mean of the three. NaNs are dropped from the mean.
    def _to_unit(v: float, kind: str) -> float | None:
        if v != v:                                           # NaN
            return None
        if kind == "acc":
            return max(0.0, min(1.0, v))
        if kind == "mae":
            return max(0.0, min(1.0, 1.0 - v))
        if kind == "r":
            return max(0.0, min(1.0, v))
        raise ValueError(kind)

    pieces = [
        _to_unit(per_task["staleness"]["accuracy"], "acc"),
        _to_unit(per_task["duration_recall"]["log_mae"], "mae"),
        _to_unit(per_task["adaptive"]["pearson_r"], "r"),
    ]
    pieces = [p for p in pieces if p is not None]
    composite = statistics.mean(pieces) if pieces else float("nan")

    return {
        "n_sessions": len(rows),
        "per_bucket": per_bucket,
        "per_task": per_task,
        "composite_score": composite,
        "bootstrap_n": n_boot,
    }


# -- CLI --------------------------------------------------------------------

def _parse_inject_layers(s: str) -> tuple[int, ...]:
    if not s:
        return ()
    return tuple(int(x) for x in s.split(","))


def _parse_timescales(s: str) -> tuple[int, ...]:
    if not s:
        from .adapters.ci_adapter import V15_TIMESCALES
        return V15_TIMESCALES
    return tuple(int(x) for x in s.split(","))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True,
                   choices=["vanilla", "prompt", "ci"],
                   help="Which adapter to evaluate")
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct",
                   help="Base HF model id (must match the checkpoint when --adapter ci)")
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="JSONL produced by generate_tau_sessions.py")
    p.add_argument("--out", default=None,
                   help="Report path. Default: reports/external_tau_bench_<adapter>.json")
    p.add_argument("--checkpoint", default=None,
                   help="CI adapter checkpoint (.pt) from the v15.0 release")
    p.add_argument("--timescales", default="",
                   help="CI adapter chrono timescales. Default: v15 (15 scales)")
    p.add_argument("--inject-layers", default="",
                   help="CI adapter injection layers. Default: all but last (matches v15)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--injection-type", default="film",
                   choices=["film", "additive"])
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--max-new-tokens", type=int, default=48,
                   help="Generation budget per session")
    p.add_argument("--n", type=int, default=0,
                   help="If >0, run only the first N sessions (smoke mode)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations for CIs")
    args = p.parse_args(argv)

    sessions = load_sessions(args.data)
    if args.n > 0:
        sessions = sessions[: args.n]
    print(f"loaded {len(sessions)} sessions from {args.data}")

    adapter_kwargs: dict[str, Any] = dict(
        base_model=args.base,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    if args.adapter == "ci":
        if not args.checkpoint:
            print("error: --checkpoint is required for --adapter ci",
                  file=sys.stderr)
            return 2
        adapter_kwargs.update(
            checkpoint=args.checkpoint,
            timescales=_parse_timescales(args.timescales),
            inject_layers=_parse_inject_layers(args.inject_layers),
            lora_rank=args.lora_rank,
            injection_type=args.injection_type,
        )

    print(f"running adapter={args.adapter}, base={args.base}, device={args.device}")
    t0 = time.time()
    rows = run(args.adapter, sessions, adapter_kwargs)
    wall = time.time() - t0
    print(f"adapter done in {wall:.1f}s")

    summary = aggregate(rows, n_boot=args.n_boot)
    report = {
        "adapter": args.adapter,
        "base_model": args.base,
        "checkpoint": args.checkpoint,
        "dataset": args.data,
        "n_sessions": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "wall_seconds": wall,
        "summary": summary,
        "sessions": rows,
    }

    out_path = args.out or f"reports/external_tau_bench_{args.adapter}.json"
    write_report(out_path, report)
    print(f"wrote {out_path}")
    print(f"composite_score = {summary['composite_score']:.4f}")
    for task, m in summary["per_task"].items():
        print(f"  {task}: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
