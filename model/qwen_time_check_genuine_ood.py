"""Genuine OOD + T3 multi-week robustness eval.

Reviewer audit found:
  - v1 T1b "OOD" tau in [2s, 14d] mostly overlaps training [1s, 7d].
    Only [7d, 14d] tail is held out. Move test range to [7d, 28d].
  - T3 v14 weekend_signal=1.0 might be tau-bin memorization. Evaluate
    Saturday at week 2/3/4 (tau = 12*86400, 19*86400, 26*86400).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import (
    load_trainable,
    greedy_decode,
    parse_duration_to_seconds,
)


CLOCK_PROMPT = (
    "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

PHASE_PROMPT = "<|im_start|>user\nGood morning.<|im_end|>\n<|im_start|>assistant\n"

WEEKEND_WORDS = ["weekend", "saturday", "sunday", "fun", "relax", "off"]
WEEKDAY_WORDS = ["weekday", "monday", "tuesday", "wednesday", "thursday",
                 "friday", "work", "busy"]


def pearson(pred, truth) -> float:
    if len(pred) < 4:
        return float("nan")
    mp, mt = statistics.mean(pred), statistics.mean(truth)
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, truth))
    denom = (sum((p - mp) ** 2 for p in pred) *
             sum((t - mt) ** 2 for t in truth)) ** 0.5
    return num / denom if denom > 0 else 0.0


def bootstrap_pearson_ci(pred, truth, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(pred)
    rs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        p = [pred[i] for i in idx]; t = [truth[i] for i in idx]
        rs.append(pearson(p, t))
    rs.sort()
    return {"r": pearson(pred, truth),
            "ci_low": rs[int(0.025 * n_boot)],
            "ci_high": rs[int(0.975 * n_boot)],
            "n": n}


def t1b_truly_ood(model, device, n=30, tau_lo_days=7, tau_hi_days=28):
    """Tau drawn log-uniform in [tau_lo_days, tau_hi_days] -- truly held out
    above the v11 training upper bound of 7 days."""
    rng = random.Random(7777)
    pred = []; truth = []; log_errs = []; examples = []
    tau_lo = tau_lo_days * 86400.0
    tau_hi = tau_hi_days * 86400.0
    for _ in range(n):
        tau = math.exp(rng.uniform(math.log(tau_lo), math.log(tau_hi)))
        resp = greedy_decode(model, CLOCK_PROMPT, tau_t=tau, device=device)
        sec = parse_duration_to_seconds(resp)
        if sec == sec and sec > 0:
            pred.append(sec); truth.append(tau)
            log_errs.append(abs(math.log10(sec) - math.log10(tau)))
            if len(examples) < 8:
                examples.append({"true_tau_days": tau / 86400.0,
                                 "resp": resp, "parsed": sec})
    if len(pred) < 4:
        return {"n": len(pred), "pearson_r": float("nan"),
                "log_mae": float("nan"), "examples": examples}
    ci = bootstrap_pearson_ci(pred, truth)
    return {"n": len(pred), "pearson_r": ci["r"],
            "pearson_ci_low": ci["ci_low"], "pearson_ci_high": ci["ci_high"],
            "log_mae": statistics.mean(log_errs),
            "tau_range_days": [tau_lo_days, tau_hi_days],
            "examples": examples}


def has_weekend_word(text):
    t = text.lower()
    return float(any(w in t for w in WEEKEND_WORDS))


def t3_multi_week(model, device, n_per_week=20):
    """Test weekend phase generalization across multiple weeks.
    If model genuinely uses chronometric encoder, weekend response
    should hold at week 2 Sat (tau=12*86400), week 3 (19*86400), etc.
    If model just memorized the v14 training tau bins, signal collapses
    when tau wraps around the trained 7-day window."""
    rng = random.Random(31337)
    results = {}
    for week in range(1, 5):  # weeks 1-4
        # Saturday in week k: tau = (k-1)*7 + 5 days
        tau_sat = ((week - 1) * 7 + 5) * 86400.0 + 12 * 3600
        # Wednesday in week k: tau = (k-1)*7 + 2 days
        tau_wed = ((week - 1) * 7 + 2) * 86400.0 + 12 * 3600
        sat_signals = []; wed_signals = []
        sat_examples = []; wed_examples = []
        for _ in range(n_per_week):
            sat_resp = greedy_decode(model, PHASE_PROMPT, tau_t=tau_sat,
                                     device=device)
            wed_resp = greedy_decode(model, PHASE_PROMPT, tau_t=tau_wed,
                                     device=device)
            sat_signals.append(has_weekend_word(sat_resp))
            wed_signals.append(has_weekend_word(wed_resp))
            if len(sat_examples) < 2:
                sat_examples.append(sat_resp)
                wed_examples.append(wed_resp)
        signal = statistics.mean(sat_signals) - statistics.mean(wed_signals)
        results[f"week_{week}"] = {
            "tau_sat_days": tau_sat / 86400.0,
            "tau_wed_days": tau_wed / 86400.0,
            "sat_weekend_rate": statistics.mean(sat_signals),
            "wed_weekend_rate": statistics.mean(wed_signals),
            "signal": signal,
            "sat_examples": sat_examples,
            "wed_examples": wed_examples,
        }
        print(f"  Week {week}: tau_sat={tau_sat/86400:.1f}d  "
              f"sat_weekend_rate={statistics.mean(sat_signals):.2f}  "
              f"wed_weekend_rate={statistics.mean(wed_signals):.2f}  "
              f"signal={signal:+.3f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--timescales", type=str, default="")
    p.add_argument("--t1b-lo-days", type=int, default=7)
    p.add_argument("--t1b-hi-days", type=int, default=28)
    p.add_argument("--t1b-n", type=int, default=30)
    args = p.parse_args()

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
        print(f"  Override timescales: {cfg.timescales}")
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    print(f"\n=== T1b GENUINE OOD ({args.t1b_lo_days}-{args.t1b_hi_days} days, "
          f"n={args.t1b_n}) ===")
    t1b = t1b_truly_ood(model, args.device, n=args.t1b_n,
                        tau_lo_days=args.t1b_lo_days,
                        tau_hi_days=args.t1b_hi_days)
    print(f"  r={t1b.get('pearson_r', float('nan')):.3f}, "
          f"CI=[{t1b.get('pearson_ci_low', float('nan')):.3f}, "
          f"{t1b.get('pearson_ci_high', float('nan')):.3f}]  "
          f"log_mae={t1b.get('log_mae', float('nan')):.3f}  n={t1b.get('n')}")

    print(f"\n=== T3 MULTI-WEEK (Sat at weeks 1-4) ===")
    t3mw = t3_multi_week(model, args.device)

    summary = {
        "T1b_genuine_OOD": t1b,
        "T3_multi_week": t3mw,
        "PASS_T1b_genuine_OOD": (t1b.get("pearson_r", 0) >= 0.5
                                 and t1b.get("log_mae", 99) < 0.7),
        "PASS_T3_holds_all_weeks": all(
            t3mw[f"week_{w}"]["signal"] >= 0.3 for w in range(1, 5)
        ),
        "T3_min_week_signal": min(t3mw[f"week_{w}"]["signal"]
                                  for w in range(1, 5)),
    }
    print("\n=== VERDICT ===")
    for k, v in summary.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
