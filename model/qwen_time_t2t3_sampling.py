"""T2 / T3 sampling rerun.

Reviewer attack: T2 and T3 effective n=1 under greedy decoding
(identical prompt + identical tau + deterministic argmax = same
output replicated n times). The 'delta=1.0 across 30 trials' / 'signal
across 20 trials' reports inflate denominator. This script reruns T2
and T3 at temperature=0.7 with 30 independent torch seeds per condition
so the variance bar is real.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import torch

from model.qwen_time import QwenTimeConfig, build_qwen_time
from model.qwen_time_check import (
    load_trainable,
    greedy_decode,
    parse_duration_to_seconds,
)


ACK_KEYWORDS = ["welcome back", "been a while", "long time", "it has been", "it's been"]
WEEKEND_WORDS = ["weekend", "saturday", "sunday", "fun", "relax", "off"]
WEEKDAY_WORDS = ["weekday", "monday", "tuesday", "wednesday", "thursday",
                 "friday", "work", "busy"]


def has_ack(text):
    t = text.lower()
    return any(k in t for k in ACK_KEYWORDS)


def has_word(text, words):
    t = text.lower()
    return float(any(w in t for w in words))


def t2_sampling(model, device, n_samples=30, temperature=0.7):
    prompt = (
        "<|im_start|>user\nTell me a fun fact.<|im_end|>\n"
        "<|im_start|>assistant\nOctopuses have three hearts.<|im_end|>\n"
        "<|im_start|>user\nHi again.<|im_end|>\n<|im_start|>assistant\n"
    )
    ack_small = []
    ack_large = []
    small_resps = []
    large_resps = []
    for i in range(n_samples):
        s = greedy_decode(model, prompt, tau_t=10.0, device=device,
                         temperature=temperature, seed=10000 + i)
        l = greedy_decode(model, prompt, tau_t=86400.0, device=device,
                         temperature=temperature, seed=20000 + i)
        ack_small.append(int(has_ack(s)))
        ack_large.append(int(has_ack(l)))
        if i < 6:
            small_resps.append(s)
            large_resps.append(l)
    delta = (sum(ack_large) - sum(ack_small)) / n_samples
    return {
        "n_samples": n_samples,
        "temperature": temperature,
        "ack_rate_small_tau": sum(ack_small) / n_samples,
        "ack_rate_large_tau": sum(ack_large) / n_samples,
        "delta": delta,
        "unique_small_responses": len(set(small_resps)),
        "unique_large_responses": len(set(large_resps)),
        "sample_small_responses": small_resps[:3],
        "sample_large_responses": large_resps[:3],
    }


def t3_sampling(model, device, n_samples=30, temperature=0.7):
    prompt = "<|im_start|>user\nGood morning.<|im_end|>\n<|im_start|>assistant\n"
    weekday_signals = []
    weekend_signals = []
    wd_resps = []
    we_resps = []
    for i in range(n_samples):
        wd = greedy_decode(model, prompt, tau_t=2 * 86400.0, device=device,
                          temperature=temperature, seed=30000 + i)
        we = greedy_decode(model, prompt, tau_t=5 * 86400.0, device=device,
                          temperature=temperature, seed=40000 + i)
        # weekend signal: weekend word on weekend prompt minus on weekday prompt
        weekend_signals.append(has_word(we, WEEKEND_WORDS) - has_word(wd, WEEKEND_WORDS))
        weekday_signals.append(has_word(wd, WEEKDAY_WORDS) - has_word(we, WEEKDAY_WORDS))
        if i < 6:
            wd_resps.append(wd)
            we_resps.append(we)
    return {
        "n_samples": n_samples,
        "temperature": temperature,
        "weekend_signal_mean": statistics.mean(weekend_signals),
        "weekend_signal_std": statistics.stdev(weekend_signals) if len(weekend_signals) > 1 else 0.0,
        "weekday_signal_mean": statistics.mean(weekday_signals),
        "weekday_signal_std": statistics.stdev(weekday_signals) if len(weekday_signals) > 1 else 0.0,
        "unique_wd_responses": len(set(wd_resps)),
        "unique_we_responses": len(set(we_resps)),
        "sample_wd_responses": wd_resps[:3],
        "sample_we_responses": we_resps[:3],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--timescales", type=str, default="")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--n-samples", type=int, default=30)
    args = p.parse_args()

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    print(f"\n=== T2 sampling (temperature={args.temperature}, n={args.n_samples} seeds) ===")
    t2 = t2_sampling(model, args.device, n_samples=args.n_samples,
                    temperature=args.temperature)
    print(f"  ack_small={t2['ack_rate_small_tau']:.2f}  ack_large={t2['ack_rate_large_tau']:.2f}  "
          f"delta={t2['delta']:+.2f}  unique_small={t2['unique_small_responses']}  unique_large={t2['unique_large_responses']}")

    print(f"\n=== T3 sampling (temperature={args.temperature}, n={args.n_samples} seeds) ===")
    t3 = t3_sampling(model, args.device, n_samples=args.n_samples,
                    temperature=args.temperature)
    print(f"  weekend signal: {t3['weekend_signal_mean']:+.3f} ± {t3['weekend_signal_std']:.3f}  "
          f"unique_wd={t3['unique_wd_responses']} unique_we={t3['unique_we_responses']}")
    print(f"  weekday signal: {t3['weekday_signal_mean']:+.3f} ± {t3['weekday_signal_std']:.3f}")

    out = {"T2_sampling": t2, "T3_sampling": t3,
           "PASS_T2_delta": t2["delta"] >= 0.5,
           "PASS_T3_weekend": t3["weekend_signal_mean"] >= 0.3}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
