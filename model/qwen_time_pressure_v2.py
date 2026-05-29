"""Pressure test v2: high statistical power.

v1 (model/qwen_time_pressure.py) used 5 prompts and max_new=80, causing
right-censoring (3 of 5 short-tau responses clipped at the cap) and an
underpowered headline (+9 tokens dominated by 1 outlier prompt).

v2 fixes:
  - 30 neutral technical prompts
  - max_new=256 (uncensored generation)
  - bootstrap 95% CI on delta means
  - per-prompt diff distribution reported
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
from model.qwen_time_check import load_trainable


PROMPTS_NEUTRAL = [
    "Explain how photosynthesis works.",
    "What is the difference between RAM and disk?",
    "How does a transformer language model attend to tokens?",
    "What causes thunderstorms?",
    "Describe how a bicycle stays upright.",
    "Why do leaves change color in autumn?",
    "How does a refrigerator keep food cold?",
    "What is the difference between TCP and UDP?",
    "Explain how rainbows form.",
    "How does a hard drive store data?",
    "What is the function of mitochondria?",
    "How does a vaccine train the immune system?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "How does a solar panel generate electricity?",
    "Explain the water cycle.",
    "How does a microwave oven heat food?",
    "What is the difference between HTTP and HTTPS?",
    "How does GPS know where you are?",
    "What causes the tides?",
    "Explain how a combustion engine works.",
    "How does a touchscreen detect fingers?",
    "Why does ice float on water?",
    "How does a battery store energy?",
    "What is the difference between RNA and DNA?",
    "How does a search engine rank results?",
    "Why do we dream?",
    "How does sound travel through air?",
    "Explain how a wind turbine generates power.",
    "What is the role of enzymes in digestion?",
]


def make_prompt(question: str, with_deadline: str = "") -> str:
    user = question if not with_deadline else f"{with_deadline} {question}"
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.no_grad()
def gen_len(model, prompt: str, tau_t: float, device: str,
            max_new: int = 256) -> int:
    """Greedy decode, count tokens until <|im_end|> or eos. max_new=256
    raises cap from v1's 80 to remove right-censoring."""
    tok = model.tokenizer
    ids = tok.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    cur = ids
    im_end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    n = 0
    for _ in range(max_new):
        out = model(cur, tau_t=tau_t)
        next_id = int(out["logits"][-1].argmax().item())
        if im_end is not None and next_id == im_end:
            break
        if next_id == tok.eos_token_id:
            break
        cur = torch.cat([cur, torch.tensor([next_id], device=device)])
        n += 1
    return n


def bootstrap_ci(diffs: list, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Bootstrap percentile CI on mean of diffs."""
    rng = random.Random(seed)
    n = len(diffs)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": n}
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return {"mean": statistics.mean(diffs),
            "ci_low": lo, "ci_high": hi,
            "n": n, "n_boot": n_boot,
            "fraction_positive": sum(1 for d in diffs if d > 0) / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--timescales", type=str, default="")
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

    n_prompts = len(PROMPTS_NEUTRAL)
    print(f"\n=== pressure v2: n={n_prompts} prompts, max_new={args.max_new} ===\n")

    p1_diffs = []   # text + tau, long - short
    p2_diffs = []   # tau only, long - short
    p3_diffs = []   # alpha=0 + text, long - short
    p1_short, p1_long = [], []
    p2_short, p2_long = [], []
    p3_short, p3_long = [], []
    censored = {"p1_short": 0, "p1_long": 0, "p2_short": 0, "p2_long": 0,
                "p3_short": 0, "p3_long": 0}
    examples = []

    # P1 + P2 use the trained alpha (intact)
    for i, q in enumerate(PROMPTS_NEUTRAL):
        # P1: text deadline + matching tau
        short_p1 = make_prompt(q, "You have only 30 seconds, be brief.")
        long_p1 = make_prompt(q, "You have 1 hour, take your time.")
        sl1 = gen_len(model, short_p1, tau_t=30.0, device=args.device,
                      max_new=args.max_new)
        ll1 = gen_len(model, long_p1, tau_t=3600.0, device=args.device,
                      max_new=args.max_new)
        if sl1 >= args.max_new - 1: censored["p1_short"] += 1
        if ll1 >= args.max_new - 1: censored["p1_long"] += 1
        p1_short.append(sl1); p1_long.append(ll1); p1_diffs.append(ll1 - sl1)

        # P2: neutral text, only tau differs
        p_neutral = make_prompt(q)
        sl2 = gen_len(model, p_neutral, tau_t=30.0, device=args.device,
                      max_new=args.max_new)
        ll2 = gen_len(model, p_neutral, tau_t=3600.0, device=args.device,
                      max_new=args.max_new)
        if sl2 >= args.max_new - 1: censored["p2_short"] += 1
        if ll2 >= args.max_new - 1: censored["p2_long"] += 1
        p2_short.append(sl2); p2_long.append(ll2); p2_diffs.append(ll2 - sl2)
        if i < 4:
            examples.append({"q": q,
                             "P1_short": sl1, "P1_long": ll1, "P1_delta": ll1-sl1,
                             "P2_short": sl2, "P2_long": ll2, "P2_delta": ll2-sl2})
        print(f"  [{i+1:2d}/{n_prompts}] P1 d={ll1-sl1:+4d}  P2 d={ll2-sl2:+4d}  {q[:50]}")

    # P3: alpha=0 + text deadline
    print("\n=== P3: alpha=0 (chrono OFF) + deadline text ===")
    for inj in model.chrono_injectors.values():
        with torch.no_grad():
            inj.alpha.zero_()
    for i, q in enumerate(PROMPTS_NEUTRAL):
        short_p = make_prompt(q, "You have only 30 seconds, be brief.")
        long_p = make_prompt(q, "You have 1 hour, take your time.")
        sl = gen_len(model, short_p, tau_t=30.0, device=args.device,
                     max_new=args.max_new)
        ll = gen_len(model, long_p, tau_t=3600.0, device=args.device,
                     max_new=args.max_new)
        if sl >= args.max_new - 1: censored["p3_short"] += 1
        if ll >= args.max_new - 1: censored["p3_long"] += 1
        p3_short.append(sl); p3_long.append(ll); p3_diffs.append(ll - sl)
        print(f"  [{i+1:2d}/{n_prompts}] P3 d={ll-sl:+4d}  {q[:50]}")

    P1_ci = bootstrap_ci(p1_diffs)
    P2_ci = bootstrap_ci(p2_diffs)
    P3_ci = bootstrap_ci(p3_diffs)

    # chrono contribution = P1 - P3 (paired diffs)
    chrono_diffs = [p1_diffs[i] - p3_diffs[i] for i in range(len(p1_diffs))]
    chrono_ci = bootstrap_ci(chrono_diffs)

    verdict = {
        "n_prompts": n_prompts,
        "max_new": args.max_new,
        "censored": censored,
        "P1_text_plus_chrono": P1_ci,
        "P2_chrono_only": P2_ci,
        "P3_alpha_off_text_only": P3_ci,
        "chrono_contribution_P1_minus_P3": chrono_ci,
        "PASS_P1_delta_positive_CI_excludes_zero": P1_ci["ci_low"] > 0,
        "PASS_P2_chrono_alone_CI_excludes_zero": P2_ci["ci_low"] > 0,
        "PASS_chrono_contribution_CI_excludes_zero": chrono_ci["ci_low"] > 0,
    }
    print("\n=== VERDICT (bootstrap 95% CI on paired diffs) ===")
    for k, v in verdict.items():
        if isinstance(v, dict):
            m = v.get('mean', float('nan'))
            lo = v.get('ci_low', float('nan'))
            hi = v.get('ci_high', float('nan'))
            n = v.get('n', '?')
            pf = v.get('fraction_positive', '?')
            def _fmt(x):
                if isinstance(x, (int, float)) and x == x:
                    return f"{x:+.2f}"
                return str(x)
            print(f"  {k}: mean={_fmt(m)} CI=[{_fmt(lo)}, {_fmt(hi)}] n={n} pos_frac={pf}")
        else:
            print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "verdict": verdict,
            "p1_diffs": p1_diffs, "p2_diffs": p2_diffs, "p3_diffs": p3_diffs,
            "p1_short": p1_short, "p1_long": p1_long,
            "p2_short": p2_short, "p2_long": p2_long,
            "p3_short": p3_short, "p3_long": p3_long,
            "chrono_diffs": chrono_diffs,
            "examples": examples,
        }, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
