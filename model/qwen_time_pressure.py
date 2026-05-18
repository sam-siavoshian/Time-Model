"""Behavioral pressure test (PAPER Gap 4: no deadline behavior).

The strongest claim of "time experience" is not just answering 'how long
has it been' -- it is that elapsed-time signal shifts OTHER behavior.
A model that knows it has only 30 seconds to answer should answer
shorter than one that has 1 day. v11 was NOT trained on deadline
conversations; this is therefore an OOD generalization test:

  Does the chrono signal, learned from clock / silent-gap / phase tasks,
  also bias response length when paired with deadline prompts?

We pair a deadline prompt with the matching tau_t. The deadline tau
encodes 'how much time remains' (model receives this as chi_t while
generating). The PROMPT also mentions the deadline in text.

Variants:
  P1: long deadline (1 hour) vs short deadline (30s)
  P2: ablation -- same prompt text but swap tau values

Metric:
  Token count of response.
  - If chrono signal AND prompt text both push in same direction, response
    under tau=30s should be shorter than under tau=3600s by >= 20%.
  - Ablation P2 isolates chrono contribution: same TEXT, only tau changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

import torch

from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable, greedy_decode


PROMPTS_NEUTRAL = [
    "Explain how photosynthesis works.",
    "What is the difference between RAM and disk?",
    "How does a transformer language model attend to tokens?",
    "What causes thunderstorms?",
    "Describe how a bicycle stays upright.",
]


def make_prompt(question: str, with_deadline: str = "") -> str:
    user = question if not with_deadline else f"{with_deadline} {question}"
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.no_grad()
def gen_len(model, prompt: str, tau_t: float, device: str,
            max_new: int = 80) -> int:
    """Greedy decode, count tokens until <|im_end|> or eos."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/qwen_time_v11_pressure.json")
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = p.parse_args()

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    print(f"Loading {cfg.base_model_name}...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"Loading {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    # P1: deadline text + matching tau
    print("\n=== P1: deadline text + matching tau (text + chrono both informed) ===")
    short_lens = []
    long_lens = []
    examples = []
    for q in PROMPTS_NEUTRAL:
        short_p = make_prompt(q, "You have only 30 seconds, be brief.")
        long_p = make_prompt(q, "You have 1 hour, take your time.")
        sl = gen_len(model, short_p, tau_t=30.0, device=args.device)
        ll = gen_len(model, long_p, tau_t=3600.0, device=args.device)
        short_lens.append(sl)
        long_lens.append(ll)
        examples.append({"q": q, "short_tokens": sl, "long_tokens": ll})
        print(f"  Q={q[:40]:40s}  short={sl:3d}  long={ll:3d}  delta={ll-sl:+d}")
    P1 = {
        "mean_short": statistics.mean(short_lens),
        "mean_long": statistics.mean(long_lens),
        "delta": statistics.mean(long_lens) - statistics.mean(short_lens),
        "pct_short_of_long": statistics.mean(short_lens) / max(1, statistics.mean(long_lens)),
        "examples": examples,
    }
    print(f"  mean_short={P1['mean_short']:.1f}, mean_long={P1['mean_long']:.1f}, "
          f"delta={P1['delta']:+.1f}, pct={P1['pct_short_of_long']:.2f}")

    # P2: ablation -- same TEXT, swap tau only
    print("\n=== P2: ablation -- same neutral text, swap tau ===")
    short_lens_a = []
    long_lens_a = []
    ex2 = []
    for q in PROMPTS_NEUTRAL:
        p_q = make_prompt(q)
        sl = gen_len(model, p_q, tau_t=30.0, device=args.device)
        ll = gen_len(model, p_q, tau_t=3600.0, device=args.device)
        short_lens_a.append(sl)
        long_lens_a.append(ll)
        ex2.append({"q": q, "short_tokens": sl, "long_tokens": ll})
        print(f"  Q={q[:40]:40s}  tau30={sl:3d}  tau3600={ll:3d}  delta={ll-sl:+d}")
    P2 = {
        "mean_short": statistics.mean(short_lens_a),
        "mean_long": statistics.mean(long_lens_a),
        "delta": statistics.mean(long_lens_a) - statistics.mean(short_lens_a),
        "examples": ex2,
    }
    print(f"  mean_short={P2['mean_short']:.1f}, mean_long={P2['mean_long']:.1f}, "
          f"delta={P2['delta']:+.1f}")

    # P3: alpha=0 baseline -- text deadline alone, chrono off
    print("\n=== P3: alpha=0 (chrono OFF) + deadline text ===")
    for inj in model.chrono_injectors.values():
        with torch.no_grad():
            inj.alpha.zero_()
    short_lens_b = []
    long_lens_b = []
    for q in PROMPTS_NEUTRAL:
        short_p = make_prompt(q, "You have only 30 seconds, be brief.")
        long_p = make_prompt(q, "You have 1 hour, take your time.")
        sl = gen_len(model, short_p, tau_t=30.0, device=args.device)
        ll = gen_len(model, long_p, tau_t=3600.0, device=args.device)
        short_lens_b.append(sl)
        long_lens_b.append(ll)
        print(f"  Q={q[:40]:40s}  short={sl:3d}  long={ll:3d}  delta={ll-sl:+d}")
    P3 = {
        "mean_short": statistics.mean(short_lens_b),
        "mean_long": statistics.mean(long_lens_b),
        "delta": statistics.mean(long_lens_b) - statistics.mean(short_lens_b),
    }

    verdict = {
        "P1_delta_text_plus_chrono": P1["delta"],
        "P2_delta_chrono_only": P2["delta"],
        "P3_delta_text_only_chrono_off": P3["delta"],
        "chrono_contribution": P1["delta"] - P3["delta"],
        "PASS_P1_text_plus_chrono": P1["delta"] >= 5,
        "PASS_P2_chrono_alone_shortens": P2["delta"] >= 2,
        "PASS_chrono_adds_beyond_text": (P1["delta"] - P3["delta"]) >= 2,
    }
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "P1": P1, "P2": P2, "P3": P3}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
