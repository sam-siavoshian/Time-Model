"""Time-aware training data for QwenTime.

Generates conversations whose CORRECT response depends on the elapsed
wall-time tau at which the model is invoked. Three task families, mixed
in the same training stream so the model learns to use chi_t for all of
them simultaneously:

  T1 CLOCK: 'how long has it been?' / 'what time is it?'
     -> assistant answers with a duration string parameterized by tau.

  T2 SILENT-GAP ACK: in a multi-turn conversation, when a user turn
     arrives after a large Delta-tau, the assistant acknowledges the
     gap before responding. After a small Delta-tau, no ack.

  T3 PHASE: same Delta-tau = 86400 seconds (1 day) presented at
     different phase positions inside a 7-day week. Assistant
     greets accordingly (weekday vs weekend).

Each record exposes (text, tau_per_chunk, answer, mode).
"""

from __future__ import annotations

import argparse
import json
import math
import math as _math
import random
from pathlib import Path


V15_DATA_MIX_DEFAULT = "0.40,0.30,0.30"


def _fmt_seconds(secs: float) -> str:
    """Grammatical duration string (handles singular vs plural)."""
    if secs < 60:
        n = int(round(secs))
        return f"{n} second{'s' if n != 1 else ''}"
    if secs < 3600:
        n = int(round(secs / 60))
        return f"{n} minute{'s' if n != 1 else ''}"
    if secs < 86400:
        n = int(round(secs / 3600))
        return f"about {n} hour{'s' if n != 1 else ''}"
    n = int(round(secs / 86400))
    return f"about {n} day{'s' if n != 1 else ''}"


def gen_clock_conversation(rng: random.Random) -> dict:
    """Single-turn: user asks how long it has been; assistant gives a
    duration string keyed to tau_t. tau is sampled LOG-UNIFORMLY in
    [1s, 7 days] -- continuous, NOT bucketed. The model has to learn
    a real duration -> string map, not a 1-of-8 lookup that would let
    template-matching pass T1 trivially.
    """
    tau = math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
    duration_str = _fmt_seconds(tau)
    user = rng.choice([
        "How long has it been since we started?",
        "What time is it now?",
        "How much time has passed?",
        "How long have we been talking?",
    ])
    assistant = f"It has been {duration_str}."
    text = (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    answer = duration_str
    return {
        "mode": "clock",
        "tau_t": float(tau),
        "text": text,
        "answer": answer,
        "prefix_text": (
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "answer_text": f"It has been {duration_str}.<|im_end|>",
    }


def gen_silent_gap_conversation(rng: random.Random) -> dict:
    """Two-turn conversation. First turn at tau=0. Second turn after a
    Delta-tau gap. If gap is large (>30min), assistant acks the gap; if
    small, no ack. delta sampled log-uniformly so the model can't memorize
    a 7-bucket map.
    """
    delta = math.exp(rng.uniform(math.log(5.0), math.log(7 * 86400.0)))
    large = delta > 1800
    first_user = rng.choice([
        "Tell me a fun fact.",
        "What's the weather usually like in Tokyo?",
        "Recommend a podcast.",
        "Explain photosynthesis briefly.",
    ])
    first_assistant = rng.choice([
        "Sure, here's one: octopuses have three hearts.",
        "Generally warm and humid in summer.",
        "Try Radiolab.",
        "Plants convert sunlight into sugar via chlorophyll.",
    ])
    second_user = rng.choice([
        "Hi again.",
        "Hey.",
        "Hello.",
        "Are you there?",
    ])
    if large:
        second_assistant = f"Welcome back, it has been {_fmt_seconds(delta)}. What can I help with?"
    else:
        second_assistant = "Hi, I am still here. What's next?"
    text = (
        f"<|im_start|>user\n{first_user}<|im_end|>\n"
        f"<|im_start|>assistant\n{first_assistant}<|im_end|>\n"
        f"<|im_start|>user\n{second_user}<|im_end|>\n"
        f"<|im_start|>assistant\n{second_assistant}<|im_end|>"
    )
    prefix = (
        f"<|im_start|>user\n{first_user}<|im_end|>\n"
        f"<|im_start|>assistant\n{first_assistant}<|im_end|>\n"
        f"<|im_start|>user\n{second_user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "mode": "silent_gap",
        "tau_t": float(delta),                                 # tau at second turn
        "delta_tau": float(delta),
        "is_large_gap": large,
        "text": text,
        "answer": second_assistant,
        "prefix_text": prefix,
        "answer_text": second_assistant + "<|im_end|>",
    }


def gen_phase_conversation(rng: random.Random, balance_weekend: bool = True) -> dict:
    """Multi-scale phase test. tau drawn UNIFORMLY across the 7-day cycle
    with fractional hours, so phase signal lives in the chi sin/cos
    components at the 604800s scale, not in an integer-day lookup.

    balance_weekend=True forces 50/50 weekday/weekend sampling. Natural
    uniform over 7 days gives 5/7 weekday vs 2/7 weekend, which makes
    the model learn 'always weekday' as a prior (T3 fail mode in
    v11/v12/v13). v14 default flips a fair coin first, then picks tau
    within the corresponding day window.
    """
    if balance_weekend:
        is_weekend_target = rng.random() < 0.5
        if is_weekend_target:
            day = rng.choice([5, 6])
        else:
            day = rng.choice([0, 1, 2, 3, 4])
        tau = day * 86400.0 + rng.uniform(0, 86400.0)
    else:
        tau = rng.uniform(0, 7 * 86400)
    day_of_week = int(tau // 86400) % 7
    is_weekend = day_of_week in (5, 6)
    user = rng.choice([
        "Good morning.",
        "Hey there.",
        "How's your day?",
    ])
    if is_weekend:
        assistant = rng.choice([
            "Happy weekend. Anything fun planned?",
            "Hope you are enjoying the weekend.",
            "Weekend mode. What is up?",
        ])
    else:
        assistant = rng.choice([
            "Hope your weekday is going well.",
            "Good day. What can I help with?",
            "Weekday vibes. What is on your list?",
        ])
    text = (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {
        "mode": "phase",
        "tau_t": tau,
        "day_of_week": day_of_week,
        "text": text,
        "answer": assistant,
        "prefix_text": f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        "answer_text": assistant + "<|im_end|>",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mix", type=str, default=V15_DATA_MIX_DEFAULT,
                   help="comma-sep mix of clock/silent_gap/phase probabilities")
    args = p.parse_args()
    rng = random.Random(args.seed)
    mix = [float(x) for x in args.mix.split(",")]
    assert abs(sum(mix) - 1.0) < 1e-3
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    counts = {"clock": 0, "silent_gap": 0, "phase": 0}
    with open(args.out, "w") as f:
        for i in range(args.n):
            r = rng.random()
            if r < mix[0]:
                rec = gen_clock_conversation(rng)
            elif r < mix[0] + mix[1]:
                rec = gen_silent_gap_conversation(rng)
            else:
                rec = gen_phase_conversation(rng)
            counts[rec["mode"]] += 1
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {args.n} -> {args.out}  counts={counts}")


if __name__ == "__main__":
    main()
