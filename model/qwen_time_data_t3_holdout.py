"""W8/Q9: train data that EXCLUDES Sunday from the PHASE task.
T3 evaluation then probes Sunday tau. If the chrono channel generalizes
phase information from Mon-Sat to Sun, the channel is doing real phase
work; if it cannot, T3 is supervised classification recall (not phase
discovery) and we should retract or reframe.

Identical to qwen_time_data.py except gen_phase_conversation samples
day_of_week in {0,1,2,3,4,5} only (Mon-Sat). Eval uses day 6 (Sun).
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.qwen_time_data import (
    _fmt_seconds, gen_clock_conversation, gen_silent_gap_conversation,
)


def gen_phase_conversation_no_sun(rng: random.Random) -> dict:
    """Phase task but day_of_week in {0..5} only (Mon-Sat). Sunday held out."""
    # balance weekday/weekend within Mon-Sat. weekday: {0..4}, weekend: {5}.
    is_weekend_target = rng.random() < 0.5
    if is_weekend_target:
        day = 5  # Saturday only (Sunday held out)
    else:
        day = rng.randint(0, 4)
    tau = day * 86400.0 + rng.uniform(0, 86400.0)
    day_of_week = int(tau // 86400) % 7
    is_weekend = day_of_week in (5, 6)
    user = rng.choice([
        "Good morning.", "Hi there.", "Hello!", "Hey.",
        "What is today like?",
    ])
    if is_weekend:
        assistant = rng.choice([
            "Happy weekend. Anything fun planned?",
            "Hope you are enjoying the weekend.",
            "Weekend vibes. What is on your list?",
        ])
    else:
        assistant = rng.choice([
            "Good day. What can I help with?",
            "Hope your weekday is going well.",
            "Weekday vibes. What is on your list?",
        ])
    text = (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {
        "mode": "phase_no_sun",
        "tau_t": float(tau),
        "day_of_week": day_of_week,
        "is_weekend_label": is_weekend,
        "text": text,
        "answer": assistant,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=str, default="0.40,0.30,0.30",
                    help="CLOCK,GAP,PHASE shares; PHASE excludes Sun")
    args = ap.parse_args()
    mix = [float(x) for x in args.mix.split(",")]
    rng = random.Random(args.seed)
    out = []
    for _ in range(args.n):
        r = rng.random()
        if r < mix[0]:
            out.append(gen_clock_conversation(rng))
        elif r < mix[0] + mix[1]:
            out.append(gen_silent_gap_conversation(rng))
        else:
            out.append(gen_phase_conversation_no_sun(rng))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_phase = sum(1 for r in out if r["mode"] == "phase_no_sun")
    print(f"wrote {len(out)} records ({n_phase} phase, no Sun) to {args.out}")


if __name__ == "__main__":
    main()
