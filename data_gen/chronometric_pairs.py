"""Chronometric ablation pairs for Prediction 6 eval.

Each pair: identical visible text, two different Δτ values.
- Δτ-real arm:    real elapsed seconds matching the world simulator's tick
- Δτ-ablated arm: a constant (e.g. 60s) — what the model sees if time
                  substrate is fake

Test: on duration-sensitive questions (decay, delayed transition, periodic
phase), the model's accuracy should DROP on the Δτ-ablated arm.
On duration-insensitive questions, accuracy should be UNCHANGED.

Reuses Latent World simulator + post-hoc Δτ override on the final silent gap.

Usage:
  uv run python -m data_gen.chronometric_pairs --n 10000 --out data/chronometric_pairs/pairs.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from data_gen.latent_world_sim import generate_stream
from data_gen.schemas import ChronometricPair, LatentWorldQuestion


ABLATION_DELTA_MINUTES = 1.0                                # constant for ablation arm


def generate_pair(seed: int) -> ChronometricPair:
    # Generate the underlying stream (real Δτ from simulator)
    stream = generate_stream(seed=seed, target_tokens=1024)

    # The final silent gap defines Δτ_real
    gap_events = [ev for ev in stream.events if ev.event_type == "silent_gap"]
    delta_real = gap_events[-1].delta_tau_minutes if gap_events else 0.0

    # Visible text is identical in both arms (the silent-gap text contains
    # the duration, but at inference time we strip / replace it with χ_t)
    visible = "\n".join(ev.text for ev in stream.events)

    # Pick one duration-sensitive and one duration-insensitive question
    ds_q = next((q for q in stream.questions if q.duration_sensitive), None)
    di_q = next((q for q in stream.questions if not q.duration_sensitive), None)

    # Fallback to first/last if no clean split
    if ds_q is None and stream.questions:
        ds_q = stream.questions[0]
    if di_q is None and len(stream.questions) > 1:
        di_q = stream.questions[-1]
    if ds_q is None or di_q is None:
        # extremely degenerate stream, fabricate a placeholder
        ds_q = LatentWorldQuestion(
            text="(no duration-sensitive question available)",
            answer="N/A", rationale="", duration_sensitive=True,
            question_type="current_state",
        )
        di_q = LatentWorldQuestion(
            text="(no duration-insensitive question available)",
            answer="N/A", rationale="", duration_sensitive=False,
            question_type="current_state",
        )

    return ChronometricPair(
        paired_stream_id=str(uuid4()),
        delta_tau_real_minutes=delta_real,
        delta_tau_ablated_minutes=ABLATION_DELTA_MINUTES,
        visible_text=visible,
        duration_sensitive_q=ds_q,
        duration_insensitive_q=di_q,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000, help="number of pairs")
    p.add_argument("--out", type=str, default="data/chronometric_pairs/pairs.jsonl")
    p.add_argument("--seed", type=int, default=40000000, help="base seed")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for i in tqdm(range(args.n), desc=out.name):
            pair = generate_pair(seed=args.seed + i)
            f.write(pair.model_dump_json() + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {args.n} pairs to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
