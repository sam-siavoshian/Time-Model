"""Contradiction pairs for Prediction 7 eval.

For each ambiguity family with at least 2 senses, generate pairs that share:
  - same family
  - same ambiguous current input
  - same question
But differ in:
  - memory state (memory_A_facts vs memory_B_facts)
  - correct answer (correct_under_memory_a vs correct_under_memory_b)
Plus:
  - explicit_input: a variant of the current input that resolves the ambiguity,
                   so the correct answer is fixed regardless of memory

Eval Prediction 7:
  - With ambiguous_input: KL(p(y | input, M_A) || p(y | input, M_B)) >= 0.5
  - With explicit_input:  KL(p(y | input, M_A) || p(y | input, M_B)) <= 0.1
  (memory should bend interpretation on ambiguous, but be overridden by explicit)

Usage:
  uv run python -m data_gen.contradiction_pairs --n 5000 --out data/contradiction_pairs/pairs.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from data_gen.ambiguity_families import FAMILIES
from data_gen.schemas import ContradictionPair


# Explicit-input templates per family (resolves the ambiguity directly)
EXPLICIT_TEMPLATES = {
    "bat_animal_vs_sport": {
        "animal": "He saw the bat — a furry winged mammal — near the entrance.",
        "sport": "He saw the bat — a wooden baseball club — near the entrance.",
    },
    "bank_river_vs_financial": {
        "river": "She walked toward the bank of the river.",
        "financial": "She walked toward the bank to deposit a check.",
    },
    "crane_bird_vs_machine": {
        "bird": "The crane, a tall wading bird, moved slowly across the field.",
        "machine": "The construction crane moved slowly across the field.",
    },
    "mip_animal_vs_vehicle": {
        "animal": "The child saw a mip — a small winged forest animal — near the cave entrance.",
        "vehicle": "The child saw a mip — a small mining vehicle — near the cave entrance.",
    },
    "rin_social_reliability": {
        # For Rin, explicit context simply restates the truth verifiably
        "blue_liar": (
            "Verified independently: the blue ball is light, the red book is light. "
            "Now Rin said the blue ball is heavy and the red book is light."
        ),
        "red_liar": (
            "Verified independently: the blue ball is heavy, the red book is heavy. "
            "Now Rin said the blue ball is heavy and the red book is light."
        ),
    },
}


def generate_pair(rng: random.Random) -> ContradictionPair | None:
    family = rng.choice(FAMILIES)
    senses = list(family["senses"].keys())
    if len(senses) < 2:
        return None
    sense_a, sense_b = rng.sample(senses, 2)

    facts_a = rng.sample(family["senses"][sense_a]["facts"], min(4, len(family["senses"][sense_a]["facts"])))
    facts_b = rng.sample(family["senses"][sense_b]["facts"], min(4, len(family["senses"][sense_b]["facts"])))

    amb_input = rng.choice(family["ambiguous_inputs"])
    explicit_input = EXPLICIT_TEMPLATES.get(family["family"], {}).get(sense_a, amb_input)
    q = rng.choice(family["questions"])
    correct_a = q["correct_by_sense"][sense_a]
    correct_b = q["correct_by_sense"][sense_b]
    # Under explicit input, correct = whichever sense the explicit text resolves to (sense_a here)
    correct_explicit = correct_a

    return ContradictionPair(
        memory_state_a_id=family["senses"][sense_a]["memory_id_tag"],
        memory_state_b_id=family["senses"][sense_b]["memory_id_tag"],
        memory_a_facts=facts_a,
        memory_b_facts=facts_b,
        ambiguous_input=amb_input,
        explicit_input=explicit_input,
        question=q["text"],
        options=dict(q["options"]),
        correct_under_memory_a=correct_a,
        correct_under_memory_b=correct_b,
        correct_under_explicit=correct_explicit,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="number of pairs")
    p.add_argument("--out", type=str, default="data/contradiction_pairs/pairs.jsonl")
    p.add_argument("--seed", type=int, default=50000000, help="base seed")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written = 0
    with open(out, "w") as f:
        bar = tqdm(total=args.n, desc=out.name)
        while written < args.n:
            pair = generate_pair(rng)
            if pair is None:
                continue
            f.write(pair.model_dump_json() + "\n")
            written += 1
            bar.update(1)
        bar.close()

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {written} pairs to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
