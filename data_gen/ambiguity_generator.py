"""Memory-Biased Ambiguity Suite generator.

Generates three-phase examples:
  phase1_memory: list of facts establishing a prior (one specific "sense")
  phase2_input: short ambiguous current input (fits multiple senses)
  phase3_question: question whose correct answer depends on the prior

For each family × sense × question × ambiguous_input combo, generates many
examples by sampling 3-5 facts from the sense's fact pool. Deterministic by
seed.

Usage:
  uv run python -m data_gen.ambiguity_generator --n 100000 --out data/ambiguity/train.jsonl
  uv run python -m data_gen.ambiguity_generator --n 10000  --out data/ambiguity/valid.jsonl --seed 999999
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from data_gen.ambiguity_families import FAMILIES
from data_gen.schemas import AmbiguityExample


def generate_example(rng: random.Random) -> AmbiguityExample:
    family = rng.choice(FAMILIES)
    sense_name = rng.choice(list(family["senses"].keys()))
    sense = family["senses"][sense_name]

    # Sample 3-5 facts from sense.facts (forms the prior)
    n_facts = rng.randint(3, min(5, len(sense["facts"])))
    facts = rng.sample(sense["facts"], n_facts)

    # Pick an ambiguous input
    amb_input = rng.choice(family["ambiguous_inputs"])

    # Pick a question (whose correct answer depends on the active sense)
    q = rng.choice(family["questions"])
    correct = q["correct_by_sense"][sense_name]

    return AmbiguityExample(
        family=family["family"],
        memory_state_id=sense["memory_id_tag"],
        phase1_memory=facts,
        phase2_input=amb_input,
        phase3_question=q["text"],
        options=dict(q["options"]),                          # shallow copy
        correct_answer=correct,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100000, help="number of examples")
    p.add_argument("--out", type=str, default="data/ambiguity/train.jsonl")
    p.add_argument("--seed", type=int, default=20260512, help="base seed")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    with open(out, "w") as f:
        for _ in tqdm(range(args.n), desc=out.name):
            ex = generate_example(rng)
            f.write(ex.model_dump_json() + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {args.n} examples to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
