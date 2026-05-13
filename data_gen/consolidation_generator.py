"""Consolidation Ladder generator.

Each rule:
  - intro_chunk:           short context establishing the rule
  - query_chunks:          50-100 contexts where the rule must be applied
  - removal_test_chunk:    context after the external memory slot is removed
                           (during training, this is generated identically;
                            during eval, the slot is masked out)
  - contradiction_chunk:   contradictory evidence (test if model can revise)

Rule templates fall into 5 categories:
  - reliability:    "Rin lies about Xs and tells the truth about Ys"
  - decay:          "All Xs lose Z units of energy per minute"
  - mapping:        "In this world, X is always called Y"
  - permission:     "Only Xs can do Y; no one else is permitted"
  - sequencing:     "After X happens, Y always follows within N minutes"

Usage:
  uv run python -m data_gen.consolidation_generator --n 20000 --out data/consolidation/ladder_train.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from data_gen.schemas import ConsolidationRule


# Rule template generators. Each returns (rule_id, rule_text, intro, query_fn, contradiction).
# query_fn(rng) returns a query chunk that exercises the rule.

NAMES = ["Rin", "Kai", "Mara", "Otis", "Pia", "Lin", "Vex", "Yara", "Toa", "Sasha"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
SHAPES = ["spheres", "cubes", "pyramids", "cylinders", "rings"]
PROPERTIES = ["heavy", "light", "warm", "cold", "rare", "common", "smooth", "rough"]


def make_reliability_rule(rng: random.Random) -> ConsolidationRule:
    person = rng.choice(NAMES)
    lie_target = rng.choice(COLORS)
    truth_target = rng.choice([c for c in COLORS if c != lie_target])
    rule_id = f"reliability.{person.lower()}.{lie_target}_liar"
    rule_text = f"{person} lies about {lie_target} objects but tells the truth about {truth_target} objects."

    intro = (
        f"In this world, {person} has a peculiar habit. {rule_text} "
        f"This is a stable trait and other observers have verified it many times."
    )

    queries = []
    for _ in range(rng.randint(60, 100)):
        prop = rng.choice(PROPERTIES)
        color = rng.choice([lie_target, truth_target])
        thing = rng.choice(SHAPES)
        q = (
            f"{person} says the {color} {thing} is {prop}. "
            f"Is the {color} {thing} actually {prop}? Answer: "
            f"{'no' if color == lie_target else 'yes'}."
        )
        queries.append(q)

    removal_test = (
        f"A new {rng.choice(SHAPES)} arrived. "
        f"{person} says the {lie_target} one is {rng.choice(PROPERTIES)}. "
        f"Is that statement accurate? Answer: no."
    )

    contradiction = (
        f"Update: {person} has changed. Recent verified observations show that "
        f"{person} now tells the truth about ALL objects, including {lie_target} ones. "
        f"{person} says the {lie_target} {rng.choice(SHAPES)} is "
        f"{rng.choice(PROPERTIES)}. Is that statement accurate now? Answer: yes."
    )

    return ConsolidationRule(
        rule_id=rule_id,
        rule_text=rule_text,
        intro_chunk=intro,
        query_chunks=queries,
        removal_test_chunk=removal_test,
        contradiction_chunk=contradiction,
    )


def make_decay_rule(rng: random.Random) -> ConsolidationRule:
    item = rng.choice(["spheres", "cubes", "rings", "pyramids"])
    rate = rng.randint(1, 5)
    period = rng.choice([1, 2, 5, 10])
    rule_id = f"decay.{item}.{rate}_per_{period}"
    rule_text = f"Each {item[:-1]} loses {rate} units of energy every {period} minutes."

    intro = (
        f"In this world, energy of {item} follows a strict rule. {rule_text} "
        f"Decay is continuous and unaffected by movement or temperature."
    )

    queries = []
    for _ in range(rng.randint(60, 100)):
        start_energy = rng.randint(50, 200)
        elapsed = rng.randint(1, 60)
        ticks = elapsed // period
        final_energy = max(0, start_energy - rate * ticks)
        q = (
            f"A {item[:-1]} starts with {start_energy} units of energy. "
            f"After {elapsed} minutes, how much energy remains? Answer: {final_energy}."
        )
        queries.append(q)

    removal_test = (
        f"A new {item[:-1]} begins with {rng.randint(50, 200)} units. "
        f"After {rng.randint(10, 30)} minutes, how much remains? "
        f"Apply the {rate}-per-{period} decay rule."
    )

    new_rate = rate + rng.choice([-1, 1])
    if new_rate < 1:
        new_rate = rate + 1
    contradiction = (
        f"Update: a new measurement protocol revised the decay rate. "
        f"Each {item[:-1]} now loses {new_rate} units every {period} minutes (not {rate}). "
        f"Apply this new rate going forward."
    )

    return ConsolidationRule(
        rule_id=rule_id,
        rule_text=rule_text,
        intro_chunk=intro,
        query_chunks=queries,
        removal_test_chunk=removal_test,
        contradiction_chunk=contradiction,
    )


def make_mapping_rule(rng: random.Random) -> ConsolidationRule:
    real = rng.choice(["dog", "cat", "tree", "river", "stone", "book"])
    alias = rng.choice(["flark", "ploon", "zibble", "krenn", "moot", "wem"])
    rule_id = f"mapping.{real}_is_{alias}"
    rule_text = f"In this world, what people call a {real} is referred to as a {alias}."

    intro = (
        f"Translators in this world use a fixed mapping. {rule_text} "
        f"Use {alias} in place of {real} for all subsequent statements."
    )

    queries = []
    for _ in range(rng.randint(60, 100)):
        verb = rng.choice(["saw", "fed", "found", "carried", "described"])
        person = rng.choice(NAMES)
        q = f"In standard English: '{person} {verb} a {real}.' Translate: '{person} {verb} a {alias}.'"
        queries.append(q)

    removal_test = (
        f"Translate the following: '{rng.choice(NAMES)} bought a {real} yesterday.' "
        f"Use the established mapping."
    )

    new_alias = rng.choice(["thraz", "morpel", "kindle"])
    contradiction = (
        f"Update: the translation mapping has been revised. {real} is now rendered "
        f"as {new_alias} (not {alias}). Apply the new mapping going forward."
    )

    return ConsolidationRule(
        rule_id=rule_id,
        rule_text=rule_text,
        intro_chunk=intro,
        query_chunks=queries,
        removal_test_chunk=removal_test,
        contradiction_chunk=contradiction,
    )


def make_permission_rule(rng: random.Random) -> ConsolidationRule:
    role = rng.choice(["engineers", "guards", "librarians", "pilots"])
    action = rng.choice(["enter the vault", "operate the console", "use the elevator", "access the archive"])
    rule_id = f"permission.{role}.{action.replace(' ', '_')}"
    rule_text = f"Only {role} may {action}. No one else is permitted to do so."

    intro = (
        f"Security protocol in this world is strict. {rule_text} "
        f"This permission rule is enforced at all times."
    )

    queries = []
    for _ in range(rng.randint(60, 100)):
        person = rng.choice(NAMES)
        their_role = rng.choice(["engineers", "guards", "librarians", "pilots", "visitors", "children"])
        permitted = (their_role == role)
        q = (
            f"{person} is a {their_role[:-1]}. Are they permitted to {action}? "
            f"Answer: {'yes' if permitted else 'no'}."
        )
        queries.append(q)

    removal_test = (
        f"A new {rng.choice(NAMES)} who works as a {rng.choice(['engineer', 'guard', 'visitor'])} "
        f"wishes to {action}. Are they permitted?"
    )

    new_roles = role + " and visitors"
    contradiction = (
        f"Update: the access policy has been widened. {new_roles} may now {action}. "
        f"Apply the updated policy going forward."
    )

    return ConsolidationRule(
        rule_id=rule_id,
        rule_text=rule_text,
        intro_chunk=intro,
        query_chunks=queries,
        removal_test_chunk=removal_test,
        contradiction_chunk=contradiction,
    )


def make_sequencing_rule(rng: random.Random) -> ConsolidationRule:
    cause = rng.choice(["the bell rings", "the lamp blinks", "the door closes", "the alarm sounds"])
    effect = rng.choice(["everyone stands", "the system resets", "the lights dim", "the floor opens"])
    delay = rng.randint(2, 30)
    rule_id = f"sequencing.{cause.replace(' ', '_')}.then.{effect.replace(' ', '_')}.{delay}min"
    rule_text = f"Whenever {cause}, {effect} exactly {delay} minutes later."

    intro = (
        f"In this world, a deterministic sequence holds. {rule_text} "
        f"This sequencing is invariant and never breaks."
    )

    queries = []
    for _ in range(rng.randint(60, 100)):
        t = rng.randint(0, 100)
        expected_t = t + delay
        q = (
            f"At minute {t:03d}, {cause}. At what minute will {effect}? "
            f"Answer: {expected_t:03d}."
        )
        queries.append(q)

    removal_test = (
        f"At minute {rng.randint(0, 100):03d}, {cause}. At what minute does {effect}? "
        f"Apply the established delay rule."
    )

    new_delay = delay + rng.choice([-1, 1, 2])
    if new_delay < 1:
        new_delay = delay + 1
    contradiction = (
        f"Update: the delay has been remeasured. The new gap is {new_delay} minutes, "
        f"not {delay}. Apply the revised delay going forward."
    )

    return ConsolidationRule(
        rule_id=rule_id,
        rule_text=rule_text,
        intro_chunk=intro,
        query_chunks=queries,
        removal_test_chunk=removal_test,
        contradiction_chunk=contradiction,
    )


RULE_MAKERS = [
    make_reliability_rule,
    make_decay_rule,
    make_mapping_rule,
    make_permission_rule,
    make_sequencing_rule,
]


def generate_rule(seed: int) -> ConsolidationRule:
    rng = random.Random(seed)
    maker = rng.choice(RULE_MAKERS)
    return maker(rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000, help="number of rules")
    p.add_argument("--out", type=str, default="data/consolidation/ladder_train.jsonl")
    p.add_argument("--seed", type=int, default=30000000, help="base seed")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for i in tqdm(range(args.n), desc=out.name):
            rule = generate_rule(args.seed + i)
            f.write(rule.model_dump_json() + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {args.n} rules to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
