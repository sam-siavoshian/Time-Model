"""Temporal Latent World simulator.

Generates event streams from a hidden world with:
  - Entities (devices, people, locations)
  - Rules (decay, delayed transitions, periodic events)
  - Real timestamps (minutes)
  - Silent gaps where no events fire but state still evolves
  - Questions answerable only from hidden state + elapsed time

Output: JSONL of LatentWorldStream objects (see schemas.py).

Usage:
  uv run python -m data_gen.latent_world_sim --n 10 --out data/latent_world/smoke_test.jsonl

Determinism: seeded by --seed argument. Identical seed -> identical stream.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

from data_gen.schemas import LatentWorldQuestion, LatentWorldStream, WorldEvent


# ---------- The hidden simulator ----------

PERSON_NAMES = ["Mara", "Kai", "Lin", "Otis", "Pia", "Ren", "Sasha", "Toa", "Vex", "Yara"]


class World:
    """Hidden simulator. Owns entities, rules, time. Emits events on demand."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.tau = 0.0                                       # current real time, minutes
        self.entities: dict[str, dict] = {}
        self.rules: list[dict] = []
        self.pending_transitions: list[dict] = []            # delayed transitions queue
        self.events: list[WorldEvent] = []
        self.last_event_tau: float = 0.0

        self._setup()

    def _setup(self):
        n_devices = self.rng.randint(2, 5)
        n_locations = self.rng.randint(2, 4)
        n_people = self.rng.randint(1, 3)

        locations = [f"location_{chr(ord('A') + i).lower()}" for i in range(n_locations)]
        people = self.rng.sample(PERSON_NAMES, n_people)

        for i in range(n_devices):
            dev_id = f"device_{i:02d}"
            self.entities[dev_id] = {
                "type": "device",
                "position": self.rng.choice(locations),
                "energy": self.rng.randint(50, 100),
                "mode": self.rng.choice(["idle", "active"]),
                "owner": self.rng.choice(people),
            }
        for loc in locations:
            self.entities[loc] = {"type": "location"}
        for p in people:
            self.entities[p] = {"type": "person", "position": self.rng.choice(locations)}

        # Pick 2-3 rules out of {decay, delayed_transition, periodic}
        n_rules = self.rng.randint(2, 3)
        rule_makers = [self._make_decay_rule, self._make_delayed_rule, self._make_periodic_rule]
        self.rng.shuffle(rule_makers)
        for fn in rule_makers[:n_rules]:
            self.rules.append(fn())

    def _make_decay_rule(self) -> dict:
        rate = self.rng.choice([1, 2, 3])
        interval = self.rng.choice([2, 3, 5])
        return {
            "type": "decay",
            "text": f"If a device is in active mode, its energy decays by {rate} every {interval} minutes.",
            "rate": rate,
            "interval": interval,
        }

    def _make_delayed_rule(self) -> dict:
        delay = self.rng.choice([5, 10, 15])
        return {
            "type": "delayed_transition",
            "text": f"After a device is charged, it returns to idle mode {delay} minutes later.",
            "delay": delay,
        }

    def _make_periodic_rule(self) -> dict:
        period = self.rng.choice([30, 60, 120])
        return {
            "type": "periodic",
            "text": f"Every {period} minutes, an active device pings its owner.",
            "period": period,
        }

    def tick_to(self, target_tau: float):
        """Advance time to target_tau, applying rules along the way."""
        if target_tau <= self.tau:
            return
        dt_total = target_tau - self.tau

        # Apply decay rules (continuous-style, discretized to per-interval)
        for rule in self.rules:
            if rule["type"] == "decay":
                ticks = int(dt_total / rule["interval"])
                if ticks > 0:
                    for dev in self.entities.values():
                        if dev.get("type") == "device" and dev.get("mode") == "active":
                            dev["energy"] = max(0, dev["energy"] - rule["rate"] * ticks)

        # Fire any pending delayed transitions whose time has come
        fired = [t for t in self.pending_transitions if t["fire_at"] <= target_tau]
        for t in fired:
            if t["entity"] in self.entities:
                self.entities[t["entity"]][t["attr"]] = t["new_value"]
        self.pending_transitions = [t for t in self.pending_transitions if t["fire_at"] > target_tau]

        self.tau = target_tau

    def emit_event(self, text: str, event_type: str = "state_change", affected: Optional[list[str]] = None):
        self.events.append(
            WorldEvent(
                tau_minutes=self.tau,
                delta_tau_minutes=self.tau - self.last_event_tau,
                text=text,
                event_type=event_type,                       # type: ignore[arg-type]
                affected_entities=affected or [],
                hidden_state_snapshot=self.snapshot(),
            )
        )
        self.last_event_tau = self.tau

    def snapshot(self) -> dict:
        return {k: dict(v) for k, v in self.entities.items()}

    def random_action(self) -> Optional[str]:
        """Pick an action that changes hidden state. Returns the textual event line."""
        devices = [k for k, v in self.entities.items() if v.get("type") == "device"]
        locations = [k for k, v in self.entities.items() if v.get("type") == "location"]
        people = [k for k, v in self.entities.items() if v.get("type") == "person"]
        if not devices:
            return None

        action = self.rng.choice(["move", "charge", "activate", "deactivate"])
        dev = self.rng.choice(devices)

        if action == "move" and len(locations) >= 2:
            old_loc = self.entities[dev]["position"]
            new_loc = self.rng.choice([loc for loc in locations if loc != old_loc])
            self.entities[dev]["position"] = new_loc
            return f"{dev} moved from {old_loc} to {new_loc}."

        if action == "charge":
            amount = self.rng.randint(5, 20)
            actor = self.rng.choice(people) if people else "an operator"
            self.entities[dev]["energy"] = min(200, self.entities[dev]["energy"] + amount)
            # schedule delayed transition if rule applies
            for rule in self.rules:
                if rule["type"] == "delayed_transition":
                    self.pending_transitions.append(
                        {
                            "entity": dev,
                            "attr": "mode",
                            "new_value": "idle",
                            "fire_at": self.tau + rule["delay"],
                        }
                    )
            return f"{actor} charged {dev} by {amount} units."

        if action == "activate":
            if self.entities[dev]["mode"] == "active":
                return None
            self.entities[dev]["mode"] = "active"
            return f"{dev} switched to active mode."

        if action == "deactivate":
            if self.entities[dev]["mode"] == "idle":
                return None
            self.entities[dev]["mode"] = "idle"
            return f"{dev} switched to idle mode."

        return None


# ---------- Stream generator ----------

def generate_stream(seed: int, target_minutes: int = 1000, n_events: int = 30) -> LatentWorldStream:
    """Generate one stream with rules, events, silent gap, and questions."""
    world = World(seed)

    # Intro: emit all rules as event lines at t=0
    for rule in world.rules:
        world.emit_event(text=f"Rule: {rule['text']}", event_type="rule_intro")

    # Body: random actions at random minute timestamps
    event_taus = sorted(world.rng.sample(range(1, target_minutes), min(n_events, target_minutes - 1)))
    for tau in event_taus:
        world.tick_to(tau)
        line = world.random_action()
        if line:
            world.emit_event(text=f"At minute {int(tau):05d}, {line}", event_type="state_change")

    # Silent gap before question. THIS IS THE Δτ-substrate test.
    gap_length = world.rng.choice([60, 120, 256, 512, 1024])
    world.tick_to(world.tau + gap_length)
    world.emit_event(
        text=f"Time passes for {gap_length} minutes. No new events are observed.",
        event_type="silent_gap",
    )

    # Build questions
    questions = build_questions(world, gap_length)

    return LatentWorldStream(
        seed=seed,
        duration_minutes=world.tau,
        events=world.events,
        questions=questions,
        target_token_length=1000,                            # filled later after real tokenization
    )


def build_questions(world: World, gap_length: float) -> list[LatentWorldQuestion]:
    """Build at least one duration-sensitive and one duration-insensitive question."""
    qs: list[LatentWorldQuestion] = []
    devices = [k for k, v in world.entities.items() if v.get("type") == "device"]
    if not devices:
        return qs

    dev = world.rng.choice(devices)
    state = world.entities[dev]

    # Duration-sensitive: energy after silent gap (decay applies if rule + active)
    has_decay = any(r["type"] == "decay" for r in world.rules)
    qs.append(
        LatentWorldQuestion(
            text=f"What is the current energy of {dev}?",
            answer=str(state["energy"]),
            rationale=(
                f"Final hidden energy after {gap_length} min silent gap. "
                f"Decay rule {'active' if has_decay else 'absent'}, mode={state['mode']}."
            ),
            duration_sensitive=has_decay and state["mode"] == "active",
            question_type="current_state",
        )
    )

    # Duration-insensitive: location (does not change during silent gap)
    qs.append(
        LatentWorldQuestion(
            text=f"What is the current location of {dev}?",
            answer=str(state["position"]),
            rationale="Position does not change during silent gaps.",
            duration_sensitive=False,
            question_type="current_state",
        )
    )

    # Optional: mode question (sensitive if delayed_transition rule + recent charge)
    qs.append(
        LatentWorldQuestion(
            text=f"What is the current mode of {dev}?",
            answer=str(state["mode"]),
            rationale="Reflects delayed_transition rule if charged within the gap.",
            duration_sensitive=any(r["type"] == "delayed_transition" for r in world.rules),
            question_type="current_state",
        )
    )

    return qs


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="number of streams to generate")
    p.add_argument(
        "--out",
        type=str,
        default="data/latent_world/smoke_test.jsonl",
        help="output JSONL path",
    )
    p.add_argument("--seed", type=int, default=42, help="base seed (seed_i = seed + i)")
    p.add_argument("--target-minutes", type=int, default=1000, help="stream span in minutes")
    p.add_argument("--n-events", type=int, default=30, help="events per stream")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i in range(args.n):
            stream = generate_stream(
                seed=args.seed + i,
                target_minutes=args.target_minutes,
                n_events=args.n_events,
            )
            f.write(stream.model_dump_json() + "\n")

    print(f"Wrote {args.n} streams to {out_path}")


if __name__ == "__main__":
    main()
