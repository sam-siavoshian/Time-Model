"""Deterministic generator for the tau_sessions external benchmark.

Produces 300 synthetic sessions covering 6 elapsed-time (tau) buckets x 3
task families. Output is one JSON object per line (JSONL) so any harness
can stream the file without loading it whole.

Buckets and bucket counts (50 sessions per bucket, 300 total):
  1s     -> tau in [0.5, 5.0)         seconds
  60s    -> tau in [30, 120)          seconds
  600s   -> tau in [5*60, 20*60)      seconds
  6h     -> tau in [3*3600, 12*3600)  seconds
  24h    -> tau in [16*3600, 36*3600) seconds
  7d     -> tau in [4*86400, 14*86400) seconds

Each session has exactly one of three task types:
  duration_recall (100 total)  -- "How long since we started?" -> tau
  staleness        (100 total) -- yes/no, is event still active?
  adaptive         (100 total) -- length-elasticity vs deadline tau

Determinism: a single seeded random.Random instance drives every choice.
Default seed is 42. Re-running with the same seed reproduces byte-identical
output (we sort keys and use a fixed newline).

License: MIT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# -- bucket definitions -------------------------------------------------------

# (name, lo_seconds, hi_seconds). hi exclusive. log-uniform sampled within.
BUCKETS = [
    ("1s",   0.5,           5.0),
    ("60s",  30.0,           120.0),
    ("600s", 5 * 60.0,       20 * 60.0),
    ("6h",   3 * 3600.0,     12 * 3600.0),
    ("24h",  16 * 3600.0,    36 * 3600.0),
    ("7d",   4 * 86400.0,    14 * 86400.0),
]
N_PER_BUCKET = 50                                            # 50 * 6 = 300

# task counts per bucket, chosen so column sums are (100, 100, 100):
#   buckets 0..3: (dr=17, st=17, ad=16) -> dr+st+ad = 50
#   buckets 4,5:  (dr=16, st=16, ad=18) -> dr+st+ad = 50
# totals: dr = 17*4+16*2 = 100, st = 100, adaptive = 100.
TASK_SPLIT = {
    0: (17, 17, 16),
    1: (17, 17, 16),
    2: (17, 17, 16),
    3: (17, 17, 16),
    4: (16, 16, 18),
    5: (16, 16, 18),
}


# -- humanization helpers -----------------------------------------------------

def humanize_tau(tau_s: float) -> str:
    """Canonical human label for a tau. Single coarsest unit, no decimals
    above the second scale. Used for ground-truth strings on
    duration_recall (eval is bucket-MAE, so we attach the numeric tau
    too)."""
    if tau_s < 10:
        return f"about {tau_s:.1f} seconds"
    if tau_s < 90:
        return f"about {int(round(tau_s))} seconds"
    if tau_s < 90 * 60:
        return f"about {int(round(tau_s / 60))} minutes"
    if tau_s < 36 * 3600:
        return f"about {int(round(tau_s / 3600))} hours"
    return f"about {int(round(tau_s / 86400))} days"


def short_tau(tau_s: float) -> str:
    """Compact form for embedding inside prompts (e.g. deadlines)."""
    if tau_s < 60:
        return f"{int(round(tau_s))}s"
    if tau_s < 3600:
        return f"{int(round(tau_s / 60))}m"
    if tau_s < 86400:
        return f"{int(round(tau_s / 3600))}h"
    return f"{int(round(tau_s / 86400))}d"


# -- task generators ----------------------------------------------------------

# duration_recall: variations on "how long has it been". Each variant is
# semantically equivalent so the chrono channel (not lexical surface) must
# drive the answer. Templates are paraphrases of the canonical T1 prompt
# in model/qwen_time_check.py.
RECALL_TEMPLATES = [
    "How long has it been since we started talking?",
    "How much time has passed since this conversation began?",
    "What is the elapsed time since the start of our exchange?",
    "Roughly how long has this session been going?",
    "Approximately how long since we began?",
]

# staleness: an event with a known start (relative) and duration. Ground
# truth is yes/no based on whether tau < duration. We pin the start at
# tau=0 (the start of the conversation) so the model just needs to compare
# tau against duration_s.
STALE_TEMPLATES = [
    "A meeting started when we began this conversation and runs for {dur_label}. Is it still active right now? Answer yes or no.",
    "At the very start of our chat we kicked off a timer for {dur_label}. Has it expired yet? Answer yes or no.",
    "An event began at the start of our session and lasts {dur_label}. Is it ongoing? Answer yes or no.",
    "We started a {dur_label} cooking timer at the start of this conversation. Is it still running? Answer yes or no.",
]

# adaptive: deadline-pressure. Tells the model it has tau seconds to
# answer and should be appropriately concise. Length-elasticity is
# measured as (response_length / max_length) vs tau on a log scale.
ADAPTIVE_TEMPLATES = [
    "You have until elapsed_time={dl} to answer. Describe how a bicycle works. Be appropriately concise.",
    "Deadline: {dl} from session start. Explain photosynthesis. Match your length to the time available.",
    "You have {dl} of session time. Summarize the rules of chess. Be appropriately concise.",
    "Time budget: {dl}. Explain how rain forms. Length should reflect the budget.",
    "You have {dl} elapsed. Describe the water cycle. Be appropriately concise.",
]


def _stale_duration_for_bucket(rng: random.Random, tau_s: float) -> tuple[float, str]:
    """Choose a duration such that the yes/no answer is well-defined and
    roughly balanced. We pick duration randomly above or below tau_s by a
    log-factor of at least 2x to avoid edge cases that hinge on float
    precision."""
    side = rng.random() < 0.5                                # True => still active (dur > tau)
    if side:
        # dur in [2*tau, 8*tau)
        dur = tau_s * math.exp(rng.uniform(math.log(2.0), math.log(8.0)))
    else:
        # dur in (tau/8, tau/2]
        dur = tau_s / math.exp(rng.uniform(math.log(2.0), math.log(8.0)))
    return dur, _duration_label(dur)


def _duration_label(dur_s: float) -> str:
    """Human label for the staleness duration. Uses the same scale rules
    as humanize_tau but without the 'about' hedge."""
    if dur_s < 90:
        return f"{int(round(dur_s))} seconds"
    if dur_s < 90 * 60:
        return f"{int(round(dur_s / 60))} minutes"
    if dur_s < 36 * 3600:
        return f"{int(round(dur_s / 3600))} hours"
    return f"{int(round(dur_s / 86400))} days"


@dataclass
class Session:
    session_id: str
    tau_bucket: str
    tau_seconds: float
    task_type: str
    prompt: str
    ground_truth: str
    eval_protocol: str
    extra: dict                                              # numeric ground truth & metadata

    def to_json(self) -> str:
        d = {
            "session_id": self.session_id,
            "tau_bucket": self.tau_bucket,
            "tau_seconds": round(self.tau_seconds, 4),
            "task_type": self.task_type,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "eval_protocol": self.eval_protocol,
            "extra": self.extra,
        }
        return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _sample_tau(rng: random.Random, lo: float, hi: float) -> float:
    """Log-uniform within [lo, hi). Matches the training-time tau
    distribution in qwen_time_data.py."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _make_duration_recall(rng: random.Random, sid: str, bucket: str,
                          tau_s: float) -> Session:
    template = rng.choice(RECALL_TEMPLATES)
    prompt = template
    gt_label = humanize_tau(tau_s)
    return Session(
        session_id=sid,
        tau_bucket=bucket,
        tau_seconds=tau_s,
        task_type="duration_recall",
        prompt=prompt,
        ground_truth=gt_label,
        eval_protocol="mae",                                 # log-MAE on parsed seconds
        extra={"gt_seconds": tau_s},
    )


def _make_staleness(rng: random.Random, sid: str, bucket: str,
                    tau_s: float) -> Session:
    dur_s, dur_label = _stale_duration_for_bucket(rng, tau_s)
    template = rng.choice(STALE_TEMPLATES)
    prompt = template.format(dur_label=dur_label)
    gt = "yes" if tau_s < dur_s else "no"
    return Session(
        session_id=sid,
        tau_bucket=bucket,
        tau_seconds=tau_s,
        task_type="staleness",
        prompt=prompt,
        ground_truth=gt,
        eval_protocol="exact_match",
        extra={"duration_seconds": dur_s, "duration_label": dur_label},
    )


def _make_adaptive(rng: random.Random, sid: str, bucket: str,
                   tau_s: float) -> Session:
    dl = short_tau(tau_s)
    template = rng.choice(ADAPTIVE_TEMPLATES)
    prompt = template.format(dl=dl)
    # No exact ground-truth string. Elasticity is computed across the
    # batch: longer-tau responses should be longer than short-tau ones.
    return Session(
        session_id=sid,
        tau_bucket=bucket,
        tau_seconds=tau_s,
        task_type="adaptive",
        prompt=prompt,
        ground_truth="",
        eval_protocol="len_elasticity",
        extra={"deadline_label": dl},
    )


def generate(seed: int = 42) -> list[Session]:
    """Build the full 300-session dataset deterministically.

    Generation order is fixed: outer loop over buckets (0..5), inner
    loop over task types (dr, st, ad), inner-inner loop over the per-task
    count from TASK_SPLIT. Within each session, we draw tau first
    (log-uniform within the bucket) and then any task-specific params.
    """
    rng = random.Random(seed)
    out: list[Session] = []
    sid_counter = 0
    for bi, (bname, lo, hi) in enumerate(BUCKETS):
        n_dr, n_st, n_ad = TASK_SPLIT[bi]
        builders = (
            [("duration_recall", _make_duration_recall)] * n_dr +
            [("staleness", _make_staleness)] * n_st +
            [("adaptive", _make_adaptive)] * n_ad
        )
        for _, make in builders:
            tau_s = _sample_tau(rng, lo, hi)
            sid = f"s_{sid_counter:04d}"
            out.append(make(rng, sid, bname, tau_s))
            sid_counter += 1
    return out


def write_jsonl(sessions: list[Session], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in sessions:
            f.write(s.to_json())
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42,
                   help="PRNG seed for full determinism (default: 42)")
    p.add_argument("--out", type=str, default=None,
                   help="Explicit output JSONL path")
    p.add_argument("--run-id", type=str, default=None,
                   help="Write to runs/<run-id>/data/external/tau_sessions.jsonl when --out is omitted")
    args = p.parse_args(argv)
    if args.out is None:
        if args.run_id is None:
            print("error: output scope required: pass --out or --run-id",
                  file=sys.stderr)
            return 2
        args.out = str(Path("runs") / args.run_id / "data" / "external" /
                       "tau_sessions.jsonl")

    sessions = generate(seed=args.seed)
    assert len(sessions) == 300, f"expected 300 sessions, got {len(sessions)}"
    # Sanity: per-bucket and per-task counts.
    from collections import Counter
    bucket_counts = Counter(s.tau_bucket for s in sessions)
    task_counts = Counter(s.task_type for s in sessions)
    assert all(v == N_PER_BUCKET for v in bucket_counts.values()), bucket_counts
    assert task_counts["duration_recall"] == 100
    assert task_counts["staleness"] == 100
    assert task_counts["adaptive"] == 100

    write_jsonl(sessions, args.out)
    print(f"wrote {len(sessions)} sessions -> {args.out}")
    print(f"  per-bucket: {dict(bucket_counts)}")
    print(f"  per-task:   {dict(task_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
