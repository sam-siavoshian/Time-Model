"""W6/Phase-B alternative prompt baseline: ISO timestamp prefix.

Same as qwen_time_data_prompt but uses ISO-style timestamp in prefix
('[timestamp: 2024-01-15T14:42:00Z]' computed from a reference epoch +
tau seconds) instead of the relative-duration '[elapsed: 3h 42m]'.
Tests whether absolute-timestamp prompt format helps or hurts vs.
the relative-duration prefix.
"""
from __future__ import annotations
import argparse, json, math, random, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.qwen_time_data import (
    gen_clock_conversation, gen_silent_gap_conversation, gen_phase_conversation,
)

REFERENCE_EPOCH = datetime.datetime(2024, 1, 15, 14, 42, 0, tzinfo=datetime.timezone.utc)


def tau_to_iso(tau: float) -> str:
    t = REFERENCE_EPOCH + datetime.timedelta(seconds=tau)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def inject_iso_in_text(record: dict) -> dict:
    tau = record.get("tau_t")
    if tau is None:
        tau = record.get("tau_b", record.get("tau", 0.0))
    prefix = f"[timestamp: {tau_to_iso(float(tau))}]"
    text = record["text"]
    marker = "<|im_start|>user\n"
    if marker in text:
        text = text.replace(marker, f"{marker}{prefix} ", 1)
    record = dict(record)
    record["text"] = text
    if "prefix_text" in record:
        record["prefix_text"] = record["prefix_text"].replace(
            marker, f"{marker}{prefix} ", 1)
    record["mode"] = record.get("mode", "?") + "+prompt_iso"
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=str, default="0.40,0.30,0.30")
    args = ap.parse_args()
    mix = [float(x) for x in args.mix.split(",")]
    assert abs(sum(mix) - 1.0) < 1e-3
    rng = random.Random(args.seed)
    out = []
    for _ in range(args.n):
        r = rng.random()
        if r < mix[0]: rec = gen_clock_conversation(rng)
        elif r < mix[0]+mix[1]: rec = gen_silent_gap_conversation(rng)
        else: rec = gen_phase_conversation(rng)
        out.append(inject_iso_in_text(rec))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} -> {args.out}")
    print(f"example: {out[0]['text'][:200]!r}")


if __name__ == "__main__":
    main()
