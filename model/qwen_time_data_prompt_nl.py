"""W6/Phase-B alternative prompt baseline: natural-language time prefix.

Uses natural-language duration phrasing ('About 3 hours have passed
since we started.') instead of the bracketed '[elapsed: 3h 42m]'
format. Tests whether the prompt-baseline result depends on the
specific format.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.qwen_time_data import (
    gen_clock_conversation, gen_silent_gap_conversation, gen_phase_conversation,
)


def tau_to_nl(tau: float) -> str:
    if tau < 60:
        return f"About {tau:.0f} seconds have passed since we started."
    if tau < 3600:
        return f"About {tau/60:.0f} minutes have passed since we started."
    if tau < 86400:
        h = int(tau // 3600); m = int((tau % 3600) // 60)
        return f"About {h} hours and {m} minutes have passed since we started."
    d = int(tau // 86400); h = int((tau % 86400) // 3600)
    return f"About {d} days and {h} hours have passed since we started."


def inject_nl_in_text(record: dict) -> dict:
    tau = record.get("tau_t")
    if tau is None:
        tau = record.get("tau_b", record.get("tau", 0.0))
    prefix = tau_to_nl(float(tau))
    text = record["text"]
    marker = "<|im_start|>user\n"
    if marker in text:
        text = text.replace(marker, f"{marker}{prefix} ", 1)
    record = dict(record)
    record["text"] = text
    if "prefix_text" in record:
        record["prefix_text"] = record["prefix_text"].replace(
            marker, f"{marker}{prefix} ", 1)
    record["mode"] = record.get("mode", "?") + "+prompt_nl"
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
        out.append(inject_nl_in_text(rec))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} -> {args.out}")
    print(f"example: {out[0]['text'][:200]!r}")


if __name__ == "__main__":
    main()
