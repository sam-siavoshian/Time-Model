"""W6 baseline: same training distribution as qwen_time_data.py but with
tau injected as text in the user prompt instead of via the chrono channel.

The user message gains a "[elapsed: X seconds] " prefix where X is the
same continuous tau used by the chrono encoder in the standard CI training.
Otherwise identical: same 3 task families (CLOCK/SILENT-GAP/PHASE), same
log-uniform tau distribution, same response strings.

Pair with training run that has --freeze-alpha so the chrono channel is
forced off and tau information can only enter via the prompt text. This
is the natural baseline the reviewer asked for in W6.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

# import existing generators
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.qwen_time_data import (
    _fmt_seconds, gen_clock_conversation, gen_silent_gap_conversation,
    gen_phase_conversation,
)


def tau_text(tau: float) -> str:
    """Format tau as a compact text prefix.
    e.g. 79347 -> '[elapsed: 22h 4m]', 5.5 -> '[elapsed: 5.5s]'.
    """
    if tau < 60:
        return f"[elapsed: {tau:.1f}s]"
    if tau < 3600:
        m = tau / 60
        return f"[elapsed: {m:.1f}m]"
    if tau < 86400:
        h = int(tau // 3600); m = int((tau % 3600) // 60)
        return f"[elapsed: {h}h {m}m]"
    d = int(tau // 86400); h = int((tau % 86400) // 3600)
    return f"[elapsed: {d}d {h}h]"


def inject_tau_in_text(record: dict) -> dict:
    """Take a record from one of the existing generators and rewrite its
    text so the user prompt starts with a [elapsed: ...] prefix encoding
    the canonical tau the chrono channel would have used.
    """
    tau = record.get("tau_t")
    if tau is None:
        # silent-gap uses tau_b (gap)
        tau = record.get("tau_b", record.get("tau", 0.0))
    prefix = tau_text(float(tau))
    text = record["text"]
    # find first <|im_start|>user\n and inject prefix after newline
    marker = "<|im_start|>user\n"
    if marker in text:
        text = text.replace(marker, f"{marker}{prefix} ", 1)
    record = dict(record)
    record["text"] = text
    if "prefix_text" in record:
        record["prefix_text"] = record["prefix_text"].replace(
            marker, f"{marker}{prefix} ", 1)
    record["mode"] = record.get("mode", "?") + "+prompt_tau"
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=str, default="0.40,0.30,0.30",
                    help="CLOCK,GAP,PHASE shares; same as v15")
    args = ap.parse_args()

    mix = [float(x) for x in args.mix.split(",")]
    assert abs(sum(mix) - 1.0) < 1e-3
    rng = random.Random(args.seed)
    out = []
    for _ in range(args.n):
        r = rng.random()
        if r < mix[0]:
            rec = gen_clock_conversation(rng)
        elif r < mix[0] + mix[1]:
            rec = gen_silent_gap_conversation(rng)
        else:
            rec = gen_phase_conversation(rng)
        out.append(inject_tau_in_text(rec))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} records to {args.out}")
    print(f"example: {out[0]['text'][:200]!r}")


if __name__ == "__main__":
    main()
