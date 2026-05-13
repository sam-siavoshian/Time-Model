"""View a Latent World stream as the model would see it (text only, no state snapshots).

Usage:
  uv run python -m data_gen.view_stream data/latent_world/smoke_test.jsonl 0
  uv run python -m data_gen.view_stream data/latent_world/smoke_test.jsonl --all
"""

import argparse
import json
from pathlib import Path


def render_stream(stream: dict) -> str:
    lines = []
    lines.append(f"=== STREAM {stream['id'][:8]} (seed={stream['seed']}) ===")
    lines.append(f"Duration: {stream['duration_minutes']:.0f} min total")
    lines.append(f"Events: {len(stream['events'])}, Questions: {len(stream['questions'])}")
    lines.append("")
    lines.append("--- TEXT STREAM (what model sees) ---")
    for ev in stream["events"]:
        prefix = "[RULE]" if ev["event_type"] == "rule_intro" else "[GAP]" if ev["event_type"] == "silent_gap" else f"[t={ev['tau_minutes']:.0f}]"
        lines.append(f"  {prefix}  {ev['text']}")
    lines.append("")
    lines.append("--- QUESTIONS ---")
    for i, q in enumerate(stream["questions"]):
        dsens = "Δτ-sensitive" if q["duration_sensitive"] else "Δτ-insensitive"
        lines.append(f"  Q{i+1} [{dsens}, {q['question_type']}]")
        lines.append(f"    text: {q['text']}")
        lines.append(f"    answer: {q['answer']}")
        lines.append(f"    rationale: {q['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str, help="JSONL path")
    p.add_argument("index", type=int, nargs="?", default=0, help="stream index (default 0)")
    p.add_argument("--all", action="store_true", help="render all streams")
    p.add_argument("--limit", type=int, default=None, help="cap when --all")
    args = p.parse_args()

    path = Path(args.path)
    streams = [json.loads(line) for line in open(path)]

    if args.all:
        n = len(streams) if args.limit is None else min(args.limit, len(streams))
        for i in range(n):
            print(render_stream(streams[i]))
    else:
        print(render_stream(streams[args.index]))


if __name__ == "__main__":
    main()
