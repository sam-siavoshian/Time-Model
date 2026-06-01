"""Convert Track C records into the generic QwenTime trainer schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from eval.track_c.generate import render_model_prompt


VALID_CONDITIONS = ("hidden_only", "prompt_timestamp")


def _chat_prefix(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_records(items: Iterable[dict], *, condition: str = "hidden_only") -> list[dict]:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown Track C training condition {condition!r}")
    records: list[dict] = []
    for item in items:
        if condition == "prompt_timestamp":
            prompt = render_model_prompt(item, elapsed_seconds_text=int(item["tau_seconds"]))
            tau_t = 0.0
        else:
            prompt = render_model_prompt(item, elapsed_seconds_text=None)
            tau_t = float(item["tau_seconds"])
        prefix = _chat_prefix(prompt)
        answer_text = f"{item['gold_letter']}<|im_end|>"
        records.append({
            "mode": f"track_c_{condition}",
            "track": "track_c",
            "tau_t": tau_t,
            "prefix_text": prefix,
            "answer_text": answer_text,
            "text": prefix + answer_text,
            "answer": item["gold_letter"],
            "action": item["gold_action"],
            "family": item["family"],
            "template_id": item["template_id"],
            "condition": condition,
            "tau_seconds": item["tau_seconds"],
            "source_item_id": item["id"],
            "split": item["split"],
            "num_constraints": item["num_constraints"],
        })
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def resolve_out(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    if args.run_id:
        return Path("runs") / args.run_id / "data" / "track_c" / f"track_c_train_seed{args.seed}_{args.condition}.jsonl"
    raise SystemExit("output scope required: pass --out or --run-id")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", choices=VALID_CONDITIONS, default="hidden_only")
    args = ap.parse_args()
    records = build_records(iter_jsonl(Path(args.items)), condition=args.condition)
    out = resolve_out(args)
    n = write_jsonl(out, records)
    print(f"wrote {n} Track C trainer records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

