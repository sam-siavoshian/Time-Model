"""Track B TPS policy-training data.

Converts TPS benchmark items into the generic QwenTime trainer schema:
``mode``, ``tau_t``, ``prefix_text``, ``answer_text``, and ``text``.

The default split is intentionally narrow: hidden-only TPS policy labels,
training templates only, and no held-out family. This trains the chrono channel
on downstream action selection without mixing in Track A clock/silent/phase
records or visible timestamp text.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from eval.tps.benchmark import ACTION_TO_LETTER, Item, held_out_family, held_out_template, iter_items


DEFAULT_CONDITIONS = ("hidden_only",)
VALID_SPLITS = ("train", "eval", "all")


def _chat_prefix(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _include_item(item: Item, split: str, conditions: set[str]) -> bool:
    if item.condition not in conditions:
        return False
    is_heldout = held_out_template(item.template_idx) or held_out_family(item.family)
    if split == "train":
        return not is_heldout
    if split == "eval":
        return is_heldout
    if split == "all":
        return True
    raise ValueError(f"unknown split {split!r}")


def build_records(
    *,
    split: str = "train",
    seed: int = 0,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
) -> list[dict]:
    """Return deterministic TPS policy records in trainer JSONL shape."""
    condition_set = {c.strip() for c in conditions if c.strip()}
    if not condition_set:
        raise ValueError("at least one condition is required")
    records: list[dict] = []
    for item in iter_items():
        if not _include_item(item, split, condition_set):
            continue
        letter = ACTION_TO_LETTER[item.gold_scalar]
        prefix = _chat_prefix(item.prompt)
        answer_text = f"{letter}<|im_end|>"
        records.append({
            "mode": f"tps_policy_{item.condition}",
            "track": "track_b_policy",
            "tau_t": float(item.tau_ci_s or 0.0),
            "prefix_text": prefix,
            "answer_text": answer_text,
            "text": prefix + answer_text,
            "answer": letter,
            "action": item.gold_scalar,
            "family": item.family,
            "template_idx": item.template_idx,
            "condition": item.condition,
            "tau_ci_s": item.tau_ci_s,
            "tau_prompt_s": item.tau_prompt_s,
            "held_out_template": item.held_out_template,
            "held_out_family": item.held_out_family,
            "source_item_id": item.item_id,
            "split": split,
        })
    rng = random.Random(seed)
    rng.shuffle(records)
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
        return Path("runs") / args.run_id / "data" / "track_b" / f"tps_train_seed{args.seed}.jsonl"
    raise SystemExit("output scope required: pass --out or --run-id")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=VALID_SPLITS, default="train")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                    help="Comma-separated TPS conditions to include. Default: hidden_only.")
    args = ap.parse_args()
    out = resolve_out(args)
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    records = build_records(split=args.split, seed=args.seed, conditions=conditions)
    n = write_jsonl(out, records)
    print(f"wrote {n} TPS policy records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
