"""TPS forced-choice eval.

Greedy-decode up to N tokens, regex-extract the first letter A/B/C/D.
This works uniformly for CI adapters (which thread tau_t through forward)
and vanilla/prompt adapters (which do not).

Usage:
  uv run python -m eval.tps.run_tps \
    --adapter ci --checkpoint /path/to/v15s_seed0.pt \
    --items runs/tps/data/tps/items.jsonl --out runs/tps/reports/tps/ci_v15s_s0.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Iterable

import torch


LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> str | None:
    m = LETTER_RE.search(text.strip())
    if m:
        return m.group(1)
    # fallback: first alpha char
    for ch in text.strip():
        if ch in "ABCD":
            return ch
    return None


def letter_to_action(letter: str | None) -> str | None:
    return {"A": "REUSE", "B": "REFRESH", "C": "ASK", "D": "SUMMARIZE"}.get(letter or "", None)


def build_adapter(name: str, checkpoint: str | None, base_model: str, max_new_tokens: int):
    from eval.external.adapters.vanilla_adapter import VanillaAdapter
    from eval.external.adapters.prompt_adapter import PromptAdapter
    from eval.external.adapters.ci_adapter import CIAdapter

    if name == "vanilla":
        return VanillaAdapter(base_model=base_model, max_new_tokens=max_new_tokens)
    if name == "prompt":
        return PromptAdapter(base_model=base_model, max_new_tokens=max_new_tokens)
    if name == "ci":
        if not checkpoint:
            raise SystemExit("--adapter ci requires --checkpoint")
        return CIAdapter(base_model=base_model, max_new_tokens=max_new_tokens, checkpoint=checkpoint)
    raise SystemExit(f"unknown adapter {name!r}")


def iter_items_from_path(path: str) -> Iterable[dict]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=["vanilla", "prompt", "ci"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--memory-fraction", type=float, default=0.0,
                    help="if >0, set CUDA per-process memory fraction to avoid Omar contention")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    if args.memory_fraction > 0 and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), 0)
        except Exception as exc:                                # noqa: BLE001
            print(f"WARN: could not cap CUDA memory fraction: {exc}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sweep_id = f"{args.adapter}|{os.path.basename(args.checkpoint or 'base')}"
    print(f"== TPS sweep: {sweep_id} ==", flush=True)

    adapter = build_adapter(args.adapter, args.checkpoint, args.base_model, args.max_new_tokens)
    t0 = time.time()
    adapter.load()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    items = list(iter_items_from_path(args.items))
    if args.limit > 0:
        items = items[: args.limit]

    results: list[dict] = []
    t_eval0 = time.time()
    for idx, item in enumerate(items):
        prompt_text = item["prompt"]
        tau_ci = float(item["tau_ci_s"]) if item["tau_ci_s"] is not None else 0.0
        try:
            text = adapter.generate(prompt_text, tau_seconds=tau_ci)
        except Exception as exc:                                # noqa: BLE001
            text = f"<ERROR: {exc}>"
        letter = parse_letter(text)
        action = letter_to_action(letter)
        results.append({
            "item_id": item["item_id"],
            "family": item["family"],
            "template_idx": item["template_idx"],
            "condition": item["condition"],
            "tau_ci_s": item["tau_ci_s"],
            "tau_prompt_s": item["tau_prompt_s"],
            "gold_scalar": item["gold_scalar"],
            "gold_prompt": item["gold_prompt"],
            "held_out_template": item["held_out_template"],
            "held_out_family": item["held_out_family"],
            "raw_text": text[:200],
            "letter": letter,
            "action": action,
        })
        if (idx + 1) % args.progress_every == 0:
            dt = time.time() - t_eval0
            rate = (idx + 1) / max(dt, 1e-9)
            eta = (len(items) - idx - 1) / max(rate, 1e-9)
            print(f"  [{idx+1}/{len(items)}] {rate:.2f} item/s, eta {eta/60:.1f} min", flush=True)

    out_payload = {
        "sweep_id": sweep_id,
        "adapter": args.adapter,
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "n_items": len(results),
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(out_payload, fh)
    print(f"wrote {len(results)} results to {args.out} (total {time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
