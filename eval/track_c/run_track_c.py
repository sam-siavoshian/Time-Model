"""Track C forced-choice evaluator."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import torch

from eval.track_c.generate import label_for_state, predicate_signature_for_action, render_model_prompt
from eval.track_c.schemas import LETTERS, TrackCPrediction


LETTER_RE = re.compile(r"\b([ABCD])\b")
CONDITIONS = (
    "ci_hidden_time",
    "no_time_control",
    "shuffled_time_control",
    "prompt_timestamp",
    "both_agree",
    "conflict",
)


def parse_letter(text: str) -> str | None:
    m = LETTER_RE.search(text.strip())
    if m:
        return m.group(1)
    for ch in text.strip():
        if ch in LETTERS:
            return ch
    return None


def iter_items(path: str) -> Iterable[dict]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def portable_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    root = Path(__file__).resolve().parents[2]
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.name


def build_adapter(name: str, checkpoint: str | None, base_model: str, max_new_tokens: int, chunk_length: int):
    from eval.external.adapters.ci_adapter import CIAdapter
    from eval.external.adapters.vanilla_adapter import VanillaAdapter

    if name == "vanilla":
        return VanillaAdapter(base_model=base_model, max_new_tokens=max_new_tokens)
    if name == "ci":
        if not checkpoint:
            raise SystemExit("--adapter ci requires --checkpoint")
        return CIAdapter(
            base_model=base_model,
            max_new_tokens=max_new_tokens,
            checkpoint=checkpoint,
            chunk_length=chunk_length,
        )
    raise SystemExit(f"unknown adapter {name!r}")


@torch.no_grad()
def score_letters_logprob(adapter, prompt: str, tau_seconds: float) -> tuple[str | None, dict[str, float]]:
    if not getattr(adapter, "_loaded", False):
        raise RuntimeError("call load() before scoring")
    wrapped = adapter.construct_prompt(prompt, tau_seconds)
    ids = adapter.tokenizer.encode(wrapped, return_tensors="pt").to(adapter.device)
    if ids.dim() == 2 and adapter.name == "ci":
        out = adapter.model(ids.squeeze(0), tau_t=float(tau_seconds))
    else:
        out = adapter.model(ids)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    if logits.dim() == 3:
        logits = logits[0]
    next_logprobs = torch.log_softmax(logits[-1].float(), dim=-1)
    scores: dict[str, float] = {}
    for letter in LETTERS:
        variants = []
        for piece in (letter, f" {letter}"):
            token_ids = adapter.tokenizer.encode(piece, add_special_tokens=False)
            if token_ids:
                variants.append(float(next_logprobs[int(token_ids[0])].item()))
        scores[letter] = max(variants) if variants else float("-inf")
    return max(scores, key=scores.get), scores


def recompute_action(item: dict, tau_seconds: int) -> str:
    action, _, _, _ = label_for_state(
        item["family"],
        item["hidden_state_json"],
        int(tau_seconds),
        item.get("active_constraints") or [],
    )
    return action


def shifted_tau(items: list[dict], idx: int) -> int:
    current = int(items[idx]["tau_seconds"])
    for offset in range(1, len(items) + 1):
        candidate = int(items[(idx + offset) % len(items)]["tau_seconds"])
        if candidate != current:
            return candidate
    return current


def condition_inputs(item: dict, items: list[dict], idx: int, condition: str) -> tuple[str, int, int | None, int | None, str | None]:
    tau = int(item["tau_seconds"])
    shuffled = shifted_tau(items, idx)
    if condition == "ci_hidden_time":
        return render_model_prompt(item), tau, None, None, None
    if condition == "no_time_control":
        return render_model_prompt(item), 0, None, None, None
    if condition == "shuffled_time_control":
        return render_model_prompt(item), shuffled, None, shuffled, recompute_action(item, shuffled)
    if condition == "prompt_timestamp":
        return render_model_prompt(item, elapsed_seconds_text=tau), 0, tau, None, None
    if condition == "both_agree":
        return render_model_prompt(item, elapsed_seconds_text=tau), tau, tau, None, None
    if condition == "conflict":
        return render_model_prompt(item, elapsed_seconds_text=shuffled), tau, shuffled, None, recompute_action(item, shuffled)
    raise ValueError(f"unknown Track C condition {condition!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=("vanilla", "ci"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--chunk-length", type=int, default=512)
    ap.add_argument("--scoring", choices=("generate", "logprob"), default="logprob")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model_tag = args.model_tag or f"{args.adapter}_{Path(args.checkpoint or 'base').stem}"
    adapter = build_adapter(args.adapter, args.checkpoint, args.base_model, args.max_new_tokens, args.chunk_length)
    t0 = time.time()
    adapter.load()
    items = list(iter_items(args.items))
    if args.limit > 0:
        items = items[: args.limit]
    results: list[dict] = []
    for idx, item in enumerate(items):
        prompt, tau_forward, tau_prompt, shuffled_tau, alternate_gold = condition_inputs(item, items, idx, args.condition)
        letter_scores: dict[str, float] | None = None
        if args.scoring == "logprob":
            try:
                letter, letter_scores = score_letters_logprob(adapter, prompt, tau_forward)
                raw_text = f"<LOGPROB:{letter}>"
            except Exception as exc:  # noqa: BLE001
                letter = None
                raw_text = f"<ERROR: {exc}>"
        else:
            try:
                raw_text = adapter.generate(prompt, tau_forward)
            except Exception as exc:  # noqa: BLE001
                raw_text = f"<ERROR: {exc}>"
            letter = parse_letter(raw_text)
        action = item["choice_actions"].get(letter or "")
        predicted_predicates = (
            predicate_signature_for_action(
                item["family"],
                item["hidden_state_json"],
                int(item["tau_seconds"]),
                item.get("active_constraints") or [],
                action,
            )
            if action
            else None
        )
        gold_prompt_action = alternate_gold if args.condition == "conflict" else None
        shuffled_gold_action = alternate_gold if args.condition == "shuffled_time_control" else None
        results.append(TrackCPrediction(
            item_id=item["id"],
            seed=int(item["seed"]),
            split=item["split"],
            family=item["family"],
            condition=args.condition,
            template_id=int(item["template_id"]),
            state_id=int(item["state_id"]),
            tau_seconds=int(item["tau_seconds"]),
            tau_forward_seconds=int(tau_forward),
            tau_prompt_seconds=tau_prompt,
            shuffled_tau_seconds=shuffled_tau,
            gold_letter=item["gold_letter"],
            gold_action=item["gold_action"],
            gold_prompt_action=gold_prompt_action,
            shuffled_gold_action=shuffled_gold_action,
            letter=letter,
            action=action,
            letter_scores=letter_scores,
            raw_text=raw_text[:240],
            num_constraints=int(item["num_constraints"]),
            choice_actions=item["choice_actions"],
            hidden_state_json=item["hidden_state_json"],
            updated_state_json=item["updated_state_json"],
            active_constraints=item.get("active_constraints") or [],
            gold_predicates=item["gold_predicates"],
            predicted_predicates=predicted_predicates,
        ).to_dict())
        if (idx + 1) % args.progress_every == 0:
            print(f"  [{idx + 1}/{len(items)}] {args.condition}", flush=True)
    payload = {
        "track": "C",
        "sweep_id": f"{model_tag}|{args.condition}|{Path(args.items).stem}",
        "model_tag": model_tag,
        "adapter": args.adapter,
        "checkpoint": portable_path(args.checkpoint),
        "base_model": args.base_model,
        "condition": args.condition,
        "items": portable_path(args.items),
        "scoring": args.scoring,
        "n_items": len(results),
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {len(results)} Track C predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
