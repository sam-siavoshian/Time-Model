"""Memory recall accuracy check for Qwen IPCN.

Generates fresh memorize-recall conversations (different from training),
runs them through the model, and measures: at the recall question,
does the assistant produce the correct answer?

Three conditions per example:
  1. WITH MEMORY: model has full memory bank populated from the fact +
     distractor chunks, then we feed only the recall question chunk
     (NO fact in this chunk). Memory is the only signal.
  2. WITHOUT MEMORY (ablation): same as above, but we zero the slot
     with highest tau_write before the recall question chunk. If memory
     was the signal, recall accuracy drops here.
  3. SHUFFLED MEMORY: rebuild the bank with a different conversation's
     memories before recall. Memory pointing the wrong way; recall
     should drop more.

The KEY metric: Acc(with_memory) - Acc(without_memory). Track A had this
at zero (model didn't use memory). For Track B to demonstrate the paper
claim, this delta must be > 0.

Usage:
  uv run python -m model.qwen_recall_check --checkpoint checkpoints/qwen_ipcn.pt --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from model.qwen_ipcn import QwenIPCN, QwenIPCNConfig, build_qwen_ipcn
from model.qwen_data import gen_conversation


def load_trainable_state(model: QwenIPCN, ckpt_path: str):
    """Load only the trainable params + memory bank state from a saved
    qwen_train checkpoint. The frozen base is loaded fresh from HF.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cur_state = dict(model.named_parameters())
    n_loaded = 0
    for name, tensor in state["trainable_state"].items():
        if name in cur_state:
            cur_state[name].data.copy_(tensor.to(cur_state[name].dtype))
            n_loaded += 1
    print(f"  loaded {n_loaded} trainable tensors from {ckpt_path}")
    mem_state = state.get("memory_state", {})
    for fname, t in mem_state.items():
        if hasattr(model.memory, fname):
            getattr(model.memory, fname).copy_(t)


def feed_to_memory(model: QwenIPCN, tokenizer, text: str, chunk_length: int, device: str):
    """Tokenize text, split into chunks, feed each chunk through model so
    the memory bank populates. The recall question is fed SEPARATELY.
    """
    ids = tokenizer.encode(text, return_tensors="pt").squeeze(0).to(device)
    L = ids.shape[0]
    i = 0
    chunk_idx = 0
    while i < L:
        chunk = ids[i: i + chunk_length]
        if chunk.shape[0] < 8:
            break
        with torch.no_grad():
            _ = model(chunk, tau_t=float(chunk_idx), delta_tau=1.0)
        i += chunk_length
        chunk_idx += 1
    return chunk_idx


def answer_with_model(
    model: QwenIPCN, tokenizer, question: str, max_new_tokens: int = 12,
    device: str = "cuda", tau_t: float = 100.0,
) -> str:
    """Feed the recall question (in Qwen chat format), greedy-decode up to
    max_new_tokens, return decoded string.
    """
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    generated = []
    cur = ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(cur, tau_t=tau_t, delta_tau=1.0)
        logits = out["logits"]
        next_id = int(logits[-1].argmax().item())
        if next_id == tokenizer.eos_token_id:
            break
        im_end_id = (
            tokenizer.convert_tokens_to_ids("<|im_end|>")
            if hasattr(tokenizer, "convert_tokens_to_ids") else None
        )
        if im_end_id is not None and next_id == im_end_id:
            break
        generated.append(next_id)
        cur = torch.cat([cur, torch.tensor([next_id], device=device)])
    return tokenizer.decode(generated)


def split_fact_from_question(text: str) -> tuple[str, str]:
    """Take a full conversation rendered text and split into:
       fact_prefix: everything up to the LAST user turn (the recall question)
       recall_question: the last user turn content
    """
    parts = text.split("<|im_start|>user\n")
    last_user = parts[-1]
    last_user = last_user.split("<|im_end|>")[0]
    fact_prefix = text[: text.rfind("<|im_start|>user\n")]
    return fact_prefix, last_user


def _normalize(s: str) -> str:
    return s.strip().lower().rstrip(".")


def run_recall(model: QwenIPCN, tokenizer, n: int, chunk_length: int, device: str, seed: int = 1234):
    """Returns dict of metrics."""
    rng = random.Random(seed)
    with_correct = 0
    without_correct = 0
    shuffled_correct = 0
    total = 0
    for i in range(n):
        nd = rng.randint(6, 12)
        text, answer = gen_conversation(rng, nd)
        fact_prefix, recall_q = split_fact_from_question(text)

        # WITH MEMORY
        model.reset_memory()
        feed_to_memory(model, tokenizer, fact_prefix, chunk_length, device)
        ans_with = answer_with_model(model, tokenizer, recall_q, device=device)

        # WITHOUT MEMORY (ablate primary slot)
        model.reset_memory()
        feed_to_memory(model, tokenizer, fact_prefix, chunk_length, device)
        with torch.no_grad():
            primary = int(model.memory.tau_write.argmax().item())
            model.memory.k[primary].zero_()
            model.memory.v[primary].zero_()
        ans_without = answer_with_model(model, tokenizer, recall_q, device=device)

        # SHUFFLED MEMORY (different conversation's fact)
        rng2 = random.Random(seed + 10000 + i)
        other_text, _ = gen_conversation(rng2, nd)
        other_prefix, _ = split_fact_from_question(other_text)
        model.reset_memory()
        feed_to_memory(model, tokenizer, other_prefix, chunk_length, device)
        ans_shuffled = answer_with_model(model, tokenizer, recall_q, device=device)

        a = _normalize(answer)
        with_correct += int(a in _normalize(ans_with))
        without_correct += int(a in _normalize(ans_without))
        shuffled_correct += int(a in _normalize(ans_shuffled))
        total += 1
        if i < 5:
            print(f"  [{i}] gold={answer!r:18s} | with={ans_with!r:32s} | without={ans_without!r:32s} | shuf={ans_shuffled!r}")

    return {
        "n": total,
        "acc_with_memory": with_correct / total,
        "acc_without_memory": without_correct / total,
        "acc_shuffled_memory": shuffled_correct / total,
        "delta_with_minus_without": (with_correct - without_correct) / total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--chunk-length", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/qwen_recall.json")
    args = p.parse_args()

    cfg = QwenIPCNConfig()
    cfg.chunk_length = args.chunk_length
    print(f"Loading Qwen IPCN ({cfg.base_model_name})...")
    model = build_qwen_ipcn(cfg)
    model = model.to(args.device)
    if args.checkpoint:
        print(f"Loading trainable state from {args.checkpoint}...")
        load_trainable_state(model, args.checkpoint)
    model.train(False)
    print(f"Running recall check on {args.n} fresh conversations...")
    t0 = time.time()
    r = run_recall(model, model.tokenizer, args.n, args.chunk_length, args.device)
    print(f"\nResults (n={r['n']}):")
    print(f"  acc WITH memory:      {r['acc_with_memory']:.3f}")
    print(f"  acc WITHOUT memory:   {r['acc_without_memory']:.3f}")
    print(f"  acc SHUFFLED memory:  {r['acc_shuffled_memory']:.3f}")
    print(f"  delta (with - without): {r['delta_with_minus_without']:+.3f}")
    print(f"  wall: {time.time() - t0:.1f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
