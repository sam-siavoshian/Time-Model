"""Recall check for QwenIPCNv2 checkpoints. Mirrors qwen_recall_check.py
but uses the v2 wrapper (cross-attention injection, differentiable writes,
Identity-V).

The three conditions are identical: WITH memory, WITHOUT memory (zero the
top-tau_write slot), SHUFFLED memory (different conversation's fact).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from model.qwen_ipcn_v2 import QwenIPCNv2, QwenIPCNv2Config, build_qwen_ipcn_v2
from model.qwen_data import gen_nonce_conversation, gen_conversation_with_answer_span


def load_trainable_state(model: QwenIPCNv2, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cur_state = dict(model.named_parameters())
    n = 0
    for name, tensor in state["trainable_state"].items():
        if name in cur_state:
            cur_state[name].data.copy_(tensor.to(cur_state[name].dtype))
            n += 1
    print(f"  loaded {n} trainable tensors")
    mem_state = state.get("memory_state", {})
    for fname, t in mem_state.items():
        if hasattr(model.memory, fname):
            getattr(model.memory, fname).copy_(t)


def feed_to_memory(model: QwenIPCNv2, tokenizer, text: str, chunk_length: int, device: str):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze(0).to(device)
    L = ids.shape[0]
    i = 0
    chunk_idx = 0
    while i < L:
        chunk = ids[i: i + chunk_length]
        if chunk.shape[0] < 8:
            break
        # Pass next-token ids as write targets so Identity-V works.
        end_idx = min(i + chunk_length + 1, L)
        next_ids = ids[i + 1: end_idx]
        targets = torch.full((chunk.shape[0],), -100, dtype=torch.long, device=device)
        if next_ids.numel() > 0:
            targets[: next_ids.shape[0]] = next_ids
        with torch.no_grad():
            _ = model(chunk, tau_t=float(chunk_idx), delta_tau=1.0, write_target_ids=targets)
        i += chunk_length
        chunk_idx += 1
    return chunk_idx


def answer_with_model(
    model: QwenIPCNv2, tokenizer, question: str, max_new_tokens: int = 12,
    device: str = "cuda", tau_t: float = 100.0,
) -> str:
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
    last_user_idx = text.rfind("<|im_start|>user\n")
    last_user = text[last_user_idx + len("<|im_start|>user\n"):]
    last_user = last_user.split("<|im_end|>")[0]
    fact_prefix = text[:last_user_idx]
    return fact_prefix, last_user


def _normalize(s: str) -> str:
    return s.strip().lower().rstrip(".")


def run_recall(model: QwenIPCNv2, tokenizer, n: int, chunk_length: int, device: str,
               seed: int = 1234, use_nonce: bool = True):
    rng = random.Random(seed)
    with_correct = 0
    without_correct = 0
    shuffled_correct = 0
    total = 0
    for i in range(n):
        nd = rng.randint(8, 16)
        if use_nonce:
            text, answer, _, _ = gen_nonce_conversation(rng, nd)
        else:
            text, answer, _, _ = gen_conversation_with_answer_span(rng, nd)
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
            if model.memory._k_grad is not None:
                model.memory._k_grad = model.memory._k_grad.clone()
                model.memory._v_grad = model.memory._v_grad.clone()
                model.memory._k_grad[primary].zero_()
                model.memory._v_grad[primary].zero_()
        ans_without = answer_with_model(model, tokenizer, recall_q, device=device)

        # SHUFFLED MEMORY
        rng2 = random.Random(seed + 10000 + i)
        if use_nonce:
            other_text, _, _, _ = gen_nonce_conversation(rng2, nd)
        else:
            other_text, _, _, _ = gen_conversation_with_answer_span(rng2, nd)
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
            print(f"  [{i}] gold={answer!r:25s} | with={ans_with!r:32s} | without={ans_without!r:32s} | shuf={ans_shuffled!r}")
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
    p.add_argument("--chunk-length", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="reports/qwen_recall_v2.json")
    p.add_argument("--no-nonce", action="store_true",
                   help="use legacy pool values instead of nonce; for "
                        "comparing with old eval baselines.")
    args = p.parse_args()

    cfg = QwenIPCNv2Config()
    cfg.chunk_length = args.chunk_length
    print(f"Loading Qwen IPCN v2 ({cfg.base_model_name})...")
    model = build_qwen_ipcn_v2(cfg)
    model = model.to(args.device)
    if args.checkpoint:
        print(f"Loading trainable state from {args.checkpoint}...")
        load_trainable_state(model, args.checkpoint)
    model.train(False)
    print(f"Running recall check on {args.n} fresh conversations (nonce={not args.no_nonce})...")
    t0 = time.time()
    r = run_recall(model, model.tokenizer, args.n, args.chunk_length, args.device,
                   use_nonce=not args.no_nonce)
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
